import argparse
import json
import os
import re
import shutil
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "allenai/Olmo-3-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistralai/Mistral-Nemo-Instruct-2407",
]

DATASETS = ["mmlu", "piqa", "openbookqa", "strategyqa", "truthfulqa"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "auto"
PROMPT_MAX_TOKENS = 2048
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
SHUFFLE_SEED = 42
BATCH_SIZE = 8
DEBUG_SAVE_LIMIT = 40
TOPK_SCORES_SAVE = 10
RAW_AUDIT_LIMIT = 5
SELF_CONSISTENCY_SAMPLES = 5
SELF_CONSISTENCY_TEMPERATURE = 0.7
SELF_CONSISTENCY_TOP_P = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate instruct models on mmlu, piqa, openbookqa, strategyqa, and truthfulqa.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Model names to evaluate.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="Datasets to evaluate.")
    parser.add_argument("--samples-per-dataset", type=int, default=500, help="Maximum examples per dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for generation.")
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=SELF_CONSISTENCY_SAMPLES,
        help="Number of sampled generations per prompt for self-consistency voting.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=SELF_CONSISTENCY_TEMPERATURE,
        help="Sampling temperature for self-consistency.",
    )
    parser.add_argument(
        "--self-consistency-top-p",
        type=float,
        default=SELF_CONSISTENCY_TOP_P,
        help="Top-p for self-consistency.",
    )
    parser.add_argument("--output-dir", default="results_json", help="Directory for JSON outputs.")
    parser.add_argument("--mmlu-config", default="all", help="Config/split subset for cais/mmlu.")
    parser.add_argument("--math-split", default="test", help="Split to use for math dataset.")
    parser.add_argument("--piqa-split", default="validation", help="Split to use for PIQA.")
    return parser.parse_args()


def norm(text: Optional[str]) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;!\"'`()[]{}")


def build_chat_prompt(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return user_text


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def extract_last_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def extract_math_answer(text: str) -> Optional[str]:
    if not text:
        return None

    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    final_markers = [
        r"final answer\s*[:\-]\s*(.+)",
        r"answer\s*[:\-]\s*(.+)",
    ]
    for pattern in final_markers:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def normalize_math_answer(text: Optional[str]) -> str:
    value = norm(text)
    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.replace("\\!", "")
    value = value.replace("\\,", "")
    value = value.replace("\\%", "%")
    value = re.sub(r"^\s*the answer is\s*", "", value)
    value = re.sub(r"^\s*final answer\s*[:\-]?\s*", "", value)
    value = re.sub(r"^\s*=+\s*", "", value)
    value = re.sub(r"\\left", "", value)
    value = re.sub(r"\\right", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    return value


def extract_last_line_payload(text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "", "empty_output"
    last = lines[-1]
    match = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", last)
    if match:
        return match.group(1).strip(), "last_line_final_answer"
    return last, "last_line"


def extract_final_answer_anywhere(text: str) -> Tuple[Optional[str], str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", line)
        if match:
            return match.group(1).strip(), "full_output_final_answer"
    return None, "no_final_answer"


def strip_prompt_echo(text: str) -> str:
    out = (text or "").strip()

    # Remove common chat-template markers that some models echo back.
    replacements = [
        "<｜User｜>",
        "<｜Assistant｜>",
        "<|user|>",
        "<|assistant|>",
        "<think>",
        "</think>",
    ]
    for token in replacements:
        out = out.replace(token, "")

    lines = [line.strip() for line in out.splitlines()]
    cleaned_lines = []
    drop_prefixes = (
        "user",
        "assistant",
        "model",
    )
    for line in lines:
        if not line:
            continue
        lowered = line.lower().strip()
        if lowered in drop_prefixes:
            continue
        if lowered.startswith("user\n") or lowered.startswith("assistant\n") or lowered.startswith("model\n"):
            continue
        cleaned_lines.append(line)

    out = "\n".join(cleaned_lines).strip()

    lowered = out.lower()
    anchor_patterns = [
        "final answer:",
        "the correct answer is",
        "the more plausible solution is",
        "answer:",
    ]
    anchor_positions = [lowered.find(pattern) for pattern in anchor_patterns if lowered.find(pattern) != -1]
    if anchor_positions:
        out = out[min(anchor_positions):].strip()

    # If the model echoed a long prompt and ended with a clean letter/yes/no line, keep the tail.
    tail_lines = [line.strip() for line in out.splitlines() if line.strip()]
    if tail_lines:
        last = tail_lines[-1]
        if re.match(r"(?is)^(final\s*answer\s*[:\-]\s*)?(yes|no|[A-F])[\.\):;!,]?\s*$", last):
            out = last.strip()

    return out


def extract_text_answer_strict(text: str) -> Tuple[Optional[str], str]:
    payload_any, src_any = extract_final_answer_anywhere(text)
    if payload_any is not None:
        normalized = norm(payload_any)
        if normalized:
            return normalized, f"strict_{src_any}_text"

    payload_last, src_last = extract_last_line_payload(text)
    normalized_last = norm(payload_last)
    if normalized_last:
        return normalized_last, f"strict_{src_last}_text"

    return None, "unparsed_none_text_strict"


def label_aliases_for_eval(labels: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for label in labels:
        key = label.lower()
        aliases[key] = label
        aliases[key.replace("_", " ")] = label
        aliases[key.replace("_", "-")] = label
    return aliases


def extract_categorical_strict(text: str, labels: Sequence[str]) -> Tuple[Optional[str], str]:
    aliases = label_aliases_for_eval(labels)

    payload_last, src_last = extract_last_line_payload(text)
    norm_last = norm(payload_last)
    if norm_last in aliases:
        return aliases[norm_last], f"{src_last}_exact"

    payload_any, src_any = extract_final_answer_anywhere(text)
    if payload_any is not None:
        norm_any = norm(payload_any)
        if norm_any in aliases:
            return aliases[norm_any], f"{src_any}_exact"

    return None, "unparsed_none_categorical_strict"


def extract_choice_answer_strict(text: str, valid_choices: Sequence[str]) -> Tuple[Optional[str], str]:
    out = text or ""
    valid = "".join(dict.fromkeys(choice.upper() for choice in valid_choices))

    def normalize_letter(candidate: str) -> Optional[str]:
        cleaned = (candidate or "").strip().upper().rstrip(".):;!,")
        return cleaned if cleaned in set(valid) else None

    def last_valid_from_pattern(pattern: str) -> Optional[str]:
        matches = re.findall(pattern, out, flags=re.I | re.S)
        if not matches:
            return None
        choice = matches[-1]
        if isinstance(choice, tuple):
            choice = choice[-1]
        normalized = normalize_letter(choice)
        return normalized.lower() if normalized is not None else None

    payload_any, src_any = extract_final_answer_anywhere(out)
    if payload_any is not None:
        payload = payload_any.strip()

        match = re.match(rf"(?is)^\s*([{valid}])\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_bare_letter"

        match = re.match(rf"(?is)^\s*([{valid}])[\.\):;!,]?\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_bare_letter_punct"

        match = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_option_letter"

        match = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])[\.\):;!,]?\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_option_letter_punct"

        match = re.match(rf"(?is)^\s*\\boxed\{{\s*([{valid}])\s*\}}\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_boxed_letter"

        match = re.match(rf"(?is)^\s*(?:answer\s+is|correct\s+answer\s+is|i\s+choose)\s*([{valid}])[\.\):;!,]?\s*$", payload)
        if match:
            return match.group(1).lower(), f"strict_{src_any}_answer_is_letter"

    boxed = re.findall(rf"\\boxed\{{\s*([{valid}])\s*\}}", out, flags=re.I)
    if boxed:
        return boxed[-1].lower(), "strict_boxed_letter"

    payload_last, src_last = extract_last_line_payload(out)
    payload = payload_last.strip()

    match = re.match(rf"(?is)^\s*([{valid}])\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_bare_letter"

    match = re.match(rf"(?is)^\s*([{valid}])[\.\):;!,]?\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_bare_letter_punct"

    match = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_option_letter"

    match = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])[\.\):;!,]?\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_option_letter_punct"

    match = re.match(rf"(?is)^\s*(?:answer\s+is|correct\s+answer\s+is|i\s+choose)\s*([{valid}])\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_answer_is_letter"

    match = re.match(rf"(?is)^\s*(?:answer\s+is|correct\s+answer\s+is|i\s+choose)\s*([{valid}])[\.\):;!,]?\s*$", payload)
    if match:
        return match.group(1).lower(), f"strict_{src_last}_answer_is_letter_punct"

    # Fallbacks for models that ignore the one-line format but still state an answer clearly.
    explicit_patterns = [
        ("fallback_correct_answer_is", rf"(?is)correct\s+answer\s+is\s+([{valid}])[\.\):;!,]?\b"),
        ("fallback_more_plausible_is", rf"(?is)more\s+plausible\s+solution\s+is[:\s]+([{valid}])[\.\):;!,]?\b"),
        ("fallback_i_choose", rf"(?is)i\s+(?:would\s+)?choose\s+([{valid}])[\.\):;!,]?\b"),
        ("fallback_answer_colon", rf"(?is)answer\s*[:\-]\s*([{valid}])[\.\):;!,]?\b"),
    ]
    for rule_name, pattern in explicit_patterns:
        choice = last_valid_from_pattern(pattern)
        if choice is not None:
            return choice, rule_name

    return None, "unparsed_none_mcq_strict"


def load_first_available(candidates: Sequence[Tuple[str, Optional[str], str]]) -> Dataset:
    last_error = None
    for path, config, split in candidates:
        try:
            return load_dataset(path, config, split=split)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load dataset from candidates: {candidates}") from last_error


def load_math_dataset(split: str) -> Dataset:
    return load_dataset("hendrycks/competition_math", split=split)


def load_mmlu_dataset(config: str) -> Dataset:
    candidates = [
        ("cais/mmlu", config, "test"),
        ("hendrycks_test", config, "test"),
    ]
    return load_first_available(candidates)


def load_piqa_dataset(split: str) -> Dataset:
    return load_dataset("piqa", split=split, trust_remote_code=True)


def load_openbookqa_dataset() -> Dataset:
    return load_dataset("openbookqa", "main", split="test")


def load_strategyqa_dataset() -> Dataset:
    return load_dataset("tasksource/strategy-qa", split="train")


def load_truthfulqa_dataset() -> Dataset:
    return load_dataset("truthful_qa", "multiple_choice", split="validation")


def make_batch_inputs(tokenizer, prompts: Sequence[str]):
    if hasattr(tokenizer, "apply_chat_template"):
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=PROMPT_MAX_TOKENS,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=PROMPT_MAX_TOKENS,
        )
    inputs = inputs.to(DEVICE)
    if "attention_mask" in inputs:
        input_lengths = inputs["attention_mask"].sum(dim=1)
    else:
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    return inputs, input_lengths


def decode_batch_outputs(tokenizer, outputs, input_lengths) -> List[str]:
    decoded = []
    for i in range(outputs.size(0)):
        start = int(input_lengths[i].item()) if input_lengths is not None else 0
        gen_tokens = outputs[i, start:]
        decoded.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return decoded


def generate_batch_oom_safe(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    do_sample: bool = DO_SAMPLE,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    results: List[str] = []
    batch_size = len(prompts)
    index = 0
    while index < len(prompts):
        chunk = prompts[index : index + batch_size]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk)
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    generation_kwargs["temperature"] = temperature if temperature is not None else 1.0
                    generation_kwargs["top_p"] = top_p if top_p is not None else 1.0
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            results.extend(decode_batch_outputs(tokenizer, outputs, input_lengths))
            index += batch_size
        except torch.cuda.OutOfMemoryError:
            if DEVICE != "cuda":
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            if batch_size < 1:
                raise RuntimeError("OOM even with batch_size=1. Reduce batch size or max_new_tokens.")
            print(f"  [OOM] reducing sub-batch size -> {batch_size} and retrying...")
    return results


def prepare_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def clear_hf_cache() -> None:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for subdir in ("hub", "transformers", "modules", "xet"):
        path = os.path.join(hf_home, subdir)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def build_piqa_item(sample: Dict) -> Tuple[str, str, str]:
    prompt = (
        "Choose the more plausible solution.\n"
        f"Goal: {sample['goal']}\n"
        f"A. {sample['sol1']}\n"
        f"B. {sample['sol2']}\n"
        "Return exactly one line: Final answer: <letter>\n"
        "The letter must be A or B."
    )
    gold = "a" if int(sample["label"]) == 0 else "b"
    preview = sample["goal"]
    return prompt, gold, preview


def build_mmlu_item(sample: Dict) -> Tuple[str, str, str]:
    choices = sample["choices"]
    prompt = (
        "Answer the multiple-choice question.\n"
        f"Question: {sample['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Return exactly one line: Final answer: <letter>\n"
        "The letter must be A, B, C, or D."
    )
    answer = sample["answer"]
    if isinstance(answer, int):
        gold = ["a", "b", "c", "d"][answer]
    else:
        gold = norm(answer)
    preview = sample["question"]
    return prompt, gold, preview


def build_openbookqa_item(sample: Dict) -> Tuple[str, str, str]:
    choices = sample["choices"]
    labels = choices["label"]
    texts = choices["text"]
    prompt = (
        "Answer the science question.\n"
        f"Question: {sample['question_stem']}\n"
        + "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        + "\nReturn exactly one line: Final answer: <letter>\n"
        + "The letter must be one of the listed options."
    )
    gold = sample["answerKey"].strip().lower()
    preview = sample["question_stem"]
    return prompt, gold, preview


def build_strategyqa_item(sample: Dict) -> Tuple[str, str, str]:
    prompt = (
        "Answer the question with yes or no.\n"
        f"Question: {sample['question']}\n"
        "Return exactly one line: Final answer: <yes or no>"
    )
    gold = "yes" if bool(sample["answer"]) else "no"
    preview = sample["question"]
    return prompt, gold, preview


def build_truthfulqa_item(sample: Dict) -> Tuple[str, str, str]:
    targets = sample["mc1_targets"]
    choices = targets["choices"]
    labels = targets["labels"]
    gold_index = labels.index(1)
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    prompt = (
        "Answer the multiple-choice question truthfully.\n"
        f"Question: {sample['question']}\n"
        + "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        + "\nReturn exactly one line: Final answer: <letter>\n"
        + "The letter must be one of the listed options."
    )
    gold = letters[gold_index].lower()
    preview = sample["question"]
    return prompt, gold, preview


def evaluate_math(model, tokenizer, sample: Dict) -> Tuple[Optional[str], str, Optional[List[Tuple[str, float]]], str, str, str, Optional[str]]:
    problem = sample.get("problem") or sample.get("question") or sample.get("input") or ""
    solution = sample.get("solution") or sample.get("answer") or sample.get("output") or ""
    prompt = (
        "Solve the math problem carefully. End with the final answer only.\n"
        f"Problem:\n{problem}\n"
        "Final answer:"
    )
    prompt_text = build_chat_prompt(tokenizer, prompt)
    raw_output = generate_text(model, tokenizer, prompt_text)
    prediction = extract_math_answer(raw_output)
    gold = normalize_math_answer(extract_math_answer(str(solution)) or str(solution))
    preview = problem[:400]
    if prediction is not None:
        prediction = normalize_math_answer(prediction)
    return prediction, gold, None, preview, prompt_text, raw_output, "math_answer_extraction"


def self_consistency_vote(
    raw_outputs: Sequence[str],
    parser_fn,
) -> Tuple[Optional[str], str, List[Dict[str, Optional[str]]]]:
    parsed_runs: List[Dict[str, Optional[str]]] = []
    parsed_predictions: List[str] = []

    for raw_output in raw_outputs:
        cleaned_output = strip_prompt_echo(raw_output)
        prediction, parsed_from = parser_fn(cleaned_output)
        parsed_runs.append(
            {
                "prediction": prediction,
                "parsed_from": parsed_from,
                "raw_generation": cleaned_output,
            }
        )
        if prediction is not None:
            parsed_predictions.append(prediction)

    if not parsed_predictions:
        return None, "self_consistency_all_unparsed", parsed_runs

    counts = Counter(parsed_predictions)
    best_count = max(counts.values())
    voted_prediction = None
    for prediction in parsed_predictions:
        if counts[prediction] == best_count:
            voted_prediction = prediction
            break

    vote_summary = ",".join(f"{label}:{counts[label]}" for label, _ in counts.most_common())
    return voted_prediction, f"self_consistency_vote[{vote_summary}]", parsed_runs


def run_dataset_eval(model, tokenizer, dataset_name: str, samples_per_dataset: int, args: argparse.Namespace) -> Dict:
    if dataset_name == "piqa":
        ds = load_piqa_dataset(args.piqa_split)
        builder = build_piqa_item
        valid_choices = ["A", "B"]
        parser_fn = lambda text: extract_choice_answer_strict(text, valid_choices)
    elif dataset_name == "mmlu":
        ds = load_mmlu_dataset(args.mmlu_config)
        builder = build_mmlu_item
        valid_choices = ["A", "B", "C", "D"]
        parser_fn = lambda text: extract_choice_answer_strict(text, valid_choices)
    elif dataset_name == "openbookqa":
        ds = load_openbookqa_dataset()
        builder = build_openbookqa_item
        valid_choices = ["A", "B", "C", "D"]
        parser_fn = lambda text: extract_choice_answer_strict(text, valid_choices)
    elif dataset_name == "strategyqa":
        ds = load_strategyqa_dataset()
        builder = build_strategyqa_item
        valid_choices = None
        parser_fn = lambda text: extract_categorical_strict(text, ["yes", "no"])
    elif dataset_name == "truthfulqa":
        ds = load_truthfulqa_dataset()
        builder = build_truthfulqa_item
        valid_choices = ["A", "B", "C", "D", "E", "F"]
        parser_fn = lambda text: extract_choice_answer_strict(text, valid_choices)
    elif dataset_name == "math":
        ds = load_math_dataset(args.math_split)
        builder = None
        valid_choices = None
        parser_fn = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    sample_count = min(samples_per_dataset, len(ds))
    ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(sample_count))

    correct = 0
    total = 0
    none_predictions = 0
    parsed_count = 0
    correct_parsed = 0
    debug_examples = []
    audit_examples = []

    if dataset_name == "math":
        for idx in range(sample_count):
            sample = ds[idx]
            prediction, gold, scored_sorted, preview, prompt_text, raw_output, parsed_from = evaluate_math(model, tokenizer, sample)
            total += 1
            if not prediction:
                none_predictions += 1
            else:
                parsed_count += 1
            if len(audit_examples) < RAW_AUDIT_LIMIT:
                audit_item = {
                    "idx": idx,
                    "gold": gold,
                    "pred": prediction,
                    "parsed_from": parsed_from,
                    "prompt_preview": prompt_text[:800],
                    "raw_generation": raw_output[:800],
                    "text_preview": preview,
                }
                audit_examples.append(audit_item)
                print(f"  audit sample {idx}: gold={gold} pred={prediction} parsed_from={parsed_from}")
                print(f"  raw_generation: {raw_output[:200]!r}")
            if prediction == gold:
                correct += 1
                if prediction is not None:
                    correct_parsed += 1
            elif len(debug_examples) < DEBUG_SAVE_LIMIT:
                debug_examples.append({
                    "idx": idx,
                    "gold": gold,
                    "pred": prediction,
                    "parsed_from": parsed_from,
                    "text_preview": preview,
                    "raw_generation": raw_output[:400],
                })
            if (idx + 1) % 25 == 0 or idx + 1 == sample_count:
                print(f"  processed {idx + 1}/{sample_count}")
    else:
        for start in range(0, sample_count, args.batch_size):
            end = min(start + args.batch_size, sample_count)
            batch_samples = [ds[i] for i in range(start, end)]
            built = [builder(sample) for sample in batch_samples]
            prompts = [build_chat_prompt(tokenizer, item[0]) for item in built]
            golds = [item[1] for item in built]
            previews = [item[2] for item in built]
            sc_samples = max(1, args.self_consistency_samples)

            if sc_samples == 1:
                batch_outputs = [
                    [raw_output]
                    for raw_output in generate_batch_oom_safe(
                        model,
                        tokenizer,
                        prompts,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                ]
            else:
                expanded_prompts: List[str] = []
                for prompt in prompts:
                    expanded_prompts.extend([prompt] * sc_samples)
                flat_outputs = generate_batch_oom_safe(
                    model,
                    tokenizer,
                    expanded_prompts,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=args.self_consistency_temperature,
                    top_p=args.self_consistency_top_p,
                )
                batch_outputs = [
                    flat_outputs[i * sc_samples : (i + 1) * sc_samples]
                    for i in range(len(prompts))
                ]

            for idx, gold, preview, prompt_text, raw_outputs in zip(range(start, end), golds, previews, prompts, batch_outputs):
                prediction, parsed_from, parsed_runs = self_consistency_vote(raw_outputs, parser_fn)
                display_output = parsed_runs[0]["raw_generation"] if parsed_runs else ""
                total += 1
                if not prediction:
                    none_predictions += 1
                else:
                    parsed_count += 1

                if len(audit_examples) < RAW_AUDIT_LIMIT:
                    audit_item = {
                        "idx": idx,
                        "gold": gold,
                        "pred": prediction,
                        "parsed_from": parsed_from,
                        "prompt_preview": prompt_text[:800],
                        "raw_generation": display_output[:800],
                        "text_preview": preview,
                        "self_consistency_runs": [
                            {
                                "prediction": run["prediction"],
                                "parsed_from": run["parsed_from"],
                                "raw_generation": (run["raw_generation"] or "")[:300],
                            }
                            for run in parsed_runs[:sc_samples]
                        ],
                    }
                    audit_examples.append(audit_item)
                    print(f"  audit sample {idx}: gold={gold} pred={prediction} parsed_from={parsed_from}")
                    print(f"  raw_generation: {display_output[:200]!r}")

                if prediction == gold:
                    correct += 1
                    if prediction is not None:
                        correct_parsed += 1
                elif len(debug_examples) < DEBUG_SAVE_LIMIT:
                    debug_examples.append({
                        "idx": idx,
                        "gold": gold,
                        "pred": prediction,
                        "parsed_from": parsed_from,
                        "text_preview": preview,
                        "raw_generation": display_output[:400],
                        "self_consistency_runs": [
                            {
                                "prediction": run["prediction"],
                                "parsed_from": run["parsed_from"],
                                "raw_generation": (run["raw_generation"] or "")[:250],
                            }
                            for run in parsed_runs[:sc_samples]
                        ],
                    })
            if end % 25 == 0 or end == sample_count:
                print(f"  processed {end}/{sample_count}")

    accuracy = correct / total if total else 0.0
    parse_rate = parsed_count / total if total else 0.0
    acc_parsed = correct_parsed / parsed_count if parsed_count else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "none_predictions": none_predictions,
        "parse_rate": parse_rate,
        "acc_parsed": acc_parsed,
        "self_consistency_samples": max(1, args.self_consistency_samples),
        "audit_examples": audit_examples,
        "debug_examples": debug_examples,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {DEVICE}")

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        try:
            tokenizer, model = prepare_model(model_name)
        except Exception as exc:
            print(f"Skipping model due to load failure: {model_name}")
            print(f"Load error: {type(exc).__name__}: {exc}")
            failure_path = os.path.join(args.output_dir, f"{safe_model_name(model_name)}__load_failure.json")
            with open(failure_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model": model_name,
                        "load_error_type": type(exc).__name__,
                        "load_error": str(exc),
                    },
                    handle,
                    indent=2,
                )
            print(f"Saved: {failure_path}")
            continue

        results = {
            "model": model_name,
            "device": DEVICE,
            "samples_per_dataset": args.samples_per_dataset,
            "shuffle_seed": SHUFFLE_SEED,
            "self_consistency_samples": max(1, args.self_consistency_samples),
            "datasets": {},
        }

        for dataset_name in args.datasets:
            print("\n" + "-" * 60)
            print(f"DATASET: {dataset_name}")
            print("-" * 60)
            dataset_result = run_dataset_eval(model, tokenizer, dataset_name, args.samples_per_dataset, args)
            results["datasets"][dataset_name] = dataset_result
            print(
                f"Accuracy: {dataset_result['accuracy']:.4f} "
                f"({dataset_result['correct']}/{dataset_result['total']}), "
                f"None preds: {dataset_result['none_predictions']}"
            )
            print(
                f"Parse rate: {dataset_result['parse_rate']:.4f}, "
                f"Accuracy on parsed: {dataset_result['acc_parsed']:.4f}"
            )

        output_name = safe_model_name(model_name)
        if len(args.datasets) == 1:
            output_name = f"{output_name}__{args.datasets[0]}"
        out_path = os.path.join(args.output_dir, f"{output_name}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        print(f"\nSaved: {out_path}")

        del model
        del tokenizer
        torch.cuda.empty_cache()
        clear_hf_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
