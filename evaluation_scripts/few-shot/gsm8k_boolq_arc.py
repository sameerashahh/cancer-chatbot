import gc
import json
import os
import re
import shutil
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# HUGGING FACE LOGIN
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    raise RuntimeError("HF_TOKEN environment variable is not set.")


# ============================================================
# CONFIG
# ============================================================

MODEL_CANDIDATES = [
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "ibm-granite/granite-3.1-2b-instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "allenai/Olmo-3-7B-Instruct",
]
TARGET_MODEL_COUNT = len(MODEL_CANDIDATES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set to None for full official split evaluation
SAMPLES_PER_DATASET = 500
SHUFFLE_SEED = 42
BATCH_SIZE = 4
MAX_INPUT_TOKENS = 4096

OUTPUT_DIR = "results_gsm8k_arc_boolq_meta_style"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_EXAMPLES_PER_DATASET = 5
CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL = True

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")
print(f"Writing results to: {OUTPUT_DIR}")


# ============================================================
# CLEANUP HELPERS
# ============================================================

def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_hf_disk_cache() -> None:
    hf_cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    for cache_dir in hf_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def clear_memory_and_cache() -> None:
    clear_gpu_memory()
    if CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL:
        clear_hf_disk_cache()


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_number(token: str) -> Optional[str]:
    token = (token or "").strip().replace(",", "")
    if not token:
        return None
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError):
        return None

    out = format(value.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if out in ("", "-0"):
        out = "0"
    return out


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
    assistant_markers = ["\nassistant\n", "\nassistant", "assistant\n"]

    last_pos = -1
    last_marker = None
    lowered = out.lower()
    for marker in assistant_markers:
        pos = lowered.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker = marker

    if last_pos != -1 and last_marker is not None:
        out = out[last_pos + len(last_marker):].strip()

    return out


def apply_stop_markers(text: str, stop_markers: Optional[List[str]]) -> str:
    if not stop_markers:
        return text

    cut = len(text)
    for marker in stop_markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut].strip()


# ============================================================
# GSM8K PARSER
# ============================================================

def extract_number_gsm8k_strict(text: str) -> Tuple[Optional[str], str]:
    out = text or ""
    number_re = r"-?\d[\d,]*(?:\.\d+)?"

    match = re.findall(rf"The answer is ({number_re})\.", out, flags=re.I)
    if match:
        value = canonicalize_number(match[-1])
        if value is not None:
            return value, "gsm8k_the_answer_is"

    payload_any, src_any = extract_final_answer_anywhere(out)
    if payload_any is not None:
        nums = re.findall(number_re, payload_any)
        if nums:
            value = canonicalize_number(nums[-1])
            if value is not None:
                return value, f"{src_any}_number"

    payload_last, src_last = extract_last_line_payload(out)
    nums_last = re.findall(number_re, payload_last)
    if nums_last:
        value = canonicalize_number(nums_last[-1])
        if value is not None:
            return value, f"{src_last}_number"

    nums_any = re.findall(number_re, out)
    if nums_any:
        value = canonicalize_number(nums_any[-1])
        if value is not None:
            return value, "fallback_last_number_anywhere"

    return None, "unparsed_none_number"


# ============================================================
# BOOLQ PARSER
# ============================================================

def extract_boolq_yes_no(text: str) -> Tuple[Optional[str], str]:
    out = text or ""

    match = re.search(r"answer\s*:\s*(yes|no)\b", out, flags=re.I)
    if match:
        return match.group(1).lower(), "answer_field_yes_no"

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    for line in reversed(lines):
        normalized = normalize_text(line)
        if normalized in {"yes", "no"}:
            return normalized, "line_exact"

    normalized_all = normalize_text(out)
    if re.search(r"\byes\b", normalized_all):
        return "yes", "fallback_yes"
    if re.search(r"\bno\b", normalized_all):
        return "no", "fallback_no"

    return None, "unparsed_none_boolq"


# ============================================================
# ARC HELPERS (UNCHANGED)
# ============================================================

def normalize_arc_label(label: str) -> str:
    label = str(label).strip().upper()
    if label in {"1", "2", "3", "4", "5"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[label]
    return label


def score_continuation_logprob(model, tokenizer, prompt_text: str, continuation_text: str) -> float:
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    full_ids = tokenizer(
        prompt_text + continuation_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    cont_len = full_len - prompt_len
    if cont_len <= 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)

    target_ids = full_ids[:, 1:]
    start = prompt_len - 1
    end = full_len - 1

    total_logprob = 0.0
    for pos in range(start, end):
        token_id = target_ids[0, pos].item()
        total_logprob += log_probs[0, pos, token_id].item()

    return total_logprob


def choose_best_arc_letter(model, tokenizer, prompt_text: str, valid_letters: List[str]) -> Tuple[Optional[str], Dict[str, float]]:
    scores = {
        letter: score_continuation_logprob(model, tokenizer, prompt_text, f" {letter}")
        for letter in valid_letters
    }
    if not scores:
        return None, {}
    best_letter = max(scores.items(), key=lambda item: item[1])[0]
    return best_letter, scores


# ============================================================
# MODEL / DATASET IO
# ============================================================

def resolve_models(candidates: List[str], target_count: int) -> Tuple[List[str], Dict[str, str]]:
    selected: List[str] = []
    skipped: Dict[str, str] = {}

    for model_name in candidates:
        if len(selected) >= target_count:
            break
        try:
            _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            selected.append(model_name)
            print(f"[MODEL OK] {model_name}")
        except Exception as exc:
            skipped[model_name] = repr(exc)
            print(f"[MODEL SKIP] {model_name}: {repr(exc)}")

    if len(selected) < target_count:
        raise RuntimeError(
            f"Could only resolve {len(selected)} model(s). Need {target_count}."
        )

    return selected, skipped


def dataset_load_first(load_specs: List[Dict[str, Any]]):
    errors = []
    for spec in load_specs:
        try:
            dataset = load_dataset(spec["path"], spec.get("config"), split=spec["split"])
            return dataset, spec
        except Exception as exc:
            errors.append({"spec": spec, "error": repr(exc)})
    raise RuntimeError(f"No load spec worked: {errors}")


def select_dataset_subset(dataset, sample_limit: Optional[int], shuffle_seed: int):
    if sample_limit is None:
        return dataset
    n = min(sample_limit, len(dataset))
    return dataset.shuffle(seed=shuffle_seed).select(range(n))


def make_batch_inputs(tokenizer, prompt_texts: List[str]):
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    input_lengths = inputs["attention_mask"].sum(dim=1)
    return inputs, input_lengths


def decode_batch_outputs(tokenizer, outputs, input_lengths):
    decoded = []
    for i in range(outputs.size(0)):
        start = int(input_lengths[i].item())
        decoded.append(tokenizer.decode(outputs[i, start:], skip_special_tokens=True))
    return decoded


def generate_oom_safe(model, tokenizer, prompt_texts: List[str], max_new_tokens: int) -> List[str]:
    decoded_results: List[str] = []
    sub_batch_size = len(prompt_texts)
    index = 0

    while index < len(prompt_texts):
        chunk = prompt_texts[index:index + sub_batch_size]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded_results.extend(decode_batch_outputs(tokenizer, outputs, input_lengths))
            index += sub_batch_size
        except torch.cuda.OutOfMemoryError:
            if DEVICE != "cuda":
                raise
            torch.cuda.empty_cache()
            sub_batch_size //= 2
            if sub_batch_size < 1:
                raise RuntimeError("OOM even with batch_size=1. Reduce prompt size or max_new_tokens.")
            print(f"  [OOM] reducing sub-batch size -> {sub_batch_size} and retrying...")

    return decoded_results


# ============================================================
# FEW-SHOT PROMPTS
# ============================================================

GSM8K_8SHOT_PREFIX = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""


ARC_FEWSHOT_CACHE = None


def build_gsm8k_8shot_prompt(question: str) -> str:
    return f"{GSM8K_8SHOT_PREFIX}\n\nQ: {question}\nA:"


def get_arc_25shot_examples():
    global ARC_FEWSHOT_CACHE
    if ARC_FEWSHOT_CACHE is not None:
        return ARC_FEWSHOT_CACHE

    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed=SHUFFLE_SEED)
    shots = []

    for sample in dataset:
        question = sample.get("question")
        choices = sample.get("choices")
        answer_key = sample.get("answerKey")

        if not question or not isinstance(choices, dict) or answer_key is None:
            continue

        labels = choices.get("label") or []
        texts = choices.get("text") or []
        if not labels or not texts or len(labels) != len(texts):
            continue

        normalized_options = []
        key_map = {}
        for label, text in zip(labels, texts):
            normalized_label = normalize_arc_label(label)
            normalized_options.append((normalized_label, str(text)))
            key_map[str(label).strip().upper()] = normalized_label

        reference = key_map.get(str(answer_key).strip().upper(), normalize_arc_label(answer_key))
        valid_letters = [label for label, _ in normalized_options]
        if reference not in valid_letters:
            continue

        shots.append(
            {
                "question": question,
                "options": normalized_options,
                "answer": reference,
            }
        )

        if len(shots) == 25:
            break

    if len(shots) < 25:
        raise RuntimeError(f"Could only build {len(shots)} ARC few-shot examples, expected 25.")

    ARC_FEWSHOT_CACHE = shots
    return ARC_FEWSHOT_CACHE


def build_arc_25shot_prompt(question: str, normalized_options: List[Tuple[str, str]]) -> str:
    intro = (
        "You are given a science multiple-choice question.\n"
        "Choose the single best answer.\n"
        "Respond with only the answer letter."
    )

    sections = [intro]

    for example in get_arc_25shot_examples():
        options_text = "\n".join(f"{label}. {text}" for label, text in example["options"])
        sections.append(
            f"Question: {example['question']}\n"
            f"Options:\n{options_text}\n"
            f"Answer: {example['answer']}"
        )

    current_options = "\n".join(f"{label}. {text}" for label, text in normalized_options)
    sections.append(
        f"Question: {question}\n"
        f"Options:\n{current_options}\n"
        "Answer:"
    )

    return "\n\n".join(sections)


# ============================================================
# DATASET BUILDERS
# ============================================================

def build_gsm8k_item(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question")
    answer = sample.get("answer")
    if not question or answer is None:
        return None

    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", str(answer))
    reference = canonicalize_number(match.group(1)) if match else None
    if reference is None:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(answer))
        reference = canonicalize_number(numbers[-1]) if numbers else None
    if reference is None:
        return None

    return {
        "mode": "generate",
        "prompt": build_gsm8k_8shot_prompt(question),
        "reference": reference,
        "parse": extract_number_gsm8k_strict,
        "score": lambda pred, ref: pred == ref,
        "max_new_tokens": 512,
        "stop_markers": ["</s>", "<|im_end|>"],
    }


def build_arc_item(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question")
    choices = sample.get("choices")
    answer_key = sample.get("answerKey")
    if not question or not isinstance(choices, dict) or answer_key is None:
        return None

    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if not labels or not texts or len(labels) != len(texts):
        return None

    normalized_options = []
    key_map = {}
    for label, text in zip(labels, texts):
        normalized_label = normalize_arc_label(label)
        normalized_options.append((normalized_label, str(text)))
        key_map[str(label).strip().upper()] = normalized_label

    reference = key_map.get(str(answer_key).strip().upper(), normalize_arc_label(answer_key))
    valid_letters = [label for label, _ in normalized_options]
    if reference not in valid_letters:
        return None

    return {
        "mode": "likelihood_mcq",
        "prompt": build_arc_25shot_prompt(question, normalized_options),
        "reference": reference,
        "valid_letters": valid_letters,
        "score": lambda pred, ref: pred == ref,
    }


def build_boolq_item(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question")
    passage = sample.get("passage")
    answer = sample.get("answer")
    if question is None or passage is None or answer is None:
        return None

    prompt = (
        "Read the passage and answer the question with only yes or no.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return {
        "mode": "generate",
        "prompt": prompt,
        "reference": "yes" if bool(answer) else "no",
        "parse": extract_boolq_yes_no,
        "score": lambda pred, ref: pred == ref,
        "max_new_tokens": 6,
        "stop_markers": ["</s>", "<|im_end|>"],
    }


DATASETS = [
    {
        "name": "gsm8k",
        "load_specs": [{"path": "gsm8k", "config": "main", "split": "test"}],
        "build": build_gsm8k_item,
    },
    {
        "name": "arc_challenge",
        "load_specs": [{"path": "ai2_arc", "config": "ARC-Challenge", "split": "validation"}],
        "build": build_arc_item,
    },
    {
        "name": "boolq",
        "load_specs": [{"path": "boolq", "split": "validation"}],
        "build": build_boolq_item,
    },
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    models, skipped_models = resolve_models(MODEL_CANDIDATES, TARGET_MODEL_COUNT)

    print("\nSelected models:")
    for model_name in models:
        print(f"  - {model_name}")

    if skipped_models:
        print("\nSkipped model candidates:")
        for model_name, error in skipped_models.items():
            print(f"  - {model_name}: {error}")

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        clear_memory_and_cache()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(DEVICE)
        model.eval()

        model_results: Dict[str, Any] = {
            "model": model_name,
            "device": DEVICE,
            "samples_per_dataset": SAMPLES_PER_DATASET,
            "shuffle_seed": SHUFFLE_SEED,
            "batch_size": BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "evaluation_style": {
                "gsm8k": "8-shot CoT generate",
                "arc_challenge": "25-shot likelihood-over-choice-letter",
                "boolq": "0-shot generate",
            },
            "datasets": {},
        }

        for dataset_config in DATASETS:
            dataset_name = dataset_config["name"]
            print("\n" + "-" * 60)
            print(f"DATASET: {dataset_name}")
            print("-" * 60)

            try:
                dataset, used_load_spec = dataset_load_first(dataset_config["load_specs"])
                print(f"Using dataset source: {used_load_spec}")
                dataset = select_dataset_subset(dataset, SAMPLES_PER_DATASET, SHUFFLE_SEED)

                correct = 0
                total = 0
                none_predictions = 0
                parsed_count = 0
                correct_parsed = 0
                skipped_unbuildable = 0
                debug_printed = 0

                built_items = []
                for sample in dataset:
                    item = dataset_config["build"](sample)
                    if item is None:
                        skipped_unbuildable += 1
                        continue
                    built_items.append(item)

                if not built_items:
                    raise RuntimeError(f"No buildable items for dataset {dataset_name}")

                if built_items[0]["mode"] == "generate":
                    for start in range(0, len(built_items), BATCH_SIZE):
                        batch_items = built_items[start:start + BATCH_SIZE]
                        prompts = [item["prompt"] for item in batch_items]
                        max_new_tokens = batch_items[0]["max_new_tokens"]
                        stop_markers = batch_items[0].get("stop_markers")

                        decoded_list = generate_oom_safe(model, tokenizer, prompts, max_new_tokens)

                        for item, prompt, decoded in zip(batch_items, prompts, decoded_list):
                            cleaned = apply_stop_markers(strip_prompt_echo(decoded), stop_markers)
                            prediction, parse_source = item["parse"](cleaned)
                            reference = item["reference"]
                            total += 1

                            if debug_printed < DEBUG_EXAMPLES_PER_DATASET:
                                print("\n[EXAMPLE]")
                                print("Reference:", reference)
                                print("Prediction:", prediction, f"(extract={parse_source})")
                                print("Prompt preview:")
                                print(prompt[:1200])
                                print("Raw output:")
                                print(decoded[:1200])
                                print("Cleaned output:")
                                print(cleaned[:1200])
                                debug_printed += 1

                            if prediction is None:
                                none_predictions += 1
                            else:
                                parsed_count += 1

                            if item["score"](prediction, reference):
                                correct += 1
                                if prediction is not None:
                                    correct_parsed += 1

                        processed = min(start + BATCH_SIZE, len(built_items))
                        if processed % 25 == 0 or processed == len(built_items):
                            print(f"  processed {processed}/{len(built_items)}...")

                elif built_items[0]["mode"] == "likelihood_mcq":
                    for idx, item in enumerate(built_items, start=1):
                        prediction, score_map = choose_best_arc_letter(
                            model,
                            tokenizer,
                            item["prompt"],
                            item["valid_letters"],
                        )
                        reference = item["reference"]
                        total += 1

                        if prediction is None:
                            none_predictions += 1
                        else:
                            parsed_count += 1

                        if item["score"](prediction, reference):
                            correct += 1
                            if prediction is not None:
                                correct_parsed += 1

                        if debug_printed < DEBUG_EXAMPLES_PER_DATASET:
                            print("\n[EXAMPLE]")
                            print("Reference:", reference)
                            print("Prediction:", prediction)
                            print("Scores:", score_map)
                            print("Prompt preview:")
                            print(item["prompt"][:1400])
                            debug_printed += 1

                        if idx % 25 == 0 or idx == len(built_items):
                            print(f"  processed {idx}/{len(built_items)}...")
                else:
                    raise RuntimeError(f"Unsupported mode: {built_items[0]['mode']}")

                accuracy = correct / total if total else 0.0
                parse_rate = parsed_count / total if total else 0.0
                accuracy_parsed = correct_parsed / parsed_count if parsed_count else 0.0

                print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
                print(f"None predictions: {none_predictions}")
                print(f"Parse rate: {parse_rate:.4f} ({parsed_count}/{total})")
                print(f"Accuracy on parsed: {accuracy_parsed:.4f} ({correct_parsed}/{parsed_count})")
                print(f"Skipped unbuildable samples: {skipped_unbuildable}")

                model_results["datasets"][dataset_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "none_predictions": none_predictions,
                    "parse_rate": parse_rate,
                    "acc_parsed": accuracy_parsed,
                    "skipped_unbuildable": skipped_unbuildable,
                }

            except Exception as exc:
                print(f"[ERROR] {model_name} on {dataset_name}: {repr(exc)}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                model_results["datasets"][dataset_name] = {"error": repr(exc)}

        safe_name = model_name.replace("/", "__").replace(":", "_")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(output_path, "w") as file:
            json.dump(model_results, file, indent=2)

        print(f"\nSaved: {output_path}")

        del model
        del tokenizer
        clear_memory_and_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()