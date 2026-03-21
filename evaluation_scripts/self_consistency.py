import gc
import json
import math
import os
import re
import shutil
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
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
    "allenai/Olmo-3-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "ibm-granite/granite-3.1-2b-instruct",
    "tiiuae/Falcon3-3B-Instruct",
]
TARGET_MODEL_COUNT = len(MODEL_CANDIDATES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_DATASET = 500
SHUFFLE_SEED = 42
BATCH_SIZE = 8
MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = 2048

# Self-consistency settings
SELF_CONSISTENCY_RUNS = 5
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

OUTPUT_DIR = "results_unified_self_consistency_strict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_EXAMPLES_PER_DATASET = 5
CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL = True
SAVE_SAMPLE_DETAILS = False

WINOGRANDE_CONFIG = "winogrande_debiased"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")
print(f"Writing results to: {OUTPUT_DIR}")


# ============================================================
# CLEANUP HELPERS
# ============================================================

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_hf_disk_cache():
    hf_cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    for cache_dir in hf_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def clear_memory_and_cache():
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


def normalize_span(text: Optional[str]) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
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
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "", "empty_output"
    last = lines[-1]
    m = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", last)
    if m:
        return m.group(1).strip(), "last_line_final_answer"
    return last, "last_line"


def extract_final_answer_anywhere(text: str) -> Tuple[Optional[str], str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        m = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", line)
        if m:
            return m.group(1).strip(), "full_output_final_answer"
    return None, "no_final_answer"


def strip_prompt_echo(text: str) -> str:
    out = (text or "").strip()

    assistant_markers = [
        "<｜Assistant｜>",
        "<|assistant|>",
        "assistant",
        "assistant",
        "assistant",
    ]

    last_pos = -1
    last_marker = None

    for marker in assistant_markers:
        pos = out.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker = marker

    if last_pos != -1 and last_marker is not None:
        out = out[last_pos + len(last_marker):].strip()

    return out.strip()


def is_deepseek_model(model_name: str) -> bool:
    return "deepseek" in (model_name or "").lower()


def label_aliases_for_eval(labels: List[str]) -> Dict[str, str]:
    aliases = {}
    for label in labels:
        key = label.lower()
        aliases[key] = label
        aliases[key.replace("_", " ")] = label
        aliases[key.replace("_", "-")] = label
    return aliases
def majority_vote(predictions: List[Optional[str]]) -> Tuple[Optional[str], Dict[str, Any]]:
    parsed = [p for p in predictions if p is not None]
    if not parsed:
        return None, {
            "parsed_votes": 0,
            "vote_counter": {},
            "tie": False,
            "winner_count": 0,
        }

    counter = Counter(parsed)
    max_count = max(counter.values())
    winners = {label for label, count in counter.items() if count == max_count}

    voted = None
    for pred in parsed:
        if pred in winners:
            voted = pred
            break

    return voted, {
        "parsed_votes": len(parsed),
        "vote_counter": dict(counter),
        "tie": len(winners) > 1,
        "winner_count": max_count,
    }


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


# ============================================================
# STRICT PARSERS
# ============================================================

def extract_categorical_strict(text: str, labels: List[str]) -> Tuple[Optional[str], str]:
    aliases = label_aliases_for_eval(labels)

    payload, src = extract_last_line_payload(text)
    norm_payload = normalize_text(payload)
    if norm_payload in aliases:
        return aliases[norm_payload], f"{src}_exact"

    payload2, src2 = extract_final_answer_anywhere(text)
    if payload2 is not None:
        norm_payload2 = normalize_text(payload2)
        if norm_payload2 in aliases:
            return aliases[norm_payload2], f"{src2}_exact"

    return None, "unparsed_none_categorical_strict"


def extract_number_strict(text: str) -> Tuple[Optional[str], str]:
    out = text or ""
    number_re = r"-?\d[\d,]*(?:\.\d+)?"

    payload_any, src_any = extract_final_answer_anywhere(out)
    if payload_any is not None:
        nums_any = re.findall(number_re, payload_any)
        if nums_any:
            val = canonicalize_number(nums_any[-1])
            if val is not None:
                return val, f"strict_{src_any}_number"

    boxed = re.findall(rf"\\boxed\{{\s*({number_re})\s*\}}", out)
    if boxed:
        val = canonicalize_number(boxed[-1])
        if val is not None:
            return val, "strict_boxed_number"

    payload_last, src_last = extract_last_line_payload(out)
    nums_last = re.findall(number_re, payload_last)
    if nums_last:
        val = canonicalize_number(nums_last[-1])
        if val is not None:
            return val, f"strict_{src_last}_number"

    return None, "unparsed_none_number_strict"


def extract_mc_letter_strict(text: str, valid_letters: str) -> Tuple[Optional[str], str]:
    out = text or ""
    valid = "".join(dict.fromkeys(valid_letters.upper()))
    valid_set = set(valid)

    payload_any, src_any = extract_final_answer_anywhere(out)
    if payload_any is not None:
        p = payload_any.strip()

        m = re.match(rf"(?is)^\s*([{valid}])\s*$", p)
        if m:
            return m.group(1).upper(), f"strict_{src_any}_bare_letter"

        m = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])\s*$", p)
        if m:
            return m.group(1).upper(), f"strict_{src_any}_option_letter"

        m = re.match(rf"(?is)^\s*\\boxed\{{\s*([{valid}])\s*\}}\s*$", p)
        if m:
            return m.group(1).upper(), f"strict_{src_any}_boxed_letter"

        m = re.match(
            rf"(?is)^\s*(?:the\s+answer\s+is|answer:|correct\s+answer\s+is|i\s+choose)\s*([{valid}])\s*$",
            p,
        )
        if m:
            return m.group(1).upper(), f"strict_{src_any}_answer_phrase"

    boxed = re.findall(rf"\\boxed\{{\s*([{valid}])\s*\}}", out, flags=re.I)
    if boxed:
        letter = boxed[-1].upper()
        if letter in valid_set:
            return letter, "strict_boxed_letter"

    payload_last, src_last = extract_last_line_payload(out)
    p_last = payload_last.strip()

    m = re.match(rf"(?is)^\s*([{valid}])\s*$", p_last)
    if m:
        return m.group(1).upper(), f"strict_{src_last}_bare_letter"

    m = re.match(rf"(?is)^\s*(?:option|choice)\s*([{valid}])\s*$", p_last)
    if m:
        return m.group(1).upper(), f"strict_{src_last}_option_letter"

    m = re.match(
        rf"(?is)^\s*(?:answer\s+is|the\s+answer\s+is|correct\s+answer\s+is|i\s+choose|answer:)\s*([{valid}])\s*$",
        p_last,
    )
    if m:
        return m.group(1).upper(), f"strict_{src_last}_answer_phrase"

    return None, "unparsed_none_mcq_strict"


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
            f"Could only resolve {len(selected)} model(s). Need {target_count}. "
            "Add more accessible model candidates."
        )

    return selected, skipped


def dataset_load_first(load_specs: List[Dict[str, Any]]):
    errors = []
    for spec in load_specs:
        try:
            ds = load_dataset(spec["path"], spec.get("config"), split=spec["split"])
            return ds, spec
        except Exception as exc:
            errors.append({"spec": spec, "error": repr(exc)})
    raise RuntimeError(f"No load spec worked: {errors}")


def make_batch_inputs(tokenizer, prompts: List[str]):
    if hasattr(tokenizer, "apply_chat_template"):
        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

    inputs = inputs.to(DEVICE)
    if "attention_mask" in inputs:
        input_lengths = inputs["attention_mask"].sum(dim=1)
    else:
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    return inputs, input_lengths


def decode_batch_outputs(tokenizer, outputs, input_lengths):
    decoded = []
    for i in range(outputs.size(0)):
        start = int(input_lengths[i].item()) if input_lengths is not None else 0
        gen_tokens = outputs[i, start:]
        decoded.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return decoded


def generate_oom_safe(model, tokenizer, prompts: List[str], max_new_tokens: int):
    results = []
    bs = len(prompts)
    i = 0

    while i < len(prompts):
        chunk = prompts[i: i + bs]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else None,
                    top_p=TOP_P if DO_SAMPLE else None,
                    top_k=TOP_K if DO_SAMPLE else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            results.extend(decode_batch_outputs(tokenizer, outputs, input_lengths))
            i += bs
        except torch.cuda.OutOfMemoryError:
            if DEVICE != "cuda":
                raise
            torch.cuda.empty_cache()
            bs //= 2
            if bs < 1:
                raise RuntimeError("OOM even with batch_size=1. Reduce prompt length or max_new_tokens.")
            print(f"  [OOM] reducing sub-batch size -> {bs} and retrying...")

    return results


# ============================================================
# PROMPT BUILDERS
# ============================================================

def make_final_answer_prompt(
    instruction: str,
    answer_format: str,
    body: str,
    model_name: str,
) -> str:
    if is_deepseek_model(model_name):
        return (
            f"{instruction}\n"
            "Do not think step by step.\n"
            "Do not explain your reasoning.\n"
            "Do not output anything except the final line.\n"
            f"Output exactly one line in this format: Final answer: {answer_format}\n\n"
            f"{body}"
        )

    return (
        f"{instruction}\n"
        "Do not provide reasoning.\n"
        f"Return exactly one line and nothing else in this format: Final answer: {answer_format}\n\n"
        f"{body}"
    )
# ============================================================
# DATASET BUILDERS
# ============================================================

def build_gsm8k_item(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question")
    answer = sample.get("answer")
    if not question or answer is None:
        return None

    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", str(answer))
    ref = canonicalize_number(m.group(1)) if m else None
    if ref is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(answer))
        ref = canonicalize_number(nums[-1]) if nums else None
    if ref is None:
        return None

    prompt = make_final_answer_prompt(
        instruction="Solve the math problem.",
        answer_format="<number>",
        body=f"Problem:\n{question}",
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": extract_number_strict,
        "score": lambda pred, ref: pred == ref,
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

    normalized: List[Tuple[str, str]] = []
    key_map: Dict[str, str] = {}
    for lb, txt in zip(labels, texts):
        lb_u = str(lb).strip().upper()
        if lb_u in {"1", "2", "3", "4", "5"}:
            lb_u = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}[lb_u]
        normalized.append((lb_u, str(txt)))
        key_map[str(lb).strip().upper()] = lb_u

    ans_u = str(answer_key).strip().upper()
    ref = key_map.get(ans_u, ans_u)

    valid = "".join([lb for lb, _ in normalized])
    opts = "\n".join(f"{lb}. {txt}" for lb, txt in normalized)

    prompt = make_final_answer_prompt(
        instruction="Answer the science question.",
        answer_format="<letter>",
        body=f"Question:\n{question}\n\nOptions:\n{opts}",
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": lambda out: extract_mc_letter_strict(out, valid_letters=valid),
        "score": lambda pred, ref: pred == ref,
    }
def build_boolq_item(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question")
    passage = sample.get("passage")
    answer = sample.get("answer")
    if question is None or passage is None or answer is None:
        return None

    ref = "yes" if bool(answer) else "no"

    prompt = make_final_answer_prompt(
        instruction="Answer the question based on the passage.",
        answer_format="<yes or no>",
        body=(
            f"Passage:\n{passage}\n"
            f"Question:\n{question}\n\n"
            "Label must be one of: yes, no."
        ),
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": lambda out: extract_categorical_strict(out, ["no", "yes"]),
        "score": lambda pred, ref: pred == ref,
    }
def build_hellaswag_item(sample: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    ctx = sample.get("ctx")
    endings = sample.get("endings")
    label = sample.get("label")

    if ctx is None or endings is None or label is None:
        return None
    if not isinstance(endings, list) or len(endings) != 4:
        return None

    endings = [str(x).strip() for x in endings]
    if any(not x for x in endings):
        return None

    try:
        label_int = int(label)
    except Exception:
        return None
    if label_int not in {0, 1, 2, 3}:
        return None

    ref = ["A", "B", "C", "D"][label_int]

    body = (
        f"Context:\n{str(ctx).strip()}\n\n"
        f"Options:\n"
        f"A. {endings[0]}\n"
        f"B. {endings[1]}\n"
        f"C. {endings[2]}\n"
        f"D. {endings[3]}"
    )
    prompt = make_final_answer_prompt(
        instruction="Choose the most plausible continuation of the context.",
        answer_format="<letter>",
        body=body,
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": lambda out: extract_mc_letter_strict(out, valid_letters="ABCD"),
        "score": lambda pred, ref: pred == ref,
    }
def build_winogrande_item(sample: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    sentence = sample.get("sentence")
    option1 = sample.get("option1")
    option2 = sample.get("option2")
    answer = sample.get("answer")

    if sentence is None or option1 is None or option2 is None or answer is None:
        return None

    sentence = str(sentence)
    option1 = str(option1)
    option2 = str(option2)
    answer = str(answer).strip()

    if "_" not in sentence:
        return None
    if answer not in {"1", "2"}:
        return None

    ref = "A" if answer == "1" else "B"

    body = (
        f"Sentence:\n{sentence}\n\n"
        f"Options:\nA. {option1}\nB. {option2}"
    )
    prompt = make_final_answer_prompt(
        instruction="Choose the option that best fills the blank.",
        answer_format="<letter>",
        body=body,
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": lambda out: extract_mc_letter_strict(out, valid_letters="AB"),
        "score": lambda pred, ref: pred == ref,
    }
DATASETS = [
    {
        "name": "gsm8k",
        "load_specs": [{"path": "gsm8k", "config": "main", "split": "test"}],
        "build": build_gsm8k_item,
        "max_new_tokens": 256,
    },
    {
        "name": "arc_challenge",
        "load_specs": [{"path": "ai2_arc", "config": "ARC-Challenge", "split": "validation"}],
        "build": build_arc_item,
        "max_new_tokens": 96,
    },
    {
        "name": "boolq",
        "load_specs": [{"path": "boolq", "split": "validation"}],
        "build": build_boolq_item,
        "max_new_tokens": 32,
    },
    {
        "name": "hellaswag",
        "load_specs": [{"path": "Rowan/hellaswag", "split": "validation"}],
        "build": build_hellaswag_item,
        "max_new_tokens": 24,
    },
    {
        "name": "winogrande",
        "load_specs": [{"path": "allenai/winogrande", "config": WINOGRANDE_CONFIG, "split": "validation"}],
        "build": build_winogrande_item,
        "max_new_tokens": 24,
    },
]


# ============================================================
# SELF-CONSISTENCY EVALUATION
# ============================================================

def evaluate_item_self_consistency(
    model,
    tokenizer,
    item: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompts = [item["prompt"]] * SELF_CONSISTENCY_RUNS
    decoded_list = generate_oom_safe(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
    )

    cleaned_outputs = [strip_prompt_echo(x) for x in decoded_list]
    parsed_preds: List[Optional[str]] = []
    parse_sources: List[str] = []

    for cleaned in cleaned_outputs:
        pred, parse_source = item["parse"](cleaned)
        parsed_preds.append(pred)
        parse_sources.append(parse_source)

    voted_pred, vote_meta = majority_vote(parsed_preds)
    voted_correct = item["score"](voted_pred, item["reference"])

    trial_correct = [item["score"](pred, item["reference"]) for pred in parsed_preds]
    trial_parsed = [pred is not None for pred in parsed_preds]

    return {
        "reference": item["reference"],
        "voted_prediction": voted_pred,
        "voted_correct": voted_correct,
        "decoded_list": decoded_list,
        "cleaned_outputs": cleaned_outputs,
        "parsed_predictions": parsed_preds,
        "parse_sources": parse_sources,
        "vote_meta": vote_meta,
        "trial_correct": trial_correct,
        "trial_parsed": trial_parsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(SHUFFLE_SEED)
    torch.manual_seed(SHUFFLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SHUFFLE_SEED)

    models, skipped_models = resolve_models(MODEL_CANDIDATES, TARGET_MODEL_COUNT)

    print("\nSelected models:")
    for model_name in models:
        print(f"  - {model_name}")

    if skipped_models:
        print("\nSkipped model candidates:")
        for model_name, err in skipped_models.items():
            print(f"  - {model_name}: {err}")

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
            "max_new_tokens_default": MAX_NEW_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "parsing_mode": "strict",
            "winogrande_config": WINOGRANDE_CONFIG,
            "self_consistency": {
                "runs": SELF_CONSISTENCY_RUNS,
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "vote_rule": "majority_over_parsed_predictions_only",
                "tie_break_rule": "first_prediction_among_top_tied_labels",
            },
            "datasets": {},
        }

        for ds_cfg in DATASETS:
            ds_name = ds_cfg["name"]
            print("\n" + "-" * 60)
            print(f"DATASET: {ds_name}")
            print("-" * 60)

            try:
                ds, used_load_spec = dataset_load_first(ds_cfg["load_specs"])
                print(f"Using dataset source: {used_load_spec}")

                n = min(SAMPLES_PER_DATASET, len(ds))
                ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(n))

                correct = 0
                total = 0
                none_predictions = 0
                parsed_count = 0
                correct_parsed = 0
                skipped_unbuildable = 0
                debug_printed = 0

                trial_correct_counts = [0 for _ in range(SELF_CONSISTENCY_RUNS)]
                trial_none_counts = [0 for _ in range(SELF_CONSISTENCY_RUNS)]
                trial_parsed_counts = [0 for _ in range(SELF_CONSISTENCY_RUNS)]
                ties_after_vote = 0
                all_none_vote = 0
                sample_details = []

                ds_max_new_tokens = ds_cfg.get("max_new_tokens", MAX_NEW_TOKENS)

                for start in range(0, n, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n)
                    batch = [ds[i] for i in range(start, end)]

                    built_items = []
                    for sample in batch:
                        try:
                            item = ds_cfg["build"](sample, model_name)
                        except TypeError:
                            item = ds_cfg["build"](sample)
                        if item is None:
                            skipped_unbuildable += 1
                            continue
                        built_items.append(item)

                    if not built_items:
                        continue

                    for item in built_items:
                        eval_result = evaluate_item_self_consistency(
                            model=model,
                            tokenizer=tokenizer,
                            item=item,
                            max_new_tokens=ds_max_new_tokens,
                        )

                        ref = eval_result["reference"]
                        voted_pred = eval_result["voted_prediction"]
                        total += 1

                        if voted_pred is None:
                            none_predictions += 1
                            all_none_vote += 1
                        else:
                            parsed_count += 1

                        if eval_result["voted_correct"]:
                            correct += 1
                            if voted_pred is not None:
                                correct_parsed += 1

                        if eval_result["vote_meta"]["tie"]:
                            ties_after_vote += 1

                        for idx in range(SELF_CONSISTENCY_RUNS):
                            if eval_result["trial_parsed"][idx]:
                                trial_parsed_counts[idx] += 1
                            else:
                                trial_none_counts[idx] += 1
                            if eval_result["trial_correct"][idx]:
                                trial_correct_counts[idx] += 1

                        if debug_printed < DEBUG_EXAMPLES_PER_DATASET:
                            print("\n[EXAMPLE]")
                            print("Reference:", ref)
                            print("Trial predictions:", eval_result["parsed_predictions"])
                            print("Vote meta:", eval_result["vote_meta"])
                            print("Voted prediction:", voted_pred)
                            print("Cleaned output[0]:")
                            print(eval_result["cleaned_outputs"][0][:1000])
                            debug_printed += 1

                        if SAVE_SAMPLE_DETAILS:
                            sample_details.append({
                                "reference": ref,
                                "voted_prediction": voted_pred,
                                "voted_correct": eval_result["voted_correct"],
                                "parsed_predictions": eval_result["parsed_predictions"],
                                "parse_sources": eval_result["parse_sources"],
                                "vote_meta": eval_result["vote_meta"],
                            })

                    if end % 25 == 0 or end == n:
                        print(f"  processed {end}/{n}...")

                acc = correct / total if total else 0.0
                parse_rate = parsed_count / total if total else 0.0
                acc_parsed = correct_parsed / parsed_count if parsed_count else 0.0

                trial_accuracies = [c / total if total else 0.0 for c in trial_correct_counts]
                trial_parse_rates = [c / total if total else 0.0 for c in trial_parsed_counts]

                print(f"Vote accuracy: {acc:.4f} ({correct}/{total})")
                print(f"Vote none predictions: {none_predictions}")
                print(f"Vote parse rate: {parse_rate:.4f} ({parsed_count}/{total})")
                print(f"Vote accuracy on parsed: {acc_parsed:.4f} ({correct_parsed}/{parsed_count})")
                print(f"Vote ties: {ties_after_vote}")
                print(f"Skipped unbuildable samples: {skipped_unbuildable}")
                print(f"Mean single-trial accuracy: {mean(trial_accuracies):.4f}")
                print(f"Mean single-trial parse rate: {mean(trial_parse_rates):.4f}")

                ds_result = {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                    "none_predictions": none_predictions,
                    "parse_rate": parse_rate,
                    "acc_parsed": acc_parsed,
                    "skipped_unbuildable": skipped_unbuildable,
                    "vote_ties": ties_after_vote,
                    "all_none_vote": all_none_vote,
                    "trial_correct_counts": trial_correct_counts,
                    "trial_none_counts": trial_none_counts,
                    "trial_parsed_counts": trial_parsed_counts,
                    "trial_accuracies": trial_accuracies,
                    "trial_parse_rates": trial_parse_rates,
                    "trial_accuracy_mean": mean(trial_accuracies),
                    "trial_accuracy_std": stddev(trial_accuracies),
                    "trial_parse_rate_mean": mean(trial_parse_rates),
                    "trial_parse_rate_std": stddev(trial_parse_rates),
                }

                if SAVE_SAMPLE_DETAILS:
                    ds_result["sample_details"] = sample_details

                model_results["datasets"][ds_name] = ds_result

            except Exception as exc:
                print(f"[ERROR] {model_name} on {ds_name}: {repr(exc)}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                model_results["datasets"][ds_name] = {"error": repr(exc)}
                continue

        safe_name = model_name.replace("/", "__").replace(":", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)

        print(f"\nSaved: {out_path}")

        del model
        del tokenizer
        clear_memory_and_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
