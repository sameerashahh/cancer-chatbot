import gc
import json
import os
import re
import shutil
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
    "tiiuae/Falcon3-3B-Instruct"
]
TARGET_MODEL_COUNT = len(MODEL_CANDIDATES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_DATASET = 500
SHUFFLE_SEED = 42
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = 2048
DO_SAMPLE = False

OUTPUT_DIR = "results_hellaswag_strict"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_EXAMPLES_PER_DATASET = 5
CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL = False

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
        "\nassistant\n",
        "\nassistant",
        "assistant\n",
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


# ============================================================
# STRICT PARSER
# ============================================================

def extract_mc_letter_strict(
    text: str,
    valid_letters: str,
) -> Tuple[Optional[str], str]:
    out = text or ""
    valid = "".join(dict.fromkeys(valid_letters.upper()))
    valid_set = set(valid)

    # 1) Explicit "Final answer: <letter>"
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

    # 2) Boxed letter anywhere
    boxed = re.findall(rf"\\boxed\{{\s*([{valid}])\s*\}}", out, flags=re.I)
    if boxed:
        letter = boxed[-1].upper()
        if letter in valid_set:
            return letter, "strict_boxed_letter"

    # 3) Clean last line only
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


def make_batch_inputs(tokenizer, prompts):
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


def generate_oom_safe(model, tokenizer, prompts, max_new_tokens):
    results = []
    bs = len(prompts)
    i = 0

    while i < len(prompts):
        chunk = prompts[i : i + bs]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
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
# PROMPT BUILDER
# ============================================================

def make_mc_prompt(
    instruction: str,
    body: str,
    model_name: str,
) -> str:
    if is_deepseek_model(model_name):
        return (
            f"{instruction}\n"
            "Do not think step by step.\n"
            "Do not explain your reasoning.\n"
            "Do not output anything except the final line.\n"
            "Output exactly one line in this format: Final answer: <letter>\n\n"
            f"{body}"
        )

    return (
        f"{instruction}\n"
        "Do not provide reasoning.\n"
        "Return exactly one line and nothing else in this format: Final answer: <letter>\n\n"
        f"{body}"
    )


# ============================================================
# DATASET BUILDER
# ============================================================

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

    prompt = make_mc_prompt(
        instruction="Choose the most plausible continuation of the context.",
        body=body,
        model_name=model_name,
    )

    return {
        "prompt": prompt,
        "reference": ref,
        "parse": lambda out: extract_mc_letter_strict(out, valid_letters="ABCD"),
        "score": lambda pred, ref: pred == ref,
    }


DATASETS = [
    {
        "name": "hellaswag",
        "load_specs": [
            {"path": "Rowan/hellaswag", "split": "validation"},
        ],
        "build": build_hellaswag_item,
        "max_new_tokens": 24,
    },
]


# ============================================================
# MAIN
# ============================================================

def main():
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

                ds_max_new_tokens = ds_cfg.get("max_new_tokens", MAX_NEW_TOKENS)

                for start in range(0, n, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n)
                    batch = [ds[i] for i in range(start, end)]

                    built_items = []
                    for sample in batch:
                        item = ds_cfg["build"](sample, model_name)
                        if item is None:
                            skipped_unbuildable += 1
                            continue
                        built_items.append(item)

                    if not built_items:
                        continue

                    prompts = [item["prompt"] for item in built_items]
                    refs = [item["reference"] for item in built_items]

                    decoded_list = generate_oom_safe(
                        model,
                        tokenizer,
                        prompts,
                        max_new_tokens=ds_max_new_tokens,
                    )

                    for item, decoded, ref in zip(built_items, decoded_list, refs):
                        cleaned_decoded = strip_prompt_echo(decoded)
                        pred, parse_source = item["parse"](cleaned_decoded)
                        total += 1

                        if debug_printed < DEBUG_EXAMPLES_PER_DATASET:
                            print("\n[EXAMPLE]")
                            print("Reference:", ref)
                            print("Prediction:", pred, f"(extract={parse_source})")
                            print("Raw output:")
                            print(decoded[:1000])
                            print("Cleaned output:")
                            print(cleaned_decoded[:1000])
                            debug_printed += 1

                        if pred is None:
                            none_predictions += 1
                        else:
                            parsed_count += 1

                        if item["score"](pred, ref):
                            correct += 1
                            if pred is not None:
                                correct_parsed += 1

                    if end % 50 == 0 or end == n:
                        print(f"  processed {end}/{n}...")

                acc = correct / total if total else 0.0
                parse_rate = parsed_count / total if total else 0.0
                acc_parsed = correct_parsed / parsed_count if parsed_count else 0.0

                print(f"Accuracy: {acc:.4f} ({correct}/{total})")
                print(f"None predictions: {none_predictions}")
                print(f"Parse rate: {parse_rate:.4f} ({parsed_count}/{total})")
                print(f"Accuracy on parsed: {acc_parsed:.4f} ({correct_parsed}/{parsed_count})")
                print(f"Skipped unbuildable samples: {skipped_unbuildable}")

                model_results["datasets"][ds_name] = {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                    "none_predictions": none_predictions,
                    "parse_rate": parse_rate,
                    "acc_parsed": acc_parsed,
                    "skipped_unbuildable": skipped_unbuildable,
                }

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