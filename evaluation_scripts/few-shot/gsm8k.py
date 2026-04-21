import gc
import json
import os
import re
import shutil
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
    "meta-llama/Llama-3.2-1B-Instruct",
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

# Set to None to run full GSM8K test split
SAMPLES_PER_DATASET = 500
SHUFFLE_SEED = 42
BATCH_SIZE = 8
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS_GSM8K = 512

OUTPUT_DIR = "results_gsm8k_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEBUG_EXAMPLES = 5

# Keep True, but only clear model/tokenizer caches, not datasets cache
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


def clear_model_disk_cache() -> None:
    hf_cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
        # Intentionally NOT deleting datasets cache
        # Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    for cache_dir in hf_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def clear_memory_and_cache() -> None:
    clear_gpu_memory()
    if CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL:
        clear_model_disk_cache()


# ============================================================
# HELPERS
# ============================================================

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


def strip_prompt_echo(text: str) -> str:
    out = (text or "").strip()
    assistant_markers = [
        "\nassistant\n",
        "\nassistant",
        "assistant\n",
    ]

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


def extract_last_line_payload(text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "", "empty_output"

    last = lines[-1]
    match = re.match(r"(?is)^final\s*answer\s*[:\-]\s*(.+)$", last)
    if match:
        return match.group(1).strip(), "last_line_final_answer"
    return last, "last_line"


# ============================================================
# GSM8K PARSER
# ============================================================

def extract_number_gsm8k(text: str) -> Tuple[Optional[str], str]:
    """
    Prefer the earliest clear answer signal in the completion.
    This avoids being corrupted when the model starts generating
    a new Q/A example after answering the real question.
    """
    out = text or ""
    number_re = r"-?\d[\d,]*(?:\.\d+)?"

    # 1) Exact benchmark-style phrase
    matches = list(re.finditer(rf"The answer is ({number_re})\.", out, flags=re.I))
    if matches:
        value = canonicalize_number(matches[0].group(1))
        if value is not None:
            return value, "gsm8k_the_answer_is_first"

    # 2) Accept "Final answer: X" if model uses it
    final_matches = list(re.finditer(rf"final\s*answer\s*[:\-]\s*({number_re})", out, flags=re.I))
    if final_matches:
        value = canonicalize_number(final_matches[0].group(1))
        if value is not None:
            return value, "gsm8k_final_answer_first"

    # 3) Last line number
    payload_last, src_last = extract_last_line_payload(out)
    nums_last = re.findall(number_re, payload_last)
    if nums_last:
        value = canonicalize_number(nums_last[-1])
        if value is not None:
            return value, f"{src_last}_number"

    # 4) Conservative fallback: first number anywhere in completion
    nums_any = re.findall(number_re, out)
    if nums_any:
        value = canonicalize_number(nums_any[0])
        if value is not None:
            return value, "fallback_first_number_anywhere"

    return None, "unparsed_none_number"


# ============================================================
# EXACT GSM8K 8-SHOT PROMPT
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


def build_gsm8k_prompt(question: str) -> str:
    return f"{GSM8K_8SHOT_PREFIX}\n\nQ: {question}\nA:"


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
# GSM8K DATASET BUILDER
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
        "prompt": build_gsm8k_prompt(question),
        "reference": reference,
        "parse": extract_number_gsm8k,
        "score": lambda pred, ref: pred == ref,
        "max_new_tokens": MAX_NEW_TOKENS_GSM8K,
        "stop_markers": ["\nQ:", "\n\nQ:", "</s>", "<|im_end|>"],
    }


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

    # Load dataset once; keep datasets cache across model runs
    dataset = load_dataset("gsm8k", "main", split="test")
    if SAMPLES_PER_DATASET is not None:
        dataset = dataset.shuffle(seed=SHUFFLE_SEED).select(
            range(min(SAMPLES_PER_DATASET, len(dataset)))
        )

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

        results: Dict[str, Any] = {
            "model": model_name,
            "device": DEVICE,
            "samples_per_dataset": SAMPLES_PER_DATASET,
            "shuffle_seed": SHUFFLE_SEED,
            "batch_size": BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "evaluation_style": {
                "gsm8k": "8-shot CoT generate, stop on next Q:"
            },
        }

        correct = 0
        total = 0
        none_predictions = 0
        parsed_count = 0
        correct_parsed = 0
        skipped_unbuildable = 0
        debug_printed = 0

        built_items = []
        for sample in dataset:
            item = build_gsm8k_item(sample)
            if item is None:
                skipped_unbuildable += 1
                continue
            built_items.append(item)

        for start in range(0, len(built_items), BATCH_SIZE):
            batch_items = built_items[start:start + BATCH_SIZE]
            prompts = [item["prompt"] for item in batch_items]
            max_new_tokens = batch_items[0]["max_new_tokens"]
            stop_markers = batch_items[0]["stop_markers"]

            decoded_list = generate_oom_safe(model, tokenizer, prompts, max_new_tokens)

            for item, prompt, decoded in zip(batch_items, prompts, decoded_list):
                cleaned = apply_stop_markers(strip_prompt_echo(decoded), stop_markers)
                prediction, parse_source = item["parse"](cleaned)
                reference = item["reference"]
                total += 1

                if debug_printed < DEBUG_EXAMPLES:
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

        accuracy = correct / total if total else 0.0
        parse_rate = parsed_count / total if total else 0.0
        accuracy_parsed = correct_parsed / parsed_count if parsed_count else 0.0

        results["gsm8k"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "none_predictions": none_predictions,
            "parse_rate": parse_rate,
            "acc_parsed": accuracy_parsed,
            "skipped_unbuildable": skipped_unbuildable,
        }

        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"None predictions: {none_predictions}")
        print(f"Parse rate: {parse_rate:.4f} ({parsed_count}/{total})")
        print(f"Accuracy on parsed: {accuracy_parsed:.4f} ({correct_parsed}/{parsed_count})")
        print(f"Skipped unbuildable samples: {skipped_unbuildable}")

        safe_name = model_name.replace("/", "__").replace(":", "_")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(output_path, "w") as file:
            json.dump(results, file, indent=2)

        print(f"\nSaved: {output_path}")

        del model
        del tokenizer
        clear_memory_and_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()