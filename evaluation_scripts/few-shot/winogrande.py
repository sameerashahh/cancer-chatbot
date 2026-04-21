import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
SAMPLES_PER_DATASET = 500
SHUFFLE_SEED = 42
MAX_INPUT_TOKENS = 4096
BATCH_SIZE = 1
NUM_SHOTS = 5

WINOGRANDE_CONFIG = "winogrande_debiased"
OUTPUT_DIR = "results_winogrande_only"
DEBUG_EXAMPLES = 5
CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        # Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    for cache_dir in hf_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def clear_memory_and_cache():
    clear_gpu_memory()
    if CLEAR_HF_DISK_CACHE_AFTER_EACH_MODEL:
        clear_hf_disk_cache()


# ============================================================
# MODEL HELPERS
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


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(DEVICE)
    model.eval()

    return model, tokenizer


# ============================================================
# WINOGRANDE PROMPT HELPERS
# ============================================================

def split_winogrande_sentence(sentence: str) -> Tuple[str, str]:
    if "_" not in sentence:
        raise ValueError(f"Sentence has no blank: {sentence}")
    left, right = sentence.split("_", 1)
    return left, right


def render_winogrande_completion(sentence: str, option: str) -> str:
    left, right = split_winogrande_sentence(sentence)
    return left + option + right


def format_shot(sample: Dict[str, Any]) -> str:
    sentence = str(sample["sentence"])
    option1 = str(sample["option1"])
    option2 = str(sample["option2"])
    answer = str(sample["answer"]).strip()

    correct_option = option1 if answer == "1" else option2
    completed = render_winogrande_completion(sentence, correct_option)

    return (
        "Fill in the blank with the correct option.\n"
        f"Sentence: {sentence}\n"
        f"Option 1: {option1}\n"
        f"Option 2: {option2}\n"
        f"Answer: {completed}\n"
    )


def build_fewshot_prefix(fewshot_examples: List[Dict[str, Any]]) -> str:
    blocks = [format_shot(ex) for ex in fewshot_examples]
    return "\n".join(blocks).strip() + "\n\n"


def build_prefix_and_continuations(sample: Dict[str, Any], fewshot_prefix: str) -> Dict[str, Any]:
    sentence = str(sample["sentence"])
    option1 = str(sample["option1"])
    option2 = str(sample["option2"])
    answer = str(sample["answer"]).strip()

    left, right = split_winogrande_sentence(sentence)

    # Prefix ends right before the missing choice.
    prefix = (
        fewshot_prefix
        + "Fill in the blank with the correct option.\n"
        + f"Sentence: {sentence}\n"
        + f"Option 1: {option1}\n"
        + f"Option 2: {option2}\n"
        + "Answer: "
        + left
    )

    continuation_1 = option1 + right
    continuation_2 = option2 + right

    return {
        "prefix": prefix,
        "choice1_text": continuation_1,
        "choice2_text": continuation_2,
        "gold": "1" if answer == "1" else "2",
        "gold_letter": "A" if answer == "1" else "B",
        "completed_1": left + continuation_1,
        "completed_2": left + continuation_2,
    }


# ============================================================
# LOG-LIKELIHOOD SCORING
# ============================================================

def score_continuation_loglikelihood(
    model,
    tokenizer,
    prefix_text: str,
    continuation_text: str,
    max_input_tokens: int,
) -> float:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation_text, add_special_tokens=False)["input_ids"]

    if len(continuation_ids) == 0:
        raise ValueError("Continuation tokenized to empty sequence.")

    total_ids = prefix_ids + continuation_ids

    # Keep all continuation tokens; truncate prefix from the left if needed.
    if len(total_ids) > max_input_tokens:
        overflow = len(total_ids) - max_input_tokens
        if overflow >= len(prefix_ids):
            raise RuntimeError(
                "Continuation is too long to fit within MAX_INPUT_TOKENS. "
                "Increase MAX_INPUT_TOKENS."
            )
        prefix_ids = prefix_ids[overflow:]
        total_ids = prefix_ids + continuation_ids

    input_ids = torch.tensor([total_ids], device=DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    prefix_len = len(prefix_ids)
    cont_len = len(continuation_ids)

    # Positions in target_ids corresponding to continuation tokens
    start_idx = prefix_len - 1
    end_idx = start_idx + cont_len

    cont_log_probs = log_probs[:, start_idx:end_idx, :]
    cont_targets = target_ids[:, start_idx:end_idx]

    token_log_probs = cont_log_probs.gather(2, cont_targets.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def predict_winogrande_choice(
    model,
    tokenizer,
    prefix: str,
    continuation_1: str,
    continuation_2: str,
) -> Dict[str, Any]:
    ll_1 = score_continuation_loglikelihood(
        model=model,
        tokenizer=tokenizer,
        prefix_text=prefix,
        continuation_text=continuation_1,
        max_input_tokens=MAX_INPUT_TOKENS,
    )
    ll_2 = score_continuation_loglikelihood(
        model=model,
        tokenizer=tokenizer,
        prefix_text=prefix,
        continuation_text=continuation_2,
        max_input_tokens=MAX_INPUT_TOKENS,
    )

    pred = "1" if ll_1 > ll_2 else "2"
    pred_letter = "A" if pred == "1" else "B"

    return {
        "ll_option1": ll_1,
        "ll_option2": ll_2,
        "pred": pred,
        "pred_letter": pred_letter,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_winogrande_splits():
    train_ds = load_dataset("allenai/winogrande", WINOGRANDE_CONFIG, split="train")
    val_ds = load_dataset("allenai/winogrande", WINOGRANDE_CONFIG, split="validation")
    return train_ds, val_ds


def sample_fewshot_examples(train_ds, num_shots: int, seed: int):
    shuffled = train_ds.shuffle(seed=seed)
    fewshot = []
    for ex in shuffled:
        if (
            ex.get("sentence") is not None
            and ex.get("option1") is not None
            and ex.get("option2") is not None
            and str(ex.get("answer")).strip() in {"1", "2"}
            and "_" in str(ex.get("sentence"))
        ):
            fewshot.append(ex)
        if len(fewshot) == num_shots:
            break

    if len(fewshot) < num_shots:
        raise RuntimeError(f"Could only collect {len(fewshot)} few-shot examples.")

    return fewshot


def sample_eval_examples(val_ds, num_samples: int, seed: int):
    shuffled = val_ds.shuffle(seed=seed)
    valid_examples = []
    for ex in shuffled:
        if (
            ex.get("sentence") is not None
            and ex.get("option1") is not None
            and ex.get("option2") is not None
            and str(ex.get("answer")).strip() in {"1", "2"}
            and "_" in str(ex.get("sentence"))
        ):
            valid_examples.append(ex)
        if len(valid_examples) == num_samples:
            break

    return valid_examples


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

    print("\nLoading Winogrande...")
    train_ds, val_ds = load_winogrande_splits()
    fewshot_examples = sample_fewshot_examples(train_ds, NUM_SHOTS, SHUFFLE_SEED)
    eval_examples = sample_eval_examples(val_ds, SAMPLES_PER_DATASET, SHUFFLE_SEED)
    fewshot_prefix = build_fewshot_prefix(fewshot_examples)

    print(f"Using {len(fewshot_examples)} few-shot examples.")
    print(f"Evaluating on {len(eval_examples)} validation examples.")

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        clear_memory_and_cache()
        model, tokenizer = load_model_and_tokenizer(model_name)

        correct = 0
        total = 0
        debug_printed = 0
        errors = 0

        per_example_records = []

        for idx, sample in enumerate(eval_examples):
            try:
                built = build_prefix_and_continuations(sample, fewshot_prefix)

                result = predict_winogrande_choice(
                    model=model,
                    tokenizer=tokenizer,
                    prefix=built["prefix"],
                    continuation_1=built["choice1_text"],
                    continuation_2=built["choice2_text"],
                )

                total += 1
                is_correct = result["pred"] == built["gold"]
                if is_correct:
                    correct += 1

                if debug_printed < DEBUG_EXAMPLES:
                    print("\n[EXAMPLE]")
                    print("Gold:", built["gold_letter"])
                    print("Completed option A:", built["completed_1"])
                    print("Completed option B:", built["completed_2"])
                    print(f"loglik A: {result['ll_option1']:.4f}")
                    print(f"loglik B: {result['ll_option2']:.4f}")
                    print("Prediction:", result["pred_letter"])
                    print("Correct:", is_correct)
                    debug_printed += 1

                per_example_records.append(
                    {
                        "index": idx,
                        "gold": built["gold_letter"],
                        "pred": result["pred_letter"],
                        "correct": is_correct,
                        "ll_option1": result["ll_option1"],
                        "ll_option2": result["ll_option2"],
                    }
                )

                if total % 50 == 0 or total == len(eval_examples):
                    print(f"  processed {total}/{len(eval_examples)}...")

            except Exception as exc:
                errors += 1
                print(f"[ERROR] example {idx}: {repr(exc)}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        accuracy = correct / total if total else 0.0

        model_results = {
            "model": model_name,
            "device": DEVICE,
            "dataset": "winogrande",
            "winogrande_config": WINOGRANDE_CONFIG,
            "evaluation_style": "5-shot choice-based suffix log-likelihood",
            "samples_per_dataset": SAMPLES_PER_DATASET,
            "evaluated_examples": total,
            "shuffle_seed": SHUFFLE_SEED,
            "num_shots": NUM_SHOTS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
            "fewshot_examples": [
                {
                    "sentence": ex["sentence"],
                    "option1": ex["option1"],
                    "option2": ex["option2"],
                    "answer": ex["answer"],
                }
                for ex in fewshot_examples
            ],
            "predictions": per_example_records,
        }

        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Errors: {errors}")

        safe_name = model_name.replace("/", "__").replace(":", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)

        print(f"Saved: {out_path}")

        del model
        del tokenizer
        clear_memory_and_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()