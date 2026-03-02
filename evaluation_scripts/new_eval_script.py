import re
import os
import json
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "NoesisLab/Spartacus-1B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat",
    "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_PER_DATASET = 200
SHUFFLE_SEED = 42

BATCH_SIZE = 8           # requested
MAX_NEW_TOKENS = 64      # lower helps reduce rambling + duplicated "Final answer"
DO_SAMPLE = False

PRINT_EXAMPLES = 10      # requested
PRINT_TEXT_CHARS = 240
PRINT_GEN_CHARS = 320

OUTPUT_DIR = "results_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")

# ============================================================
# PARSING HELPERS (FIXED)
# ============================================================

def _last_capture(findall_result):
    """Return last capture from re.findall results (string or tuple)."""
    if not findall_result:
        return None
    last = findall_result[-1]
    if isinstance(last, tuple):
        last = last[0]
    return last

def extract_final_answer_last(out: str, choices_regex: str, flags=re.I):
    """
    Robust parsing:
      1) find ALL occurrences of "Final answer: <choice>" and take the LAST one
      2) if none, find ALL standalone <choice> and take the LAST one
    Returns: (pred_or_None, source)
    """
    # Strict: take last "Final answer: X"
    strict_matches = re.findall(rf"Final answer:\s*({choices_regex})", out, flags)
    last = _last_capture(strict_matches)
    if last is not None:
        return str(last).lower(), "strict_last"

    # Fallback: last standalone token
    fb_matches = re.findall(rf"\b({choices_regex})\b", out, flags)
    last = _last_capture(fb_matches)
    if last is not None:
        return str(last).lower(), "fallback_last"

    return None, "none"

def extract_number_final_answer_last(out: str):
    """
    For number tasks:
      1) take last "Final answer: <int>"
      2) else take last integer in text
    Returns: (pred_or_None, source)
    """
    strict = re.findall(r"Final answer:\s*(-?\d+)", out)
    last = _last_capture(strict)
    if last is not None:
        return str(last), "strict_last"

    nums = re.findall(r"-?\d+", out)
    last = _last_capture(nums)
    if last is not None:
        return str(last), "fallback_last_number"

    return None, "none"

def safe_upper_letter(x):
    return x.upper() if isinstance(x, str) else None

def snip(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + ("..." if len(s) > n else "")

def snip_gen(s: str, n: int) -> str:
    s = s.strip()
    return s[:n] + ("\n...<truncated>..." if len(s) > n else "")

# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "imdb": {
        "path": "imdb",
        "split": "test",
        "text": lambda x: x["text"],
        "reference": lambda x: {0: "negative", 1: "positive"}[x["label"]],
        "prompt": lambda t: f"""
You are a sentiment classifier.
Decide whether the sentiment is POSITIVE or NEGATIVE.

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_final_answer_last(out, r"positive|negative"),
    },

    "yelp_polarity": {
        "path": "yelp_polarity",
        "split": "test",
        "text": lambda x: x["text"],
        "reference": lambda x: {0: "negative", 1: "positive"}[x["label"]],
        "prompt": lambda t: f"""
You are a sentiment classifier.
Decide whether the sentiment is POSITIVE or NEGATIVE.

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_final_answer_last(out, r"positive|negative"),
    },

    "sst2": {
        "path": "glue",
        "config": "sst2",
        "split": "validation",
        "text": lambda x: x["sentence"],
        "reference": lambda x: {0: "negative", 1: "positive"}[x["label"]],
        "prompt": lambda t: f"""
You are a sentiment classifier.
Decide whether the sentiment is POSITIVE or NEGATIVE.

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_final_answer_last(out, r"positive|negative"),
    },

    "tweet_eval_hate": {
        "path": "tweet_eval",
        "config": "hate",
        "split": "test",
        "text": lambda x: x["text"],
        "reference": lambda x: {0: "not_hate", 1: "hate"}[x["label"]],
        "prompt": lambda t: f"""
You are a hate speech classifier.
Classify as HATE or NOT_HATE.

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: hate
or
Final answer: not_hate
""",
        "extract": lambda out: extract_final_answer_last(out, r"hate|not_hate"),
    },

    "boolq": {
        "path": "boolq",
        "split": "validation",
        "text": lambda x: f"Passage: {x['passage']}\nQuestion: {x['question']}",
        "reference": lambda x: "yes" if x["answer"] else "no",
        "prompt": lambda t: f"""
Answer the question based on the passage.

{t}

Answer with ONLY one line in exactly this format:
Final answer: yes
or
Final answer: no
""",
        "extract": lambda out: extract_final_answer_last(out, r"yes|no"),
    },

    "ag_news": {
        "path": "ag_news",
        "split": "test",
        "text": lambda x: x["text"],
        "reference": lambda x: {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}[x["label"]],
        "prompt": lambda t: f"""
Classify the news article.

Categories:
world
sports
business
sci_tech

Article:
{t}

Answer with ONLY one line in exactly this format:
Final answer: world
or Final answer: sports
or Final answer: business
or Final answer: sci_tech
""",
        "extract": lambda out: extract_final_answer_last(out, r"world|sports|business|sci_tech"),
    },

    # NOTE: dbpedia_14 & banking77 numeric IDs are not meaningful without label-name mapping.
    # We keep them here, but parsing is now correct (last final answer, else last number).
    "dbpedia_14": {
        "path": "dbpedia_14",
        "split": "test",
        "text": lambda x: f"{x['title']} {x['content']}",
        "reference": lambda x: str(x["label"]),
        "prompt": lambda t: f"""
Classify into one of 14 categories labeled 0–13.

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: <number between 0 and 13>
""",
        "extract": lambda out: extract_number_final_answer_last(out),
    },

    "banking77": {
        "path": "banking77",
        "split": "test",
        "text": lambda x: x["text"],
        "reference": lambda x: str(x["label"]),
        "prompt": lambda t: f"""
Classify into one of 77 intent labels (0–76).

Text:
{t}

Answer with ONLY one line in exactly this format:
Final answer: <number between 0 and 76>
""",
        "extract": lambda out: extract_number_final_answer_last(out),
    },

    "gsm8k": {
        "path": "gsm8k",
        "config": "main",
        "split": "test",
        "text": lambda x: x["question"],
        "reference": lambda x: re.search(r"-?\d+", x["answer"]).group(),
        "prompt": lambda t: f"""
Solve the math problem step by step.

Problem:
{t}

End with ONLY one line in exactly this format:
Final answer: <number>
""",
        "extract": lambda out: extract_number_final_answer_last(out),
    },

    "aqua_rat": {
        "path": "aqua_rat",
        "split": "test",
        "text": lambda x: f"{x['question']}\nOptions: {x['options']}",
        "reference": lambda x: safe_upper_letter(x.get("correct", x.get("answer"))),
        "prompt": lambda t: f"""
Solve the problem.

{t}

Answer with ONLY one line in exactly this format:
Final answer: A
or Final answer: B
or Final answer: C
or Final answer: D
or Final answer: E
""",
        "extract": lambda out: (
            (lambda p, src: (safe_upper_letter(p) if p is not None else None, src))(
                *extract_final_answer_last(out, r"[A-E]")
            )
        ),
    },
}

# ============================================================
# BATCH TOKENIZATION + PROMPT-STRIPPING DECODE (CAUSAL)
# ============================================================

def make_batch_inputs(tokenizer, prompts, is_seq2seq):
    if (not is_seq2seq) and hasattr(tokenizer, "apply_chat_template"):
        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=False,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)

    inputs = inputs.to(DEVICE)

    input_lengths = None
    if not is_seq2seq:
        # robust per-row length
        if "attention_mask" in inputs:
            input_lengths = inputs["attention_mask"].sum(dim=1)
        else:
            input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    return inputs, input_lengths

def decode_batch_outputs(tokenizer, outputs, input_lengths, is_seq2seq):
    if is_seq2seq:
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    decoded = []
    for i in range(outputs.size(0)):
        start = int(input_lengths[i].item()) if input_lengths is not None else 0
        gen_tokens = outputs[i, start:]
        decoded.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return decoded

# ============================================================
# OOM SAFE GENERATION
# ============================================================

def generate_oom_safe(model, tokenizer, prompts, is_seq2seq):
    results = []
    bs = len(prompts)
    i = 0
    while i < len(prompts):
        chunk = prompts[i:i+bs]
        try:
            inputs, input_lengths = make_batch_inputs(tokenizer, chunk, is_seq2seq)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id,
                )
            results.extend(decode_batch_outputs(tokenizer, outputs, input_lengths, is_seq2seq))
            i += bs
        except torch.cuda.OutOfMemoryError:
            if DEVICE != "cuda":
                raise
            torch.cuda.empty_cache()
            bs //= 2
            if bs < 1:
                raise RuntimeError("OOM even with batch_size=1. Reduce MAX_NEW_TOKENS or model size.")
            print(f"  [OOM] reducing sub-batch size -> {bs} and retrying...")
    return results

# ============================================================
# MAIN LOOP (won't stop on dataset errors)
# ============================================================

for model_name in MODELS:
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto").to(DEVICE)
        is_seq2seq = True
    else:
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype="auto"
        ).to(DEVICE)
        is_seq2seq = False

    model.eval()

    model_results = {
        "model": model_name,
        "device": DEVICE,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "shuffle_seed": SHUFFLE_SEED,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "datasets": {},
    }

    for ds_name, cfg in DATASETS.items():
        print("\n" + "-" * 60)
        print(f"DATASET: {ds_name}")
        print("-" * 60)

        try:
            ds = load_dataset(cfg["path"], cfg.get("config"), split=cfg["split"])
            n = min(SAMPLES_PER_DATASET, len(ds))
            ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(n))

            correct = 0
            total = 0
            none_preds = 0

            pred_counter = Counter()
            ref_counter = Counter()
            extract_counter = Counter()

            examples_printed = 0

            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = [ds[i] for i in range(start, end)]

                texts = [cfg["text"](s) for s in batch]
                prompts = [cfg["prompt"](t) for t in texts]
                refs = [cfg["reference"](s) for s in batch]

                decoded_list = generate_oom_safe(model, tokenizer, prompts, is_seq2seq)

                for t, decoded, ref in zip(texts, decoded_list, refs):
                    pred, src = cfg["extract"](decoded)

                    total += 1
                    ref_counter[ref] += 1
                    extract_counter[src] += 1

                    if pred is None:
                        none_preds += 1
                    else:
                        pred_counter[pred] += 1

                    if pred == ref:
                        correct += 1

                    if examples_printed < PRINT_EXAMPLES:
                        examples_printed += 1
                        print(f"\n[EXAMPLE {examples_printed:02d}]")
                        print("Text:", snip(t, PRINT_TEXT_CHARS))
                        print("Ref :", ref)
                        print("Pred:", pred, f"(extract={src})")
                        print("Gen :\n" + snip_gen(decoded, PRINT_GEN_CHARS))

                if end % 25 == 0 or end == n:
                    print(f"  processed {end}/{n}...")

            acc = correct / total if total else 0.0
            print(f"Accuracy: {acc:.4f} ({correct}/{total}), None preds: {none_preds}")
            print("Extractor usage:", dict(extract_counter))
            print("Pred dist      :", dict(pred_counter))
            print("Ref dist       :", dict(ref_counter))

            model_results["datasets"][ds_name] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "none_predictions": none_preds,
                "extractor_usage": dict(extract_counter),
                "pred_distribution": dict(pred_counter),
                "ref_distribution": dict(ref_counter),
            }

        except Exception as e:
            print(f"[ERROR] {model_name} on {ds_name}: {repr(e)}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            model_results["datasets"][ds_name] = {"error": repr(e)}
            continue

    safe_name = model_name.replace("/", "__").replace(":", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)

    print(f"\nSaved: {out_path}")

print("\nDone.")