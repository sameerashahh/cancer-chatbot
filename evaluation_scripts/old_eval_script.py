import re
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# =============================
# MODELS (10 TOTAL)
# =============================

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
MAX_NEW_TOKENS = 128

# Run 200 samples per dataset
SAMPLES_PER_DATASET = 200
SHUFFLE_SEED = 42

# Deterministic (no self-consistency, no sampling)
DO_SAMPLE = False

OUTPUT_DIR = "results_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")

# =============================
# EXTRACTION HELPERS
# =============================

def extract_with_fallback(out, pattern, fallback_pattern=None, flags=re.I):
    strict = re.search(pattern, out, flags)
    if strict:
        return strict.group(1).lower()
    if fallback_pattern:
        fallback = re.search(fallback_pattern, out, flags)
        if fallback:
            return fallback.group(1).lower()
    return None

def extract_number_with_fallback(out):
    strict = re.search(r"Final answer:\s*(-?\d+)", out)
    if strict:
        return strict.group(1)
    nums = re.findall(r"-?\d+", out)
    return nums[-1] if nums else None

def safe_upper_letter(x):
    return x.upper() if isinstance(x, str) else None

# =============================
# DATASETS (ALL 10)
# =============================

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

End your response with:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(positive|negative)",
            r"\b(positive|negative)\b"
        ),
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

End your response with:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(positive|negative)",
            r"\b(positive|negative)\b"
        ),
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

End your response with:
Final answer: positive
or
Final answer: negative
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(positive|negative)",
            r"\b(positive|negative)\b"
        ),
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

End your response with:
Final answer: hate
or
Final answer: not_hate
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(hate|not_hate)",
            r"\b(hate|not_hate)\b"
        ),
    },

    "boolq": {
        "path": "boolq",
        "split": "validation",
        "text": lambda x: f"Passage: {x['passage']}\nQuestion: {x['question']}",
        "reference": lambda x: "yes" if x["answer"] else "no",
        "prompt": lambda t: f"""
Answer the question based on the passage.

{t}

End your response with:
Final answer: yes
or
Final answer: no
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(yes|no)",
            r"\b(yes|no)\b"
        ),
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

End your response with:
Final answer: world
or sports
or business
or sci_tech
""",
        "extract": lambda out: extract_with_fallback(
            out, r"Final answer:\s*(world|sports|business|sci_tech)",
            r"\b(world|sports|business|sci_tech)\b"
        ),
    },

    "dbpedia_14": {
        "path": "dbpedia_14",
        "split": "test",
        "text": lambda x: f"{x['title']} {x['content']}",
        "reference": lambda x: str(x["label"]),
        "prompt": lambda t: f"""
Classify into one of 14 categories labeled 0–13.

Text:
{t}

End your response with:
Final answer: <number between 0 and 13>
""",
        "extract": lambda out: extract_number_with_fallback(out),
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

End your response with:
Final answer: <number between 0 and 76>
""",
        "extract": lambda out: extract_number_with_fallback(out),
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

End your response with:
Final answer: <number>
""",
        "extract": lambda out: extract_number_with_fallback(out),
    },

    "aqua_rat": {
        "path": "aqua_rat",
        "split": "test",
        "text": lambda x: f"{x['question']}\nOptions: {x['options']}",
        "reference": lambda x: safe_upper_letter(x.get("correct", x.get("answer"))),
        "prompt": lambda t: f"""
Solve the problem.

{t}

End your response with:
Final answer: A
or B
or C
or D
or E
""",
        "extract": lambda out: (
            safe_upper_letter(
                extract_with_fallback(
                    out, r"Final answer:\s*([A-E])",
                    r"\b([A-E])\b"
                )
            )
        ),
    },
}

# =============================
# EVAL LOOP (NO SELF-CONSISTENCY)
# =============================

def make_inputs(tokenizer, prompt, is_seq2seq):
    if (not is_seq2seq) and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    return inputs.to(DEVICE)

def decode_output(tokenizer, outputs, inputs, is_seq2seq):
    if is_seq2seq:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # causal: strip prompt tokens
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

for MODEL_NAME in MODELS:
    print("\n" + "=" * 80)
    print(f"MODEL: {MODEL_NAME}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Some causal LMs need an explicit pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in MODEL_NAME.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype="auto"
        ).to(DEVICE)
        is_seq2seq = False

    model.eval()

    model_results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "shuffle_seed": SHUFFLE_SEED,
        "datasets": {}
    }

    for ds_name, cfg in DATASETS.items():
        print("\n" + "-" * 60)
        print(f"DATASET: {ds_name}")
        print("-" * 60)

        dataset = load_dataset(cfg["path"], cfg.get("config"), split=cfg["split"])

        # Ensure we can take 200 samples even if dataset is smaller
        n = min(SAMPLES_PER_DATASET, len(dataset))
        dataset = dataset.shuffle(seed=SHUFFLE_SEED).select(range(n))

        correct = 0
        total = 0
        none_preds = 0

        for i in range(n):
            sample = dataset[i]
            text = cfg["text"](sample)
            prompt = cfg["prompt"](text)
            reference = cfg["reference"](sample)

            inputs = make_inputs(tokenizer, prompt, is_seq2seq)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id,
                )

            decoded = decode_output(tokenizer, outputs, inputs, is_seq2seq)
            pred = cfg["extract"](decoded)

            total += 1
            if pred is None:
                none_preds += 1
            if pred == reference:
                correct += 1

            if (i + 1) % 25 == 0:
                print(f"  processed {i+1}/{n}...")

        accuracy = correct / total if total else 0.0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total}), None preds: {none_preds}")

        model_results["datasets"][ds_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "none_predictions": none_preds,
        }

    # Write one JSON per model (contains all 10 dataset accuracies)
    safe_name = MODEL_NAME.replace("/", "__").replace(":", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(model_results, f, indent=2)

    print(f"\nSaved: {out_path}")

print("\nDone. Wrote 1 JSON file per model, each containing 10 dataset accuracies.")