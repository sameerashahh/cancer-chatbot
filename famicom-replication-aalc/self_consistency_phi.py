#!/usr/bin/env python3
"""
Self-Consistency Measurement using DeepSeek-R1-Distill-Qwen-7B on SOL
Metric: Majority agreement across N stochastic samples
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from collections import Counter

# ================= CONFIG ================= #
MODEL_NAME = "./phi_3_mini_128k"
INPUT_FILE = "questions_binary.json"
OUTPUT_FILE = "self_consistency_results.json"

NUM_PROMPTS = 1000
N_SAMPLES = 20                  # number of samples per prompt
MAX_NEW_TOKENS = 16             # keep small for binary classification
TEMPERATURE = 1.0               # must sample stochastically
BATCH_SIZE = 1                  # self-consistency requires sequential sampling

# ================= HELPERS ================= #

def extract_choice(text: str):
    text = text.strip()
    if text.startswith("(A)") or text.startswith("A"):
        return "A"
    if text.startswith("(B)") or text.startswith("B"):
        return "B"

    # fallback: scan early text
    t = text[:10].upper()
    if "A" in t: return "A"
    if "B" in t: return "B"
    return None


# ================= SOL LOADING ================= #

def load_prompts(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
        return data[:NUM_PROMPTS]

def initialize_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_NAME} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.config.use_cache = True
    return tokenizer, model, device


# ================= GENERATION ================= #

@torch.no_grad()
def sample_once(tokenizer, model, device, question: str) -> str:
    """Generate one stochastic sample from the model"""
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Move dict tensors to device
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs.to(device)}
    else:
        for k, v in inputs.items():
            inputs[k] = v.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # get output tokens (skip input prompt)
    output_ids = outputs[0]
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = output_ids[input_len:]

    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


def evaluate_self_consistency(tokenizer, model, device, question: str) -> dict:
    """Run N samples, compute majority self-consistency"""
    samples = []
    labels = []

    for _ in range(N_SAMPLES):
        out = sample_once(tokenizer, model, device, question)
        lbl = extract_choice(out)
        samples.append(out)
        labels.append(lbl)

    # remove None predictions
    filtered = [x for x in labels if x is not None]

    if len(filtered) == 0:
        return {
            "majority_label": None,
            "self_consistency": 0.0,
            "samples": samples,
            "labels": labels
        }

    counts = Counter(filtered)
    majority_label, majority_votes = counts.most_common(1)[0]
    consistency = majority_votes / N_SAMPLES

    return {
        "majority_label": majority_label,
        "self_consistency": consistency,
        "counts": dict(counts),
        "samples": samples,
        "labels": labels
    }


# ================= MAIN PROCESS ================= #

def process_prompts(tokenizer, model, device, prompts):
    results = []
    total = len(prompts)

    for i, item in enumerate(prompts):
        q = item["question"]

        eval_res = evaluate_self_consistency(tokenizer, model, device, q)

        results.append({
            "question": q[:200],
            "dataset": item.get("dataset", ""),
            "subtask": item.get("subtask", ""),
            "true_answer": item.get("true_answer", ""),

            "self_consistency": eval_res["self_consistency"],
            "majority_label": eval_res["majority_label"],
            "counts": eval_res.get("counts", {}),
            "labels": eval_res["labels"],
            "samples": eval_res["samples"]
        })

        save_results(results)
        print(f"Processed {i+1}/{total} prompts | SC={eval_res['self_consistency']:.2f}")

    return results


def save_results(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ================= MAIN ================= #

def main():
    tokenizer, model, device = initialize_model()
    prompts = load_prompts(INPUT_FILE)
    print(f"Running self-consistency on {len(prompts)} prompts...")
    results = process_prompts(tokenizer, model, device, prompts)

    # Summary statistics
    sc_vals = [r["self_consistency"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean SC: {np.mean(sc_vals):.3f}")
    print(f"Std  SC: {np.std(sc_vals):.3f}")
    print(f"Min–Max: {min(sc_vals):.3f} – {max(sc_vals):.3f}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
