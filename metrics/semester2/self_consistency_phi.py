#!/usr/bin/env python3
"""
Minimal Self-Consistency with Boxed Answer Template
Binary task: A / B
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from typing import List, Dict, Any
from tqdm import tqdm

# ================= CONFIG ================= #

MODEL_NAME = "./phi_3_mini_128k"
INPUT_FILE = "questions_binary.json"
OUTPUT_FILE = "self_consistency_results.json"

NUM_PROMPTS = 20
N_SAMPLES = 10        
MAX_NEW_TOKENS = 16
TEMPERATURE = 1.0  

# ================= PROMPT ================= #

def format_prompt(question: str) -> str:
    """
    Boxed answer template.
    Model must output exactly one letter in the box.
    """
    return f"""{question}

Answer using EXACTLY the following format:

┌─────────┐
│   A     │
└─────────┘

or

┌─────────┐
│   B     │
└─────────┘
"""

# ================= EXTRACTION ================= #

def extract_choice(text: str):
    """Extract A or B from the boxed output."""
    text = text.upper()
    if "│   A" in text:
        return "A"
    if "│   B" in text:
        return "B"
    return None

# ================= DATA ================= #

def load_prompts(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data[:NUM_PROMPTS]

def initialize_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model, device

# ================= GENERATION ================= #

@torch.no_grad()
def sample_once(tokenizer, model, device, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    gen_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# ================= SELF-CONSISTENCY ================= #

def evaluate_self_consistency(tokenizer, model, device, question: str) -> Dict[str, Any]:
    prompt = format_prompt(question)
    labels = []

    for _ in range(N_SAMPLES):
        out = sample_once(tokenizer, model, device, prompt)
        lbl = extract_choice(out)
        labels.append(lbl)

    valid_labels = [l for l in labels if l is not None]
    coverage = len(valid_labels) / N_SAMPLES

    if len(valid_labels) == 0:
        return {
            "self_consistency": 0.0,
            "coverage": 0.0,
            "majority_label": None,
            "counts": {}
        }

    counts = Counter(valid_labels)
    majority_label, majority_votes = counts.most_common(1)[0]
    self_consistency = majority_votes / N_SAMPLES   # classical SC over all samples

    return {
        "self_consistency": self_consistency,
        "coverage": coverage,
        "majority_label": majority_label,
        "counts": dict(counts)
    }

# ================= MAIN ================= #

def main():
    tokenizer, model, device = initialize_model()
    prompts = load_prompts(INPUT_FILE)

    results = []
    for item in tqdm(prompts, desc="Self-consistency"):
        q = item["question"]
        eval_res = evaluate_self_consistency(tokenizer, model, device, q)

        results.append({
            "question": q[:200],
            "true_answer": item.get("true_answer", ""),
            "self_consistency": eval_res["self_consistency"],
            "coverage": eval_res["coverage"],
            "majority_label": eval_res["majority_label"],
            "counts": eval_res["counts"]
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    sc_vals = [r["self_consistency"] for r in results]
    cov_vals = [r["coverage"] for r in results]

    print("\n=== Summary ===")
    print(f"Mean self-consistency: {np.mean(sc_vals):.3f}")
    print(f"Std self-consistency:  {np.std(sc_vals):.3f}")
    print(f"Mean coverage:         {np.mean(cov_vals):.3f}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
