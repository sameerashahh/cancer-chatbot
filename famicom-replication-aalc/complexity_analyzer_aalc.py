#!/usr/bin/env python3
"""
Prompt Complexity Measurement using DeepSeek-R1-Distill-Qwen-7B
Complexity Metric: Number of Tokens in the Model's Generated Output
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

# ================= CONFIG ================= #
MODEL_NAME = "du-lab/AALC-DeepSeek-R1-Distill-Qwen-7B-1024"
INPUT_FILE = "../basic_pipeline/questions_binary.json"
OUTPUT_FILE = "complexity_results.json"
NUM_PROMPTS = 1000
MAX_NEW_TOKENS = 256   # allow the model to produce full answers
TEMPERATURE = 0.7
BATCH_SIZE = 2          # increase to 4 if GPU allows

# ================= FUNCTIONS ================= #

def load_prompts(filepath: str) -> List[Dict[str, Any]]:
    """Load prompts (dataset with 'question')"""
    with open(filepath, "r") as f:
        data = json.load(f)
        return data[:NUM_PROMPTS]


def initialize_model():
    """Load tokenizer and model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model, device


@torch.no_grad()
def generate_responses(tokenizer, model, device, questions: List[str]) -> List[Dict[str, any]]:
    """
    Generate model answers for given questions and compute output-token complexity.
    Returns a list of dicts with 'response' and 'complexity' (token count).
    """
    messages = [{"role": "user", "content": q} for q in questions]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate answers
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Compute token lengths and decode outputs
    results = []
    for i in range(len(questions)):
        full_output = outputs[i]
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = full_output[input_len:]  # new tokens only
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        complexity = len(gen_tokens)
        results.append({
            "response": gen_text,
            "complexity": complexity
        })
    return results


def process_prompts(tokenizer, model, device, prompts: List[Dict[str, Any]]):
    """Process prompts in batches and compute complexity based on output length"""
    results = []
    total = len(prompts)

    for start in range(0, total, BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        questions = [item["question"] for item in batch]

        batch_results = generate_responses(tokenizer, model, device, questions)

        for i, res in enumerate(batch_results):
            item_result = {
                "question": questions[i][:200],
                "dataset": batch[i].get("dataset", ""),
                "subtask": batch[i].get("subtask", ""),
                "true_answer": batch[i].get("true_answer", ""),
                "response": res["response"],
                "length_complexity": res["complexity"]
            }
            results.append(item_result)

        save_results(results)
        print(f"Processed {start + len(batch)}/{total} prompts...")

    return results


def save_results(results: List[Dict[str, Any]]):
    """Save partial results to JSON"""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ================= MAIN ================= #

def main():
    tokenizer, model, device = initialize_model()
    prompts = load_prompts(INPUT_FILE)

    print(f"Processing {len(prompts)} prompts (complexity = output length)...")
    results = process_prompts(tokenizer, model, device, prompts)

    print("\n=== Summary Statistics ===")
    lengths = [r["length_complexity"] for r in results]
    print(f"Mean: {np.mean(lengths):.2f}, Std: {np.std(lengths):.2f}")
    print(f"Range: {min(lengths)} – {max(lengths)} tokens")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
