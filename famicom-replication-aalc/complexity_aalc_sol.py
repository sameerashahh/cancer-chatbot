#!/usr/bin/env python3
"""
Prompt Complexity Measurement using DeepSeek-R1-Distill-Qwen-7B
Complexity Metric: Number of Tokens in the Model's Generated Output (CoT)
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

# ================= CONFIG ================= #
MODEL_NAME = "./complexity_model_1_5B"  # path on Sol
INPUT_FILE = "questions_binary.json"
OUTPUT_FILE = "complexity_results.json"
NUM_PROMPTS = 1000
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
BATCH_SIZE = 2

# ================= FUNCTIONS ================= #

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

@torch.no_grad()
def generate_responses(tokenizer, model, device, questions: list) -> list[dict]:
    """Generate responses and compute token complexity (per prompt)"""
    results = []

    for question in questions:
        messages = [{"role": "user", "content": question}]

        # Tokenize / apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Ensure inputs is a dict and move to device
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs.to(device)}
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
        else:
            raise TypeError(f"Unexpected inputs type: {type(inputs)}")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Flatten output to tensor if needed
        output_ids = outputs if outputs.dim() == 1 else outputs[0]

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = output_ids[input_len:]  # only new tokens
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        results.append({
            "response": gen_text,
            "complexity": len(gen_tokens)
        })

    return results


def process_prompts(tokenizer, model, device, prompts: List[Dict[str, Any]]):
    results = []
    total = len(prompts)
    for start in range(0, total, BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        questions = [item["question"] for item in batch]

        batch_results = generate_responses(tokenizer, model, device, questions)

        for i, res in enumerate(batch_results):
            results.append({
                "question": questions[i][:200],
                "dataset": batch[i].get("dataset", ""),
                "subtask": batch[i].get("subtask", ""),
                "true_answer": batch[i].get("true_answer", ""),
                "response": res["response"],
                "length_complexity": res["complexity"]
            })

        save_results(results)
        print(f"Processed {start + len(batch)}/{total} prompts...")

    return results

def save_results(results: List[Dict[str, Any]]):
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
