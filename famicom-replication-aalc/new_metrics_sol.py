#!/usr/bin/env python3
"""
Prompt Evaluation Script (Token-Lite Version)

Metrics:
1. Self-Confidence Calibration (0–1)
2. Self-Verification / Self-Contradiction Detection
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

# ================= CONFIG ================= #
MODEL_NAME = "./complexity_model_1_5B"
INPUT_FILE = "questions_binary.json"
OUTPUT_FILE = "evaluation_results.json"
NUM_PROMPTS = 1000
TEMPERATURE = 0.7
BATCH_SIZE = 1

# New minimal token budgets
MAX_ANSWER_TOKENS = 256
MAX_CONFIDENCE_TOKENS = 16
MAX_VERIFICATION_TOKENS = 128

# ================= PROMPTS ================= #

CONFIDENCE_PROMPT = """
You will now rate your confidence in the answer you just gave.
Return only a number between 0 and 1.
"""

SELF_VERIFICATION_PROMPT = """
Review your previous answer carefully.

Return a JSON object in this structure:

{
  "errors_detected": <number>,
  "explanation": "<short explanation>"
}

Only output the JSON.
"""

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
def generate_output(tokenizer, model, device, messages, max_tokens):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_ids = outputs[0]
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[input_len:]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def ask_followup(tokenizer, model, device, prompt, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    return generate_output(tokenizer, model, device, messages, max_tokens)


def process_prompts(tokenizer, model, device, prompts: List[Dict[str, Any]]):
    results = []
    total = len(prompts)

    for idx, item in enumerate(prompts):
        question = item["question"]
        print(f"Processing {idx+1}/{total}...")

        # 1. MAIN ANSWER
        answer = generate_output(
            tokenizer, model, device,
            [{"role": "user", "content": question}],
            max_tokens=MAX_ANSWER_TOKENS
        )

        # 2. CONFIDENCE
        conf_text = ask_followup(
            tokenizer, model, device,
            CONFIDENCE_PROMPT,
            max_tokens=MAX_CONFIDENCE_TOKENS
        )
        try:
            confidence = float(conf_text.strip())
        except:
            confidence = None

        # 3. SELF-VERIFICATION
        verification_prompt = SELF_VERIFICATION_PROMPT.replace(
            "your previous answer", answer
        )

        verification_output = ask_followup(
            tokenizer, model, device,
            verification_prompt,
            max_tokens=MAX_VERIFICATION_TOKENS
        )

        try:
            verification = json.loads(verification_output)
        except:
            verification = {
                "errors_detected": None,
                "explanation": verification_output.strip()
            }

        # Store record
        results.append({
            "question": question,
            "dataset": item.get("dataset", ""),
            "subtask": item.get("subtask", ""),
            "true_answer": item.get("true_answer", ""),
            "response": answer,
            "confidence": confidence,
            "self_verification": verification
        })

        save_results(results)

    return results


def save_results(results: List[Dict[str, Any]]):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ================= MAIN ================= #

def main():
    tokenizer, model, device = initialize_model()
    prompts = load_prompts(INPUT_FILE)

    print(f"Evaluating {len(prompts)} prompts...")

    results = process_prompts(tokenizer, model, device, prompts)

    confs = [r["confidence"] for r in results if isinstance(r["confidence"], float)]
    errs = [r["self_verification"]["errors_detected"]
            for r in results
            if isinstance(r["self_verification"].get("errors_detected"), int)]

    print("\n=== Summary ===")
    if confs:
        print(f"Mean Confidence: {np.mean(confs):.3f}")
    if errs:
        print(f"Mean Self-Verification Errors: {np.mean(errs):.3f}")

    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
