#!/usr/bin/env python3
"""
Prompt Complexity Measurement using vLLM (GPU-based for Google Colab)
Guided Complexity Only - Faster Alternative
"""

import json
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import numpy as np

# Configuration
MODEL_NAME = "microsoft/phi-3-mini-128k-instruct"
OUTPUT_FILE = "complexity_results.json"
INPUT_FILE = "../data/prepared_prompts.json"
TEMPERATURE = 0.8
MAX_TOKENS = 10
NUM_PROMPTS = 1000  # Start small; scale up later

# Example demonstrations for guided complexity
GUIDED_EXAMPLES = """
Example 1:
Question: What is the capital of France?
Steps: 1) Recall knowledge about France, 2) Identify capital
Complexity: 2

Example 2:
Question: Solve: If John has 5 apples and Mary has 3, how many do they have together?
Steps: 1) Parse numbers, 2) Add values, 3) Return sum
Complexity: 3

Example 3:
Question: Explain why the sky is blue and how it relates to quantum mechanics
Steps: 1) Understand Rayleigh scattering, 2) Explain wave mechanics, 3) Connect to quantum theory, 4) Synthesize explanation
Complexity: 4
"""


def load_prompts(filepath: str) -> List[Dict[str, Any]]:
    """Load prepared prompts from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data[:NUM_PROMPTS]  # Only return first NUM_PROMPTS


def initialize_llm() -> LLM:
    """Initialize vLLM for Colab GPU (Phi-3-mini)"""
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",           # faster on GPU
        max_model_len=2048,
        disable_sliding_window=True,
        enable_prefix_caching=False,
        tensor_parallel_size=1,    # safe default for small GPU
    )
    return llm



def extract_number(text: str) -> float:
    """Extract numeric value right after 'Complexity:' or the last number in the string."""
    # Try to find 'Complexity: <number>'
    match = re.search(r"[Cc]omplexity\s*[:\-]?\s*(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))

    # Fallback: take last number if 'Complexity:' not found
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return float(numbers[-1])

    return 0.0


def measure_guided_complexity(llm: LLM, prompt: str) -> float:
    """
    Guided Complexity: Use human-written examples as guidance (ONLY METHOD)
    """
    query = f"""{GUIDED_EXAMPLES}

Now evaluate this question:
Question: {prompt}

Provide the complexity (number of steps) as a single number.
Complexity: """

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=0.95
    )

    outputs = llm.generate([query], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    complexity = extract_number(response)
    return min(complexity, 10.0)

BATCH_SIZE = 5  # Adjust between 2-5 depending on GPU memory

def process_prompts(llm: LLM, prompts: List[Dict]):
    """Process prompts in batches and measure guided complexity"""
    results = []
    num_prompts = len(prompts)

    for start_idx in range(0, num_prompts, BATCH_SIZE):
        batch = prompts[start_idx:start_idx + BATCH_SIZE]
        queries = []
        actual_questions = []

        for item in batch:
            prompt_text = item.get("prompt", "")
            if "### New Question:" in prompt_text:
                actual_question = prompt_text.split("### New Question:")[-1].strip()
            else:
                actual_question = prompt_text
            actual_questions.append(actual_question)

            query = f"""{GUIDED_EXAMPLES}

Now evaluate this question:
Question: {actual_question}

Provide the complexity (number of steps) as a single number.
Complexity: """
            queries.append(query)

        # Generate responses for the batch
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.95
        )
        outputs = llm.generate(queries, sampling_params)

        # Extract numbers and save results
        for idx, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            complexity = extract_number(response)
            complexity = min(complexity, 10.0)

            item_result = {
                "prompt": actual_questions[idx][:200],
                "task": batch[idx].get("task", ""),
                "subtask": batch[idx].get("subtask", ""),
                "guided": round(complexity, 2)
            }
            results.append(item_result)

        # Save after each batch
        save_results(results)
        print(f"Processed prompts {start_idx + 1} to {start_idx + len(batch)} and saved results...")

    return results


def save_results(results: List[Dict]):
    """Save results to JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    print("Initializing vLLM with Phi-3-mini (GPU-optimized for Colab)...")
    llm = initialize_llm()

    print(f"Loading first {NUM_PROMPTS} prompts from {INPUT_FILE}...")
    prompts = load_prompts(INPUT_FILE)

    print(f"Processing {len(prompts)} prompts (Guided Complexity Only)...")
    results = process_prompts(llm, prompts)

    print(f"Saving results to {OUTPUT_FILE}...")
    save_results(results)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    guided_vals = [r["guided"] for r in results]

    print(f"Guided Complexity - Mean: {np.mean(guided_vals):.2f}, Std: {np.std(guided_vals):.2f}")
    print(f"Range: {min(guided_vals):.2f} to {max(guided_vals):.2f}")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
