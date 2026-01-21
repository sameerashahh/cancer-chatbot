#!/usr/bin/env python3
"""
FIXED Prompt Complexity Measurement using vLLM
- Forces model to output ONLY a number (fewer tokens)
- Lower temperature for consistency
- Better error handling and validation
"""

import json
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_NAME = "microsoft/phi-3-mini-128k-instruct"
OUTPUT_FILE = "complexity_results_fixed.json"
INPUT_FILE = "questions_binary.json"
TEMPERATURE = 0.0  # Changed from 0.8 - deterministic output
MAX_TOKENS = 5     # Changed from 10 - just need 1-2 tokens for a number
BATCH_SIZE = 10    # Increased from 5 - faster processing

# Example demonstrations for guided complexity
# CRITICAL: Changed prompt to demand ONLY a number
GUIDED_EXAMPLES = """
Example 1:
Question: What is the capital of France?
Steps: 1) Recall knowledge about France, 2) Identify capital
Number of steps: 2

Example 2:
Question: Solve: If John has 5 apples and Mary has 3, how many do they have together?
Steps: 1) Parse numbers, 2) Add values, 3) Return sum
Number of steps: 3

Example 3:
Question: Explain why the sky is blue and how it relates to quantum mechanics
Steps: 1) Understand Rayleigh scattering, 2) Explain wave mechanics, 3) Connect to quantum theory, 4) Synthesize explanation
Number of steps: 4

Example 4:
Question: Is the sky blue?
Steps: 1) Recall basic knowledge
Number of steps: 1
"""


def load_questions(filepath: str) -> List[Dict[str, Any]]:
    """Load questions from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {filepath}")
    return data


def initialize_llm() -> LLM:
    """Initialize vLLM for Colab GPU (Phi-3-mini)"""
    print("Initializing vLLM with Phi-3-mini...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        max_model_len=2048,
        disable_sliding_window=True,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
    )
    print("✓ Model loaded")
    return llm


def extract_number(text: str) -> float:
    """
    Extract complexity number from LLM response.
    Improved to handle different response formats.
    """
    text = text.strip()
    
    # Try to extract just a standalone number at the start
    match = re.match(r'^(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    # Try to find 'Number of steps: X' or 'Steps: X'
    match = re.search(r'[Nn]umber\s+of\s+steps\s*[:\-]?\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    match = re.search(r'[Ss]teps\s*[:\-]?\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    # Try 'Complexity: X'
    match = re.search(r'[Cc]omplexity\s*[:\-]?\s*(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    
    # Try to find any number in the text (last resort)
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])  # Take FIRST number, not last
    
    # If no number found, return 0 (will be handled as error)
    return 0.0


def validate_complexity(complexity: float, response: str) -> float:
    """
    Validate and normalize complexity score.
    Returns a valid score between 1 and 10.
    """
    # If extraction failed (0), default to 2 (simple question)
    if complexity == 0.0:
        print(f"    ⚠️  No number extracted from: '{response[:50]}...' → defaulting to 2")
        return 2.0
    
    # Clip to reasonable range
    if complexity > 10.0:
        print(f"    ⚠️  Complexity {complexity} too high → capping at 10")
        return 10.0
    
    if complexity < 1.0:
        print(f"    ⚠️  Complexity {complexity} too low → setting to 1")
        return 1.0
    
    return complexity


def process_questions_in_batches(llm: LLM, questions: List[Dict]) -> List[Dict]:
    """Process questions in batches and measure guided complexity"""
    results = []
    num_questions = len(questions)
    
    print(f"\nProcessing {num_questions} questions in batches of {BATCH_SIZE}...")
    print(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
    print("="*70)
    
    for start_idx in tqdm(range(0, num_questions, BATCH_SIZE), desc="Batches"):
        batch = questions[start_idx:start_idx + BATCH_SIZE]
        queries = []
        
        for item in batch:
            # Extract the question text
            question_text = item.get("question", "")
            
            # Create query that demands ONLY a number as output
            query = f"""{GUIDED_EXAMPLES}

Now evaluate this question. Respond with ONLY the number of steps required (just the number, nothing else):

Question: {question_text}

Number of steps:"""
            
            queries.append(query)
        
        # Generate responses for the batch
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=1.0 if TEMPERATURE == 0.0 else 0.95,
            # Add stop tokens to prevent extra output
            stop=["\n", ".", ",", " "]
        )
        
        outputs = llm.generate(queries, sampling_params)
        
        # Extract numbers and save results
        for idx, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            complexity_raw = extract_number(response)
            complexity = validate_complexity(complexity_raw, response)
            
            # Occasionally show what the model is outputting
            if start_idx + idx < 5 or (start_idx + idx) % 100 == 0:
                q_preview = batch[idx].get("question", "")[:80]
                print(f"\n  Q: {q_preview}...")
                print(f"  LLM response: '{response}' → Complexity: {complexity}")
            
            item_result = {
                "question": batch[idx].get("question", ""),
                "dataset": batch[idx].get("dataset", ""),
                "subtask": batch[idx].get("subtask", ""),
                "guided": round(complexity, 1)  # Round to 1 decimal
            }
            results.append(item_result)
        
        # Save after each batch (checkpoint)
        if (start_idx + BATCH_SIZE) % 100 == 0 or start_idx + BATCH_SIZE >= num_questions:
            save_results(results)
    
    return results


def save_results(results: List[Dict]):
    """Save results to JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    print("="*70)
    print("COMPLEXITY ANALYZER - FIXED VERSION")
    print("="*70)
    print("Changes:")
    print("  ✓ Temperature: 0.8 → 0.0 (deterministic)")
    print("  ✓ Max tokens: 10 → 5 (just need a number)")
    print("  ✓ Prompt: Forces ONLY number output")
    print("  ✓ Validation: Handles errors and clips to 1-10 range")
    print("  ✓ Stop tokens: Prevents extra output")
    print("="*70)
    
    # Initialize
    llm = initialize_llm()
    questions = load_questions(INPUT_FILE)
    
    # Process
    results = process_questions_in_batches(llm, questions)
    
    # Final save
    save_results(results)
    print(f"\n✅ Complexity analysis complete!")
    print(f"   Results saved to: {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    guided_vals = [r["guided"] for r in results]
    
    print(f"Total questions: {len(results)}")
    print(f"\nComplexity scores:")
    print(f"  Mean: {np.mean(guided_vals):.2f}")
    print(f"  Std: {np.std(guided_vals):.2f}")
    print(f"  Range: {min(guided_vals):.1f} to {max(guided_vals):.1f}")
    
    # Distribution
    from collections import Counter
    dist = Counter(guided_vals)
    print(f"\nDistribution:")
    for score in sorted(dist.keys()):
        print(f"  Complexity {score}: {dist[score]} questions ({dist[score]/len(results)*100:.1f}%)")
    
    # Check for zeros (should be none now)
    zeros = sum(1 for v in guided_vals if v == 0)
    if zeros > 0:
        print(f"\n⚠️  WARNING: {zeros} questions still have complexity=0")
    else:
        print(f"\n✅ No questions with complexity=0 (good!)")


if __name__ == "__main__":
    main()


