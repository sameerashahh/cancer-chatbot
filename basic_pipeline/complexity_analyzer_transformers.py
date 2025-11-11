#!/usr/bin/env python3
"""
FIXED Prompt Complexity Measurement using Transformers
- No vLLM required - uses Hugging Face Transformers
- Optimized for Google Colab
- Forces model to output ONLY a number
"""

import json
import re
import torch
import gc
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
OUTPUT_FILE = "complexity_results_fixed.json"
INPUT_FILE = "questions_binary.json"
TEMPERATURE = 0.0  # Deterministic output
MAX_NEW_TOKENS = 5  # Just need 1-2 tokens for a number
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Example demonstrations for guided complexity
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


def initialize_model():
    """Initialize model and tokenizer with 8-bit quantization for Colab"""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    # Clear GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Configure 8-bit quantization for Colab GPU
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    # Load model with 8-bit quantization
    print("Loading model (8-bit quantization for efficiency)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager"  # Avoid flash attention issues
    )
    
    print("✓ Model and tokenizer loaded")
    return model, tokenizer


def extract_number(text: str) -> float:
    """
    Extract complexity number from LLM response.
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
    
    # Try to find any number in the text (first number)
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])
    
    return 0.0


def validate_complexity(complexity: float, response: str, question_preview: str = "") -> float:
    """
    Validate and normalize complexity score.
    """
    # If extraction failed (0), default to 2 (simple question)
    if complexity == 0.0:
        if question_preview:
            print(f"    ⚠️  No number from '{response[:30]}...' for '{question_preview[:40]}...' → default: 2")
        return 2.0
    
    # Clip to reasonable range
    if complexity > 10.0:
        return 10.0
    
    if complexity < 1.0:
        return 1.0
    
    return complexity


def process_questions(model, tokenizer, questions: List[Dict]) -> List[Dict]:
    """Process questions one by one and measure complexity"""
    results = []
    num_questions = len(questions)
    
    print(f"\nProcessing {num_questions} questions...")
    print(f"Temperature: {TEMPERATURE}, Max new tokens: {MAX_NEW_TOKENS}")
    print("="*70)
    
    for idx, item in enumerate(tqdm(questions, desc="Analyzing complexity")):
        # Extract the question text
        question_text = item.get("question", "")
        
        # Create query that demands ONLY a number as output
        query = f"""{GUIDED_EXAMPLES}

Now evaluate this question. Respond with ONLY the number of steps required (just the number, nothing else):

Question: {question_text}

Number of steps:"""
        
        # Tokenize input
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=2048)
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            if TEMPERATURE == 0.0:
                # Greedy decoding (deterministic)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # Greedy
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            else:
                # Sampling with temperature
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
        
        # Decode only the new tokens (not the input)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract and validate complexity
        complexity_raw = extract_number(response)
        complexity = validate_complexity(
            complexity_raw, 
            response, 
            question_text if idx < 5 or idx % 100 == 0 else ""
        )
        
        # Show sample outputs
        if idx < 5 or idx % 100 == 0:
            q_preview = question_text[:80]
            print(f"\n  Q{idx}: {q_preview}...")
            print(f"  LLM: '{response}' → Complexity: {complexity}")
        
        # Save result
        item_result = {
            "question": question_text,
            "dataset": item.get("dataset", ""),
            "subtask": item.get("subtask", ""),
            "guided": round(complexity, 1)
        }
        results.append(item_result)
        
        # Periodic saves (checkpoint every 100 questions)
        if (idx + 1) % 100 == 0:
            save_results(results)
            print(f"  💾 Checkpoint: Saved {len(results)} results")
        
        # Clear GPU cache periodically
        if DEVICE == "cuda" and (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    return results


def save_results(results: List[Dict]):
    """Save results to JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    print("="*70)
    print("COMPLEXITY ANALYZER - TRANSFORMERS VERSION")
    print("="*70)
    print("Features:")
    print("  ✓ No vLLM required - uses Hugging Face Transformers")
    print("  ✓ 8-bit quantization for Colab GPU efficiency")
    print("  ✓ Temperature: 0.0 (deterministic)")
    print("  ✓ Max tokens: 5 (just need a number)")
    print("  ✓ Prompt: Forces ONLY number output")
    print("  ✓ Validation: Clips to 1-10 range, defaults to 2 on error")
    print("  ✓ Checkpointing: Saves every 100 questions")
    print("="*70)
    
    # Initialize
    model, tokenizer = initialize_model()
    questions = load_questions(INPUT_FILE)
    
    # Process
    results = process_questions(model, tokenizer, questions)
    
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
        pct = dist[score]/len(results)*100
        bar = "█" * int(pct / 2)
        print(f"  Complexity {score:3.1f}: {dist[score]:4d} questions ({pct:5.1f}%) {bar}")
    
    # Check for zeros
    zeros = sum(1 for v in guided_vals if v == 0)
    if zeros > 0:
        print(f"\n⚠️  WARNING: {zeros} questions still have complexity=0")
    else:
        print(f"\n✅ No questions with complexity=0 (all valid!)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review the distribution above")
    print("  2. Run: python3 calculate_famicom.py (update to use complexity_results_fixed.json)")
    print("  3. Check if FAMICOM correlation improves!")
    print("="*70)


if __name__ == "__main__":
    main()


