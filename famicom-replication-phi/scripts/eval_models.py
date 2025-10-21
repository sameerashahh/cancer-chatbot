from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from familiarity.runner import load_prompts, compute_familiarity_scores, save_scores


DEFAULT_MODELS = [
    # Phi-3-mini-128k-instruct
    "microsoft/Phi-3-mini-128k-instruct",
    # Mistral-7B-Instruct v0.2
    "mistralai/Mistral-7B-Instruct-v0.2",
    # Llama-2-13b-chat
    "meta-llama/Llama-2-13b-chat-hf",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="data/prepared_prompts.json")
    parser.add_argument("--out_dir", type=str, default="outputs/familiarity")
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    texts = load_prompts(args.prompts, limit=args.limit)

    for model_name in args.models:
        scores = compute_familiarity_scores(
            texts,
            model_for_ppl=model_name,
            use_vllm=True,
            tensor_parallel_size=args.tp,
        )
        out_path = Path(args.out_dir) / f"{model_name.replace('/', '_')}.json"
        save_scores(scores, str(out_path))
        print(f"Saved {len(scores)} scores for {model_name} -> {out_path}")


if __name__ == "__main__":
    main()


