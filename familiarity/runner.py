from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

from .perplexity import compute_perplexity_transformers, compute_perplexity_vllm
from .similarity import compute_keyword_similarity, compute_token_similarity_over_top_perplexity


@dataclass
class FamiliarityScores:
    text: str
    ppl: float
    log_likelihood: float
    num_tokens: int
    keyword_similarity: float
    token_similarity_top_ppl: float


def load_prompts(path: str, limit: Optional[int] = None) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    texts: List[str] = []
    # Accept either a list of objects with "prompt" or raw list of strings
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "prompt" in item:
                texts.append(item["prompt"])
            elif isinstance(item, str):
                texts.append(item)
    elif isinstance(data, dict):
        # handle dict of datasets
        for _, items in data.items():
            if isinstance(items, list):
                for obj in items:
                    if isinstance(obj, dict) and "prompt" in obj:
                        texts.append(obj["prompt"])
    if limit is not None:
        texts = texts[:limit]
    return texts


def compute_familiarity_scores(
    texts: List[str],
    model_for_ppl: str,
    use_vllm: bool = True,
    tensor_parallel_size: int = 1,
) -> List[FamiliarityScores]:
    if use_vllm:
        ppl_results = compute_perplexity_vllm(texts, model_name=model_for_ppl, tensor_parallel_size=tensor_parallel_size)
        # vLLM path does not provide token-level logprobs per-token mapping in our wrapper; fall back to keyword-sim only
        token_sim_results = [None] * len(texts)
    else:
        ppl_results = compute_perplexity_transformers(texts, model_name=model_for_ppl)
        token_sim_results = compute_token_similarity_over_top_perplexity(
            texts,
            token_lists=[r.tokens for r in ppl_results],
            token_logprobs=[r.token_logprobs for r in ppl_results],
            top_m=20,
        )

    sim_results = compute_keyword_similarity(texts)

    scores: List[FamiliarityScores] = []
    for i, (p, s) in enumerate(zip(ppl_results, sim_results)):
        token_sim = 0.0
        if token_sim_results and token_sim_results[i] is not None:
            token_sim = token_sim_results[i].score
        scores.append(
            FamiliarityScores(
                text=p.text,
                ppl=p.ppl,
                log_likelihood=p.log_likelihood,
                num_tokens=p.num_tokens,
                keyword_similarity=s.score,
                token_similarity_top_ppl=token_sim,
            )
        )
    return scores


def save_scores(scores: List[FamiliarityScores], out_path: str) -> None:
    serializable = [
        {
            "ppl": sc.ppl,
            "log_likelihood": sc.log_likelihood,
            "num_tokens": sc.num_tokens,
            "keyword_similarity": sc.keyword_similarity,
            "token_similarity_top_ppl": sc.token_similarity_top_ppl,
            "text": sc.text,
        }
        for sc in scores
    ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="data/prepared_prompts.json")
    parser.add_argument("--out", type=str, default="familiarity_scores.json")
    parser.add_argument("--model", type=str, required=True, help="HF or vLLM model name")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    texts = load_prompts(args.prompts, limit=args.limit)
    try:
        from tqdm import tqdm
        iterator = tqdm(total=len(texts), desc="Scoring")
    except Exception:
        iterator = None

    scores = compute_familiarity_scores(texts, model_for_ppl=args.model, use_vllm=not args.no_vllm, tensor_parallel_size=args.tp)
    if iterator:
        iterator.update(len(texts))
    save_scores(scores, args.out)
    print(f"Saved {len(scores)} familiarity scores to {args.out}")


if __name__ == "__main__":
    main()


