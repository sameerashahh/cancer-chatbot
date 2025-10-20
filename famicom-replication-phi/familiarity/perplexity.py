from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np


@dataclass
class PerplexityResult:
    text: str
    log_likelihood: float
    num_tokens: int
    # optional: token-wise info when available (transformers path)
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None

    @property
    def ppl(self) -> float:
        if self.num_tokens == 0:
            return float("inf")
        # Per-token negative log-likelihood exponentiated
        return float(np.exp(-self.log_likelihood / max(self.num_tokens, 1)))


def compute_perplexity_transformers(
    texts: List[str],
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: Optional[str] = None,
    batch_size: int = 1,
    max_length: int = 4096,
) -> List[PerplexityResult]:
    """
    Compute perplexity using Hugging Face transformers causal LM.

    Note: Uses label-shifted LM loss (teacher forcing). For long texts, it truncates to max_length.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    results: List[PerplexityResult] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(texts), batch_size), desc="Perplexity (HF)")
    except Exception:
        iterator = range(0, len(texts), batch_size)

    for i in iterator:
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Compute token-level logprobs to derive perplexity and to support top-k selection
        with torch.no_grad():
            logits = model(**enc).logits  # [B, T, V]
            shift_logits = logits[:, :-1, :]
            shift_labels = enc["input_ids"][:, 1:]
            shift_mask = enc["attention_mask"][:, 1:]
            log_probs = shift_logits.log_softmax(dim=-1)
            gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # sum log probs over tokens
            seq_loglik = (gathered * shift_mask).sum(dim=1)
            seq_lengths = shift_mask.sum(dim=1)
            # recover strings for tokens (align with shift_labels)
            input_ids = enc["input_ids"]
            batch_tokens: List[List[str]] = []
            for ids in input_ids:
                toks = tokenizer.convert_ids_to_tokens(ids.tolist())
                batch_tokens.append(toks)

        for j, text in enumerate(batch):
            # per-token logprobs correspond to labels from position 1..T-1
            per_token_logprobs = gathered[j].detach().cpu().tolist()
            per_token_mask = shift_mask[j].detach().cpu().tolist()
            # align tokens: ignore the first token (no label) and any masked tokens
            toks = batch_tokens[j][1: 1 + len(per_token_logprobs)]
            aligned_token_logprobs: List[float] = [lp for lp, m in zip(per_token_logprobs, per_token_mask) if m == 1]
            aligned_tokens: List[str] = [t for t, m in zip(toks, per_token_mask) if m == 1]

            results.append(
                PerplexityResult(
                    text=text,
                    log_likelihood=float(seq_loglik[j].item()),
                    num_tokens=int(seq_lengths[j].item()),
                    tokens=aligned_tokens,
                    token_logprobs=aligned_token_logprobs,
                )
            )

    return results


def compute_perplexity_vllm(
    texts: List[str],
    model_name: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
) -> List[PerplexityResult]:
    """Compute perplexity using vLLM's logprob API for causal LMs."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=False,
        disable_sliding_window=True,
    )

    # Use greedy sampling but request logprobs for all tokens
    params = SamplingParams(temperature=0.0, max_tokens=0, logprobs=0, prompt_logprobs=1)
    outputs = llm.generate(texts, params)
    tokenizer = llm.get_tokenizer()

    results: List[PerplexityResult] = []
    for text, out in zip(texts, outputs):
        # vLLM returns per-token prompt logprobs in out.prompt_logprobs
        token_logprobs: List[float] = []
        for tk in out.prompt_logprobs:
            # Each tk is a dict mapping token string to logprob for the selected token
            # Get the chosen token's logprob
            if tk is None:
                continue
            chosen = next(iter(tk.values()))
            token_logprobs.append(chosen)

        # Recover tokens to support top-20 similarity
        enc = tokenizer(text, return_tensors=None)
        input_ids = enc["input_ids"] if isinstance(enc, dict) else tokenizer(text)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokens = tokens[1: 1 + len(token_logprobs)] if len(tokens) >= len(token_logprobs) + 1 else tokens[:len(token_logprobs)]

        log_likelihood = float(np.sum(token_logprobs))
        num_tokens = int(len(token_logprobs))
        results.append(PerplexityResult(text=text, log_likelihood=log_likelihood, num_tokens=num_tokens, tokens=tokens, token_logprobs=token_logprobs))

    return results


