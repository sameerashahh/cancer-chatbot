from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SimilarityResult:
    text: str
    keywords: List[str]
    score: float
    used_tokens: Optional[List[str]] = None


def extract_keywords_spacy(texts: Iterable[str], nlp_model: str = "en_core_web_sm", top_k: int = 10) -> List[List[str]]:
    import spacy

    try:
        nlp = spacy.load(nlp_model)
    except OSError:
        # allow lazy download path; user can run: python -m spacy download en_core_web_sm
        raise RuntimeError(f"spaCy model {nlp_model} not found. Install via: python -m spacy download {nlp_model}")

    results: List[List[str]] = []
    for text in texts:
        doc = nlp(text)
        # Keep content nouns/noun chunks and proper nouns; fallback to lemmas
        candidates: List[str] = []
        for chunk in doc.noun_chunks:
            token = chunk.root
            if not token.is_stop and token.is_alpha:
                candidates.append(token.lemma_.lower())
        if not candidates:
            for tok in doc:
                if tok.pos_ in {"NOUN", "PROPN", "VERB"} and not tok.is_stop and tok.is_alpha:
                    candidates.append(tok.lemma_.lower())

        # De-duplicate preserving order
        seen = set()
        deduped = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        results.append(deduped[:top_k])
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_keyword_similarity(
    texts: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    keyword_extractor: str = "spacy",
    spacy_model: str = "en_core_web_sm",
    top_k: int = 10,
) -> List[SimilarityResult]:
    """
    Compute keyword-based familiarity: extract keywords and measure intra-keyword mutual similarity
    as suggested by Li et al. (2024 referenced in the paper): higher mutual similarity implies higher familiarity.
    We return the mean pairwise cosine similarity among keyword embeddings.
    """
    if keyword_extractor != "spacy":
        raise ValueError("Only 'spacy' keyword extractor is supported currently.")

    from sentence_transformers import SentenceTransformer

    keywords_per_text = extract_keywords_spacy(texts, nlp_model=spacy_model, top_k=top_k)
    model = SentenceTransformer(embedding_model)

    results: List[SimilarityResult] = []
    for text, keywords in zip(texts, keywords_per_text):
        if not keywords:
            results.append(SimilarityResult(text=text, keywords=[], score=0.0))
            continue
        emb = model.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 1:
            score = 1.0  # single keyword
        else:
            # average of upper triangle cosine similarities
            n = emb.shape[0]
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(float(np.dot(emb[i], emb[j])))
            score = float(np.mean(sims)) if sims else 0.0

        results.append(SimilarityResult(text=text, keywords=keywords, score=score))

    return results


def compute_token_similarity_over_top_perplexity(
    texts: List[str],
    token_lists: List[Optional[List[str]]],
    token_logprobs: List[Optional[List[float]]],
    top_m: int = 20,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[SimilarityResult]:
    """
    Implements the paper's variant: compute mean pairwise cosine similarity over the M tokens
    with highest perplexity (lowest log-probability) within each prompt.
    Tokens come from the LM tokenization; subword tokens are embedded and aggregated via mean.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    results: List[SimilarityResult] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(zip(texts, token_lists, token_logprobs), total=len(texts), desc="Similarity (top-20)")
    except Exception:
        iterator = zip(texts, token_lists, token_logprobs)

    for text, toks, lps in iterator:
        if not toks or not lps:
            results.append(SimilarityResult(text=text, keywords=[], score=0.0, used_tokens=[]))
            continue
        # Sort by ascending log-prob (lowest first => highest perplexity)
        idx = list(range(len(lps)))
        idx.sort(key=lambda i: lps[i])
        chosen_idx = idx[:top_m]
        chosen_tokens = [toks[i] for i in chosen_idx]

        # Embed tokens; sentence-transformers expects strings; subwords are fine
        emb = model.encode(chosen_tokens, convert_to_numpy=True, normalize_embeddings=True)
        if emb.ndim == 1:
            score = 1.0
        else:
            n = emb.shape[0]
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(float(np.dot(emb[i], emb[j])))
            score = float(np.mean(sims)) if sims else 0.0

        results.append(SimilarityResult(text=text, keywords=chosen_tokens, score=score, used_tokens=chosen_tokens))

    return results


