"""Deterministic text embeddings via signed feature hashing."""

from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "after",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "many",
        "of",
        "on",
        "or",
        "that",
        "the",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


def content_tokens(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        raise ValueError(f"vector size mismatch: {a.size} != {b.size}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TextEmbedder:
    """Signed hashing vectorizer with unigrams and bigrams."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = content_tokens(text)
        if not tokens:
            return vec
        features: list[tuple[str, float]] = [(f"u:{t}", 1.0) for t in tokens]
        features += [
            (f"b:{a}_{b}", 0.7) for a, b in zip(tokens, tokens[1:], strict=False)
        ]
        for feature, weight in features:
            index, sign = self._hash(feature)
            vec[index] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec
