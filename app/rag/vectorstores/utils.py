"""Math helpers for vector stores (modeled on LangChain's ``vectorstores.utils``)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["cosine_similarity", "maximal_marginal_relevance"]


def cosine_similarity(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise cosine similarity between two sets of vectors.

    Args:
        x: ``n`` vectors of size ``d``.
        y: ``m`` vectors of size ``d``.

    Returns:
        An ``n x m`` array; entry ``[i, j]`` compares ``x[i]`` with ``y[j]``.
        Zero vectors have similarity 0 with everything.

    Raises:
        ValueError: If the vector sizes differ.
    """
    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    b = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a) if a.size else 0, len(b) if b.size else 0))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Vector sizes differ: {a.shape[1]} vs {b.shape[1]}")
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = (a / a_norm) @ (b / b_norm).T
    similarity[~np.isfinite(similarity)] = 0.0  # zero vectors
    return similarity


def maximal_marginal_relevance(
    query_embedding: Sequence[float],
    embedding_list: Sequence[Sequence[float]],
    *,
    k: int = 4,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Pick ``k`` candidates that are relevant to the query and different
    from each other (maximal marginal relevance).

    The first pick is the candidate closest to the query. Each next pick
    maximizes ``lambda_mult * sim(query, c) - (1 - lambda_mult) * max sim(c, s)``
    over the already selected ``s``, so near-duplicates of earlier picks lose.

    Args:
        query_embedding: The query vector.
        embedding_list: Candidate vectors (usually the top ``fetch_k`` hits).
        k: How many to select.
        lambda_mult: ``1`` = pure relevance (same as plain top-k),
            ``0`` = maximum diversity.

    Returns:
        Indices into ``embedding_list`` in selection order (best first).
        Fewer than ``k`` if there are fewer candidates.

    Raises:
        ValueError: If ``lambda_mult`` is outside [0, 1] or sizes differ.
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError(f"lambda_mult must be between 0 and 1, got {lambda_mult}")
    n = len(embedding_list)
    if k <= 0 or n == 0:
        return []

    to_query = cosine_similarity([query_embedding], embedding_list)[0]
    between = cosine_similarity(embedding_list, embedding_list)  # computed once

    selected = [int(np.argmax(to_query))]
    while len(selected) < min(k, n):
        redundancy = between[:, selected].max(axis=1)
        scores = lambda_mult * to_query - (1.0 - lambda_mult) * redundancy
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected