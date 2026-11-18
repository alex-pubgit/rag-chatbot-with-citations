"""Reciprocal Rank Fusion: merge ranked lists without comparing their raw scores."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse rankings of item ids. Each list contributes ``1 / (k + rank)`` per item.

    BM25 scores and cosine similarities live on different scales, so only the
    rank of an item in each list is used. Ties are broken by item id to keep the
    result deterministic.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
