from __future__ import annotations

import pytest

from ragbot.fusion import reciprocal_rank_fusion


def test_item_ranked_well_in_both_lists_beats_a_single_top_hit() -> None:
    keyword = [10, 20, 30]
    dense = [40, 20, 10]

    fused = [item for item, _ in reciprocal_rank_fusion([keyword, dense])]

    assert fused[:2] == [10, 20]
    assert set(fused) == {10, 20, 30, 40}


def test_scores_follow_the_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[7, 8], [9, 8, 7]], k=60))

    assert fused[7] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[8] == pytest.approx(1 / 62 + 1 / 62)
    assert fused[9] == pytest.approx(1 / 61)


def test_a_single_ranking_keeps_its_order() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([[3, 1, 2]])] == [3, 1, 2]


def test_ties_are_broken_by_item_id() -> None:
    fused = reciprocal_rank_fusion([[5, 1], [1, 5]])

    assert [item for item, _ in fused] == [1, 5]


def test_no_rankings_gives_an_empty_result() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
