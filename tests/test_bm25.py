from __future__ import annotations

import pytest

from ragbot.bm25 import BM25Index

DOCUMENTS = [
    "The swap fee is 0.25% on every network.",
    "The bridge fee is 0.05%. Bridging takes 5 to 20 minutes.",
    "Staking rewards are distributed every Monday.",
    "Network fees are paid in the native coin of the network, on every network.",
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(DOCUMENTS)


def test_best_match_comes_first(index: BM25Index) -> None:
    rows = [row for row, _ in index.search("bridge fee", top_n=4)]

    assert rows[0] == 1
    assert 2 not in rows, "a document without any query term must not be returned"


def test_query_and_documents_are_stemmed_the_same_way(index: BM25Index) -> None:
    assert index.search("bridging fees", top_n=1)[0][0] == 1
    assert index.search("stake reward", top_n=1)[0][0] == 2


def test_rare_terms_weigh_more_than_common_ones(index: BM25Index) -> None:
    assert index.idf("monday") > index.idf("fee") > 0
    assert index.idf("never-seen") > index.idf("monday")


def test_shorter_document_wins_when_term_frequency_is_equal() -> None:
    index = BM25Index(["gas fee " + "filler " * 30, "gas fee"])

    assert [row for row, _ in index.search("gas", top_n=2)] == [1, 0]


def test_scores_are_positive_and_sorted(index: BM25Index) -> None:
    results = index.search("network fee", top_n=10)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_ties_are_broken_by_position() -> None:
    index = BM25Index(["same text", "same text", "same text"])

    assert [row for row, _ in index.search("text", top_n=3)] == [0, 1, 2]


@pytest.mark.parametrize("query", ["", "the of and", "zebra"])
def test_queries_without_a_known_term_return_nothing(index: BM25Index, query: str) -> None:
    assert index.search(query, top_n=5) == []


def test_top_n_limits_the_result(index: BM25Index) -> None:
    assert len(index.search("fee network", top_n=2)) == 2
    assert index.search("fee", top_n=0) == []


def test_empty_index_is_searchable() -> None:
    assert BM25Index([]).search("fee", top_n=3) == []
