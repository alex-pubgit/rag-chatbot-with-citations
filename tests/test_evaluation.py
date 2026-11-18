from __future__ import annotations

import pytest

from conftest import REPO_ROOT
from ragbot.errors import GoldenSetError
from ragbot.evaluation import Scores, evaluate, format_table, load_golden, suggest_min_score

GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"


def test_sample_corpus_stays_above_the_regression_floor(chatbot):
    scores = evaluate(chatbot, load_golden(GOLDEN), top_k=5)
    assert (scores.answerable, scores.unanswerable) == (30, 10)
    assert scores.hit_rate >= 0.95
    assert scores.correct >= 0.75
    assert scores.refusal_recall >= 0.85


def test_threshold_suggestion_never_refuses_a_correctly_retrieved_question(chatbot):
    items = load_golden(GOLDEN)
    threshold = suggest_min_score(chatbot.retriever, items)
    assert 0.0 < threshold < 1.0
    for item in items:
        if item.source is None:
            continue
        top = chatbot.retriever.retrieve(item.question, 1)[0]
        if top.chunk.source == item.source:
            assert top.score >= threshold, item.question


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("{not json", "golden.jsonl:2"),
        ('{"source": "a.md", "facts": ["x"]}', '"question" is missing'),
        ('{"question": "q", "source": "a.md"}', 'at least one entry in "facts"'),
        ('["a list, not an object"]', "golden.jsonl:2"),
    ],
)
def test_malformed_line_is_reported_with_its_line_number(tmp_path, line, problem):
    path = tmp_path / "golden.jsonl"
    path.write_text('{"question": "fine"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(GoldenSetError) as raised:
        load_golden(path)
    assert problem in str(raised.value)


def test_empty_golden_set_is_an_error(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="no questions"):
        load_golden(path)


def test_table_shows_one_row_per_configuration():
    scores = Scores(30, 10, 1.0, 0.98, 0.8, 0.75, 0.9)
    table = format_table([("bm25", scores), ("hybrid", scores)]).splitlines()
    assert len(table) == 4
    assert table[2] == "| bm25 | 100% | 0.98 | 80% | 75% | 90% |"
