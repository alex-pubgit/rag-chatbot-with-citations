from __future__ import annotations

import pytest

from conftest import REPO_ROOT, SAMPLE_DOCS
from ragbot.cli import main

GOLDEN = str(REPO_ROOT / "eval" / "golden.jsonl")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Run from an empty folder so a developer's own .env or index is never picked up."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGBOT_INDEX_DIR", str(tmp_path / "index"))


def test_ingest_then_ask_then_eval(capsys):
    assert main(["ingest", str(SAMPLE_DOCS)]) == 0
    assert "Indexed 12 documents" in capsys.readouterr().out

    assert main(["ask", "What is the bridge fee?"]) == 0
    out = capsys.readouterr().out
    assert "0.05%" in out
    assert "[1] bridge.md - Bridge fee" in out

    assert main(["eval", "--golden", GOLDEN, "--min-hit-rate", "0.95"]) == 0
    assert "| hybrid | 100% |" in capsys.readouterr().out


def test_eval_fails_when_a_score_is_below_its_floor(capsys):
    main(["ingest", str(SAMPLE_DOCS)])
    assert main(["eval", "--golden", GOLDEN, "--min-correct", "0.99"]) == 1
    assert "correct answers" in capsys.readouterr().err


def test_second_ingest_reuses_unchanged_documents(capsys):
    main(["ingest", str(SAMPLE_DOCS)])
    capsys.readouterr()
    main(["ingest", str(SAMPLE_DOCS)])
    assert "added 0, updated 0, unchanged 12, removed 0" in capsys.readouterr().out


def test_missing_index_says_what_to_run(capsys):
    assert main(["ask", "anything"]) == 1
    assert "ragbot ingest" in capsys.readouterr().err


def test_invalid_setting_is_named(monkeypatch, capsys):
    monkeypatch.setenv("RAGBOT_TOP_K", "0")
    assert main(["ask", "anything"]) == 1
    assert "RAGBOT_TOP_K" in capsys.readouterr().err


def test_hosted_answerer_without_a_model_is_a_clear_error(monkeypatch, capsys):
    main(["ingest", str(SAMPLE_DOCS)])
    monkeypatch.setenv("RAGBOT_ANSWERER", "openai")
    assert main(["ask", "What is the bridge fee?"]) == 1
    assert "RAGBOT_LLM_MODEL" in capsys.readouterr().err
