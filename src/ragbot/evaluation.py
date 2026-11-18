"""Measure retrieval, answers and refusals against a golden set of questions.

The golden set is JSON Lines. A question the documents answer names the
document and the facts a correct answer must state:

    {"question": "What is the bridge fee?", "source": "bridge.md", "facts": ["0.05%"]}

A question the documents do not answer is just the question, and the only
correct response to it is a refusal:

    {"question": "When will the token be listed?"}
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ragbot.chatbot import Chatbot
from ragbot.errors import GoldenSetError
from ragbot.retrieval import Retriever


@dataclass(frozen=True)
class GoldenItem:
    question: str
    source: str | None
    facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scores:
    answerable: int
    unanswerable: int
    hit_rate: float  # the expected document is among the retrieved passages
    mrr: float  # mean reciprocal rank of the expected document
    correct: float  # answered, cites the expected document, states every fact
    refusal_precision: float  # of the refusals, how many were right to refuse
    refusal_recall: float  # of the unanswerable questions, how many were refused


def load_golden(path: Path) -> list[GoldenItem]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GoldenSetError(f"Cannot read the golden set '{path}': {exc.strerror}.") from exc
    items: list[GoldenItem] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            question = record.get("question")
            source = record.get("source")
            facts = tuple(record.get("facts", ()))
            if not isinstance(question, str) or not question.strip():
                raise ValueError('"question" is missing or empty')
            if source is not None and not facts:
                raise ValueError('a question with a "source" needs at least one entry in "facts"')
        except (ValueError, AttributeError, TypeError) as exc:
            raise GoldenSetError(f"{path}:{number}: {exc}") from exc
        items.append(GoldenItem(question, source, facts))
    if not items:
        raise GoldenSetError(f"The golden set '{path}' contains no questions.")
    return items


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 1.0


def evaluate(chatbot: Chatbot, items: Sequence[GoldenItem], top_k: int) -> Scores:
    hits = correct = refusals = right_refusals = 0
    reciprocal_ranks = 0.0
    for item in items:
        answer = chatbot.ask(item.question, top_k)
        refusals += answer.refused
        if item.source is None:
            right_refusals += answer.refused
            continue
        sources = [p.chunk.source for p in chatbot.retriever.retrieve(item.question, top_k)]
        if item.source in sources:
            hits += 1
            reciprocal_ranks += 1.0 / (sources.index(item.source) + 1)
        text = answer.text.lower()
        cited = {citation.source for citation in answer.citations}
        if item.source in cited and all(fact.lower() in text for fact in item.facts):
            correct += 1
    answerable = sum(item.source is not None for item in items)
    unanswerable = len(items) - answerable
    return Scores(
        answerable=answerable,
        unanswerable=unanswerable,
        hit_rate=_ratio(hits, answerable),
        mrr=_ratio(reciprocal_ranks, answerable),
        correct=_ratio(correct, answerable),
        refusal_precision=_ratio(right_refusals, refusals),
        refusal_recall=_ratio(right_refusals, unanswerable),
    )


def suggest_min_score(retriever: Retriever, items: Sequence[GoldenItem]) -> float:
    """A refusal threshold that still answers every golden question retrieved correctly.

    It is 90% of the lowest top-passage score among the answerable questions whose
    expected document ranks first, rounded down to two decimals. Anything scoring
    below that looks less like the corpus than the weakest question known to be
    answerable, so it is refused.
    """
    top_scores = []
    for item in items:
        if item.source is None:
            continue
        passages = retriever.retrieve(item.question, 1)
        if passages and passages[0].chunk.source == item.source:
            top_scores.append(passages[0].score)
    if not top_scores:
        raise GoldenSetError(
            "Cannot calibrate: no golden question retrieves its expected document first."
        )
    return math.floor(0.9 * min(top_scores) * 100) / 100


def format_table(rows: Sequence[tuple[str, Scores]]) -> str:
    lines = [
        "| retrieval | hit rate | MRR | correct answers | refusal precision | refusal recall |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {name} | {scores.hit_rate:.0%} | {scores.mrr:.2f} | {scores.correct:.0%} "
        f"| {scores.refusal_precision:.0%} | {scores.refusal_recall:.0%} |"
        for name, scores in rows
    )
    return "\n".join(lines)
