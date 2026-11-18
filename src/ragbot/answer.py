"""Answers, citations and the offline extractive answerer."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ragbot.documents import Passage
from ragbot.text import normalize_space, split_sentences, strip_list_marker, terms

REFUSAL_TEXT = "I don't know based on the provided documents."
SNIPPET_WORDS = 60
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"[ \t]+([.,;:!?])")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Citation:
    n: int
    source: str
    title: str
    heading: str
    snippet: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    refused: bool = False
    citations: tuple[Citation, ...] = ()


REFUSED = Answer(text=REFUSAL_TEXT, refused=True)


class Answerer(Protocol):
    def answer(self, question: str, passages: Sequence[Passage]) -> Answer:
        """Answer from ``passages`` only, citing passage ``i`` (1-based) as ``[i]``."""
        ...


def _snippet(text: str) -> str:
    words = normalize_space(text).split(" ")
    if len(words) <= SNIPPET_WORDS:
        return " ".join(words)
    return " ".join(words[:SNIPPET_WORDS]) + " ..."


def cite(draft: str, passages: Sequence[Passage]) -> Answer:
    """Turn the ``[n]`` markers of a draft answer into structured citations.

    Markers that point at no passage are dropped, and the rest are renumbered in
    order of first use, so the answer cites [1], [2], ... without gaps. A draft
    left with no valid citation is refused: an unsourced claim is the one thing
    this bot must not say.
    """
    used: list[int] = []

    def renumber(match: re.Match[str]) -> str:
        position = int(match.group(1)) - 1
        if not 0 <= position < len(passages):
            return ""
        if position not in used:
            used.append(position)
        return f"[{used.index(position) + 1}]"

    text = _MARKER_RE.sub(renumber, draft)
    text = _SPACE_RUN_RE.sub(" ", _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)).strip()
    if not used:
        return REFUSED
    citations = tuple(
        Citation(
            n=n,
            source=passages[position].chunk.source,
            title=passages[position].chunk.title,
            heading=passages[position].chunk.heading,
            snippet=_snippet(passages[position].chunk.text),
            score=round(passages[position].score, 4),
        )
        for n, position in enumerate(used, start=1)
    )
    return Answer(text=text, citations=citations)


class ExtractiveAnswerer:
    """Quote the passage that best covers the question. No language model.

    Coverage is the IDF-weighted share of the question's terms found in a passage,
    its title and its headings, so a rare word ("unstaking") counts for more than
    a common one ("fee"). The best-covered of the top passages is quoted, starting
    at its best-matching sentence. Below ``min_coverage`` the answer is a refusal:
    most of the question is about something the passage never mentions.
    """

    def __init__(
        self,
        idf: Callable[[str], float],
        *,
        min_coverage: float = 0.4,
        max_sentences: int = 3,
        passages_considered: int = 3,
    ) -> None:
        self._idf = idf
        self._min_coverage = min_coverage
        self._max_sentences = max_sentences
        self._passages_considered = passages_considered

    def answer(self, question: str, passages: Sequence[Passage]) -> Answer:
        weights = {term: self._idf(term) for term in dict.fromkeys(terms(question))}
        total = sum(weights.values())
        if not total or not passages:
            return REFUSED

        def matched_weight(text: str) -> float:
            return sum(weights[term] for term in weights.keys() & set(terms(text)))

        # max() keeps the first of equal candidates, so ties go to the better-ranked passage.
        position = max(
            range(min(len(passages), self._passages_considered)),
            key=lambda i: matched_weight(passages[i].chunk.index_text),
        )
        chunk = passages[position].chunk
        if matched_weight(chunk.index_text) / total < self._min_coverage:
            return REFUSED

        sentences = [strip_list_marker(s).strip() for s in split_sentences(chunk.text)]
        best = max(range(len(sentences)), key=lambda i: matched_weight(sentences[i]))
        start = min(best, max(0, len(sentences) - self._max_sentences))
        quote = " ".join(sentences[start : start + self._max_sentences])
        return cite(f"{quote} [{position + 1}]", passages)
