"""Tokenization, light stemming and sentence splitting shared by the whole pipeline.

BM25, the hashing embedder and the extractive answerer all rely on the same
notion of a "term", so that a word matched by one of them is matched by all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_TOKEN_RE = re.compile(r"[^\W_]+(?:\.\d+)*")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+\N{BULLET}]|\d{1,3}[.)])\s+")
_SENTENCE_BREAK_RE = re.compile(r"""([.!?]["')\]]*)\s+(?=["'(\[]*[^\sa-z])""")
_INITIALS_RE = re.compile(r"^(?:[A-Za-z]\.)+$")
_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "etc.", "vs.", "approx.", "no.", "fig."})

# Function words plus the scaffolding of a question ("how long", "what does X
# mean"). They say nothing about the topic, so they are ignored everywhere.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at
    be because been before being below between both but by
    can cannot could couldn
    d did didn do does doesn doing don down during
    each either else ever every
    few for from further
    get got
    had hadn has hasn have haven having he hello her here hers herself hi him himself his how
    however
    i if in into is isn it its itself
    just know
    let like ll long
    m many may me mean means might more most much must my myself
    need neither no nor not now
    of off often on once only onto or other our ours ourselves out over own
    per please
    re
    s same shall she should shouldn so some such
    t tell than thank thanks that the their theirs them themselves then there these they this
    those through to too
    under until up upon us
    ve very
    want was wasn we were weren what when where whether which while who whom whose why will with
    within without won would wouldn
    you your yours yourself yourselves
    happen happens
    """.split()
)


def normalize_space(text: str) -> str:
    """Collapse every run of whitespace into a single space."""
    return " ".join(text.split())


def count_words(text: str) -> int:
    return len(text.split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Decimals such as ``0.25`` stay in one piece."""
    return _TOKEN_RE.findall(text.lower())


def _undouble(root: str) -> str:
    """``swapp`` -> ``swap``. Short roots (``add``) and -ll/-ss/-zz words keep both letters."""
    if len(root) >= 4 and root[-1] == root[-2] and root[-1] not in "aeioulsz":
        return root[:-1]
    return root


@lru_cache(maxsize=50_000)
def stem(token: str) -> str:
    """Strip plural, -ing, -ed and a final e so that word forms meet.

    ``fees`` -> ``fee``, ``bridging`` / ``bridge`` -> ``bridg``, ``swapped`` -> ``swap``.
    This is a deliberately small English suffix stripper, not a Porter stemmer.
    """
    if len(token) <= 3 or not (token.isascii() and token.isalpha()):
        return token
    word = token
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) >= 6:
        word = _undouble(word[:-3])
    elif word.endswith("ed") and len(word) >= 5 and not word.endswith("eed"):
        word = _undouble(word[:-2])
    if word.endswith("e") and len(word) > 3 and not word.endswith("ee"):
        word = word[:-1]
    return word


def terms(text: str) -> list[str]:
    """Stemmed content words of ``text``, in order, stopwords removed."""
    return [stem(token) for token in tokenize(text) if token not in STOPWORDS]


def is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line))


def strip_list_marker(line: str) -> str:
    return _LIST_ITEM_RE.sub("", line, count=1)


@dataclass(frozen=True)
class Unit:
    """A sentence or list item plus the separator that precedes it in its section."""

    text: str
    separator: str


def _split_line(line: str) -> list[str]:
    """Split one paragraph line into sentences, keeping abbreviations intact."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(line):
        candidate = line[start : match.end(1)]
        last_word = candidate.rsplit(None, 1)[-1]
        if last_word.lower() in _ABBREVIATIONS or _INITIALS_RE.match(last_word):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = line[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_units(text: str) -> list[Unit]:
    """Split a section into sentences and list items.

    List items and table rows are kept whole. ``join_units`` puts the pieces
    back together with the original paragraph and line structure.
    """
    units: list[Unit] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        for line_number, line in enumerate(lines):
            lead = "\n\n" if line_number == 0 else "\n"
            atomic = is_list_item(line) or line.startswith("|")
            parts = [line] if atomic else _split_line(line)
            units.extend(Unit(part, lead if i == 0 else " ") for i, part in enumerate(parts))
    return units


def join_units(units: list[Unit]) -> str:
    return "".join(unit.separator + unit.text for unit in units).strip()


def split_sentences(text: str) -> list[str]:
    return [unit.text for unit in split_units(text)]
