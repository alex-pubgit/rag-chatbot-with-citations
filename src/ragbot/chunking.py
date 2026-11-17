"""Heading-aware chunking.

A chunk never crosses a heading. A section that fits in ``chunk_size`` words is
kept whole. A longer one is split on paragraph, list-item and sentence
boundaries, and each new chunk starts by repeating the last sentences of the
previous one (up to ``overlap`` words) so that a fact cut by the boundary is
still found.
"""

from __future__ import annotations

from ragbot.documents import Chunk, Document
from ragbot.text import Unit, count_words, join_units, split_units


def _hard_split(unit: Unit, chunk_size: int) -> list[Unit]:
    """Cut a single sentence that is longer than a whole chunk into word windows."""
    words = unit.text.split()
    pieces = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [Unit(piece, unit.separator if i == 0 else " ") for i, piece in enumerate(pieces)]


def _overlap_tail(units: list[Unit], overlap: int) -> list[Unit]:
    """The trailing units of a chunk that together stay within ``overlap`` words."""
    tail: list[Unit] = []
    budget = overlap
    for unit in reversed(units):
        words = count_words(unit.text)
        if words > budget:
            break
        tail.insert(0, unit)
        budget -= words
    return tail


def split_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one section into chunk texts of at most ``chunk_size`` words."""
    text = text.strip()
    if not text:
        return []
    if count_words(text) <= chunk_size:
        return [text]

    units: list[Unit] = []
    for unit in split_units(text):
        units.extend(_hard_split(unit, chunk_size))

    chunks: list[list[Unit]] = []
    current: list[Unit] = []
    current_words = 0
    for unit in units:
        words = count_words(unit.text)
        if current and current_words + words > chunk_size:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
            current_words = sum(count_words(u.text) for u in current)
            if current_words + words > chunk_size:
                current, current_words = [], 0
        current.append(unit)
        current_words += words
    chunks.append(current)
    return [join_units(chunk) for chunk in chunks]


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Turn a document into chunks that carry source, title, heading path and position."""
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for section in document.sections:
        for text in split_section(section.text, chunk_size, overlap):
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{document.source}#{position}",
                    source=document.source,
                    title=document.title,
                    heading_path=section.heading_path,
                    position=position,
                    text=text,
                )
            )
    return chunks
