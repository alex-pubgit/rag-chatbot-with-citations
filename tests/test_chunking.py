from __future__ import annotations

import itertools

import pytest

from ragbot.chunking import chunk_document, split_section
from ragbot.documents import Document, Section
from ragbot.text import count_words, split_sentences


def _sentences(count: int, words_each: int = 10) -> list[str]:
    filler = " ".join(["word"] * (words_each - 2))
    return [f"Sentence{number} {filler} end." for number in range(count)]


def _document(*sections: Section) -> Document:
    return Document(source="guide.md", title="Guide", sections=tuple(sections))


def test_short_section_is_one_chunk_with_its_metadata() -> None:
    doc = _document(Section(("Fees", "Withdrawal fees"), "Withdrawals cost 1 USDT."))

    (chunk,) = chunk_document(doc, chunk_size=50, overlap=10)

    assert chunk.id == "guide.md#0"
    assert chunk.source == "guide.md"
    assert chunk.title == "Guide"
    assert chunk.heading_path == ("Fees", "Withdrawal fees")
    assert chunk.heading == "Fees > Withdrawal fees"
    assert chunk.position == 0
    assert chunk.text == "Withdrawals cost 1 USDT."
    assert chunk.index_text == "Guide\nFees > Withdrawal fees\nWithdrawals cost 1 USDT."


def test_chunks_never_cross_a_heading() -> None:
    doc = _document(Section(("A",), "Tiny section one."), Section(("B",), "Tiny section two."))

    chunks = chunk_document(doc, chunk_size=100, overlap=20)

    assert [(c.heading_path, c.text) for c in chunks] == [
        (("A",), "Tiny section one."),
        (("B",), "Tiny section two."),
    ]
    assert [c.position for c in chunks] == [0, 1]
    assert [c.id for c in chunks] == ["guide.md#0", "guide.md#1"]


def test_long_section_splits_on_sentence_boundaries_within_the_size_limit() -> None:
    sentences = _sentences(12)
    pieces = split_section(" ".join(sentences), chunk_size=35, overlap=0)

    assert len(pieces) == 4
    assert all(count_words(piece) <= 35 for piece in pieces)
    assert [s for piece in pieces for s in split_sentences(piece)] == sentences


def test_overlap_repeats_the_trailing_sentences_of_the_previous_chunk() -> None:
    sentences = _sentences(9)
    pieces = split_section(" ".join(sentences), chunk_size=35, overlap=12)
    as_sentences = [split_sentences(piece) for piece in pieces]

    assert all(count_words(piece) <= 35 for piece in pieces)
    for previous, current in itertools.pairwise(as_sentences):
        assert current[0] == previous[-1]
    covered = {s for piece in as_sentences for s in piece}
    assert covered == set(sentences)


def test_overlap_larger_than_a_sentence_budget_is_skipped_not_truncated() -> None:
    sentences = _sentences(6)
    pieces = split_section(" ".join(sentences), chunk_size=25, overlap=5)

    assert [s for piece in pieces for s in split_sentences(piece)] == sentences


def test_a_sentence_longer_than_a_chunk_is_cut_without_losing_words() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(95))
    pieces = split_section(long_sentence, chunk_size=40, overlap=0)

    assert [count_words(piece) for piece in pieces] == [40, 40, 15]
    assert " ".join(pieces) == long_sentence


def test_paragraph_and_list_structure_survives_splitting() -> None:
    items = "\n".join(f"- item {number} " + "word " * 6 for number in range(8)).replace(" \n", "\n")
    text = f"Intro paragraph here.\n\n{items.strip()}"
    pieces = split_section(text, chunk_size=30, overlap=0)

    assert pieces[0].startswith("Intro paragraph here.\n\n- item 0")
    assert all(line.startswith("- item") for piece in pieces[1:] for line in piece.split("\n"))


def test_empty_sections_produce_no_chunks() -> None:
    assert split_section("  \n ", chunk_size=10, overlap=0) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_are_rejected(size: int, overlap: int) -> None:
    doc = _document(Section((), "Text."))

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(doc, chunk_size=size, overlap=overlap)
