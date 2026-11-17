from __future__ import annotations

import pytest

from ragbot.text import join_units, split_sentences, split_units, stem, terms, tokenize


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("fees", "fee"),
        ("bridging", "bridge"),
        ("bridged", "bridges"),
        ("swapped", "swap"),
        ("swapping", "swaps"),
        ("staking", "stake"),
        ("unstaking", "unstake"),
        ("addresses", "address"),
        ("keys", "key"),
        ("hashes", "hash"),
        ("added", "add"),
        ("currencies", "currency"),
    ],
)
def test_stem_makes_word_forms_meet(first: str, second: str) -> None:
    assert stem(first) == stem(second)


@pytest.mark.parametrize("word", ["gas", "bonus", "status", "fee", "free", "0.25", "usdt"])
def test_stem_leaves_short_and_non_plural_words_alone(word: str) -> None:
    assert stem(word) == word


def test_stem_keeps_distinct_words_distinct() -> None:
    assert stem("fee") != stem("free")
    assert stem("staking") != stem("unstaking")


def test_tokenize_lowercases_and_keeps_decimals_whole() -> None:
    assert tokenize("A 0.25% fee, v1.5 (BEP-20)") == ["a", "0.25", "fee", "v1.5", "bep", "20"]


def test_terms_drops_stopwords_and_question_scaffolding() -> None:
    assert terms("How long does the bridge usually take?") == ["bridg", "usually", "tak"]
    assert terms("What is it?") == []


def test_split_sentences_respects_decimals_and_abbreviations() -> None:
    text = "The fee is 0.25%. Stablecoin pairs cost less, e.g. USDT to USDC. Gas is separate."
    assert split_sentences(text) == [
        "The fee is 0.25%.",
        "Stablecoin pairs cost less, e.g. USDT to USDC.",
        "Gas is separate.",
    ]


def test_split_sentences_keeps_list_items_whole() -> None:
    text = "Supported networks:\n- BNB Smart Chain. The default.\n- Ethereum\n2. Polygon"
    assert split_sentences(text) == [
        "Supported networks:",
        "- BNB Smart Chain. The default.",
        "- Ethereum",
        "2. Polygon",
    ]


def test_units_round_trip_preserves_paragraphs_and_lines() -> None:
    text = "First sentence. Second sentence.\n\n- item one\n- item two\n\nLast paragraph."
    units = split_units(text)
    assert [unit.text for unit in units][:2] == ["First sentence.", "Second sentence."]
    assert join_units(units) == text
