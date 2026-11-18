from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from ragbot.embeddings.hashing import HashingEmbedder

TEXT = "The bridge fee is 0.05% of the transferred amount, with a minimum of 1 USDT equivalent."
DIGEST_SCRIPT = (
    "import hashlib, sys\n"
    "from ragbot.embeddings.hashing import HashingEmbedder\n"
    "vector = HashingEmbedder().embed_query(sys.argv[1])\n"
    "print(hashlib.sha256(vector.tobytes()).hexdigest())\n"
)


def _cosine(embedder: HashingEmbedder, first: str, second: str) -> float:
    return float(embedder.embed_query(first) @ embedder.embed_query(second))


def test_same_text_gives_the_same_vector_in_every_instance() -> None:
    first = HashingEmbedder().embed_query(TEXT)
    second = HashingEmbedder().embed_query(TEXT)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("hash_seed", ["1", "4242"])
def test_vectors_do_not_depend_on_the_process_hash_seed(hash_seed: str) -> None:
    """Python salts ``hash()`` per process. The embedder must not inherit that randomness."""
    expected = hashlib.sha256(HashingEmbedder().embed_query(TEXT).tobytes()).hexdigest()

    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT, TEXT],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected


def test_vectors_are_unit_length_float32() -> None:
    embedder = HashingEmbedder(dimension=512)
    matrix = embedder.embed_documents([TEXT, "Staking rewards are paid on Monday."])

    assert matrix.shape == (2, 512)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_text_without_content_words_is_a_zero_vector() -> None:
    embedder = HashingEmbedder()

    assert not embedder.embed_query("").any()
    assert not embedder.embed_query("what is the").any()
    assert embedder.embed_documents([]).shape == (0, embedder.dimension)


def test_shared_words_mean_higher_similarity() -> None:
    embedder = HashingEmbedder()

    on_topic = _cosine(embedder, "bridge fee", TEXT)
    off_topic = _cosine(embedder, "bridge fee", "Staking rewards are distributed every Monday.")

    assert on_topic > 0.2
    assert on_topic > off_topic + 0.15


def test_word_forms_and_small_typos_still_match() -> None:
    embedder = HashingEmbedder()

    assert _cosine(embedder, "bridging", "bridge") > 0.5
    assert _cosine(embedder, "brigde fees", "bridge fee") > 0.3
    assert _cosine(embedder, "brigde fees", "staking rewards") < 0.1


def test_fingerprint_changes_with_the_dimension() -> None:
    assert HashingEmbedder(512).fingerprint != HashingEmbedder(1024).fingerprint
