from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragbot.documents import Chunk
from ragbot.errors import IndexStoreError
from ragbot.index import EMBEDDINGS_FILE, METADATA_FILE, Index, IndexedFile


def _index() -> Index:
    chunks = [
        Chunk("bridge.md#0", "bridge.md", "Bridging", (), 0, "Move tokens between networks."),
        Chunk(
            "bridge.md#1", "bridge.md", "Bridging", ("Fees", "Minimums"), 1, "Gebühr: 0.05%, min 1."
        ),
        Chunk("guides/api.md#0", "guides/api.md", "Public API", ("Limits",), 0, "60 requests."),
    ]
    return Index(
        embedder="hashing-v1-8",
        chunk_size=160,
        chunk_overlap=30,
        files={
            "bridge.md": IndexedFile(sha256="a" * 64, chunks=2),
            "guides/api.md": IndexedFile(sha256="b" * 64, chunks=1),
        },
        chunks=chunks,
        embeddings=np.linspace(0.0, 1.0, num=24, dtype=np.float32).reshape(3, 8),
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = _index()
    original.save(tmp_path / "nested" / "index")

    loaded = Index.load(tmp_path / "nested" / "index")

    assert loaded.chunks == original.chunks
    assert loaded.files == original.files
    assert (loaded.embedder, loaded.chunk_size, loaded.chunk_overlap) == ("hashing-v1-8", 160, 30)
    assert loaded.embeddings.dtype == np.float32
    assert np.array_equal(loaded.embeddings, original.embeddings)


def test_index_is_plain_json_plus_npy(tmp_path: Path) -> None:
    _index().save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [EMBEDDINGS_FILE, METADATA_FILE]
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["chunks"][1]["heading_path"] == ["Fees", "Minimums"]


def test_documents_lists_each_source_with_its_chunk_count() -> None:
    assert _index().documents() == [
        {"source": "bridge.md", "title": "Bridging", "chunks": 2},
        {"source": "guides/api.md", "title": "Public API", "chunks": 1},
    ]


def test_missing_index_tells_the_user_to_ingest(tmp_path: Path) -> None:
    with pytest.raises(
        IndexStoreError, match=r"No index found .* Run `ragbot ingest <folder>` first"
    ):
        Index.load(tmp_path / "nowhere")


def test_corrupt_metadata_is_reported_with_a_rebuild_hint(tmp_path: Path) -> None:
    _index().save(tmp_path)
    (tmp_path / METADATA_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(IndexStoreError, match=r"cannot be read .*--rebuild"):
        Index.load(tmp_path)


def test_embedding_matrix_that_does_not_match_the_chunks_is_rejected(tmp_path: Path) -> None:
    _index().save(tmp_path)
    np.save(tmp_path / EMBEDDINGS_FILE, np.zeros((2, 8), dtype=np.float32), allow_pickle=False)

    with pytest.raises(IndexStoreError, match="inconsistent"):
        Index.load(tmp_path)


def test_pickled_embeddings_are_never_unpickled(tmp_path: Path) -> None:
    _index().save(tmp_path)
    poisoned = np.array([{"payload": "x"}, None, None], dtype=object)
    np.save(tmp_path / EMBEDDINGS_FILE, poisoned, allow_pickle=True)

    with pytest.raises(IndexStoreError, match="cannot be read"):
        Index.load(tmp_path)


def test_other_format_versions_are_rejected(tmp_path: Path) -> None:
    _index().save(tmp_path)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    metadata["format_version"] = 99
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexStoreError, match="uses format 99"):
        Index.load(tmp_path)


def test_require_embedder_explains_a_mismatch() -> None:
    index = _index()

    index.require_embedder("hashing-v1-8")
    with pytest.raises(
        IndexStoreError, match=r"built with the embedder 'hashing-v1-8'.*ragbot ingest"
    ):
        index.require_embedder("fastembed:other-model")
