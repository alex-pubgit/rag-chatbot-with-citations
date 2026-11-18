from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ragbot.embeddings.hashing import HashingEmbedder
from ragbot.errors import DocumentError, ProviderError
from ragbot.index import EMBEDDINGS_FILE, Index
from ragbot.ingest import IngestReport, ingest

DOCS = {
    "bridge.md": "# Bridging\n\nMove tokens.\n\n## Fees\n\nThe bridge fee is 0.05%.\n",
    "staking.md": "# Staking\n\nStake ACME.\n\n## Unstaking\n\nUnstaking takes 7 days.\n",
    "guides/api.txt": "The public API allows 60 requests per minute.\n",
}


class SpyEmbedder(HashingEmbedder):
    """Records every batch it is asked to embed."""

    def __init__(self, dimension: int = 256) -> None:
        super().__init__(dimension)
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        return super().embed_documents(texts)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    for name, content in DOCS.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return folder


def _ingest(docs: Path, embedder: HashingEmbedder, **overrides: int | bool) -> IngestReport:
    options = {"chunk_size": 160, "chunk_overlap": 30, **overrides}
    return ingest(docs, docs.parent / "index", embedder, **options)


def _load(docs: Path) -> Index:
    return Index.load(docs.parent / "index")


def test_first_ingest_indexes_every_document(docs: Path) -> None:
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder)
    index = _load(docs)

    assert report.added == ["bridge.md", "guides/api.txt", "staking.md"]
    assert (report.documents, report.chunks, report.embedded_chunks) == (3, 5, 5)
    assert [chunk.id for chunk in index.chunks] == [
        "bridge.md#0",
        "bridge.md#1",
        "guides/api.txt#0",
        "staking.md#0",
        "staking.md#1",
    ]
    assert index.embeddings.shape == (5, 256)
    assert index.chunks[1].heading_path == ("Fees",)


def test_unchanged_files_are_not_embedded_again(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    before = _load(docs)
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder)
    after = _load(docs)

    assert embedder.batches == []
    assert report.unchanged == ["bridge.md", "guides/api.txt", "staking.md"]
    assert (report.added, report.updated, report.removed) == ([], [], [])
    assert report.embedded_chunks == 0
    assert after.chunks == before.chunks
    assert np.array_equal(after.embeddings, before.embeddings)


def test_only_a_changed_file_is_re_embedded(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    before = _load(docs)
    (docs / "bridge.md").write_text(
        "# Bridging\n\nMove tokens.\n\n## Fees\n\nThe bridge fee is now 0.07%.\n", encoding="utf-8"
    )
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder)
    after = _load(docs)

    assert report.updated == ["bridge.md"]
    assert report.unchanged == ["guides/api.txt", "staking.md"]
    assert len(embedder.batches) == 1
    assert all(text.startswith("Bridging") for text in embedder.batches[0])
    assert after.chunks[1].text == "The bridge fee is now 0.07%."
    assert after.chunks[2:] == before.chunks[2:]
    assert np.array_equal(after.embeddings[2:], before.embeddings[2:])
    assert not np.array_equal(after.embeddings[1], before.embeddings[1])


def test_deleted_files_leave_the_index_and_new_files_join_it(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    (docs / "staking.md").unlink()
    (docs / "airdrops.md").write_text("# Airdrops\n\nNone is announced.\n", encoding="utf-8")
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder)
    index = _load(docs)

    assert report.removed == ["staking.md"]
    assert report.added == ["airdrops.md"]
    assert sorted(index.files) == ["airdrops.md", "bridge.md", "guides/api.txt"]
    assert {chunk.source for chunk in index.chunks} == set(index.files)
    assert index.embeddings.shape[0] == len(index.chunks) == 4
    assert embedder.batches == [["Airdrops\nNone is announced."]]


def test_changing_the_chunking_settings_rebuilds_everything(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder, chunk_size=80, chunk_overlap=10)

    assert report.embedded_chunks == report.chunks == 5
    assert report.unchanged == []
    assert _load(docs).chunk_size == 80


def test_switching_the_embedder_rebuilds_everything(docs: Path) -> None:
    _ingest(docs, SpyEmbedder(dimension=256))
    other = SpyEmbedder(dimension=128)

    report = _ingest(docs, other)

    assert report.embedded_chunks == 5
    assert _load(docs).embeddings.shape == (5, 128)


def test_rebuild_flag_ignores_the_existing_index(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    embedder = SpyEmbedder()

    report = _ingest(docs, embedder, rebuild=True)

    assert report.embedded_chunks == 5
    assert report.added == ["bridge.md", "guides/api.txt", "staking.md"]


def test_a_damaged_index_is_rebuilt_instead_of_crashing(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    (docs.parent / "index" / EMBEDDINGS_FILE).write_bytes(b"garbage")

    report = _ingest(docs, SpyEmbedder())

    assert report.embedded_chunks == 5
    assert _load(docs).embeddings.shape == (5, 256)


def test_unsupported_files_are_skipped_with_a_clear_log_line(
    docs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (docs / "notes.docx").write_bytes(b"binary")

    with caplog.at_level(logging.WARNING, logger="ragbot.ingest"):
        report = _ingest(docs, SpyEmbedder())

    assert report.unsupported == ["notes.docx"]
    assert "Skipping unsupported file: notes.docx (supported: .md" in caplog.text
    assert "notes.docx" not in _load(docs).files


def test_an_unreadable_file_does_not_stop_the_others(
    docs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (docs / "broken.pdf").write_bytes(b"not a pdf")

    with caplog.at_level(logging.ERROR, logger="ragbot.ingest"):
        report = _ingest(docs, SpyEmbedder())

    assert list(report.failed) == ["broken.pdf"]
    assert "broken.pdf: not a readable PDF" in report.failed["broken.pdf"]
    assert "Skipping unreadable file: broken.pdf" in caplog.text
    assert report.documents == 3
    assert "broken.pdf" not in _load(docs).files


def test_folder_without_documents_is_an_error_and_keeps_the_old_index(docs: Path) -> None:
    _ingest(docs, SpyEmbedder())
    empty = docs.parent / "empty"
    empty.mkdir()

    with pytest.raises(DocumentError, match="No readable documents found"):
        ingest(empty, docs.parent / "index", SpyEmbedder(), chunk_size=160, chunk_overlap=30)

    assert len(_load(docs).chunks) == 5


def test_missing_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="is not a folder"):
        ingest(
            tmp_path / "nope", tmp_path / "index", SpyEmbedder(), chunk_size=160, chunk_overlap=30
        )


def test_embedder_that_returns_too_few_vectors_is_reported(docs: Path) -> None:
    class ShortEmbedder(HashingEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
            return super().embed_documents(texts)[:-1]

    with pytest.raises(
        ProviderError, match=r"returned an array of shape \(4, 2048\) for 5 passages"
    ):
        _ingest(docs, ShortEmbedder())
