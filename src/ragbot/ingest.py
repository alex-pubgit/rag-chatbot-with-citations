"""Build or update the index from a folder of documents.

Re-running ingestion is incremental: a file whose content hash has not changed
keeps its chunks and embeddings, a changed file is re-chunked and re-embedded,
and a file that disappeared is dropped from the index.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragbot.chunking import chunk_document
from ragbot.documents import Chunk
from ragbot.embeddings import Embedder
from ragbot.errors import DocumentError, IndexStoreError, ProviderError
from ragbot.index import METADATA_FILE, Index, IndexedFile
from ragbot.loaders import SUPPORTED_SUFFIXES, find_documents, load_document

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    chunks: int = 0
    embedded_chunks: int = 0

    @property
    def documents(self) -> int:
        return len(self.added) + len(self.updated) + len(self.unchanged)


def _embed(embedder: Embedder, chunks: list[Chunk]) -> np.ndarray:
    """Embed the new chunks, and make sure the embedder returned one vector for each."""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = embedder.embed_documents([chunk.index_text for chunk in chunks])
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ProviderError(
            f"The embedder '{embedder.fingerprint}' returned an array of shape {vectors.shape} "
            f"for {len(chunks)} passages. Check the embedding model and endpoint."
        )
    return vectors


def build_index(
    folder: Path,
    embedder: Embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    previous: Index | None = None,
) -> tuple[Index, IngestReport]:
    """Index every supported document under ``folder``, reusing ``previous`` where files match."""
    if not folder.is_dir():
        raise DocumentError(
            f"'{folder}' is not a folder. Pass the folder that contains your documents."
        )
    supported, unsupported = find_documents(folder)
    report = IngestReport(unsupported=[p.relative_to(folder).as_posix() for p in unsupported])
    for name in report.unsupported:
        logger.warning(
            "Skipping unsupported file: %s (supported: %s)", name, ", ".join(SUPPORTED_SUFFIXES)
        )

    previous_rows: dict[str, list[int]] = {}
    if previous:
        for row, chunk in enumerate(previous.chunks):
            previous_rows.setdefault(chunk.source, []).append(row)

    files: dict[str, IndexedFile] = {}
    fresh: dict[str, list[Chunk]] = {}
    for path in supported:
        source = path.relative_to(folder).as_posix()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            known = previous.files.get(source) if previous else None
            if known and known.sha256 == digest:
                report.unchanged.append(source)
                files[source] = known
                continue
            fresh[source] = chunk_document(load_document(path, folder), chunk_size, chunk_overlap)
        except (DocumentError, OSError) as exc:
            reason = str(exc) if isinstance(exc, DocumentError) else f"{source}: {exc.strerror}"
            logger.error("Skipping unreadable file: %s", reason)
            report.failed[source] = reason
            continue
        (report.updated if known else report.added).append(source)
        files[source] = IndexedFile(sha256=digest, chunks=len(fresh[source]))

    if not files:
        raise DocumentError(
            f"No readable documents found in '{folder}'. "
            f"Supported file types: {', '.join(SUPPORTED_SUFFIXES)}."
        )

    new_chunks = [chunk for source in sorted(fresh) for chunk in fresh[source]]
    new_vectors = _embed(embedder, new_chunks)
    new_rows = {chunk.id: row for row, chunk in enumerate(new_chunks)}

    chunks: list[Chunk] = []
    vectors: list[np.ndarray] = []
    for source in sorted(files):
        if source in fresh:
            chunks.extend(fresh[source])
            vectors.extend(new_vectors[new_rows[chunk.id]] for chunk in fresh[source])
        elif previous:
            chunks.extend(previous.chunks[row] for row in previous_rows[source])
            vectors.extend(previous.embeddings[row] for row in previous_rows[source])

    if previous:
        report.removed = sorted(set(previous.files) - set(files) - set(report.failed))
    report.chunks = len(chunks)
    report.embedded_chunks = len(new_chunks)
    index = Index(
        embedder=embedder.fingerprint,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        files=files,
        chunks=chunks,
        embeddings=np.vstack(vectors).astype(np.float32),
    )
    return index, report


def _reusable_index(
    index_dir: Path, embedder: Embedder, chunk_size: int, chunk_overlap: int
) -> Index | None:
    """The existing index, if its embeddings and chunks are still valid for these settings."""
    if not (index_dir / METADATA_FILE).exists():
        return None
    try:
        index = Index.load(index_dir)
    except IndexStoreError as exc:
        logger.warning("The existing index cannot be reused and will be rebuilt: %s", exc)
        return None
    current = (embedder.fingerprint, chunk_size, chunk_overlap)
    if (index.embedder, index.chunk_size, index.chunk_overlap) != current:
        logger.info("Embedder or chunking settings changed: rebuilding the whole index")
        return None
    return index


def ingest(
    folder: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
) -> IngestReport:
    """Update the index stored in ``index_dir`` from the documents under ``folder``."""
    previous = None if rebuild else _reusable_index(index_dir, embedder, chunk_size, chunk_overlap)
    index, report = build_index(
        folder, embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap, previous=previous
    )
    index.save(index_dir)
    return report
