"""The on-disk index: chunk metadata as JSON, embeddings as a NumPy matrix.

Only JSON and ``.npy`` are used, and the matrix is loaded with
``allow_pickle=False``, so opening an index never executes code from the file.
The BM25 statistics are not stored: they are rebuilt from the chunk texts in
milliseconds when the index is loaded.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ragbot.documents import Chunk
from ragbot.errors import IndexStoreError

FORMAT_VERSION = 1
METADATA_FILE = "index.json"
EMBEDDINGS_FILE = "embeddings.npy"


@dataclass(frozen=True)
class IndexedFile:
    """What the index remembers about a source file, to skip it when it has not changed."""

    sha256: str
    chunks: int


@dataclass(frozen=True)
class Index:
    embedder: str
    chunk_size: int
    chunk_overlap: int
    files: dict[str, IndexedFile]
    chunks: list[Chunk]
    embeddings: np.ndarray

    def documents(self) -> list[dict[str, str | int]]:
        """One entry per indexed document: source, title and number of chunks."""
        titles = {chunk.source: chunk.title for chunk in self.chunks}
        counts = Counter(chunk.source for chunk in self.chunks)
        return [
            {"source": source, "title": titles[source], "chunks": counts[source]}
            for source in sorted(counts)
        ]

    def require_embedder(self, fingerprint: str) -> None:
        if self.embedder != fingerprint:
            raise IndexStoreError(
                f"The index was built with the embedder '{self.embedder}', but the current "
                f"settings select '{fingerprint}'. Run `ragbot ingest <folder>` to rebuild it."
            )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files": {source: asdict(record) for source, record in self.files.items()},
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        embeddings_tmp = directory / f"{EMBEDDINGS_FILE}.tmp"
        metadata_tmp = directory / f"{METADATA_FILE}.tmp"
        with embeddings_tmp.open("wb") as handle:
            np.save(handle, np.asarray(self.embeddings, dtype=np.float32), allow_pickle=False)
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        embeddings_tmp.replace(directory / EMBEDDINGS_FILE)
        metadata_tmp.replace(directory / METADATA_FILE)

    @classmethod
    def load(cls, directory: Path) -> Index:
        metadata_path = directory / METADATA_FILE
        embeddings_path = directory / EMBEDDINGS_FILE
        if not metadata_path.is_file() or not embeddings_path.is_file():
            raise IndexStoreError(
                f"No index found in '{directory}'. Run `ragbot ingest <folder>` first."
            )
        rebuild_hint = "Rebuild it with `ragbot ingest <folder> --rebuild`."
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["format_version"] != FORMAT_VERSION:
                raise IndexStoreError(
                    f"The index in '{directory}' uses format {metadata['format_version']}, "
                    f"this version reads format {FORMAT_VERSION}. {rebuild_hint}"
                )
            embeddings = np.load(embeddings_path, allow_pickle=False)
            chunks = [
                Chunk(**{**item, "heading_path": tuple(item["heading_path"])})
                for item in metadata["chunks"]
            ]
            files = {source: IndexedFile(**record) for source, record in metadata["files"].items()}
            index = cls(
                embedder=metadata["embedder"],
                chunk_size=metadata["chunk_size"],
                chunk_overlap=metadata["chunk_overlap"],
                files=files,
                chunks=chunks,
                embeddings=embeddings,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexStoreError(
                f"The index in '{directory}' cannot be read ({type(exc).__name__}). {rebuild_hint}"
            ) from exc
        per_file = Counter(chunk.source for chunk in chunks)
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(chunks)
            or per_file != {source: record.chunks for source, record in files.items()}
        ):
            raise IndexStoreError(
                f"The index in '{directory}' is inconsistent: its file list, {len(chunks)} chunks "
                f"and embedding matrix of shape {embeddings.shape} do not match. {rebuild_hint}"
            )
        return index
