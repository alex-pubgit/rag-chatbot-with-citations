"""Plain data types that flow through ingestion, retrieval and answering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A run of text under one heading path, for example ``("Fees", "Withdrawal fees")``."""

    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Document:
    """A loaded source file. ``source`` is its POSIX path relative to the ingested folder."""

    source: str
    title: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Chunk:
    """A passage of a document: the unit that is indexed, retrieved and cited."""

    id: str
    source: str
    title: str
    heading_path: tuple[str, ...]
    position: int
    text: str

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def index_text(self) -> str:
        """Text used for search: the title and headings give a short passage its context."""
        return "\n".join(part for part in (self.title, self.heading, self.text) if part)


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk. ``score`` is the question-to-passage cosine similarity in [0, 1]."""

    chunk: Chunk
    score: float
