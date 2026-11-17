"""Load .md, .txt, .html and .pdf files into titled documents with heading paths.

Every loader produces the same shape: a title and a list of sections, each with
the path of headings above it. Text a human reader would not see (HTML
comments, hidden elements, scripts) is dropped here, so it never reaches the
index or a prompt.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from ragbot.documents import Document, Section
from ragbot.errors import DocumentError
from ragbot.text import is_list_item, normalize_space

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".html", ".htm", ".pdf")

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.S)
_FRONT_MATTER_TITLE_RE = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.M)
_COMMENT_RE = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_RE = re.compile(r"^ {0,3}(={3,}|-{3,})[ \t]*$")
_RULE_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_TABLE_RULE_RE = re.compile(r"^\|?[ \t:|-]*-[ \t:|-]*\|[ \t:|-]*$")
_QUOTE_RE = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_BULLET_RE = re.compile(r"^[ \t]*[*+•][ \t]+")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_REFERENCE_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^>\s]+)>")
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_CODE_SPAN_RE = re.compile(r"`+([^`]+)`+")
_STRONG_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_STAR_EMPHASIS_RE = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
_UNDERSCORE_EMPHASIS_RE = re.compile(r"(?<![A-Za-z0-9])_(?=\S)(.+?)(?<=\S)_(?![A-Za-z0-9])")
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")

_HTML_SKIPPED = frozenset({"script", "style", "noscript", "template", "svg", "nav", "footer"})
_HTML_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
_HTML_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HTML_BLOCKS = frozenset(
    """address article aside blockquote body dd div dl dt fieldset figcaption figure form header
    hr main ol p pre section table tbody thead tfoot ul""".split()
)


class _Outline:
    """Collects sections while keeping track of the current heading path.

    The first level-1 heading becomes the document title and is left out of the
    heading paths, so a citation reads "Bridging / Fees", not "Bridging / Bridging > Fees".
    """

    def __init__(self) -> None:
        self.title: str | None = None
        self.sections: list[Section] = []
        self._stack: list[tuple[int, str]] = []

    def heading(self, level: int, text: str) -> None:
        if not text:
            return
        if level == 1 and self.title is None:
            self.title = text
            self._stack.clear()
            return
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    def add_text(self, text: str) -> None:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            path = tuple(heading for _, heading in self._stack)
            self.sections.append(Section(heading_path=path, text=text))


def _plain_inline(text: str) -> str:
    """Turn inline Markdown into the plain text a reader would see."""
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REFERENCE_LINK_RE.sub(r"\1", text)
    text = _AUTOLINK_RE.sub(r"\1", text)
    text = _TAG_RE.sub("", text)
    text = _CODE_SPAN_RE.sub(r"\1", text)
    text = _STRONG_RE.sub(r"\2", text)
    text = _STAR_EMPHASIS_RE.sub(r"\1", text)
    text = _UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
    return _ESCAPE_RE.sub(r"\1", text)


def _reflow(lines: list[str], clean: Callable[[str], str]) -> str:
    """Join hard-wrapped lines into one line per paragraph, list item or table row."""
    out: list[str] = []
    for raw in lines:
        line = clean(_BULLET_RE.sub("- ", _QUOTE_RE.sub("", raw))).strip()
        if not line or _TABLE_RULE_RE.match(line):
            continue
        starts_new_line = is_list_item(line) or line.startswith("|")
        if out and not starts_new_line and not out[-1].startswith("|"):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def _parse_markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    front_matter_title = None
    front_matter = _FRONT_MATTER_RE.match(text)
    if front_matter:
        title_line = _FRONT_MATTER_TITLE_RE.search(front_matter.group(1))
        if title_line:
            front_matter_title = title_line.group(1).strip("\"'")
        text = text[front_matter.end() :]
    text = _COMMENT_RE.sub("", text)

    outline = _Outline()
    blocks: list[str] = []
    paragraph: list[str] = []
    fence: str | None = None

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(_reflow(paragraph, _plain_inline))
            paragraph.clear()

    def end_section() -> None:
        end_paragraph()
        outline.add_text("\n\n".join(block for block in blocks if block))
        blocks.clear()

    for line in text.split("\n"):
        fence_match = _FENCE_RE.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1).startswith(fence):
                blocks.append("\n".join(paragraph))
                paragraph.clear()
                fence = None
            else:
                paragraph.append(line)
            continue
        if fence_match:
            end_paragraph()
            fence = fence_match.group(1)[:3]
            continue
        atx = _ATX_RE.match(line)
        if atx:
            end_section()
            outline.heading(len(atx.group(1)), _plain_inline(atx.group(2)).strip())
            continue
        setext = _SETEXT_RE.match(line)
        if setext and len(paragraph) == 1 and not is_list_item(paragraph[0]):
            heading_text = _plain_inline(paragraph.pop()).strip()
            end_section()
            outline.heading(1 if setext.group(1).startswith("=") else 2, heading_text)
            continue
        if not line.strip() or _RULE_RE.match(line):
            end_paragraph()
            continue
        paragraph.append(line)
    if fence is not None:
        blocks.append("\n".join(paragraph))
        paragraph.clear()
    end_section()
    return outline.title or front_matter_title or fallback_title, outline.sections


def _parse_text(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    outline = _Outline()
    paragraphs = [
        _reflow(block.split("\n"), lambda line: line) for block in re.split(r"\n[ \t]*\n", text)
    ]
    outline.add_text("\n\n".join(paragraph for paragraph in paragraphs if paragraph))
    return fallback_title, outline.sections


class _HtmlOutlineParser(HTMLParser):
    """Extract visible text and h1-h6 structure. Hidden or non-content elements are skipped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.outline = _Outline()
        self.page_title = ""
        self._skipped: list[str] = []
        self._in_title = False
        self._heading_level = 0
        self._inline: list[str] = []
        self._lines: list[str] = []
        self._prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        hidden = "hidden" in attributes or attributes.get("aria-hidden") == "true"
        if self._skipped or tag in _HTML_SKIPPED or hidden:
            if tag not in _HTML_VOID:
                self._skipped.append(tag)
        elif tag == "title":
            self._in_title = True
        elif tag in _HTML_HEADINGS:
            self._end_block()
            self._heading_level = int(tag[1])
        elif tag == "li":
            self._end_line()
            self._prefix = "- "
        elif tag in ("br", "tr"):
            self._end_line()
        elif tag in ("td", "th"):
            if self._inline:
                self._inline.append(" | ")
        elif tag in _HTML_BLOCKS:
            self._end_block()

    def handle_endtag(self, tag: str) -> None:
        if self._skipped:
            if tag in self._skipped:
                while self._skipped.pop() != tag:
                    pass
        elif tag == "title":
            self._in_title = False
        elif tag in _HTML_HEADINGS and self._heading_level:
            heading = normalize_space("".join(self._inline))
            self._inline.clear()
            self._flush_section()
            self.outline.heading(self._heading_level, heading)
            self._heading_level = 0
        elif tag in ("li", "tr"):
            self._end_line()
        elif tag in _HTML_BLOCKS:
            self._end_block()

    def handle_data(self, data: str) -> None:
        if self._skipped:
            return
        if self._in_title:
            self.page_title += data
        else:
            self._inline.append(data)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _end_line(self) -> None:
        text = normalize_space("".join(self._inline))
        self._inline.clear()
        if text:
            self._lines.append(self._prefix + text)
        self._prefix = ""

    def _end_block(self) -> None:
        self._end_line()
        if self._lines and self._lines[-1]:
            self._lines.append("")

    def _flush_section(self) -> None:
        self._end_line()
        self.outline.add_text("\n".join(self._lines))
        self._lines.clear()


def _parse_html(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    parser = _HtmlOutlineParser()
    parser.feed(text)
    parser.close()
    title = parser.outline.title or normalize_space(parser.page_title) or fallback_title
    return title, parser.outline.sections


def _load_pdf(path: Path, fallback_title: str) -> tuple[str, list[Section]]:
    try:
        reader = PdfReader(path)
        metadata_title = (reader.metadata.title if reader.metadata else None) or ""
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            lines = (page.extract_text() or "").split("\n")
            text = _reflow(lines, lambda line: line)
            if text:
                sections.append(Section(heading_path=(f"Page {number}",), text=text))
    except (PyPdfError, ValueError, KeyError) as exc:
        raise DocumentError(f"not a readable PDF ({exc})") from exc
    return normalize_space(metadata_title) or fallback_title, sections


_TEXT_PARSERS: dict[str, Callable[[str, str], tuple[str, list[Section]]]] = {
    ".md": _parse_markdown,
    ".markdown": _parse_markdown,
    ".txt": _parse_text,
    ".html": _parse_html,
    ".htm": _parse_html,
}


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip().capitalize() or path.name


def load_document(path: Path, root: Path) -> Document:
    """Load one supported file. Raises ``DocumentError`` with a reason if it has no usable text."""
    source = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    fallback_title = _title_from_filename(path)
    try:
        if suffix == ".pdf":
            title, sections = _load_pdf(path, fallback_title)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            title, sections = _TEXT_PARSERS[suffix](text, fallback_title)
    except OSError as exc:
        raise DocumentError(f"{source}: cannot read the file ({exc.strerror})") from exc
    except DocumentError as exc:
        raise DocumentError(f"{source}: {exc}") from exc
    if not sections:
        hint = " (scanned PDFs need OCR, which is not included)" if suffix == ".pdf" else ""
        raise DocumentError(f"{source}: no extractable text{hint}")
    return Document(source=source, title=title, sections=tuple(sections))


def find_documents(folder: Path) -> tuple[list[Path], list[Path]]:
    """Return (supported, unsupported) files under ``folder``, sorted, hidden paths skipped."""
    supported: list[Path] = []
    unsupported: list[Path] = []
    for path in sorted(folder.rglob("*"), key=lambda p: p.relative_to(folder).as_posix()):
        relative = path.relative_to(folder)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        (supported if path.suffix.lower() in SUPPORTED_SUFFIXES else unsupported).append(path)
    return supported, unsupported
