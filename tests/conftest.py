"""Shared fixtures. Everything here runs offline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pdf_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """Write a small but valid text PDF, one list of lines per page."""
    page_numbers = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for number, lines in zip(page_numbers, pages, strict=True):
        text_ops = "\n".join(f"({_pdf_string(line)}) Tj T*" for line in lines)
        stream = f"BT\n/F1 12 Tf\n72 720 Td\n14 TL\n{text_ops}\nET".encode("latin-1")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {number + 1} 0 R >>"
            ).encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%b\nendstream" % (len(stream), stream))
    trailer_extra = ""
    if title is not None:
        objects.append(f"<< /Title ({_pdf_string(title)}) >>".encode("latin-1"))
        trailer_extra = f" /Info {len(objects)} 0 R"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    size = len(objects) + 1
    out += f"trailer\n<< /Size {size} /Root 1 0 R{trailer_extra} >>\n".encode()
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    return build_pdf
