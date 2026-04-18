"""PDF size and page limits from environment; validation helpers."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber

from core.exceptions import PdfTooLargeError, PdfTooManyPagesError

DEFAULT_MAX_PDF_MB = 10
DEFAULT_MAX_PDF_PAGES = 100


def get_max_pdf_bytes() -> int:
    """Maximum upload size in bytes (``MAX_PDF_SIZE_BYTES`` or ``MAX_PDF_MB`` × 1 MiB)."""
    raw = os.getenv("MAX_PDF_SIZE_BYTES")
    if raw and raw.strip().isdigit():
        return int(raw)
    mb = os.getenv("MAX_PDF_MB", str(DEFAULT_MAX_PDF_MB)).strip()
    try:
        mb_f = float(mb)
    except ValueError:
        mb_f = float(DEFAULT_MAX_PDF_MB)
    return int(mb_f * 1024 * 1024)


def get_max_pdf_pages() -> int:
    """Maximum number of pages to process (``MAX_PDF_PAGES``, default 100)."""
    raw = os.getenv("MAX_PDF_PAGES", str(DEFAULT_MAX_PDF_PAGES)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_PDF_PAGES


def count_pdf_pages(source: Union[str, Path, BinaryIO, bytes]) -> int:
    """Return total page count without applying ``max_pages`` truncation."""
    if isinstance(source, bytes):
        src: Union[str, Path, BinaryIO] = io.BytesIO(source)
    else:
        src = source
    with pdfplumber.open(src) as pdf:
        return len(pdf.pages)


def validate_pdf_constraints(
    *,
    size_bytes: int,
    page_count: int,
) -> None:
    """Raise ``PdfTooLargeError`` or ``PdfTooManyPagesError`` when limits are exceeded."""
    max_b = get_max_pdf_bytes()
    max_p = get_max_pdf_pages()
    if size_bytes > max_b:
        raise PdfTooLargeError(
            f"PDF is too large ({size_bytes / (1024 * 1024):.2f} MiB). "
            f"Maximum allowed is {max_b / (1024 * 1024):.2f} MiB "
            f"(set MAX_PDF_MB or MAX_PDF_SIZE_BYTES).",
        )
    if page_count > max_p:
        raise PdfTooManyPagesError(
            f"PDF has too many pages ({page_count}). Maximum allowed is {max_p} "
            f"(set MAX_PDF_PAGES).",
        )


def validate_pdf_path(path: Union[str, Path]) -> int:
    """Validate file size and page count for a path; return page count."""
    p = Path(path)
    size = p.stat().st_size
    pages = count_pdf_pages(p)
    validate_pdf_constraints(size_bytes=size, page_count=pages)
    return pages


def validate_pdf_bytes(data: bytes) -> int:
    """Validate size and page count for raw PDF bytes; return page count."""
    pages = count_pdf_pages(data)
    validate_pdf_constraints(size_bytes=len(data), page_count=pages)
    return pages
