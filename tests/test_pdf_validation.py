"""Tests for PDF size/page validation helpers."""

from __future__ import annotations

import pytest

from core.exceptions import PdfTooLargeError, PdfTooManyPagesError
from core.pdf_validation import (
    get_max_pdf_bytes,
    get_max_pdf_pages,
    validate_pdf_constraints,
)


class TestPdfValidationHelpers:
    def test_validate_size_exceeded(self):
        with pytest.raises(PdfTooLargeError, match="too large"):
            validate_pdf_constraints(size_bytes=100 * 1024 * 1024, page_count=1)

    def test_validate_pages_exceeded(self, monkeypatch):
        monkeypatch.setenv("MAX_PDF_PAGES", "5")
        with pytest.raises(PdfTooManyPagesError, match="too many pages"):
            validate_pdf_constraints(size_bytes=1000, page_count=10)

    def test_get_max_pdf_pages_default(self, monkeypatch):
        monkeypatch.delenv("MAX_PDF_PAGES", raising=False)
        assert get_max_pdf_pages() == 100

    def test_get_max_pdf_bytes_from_mb(self, monkeypatch):
        monkeypatch.setenv("MAX_PDF_MB", "2")
        monkeypatch.delenv("MAX_PDF_SIZE_BYTES", raising=False)
        assert get_max_pdf_bytes() == 2 * 1024 * 1024
