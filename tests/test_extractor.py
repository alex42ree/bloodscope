"""Tests for core.extractor."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.extractor import extract_text_from_pdf


class TestExtractor:
    def test_multi_page_text(self):
        """Test extraction with mocked pdfplumber pages."""
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 content"
        page1.extract_tables.return_value = []

        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 content"
        page2.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("core.extractor.pdfplumber.open", return_value=mock_pdf):
            text = extract_text_from_pdf("dummy.pdf")

        assert "Page 1 content" in text
        assert "Page 2 content" in text

    def test_empty_page_with_tables(self):
        """When extract_text returns None, fall back to tables."""
        page = MagicMock()
        page.extract_text.return_value = None
        page.extract_tables.return_value = [
            [["Name", "Value"], ["Hemoglobin", "14.5"]],
        ]

        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("core.extractor.pdfplumber.open", return_value=mock_pdf):
            text = extract_text_from_pdf("dummy.pdf")

        assert "Hemoglobin" in text
        assert "14.5" in text

    def test_completely_empty_pdf(self):
        """PDF with no text and no tables returns empty string."""
        page = MagicMock()
        page.extract_text.return_value = None
        page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("core.extractor.pdfplumber.open", return_value=mock_pdf):
            text = extract_text_from_pdf("dummy.pdf")

        assert text.strip() == ""

    def test_max_pages_limit(self):
        """Only the first max_pages pages are processed."""
        pages = []
        for i in range(10):
            p = MagicMock()
            p.extract_text.return_value = f"Page {i}"
            pages.append(p)

        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("core.extractor.pdfplumber.open", return_value=mock_pdf):
            text = extract_text_from_pdf("dummy.pdf", max_pages=3)

        assert "Page 0" in text
        assert "Page 2" in text
        assert "Page 3" not in text
