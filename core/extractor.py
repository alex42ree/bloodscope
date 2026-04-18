"""PDF text extraction using pdfplumber."""

from __future__ import annotations

import logging
from typing import BinaryIO, Union

import pdfplumber

logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    pdf_file: Union[str, BinaryIO],
    *,
    max_pages: int = 20,
) -> str:
    """Extract the full text from all pages of a PDF.

    Parameters
    ----------
    pdf_file:
        A file path or file-like object (e.g. from ``st.file_uploader``).
    max_pages:
        Safety limit on the number of pages to process.

    Returns
    -------
    str
        Concatenated text from all pages, separated by double newlines.
        Returns an empty string if no text could be extracted.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                logger.warning("Reached max_pages limit (%d), stopping extraction", max_pages)
                break

            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            cells = [c or "" for c in row]
                            pages_text.append("\t".join(cells))

    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        logger.warning("No text could be extracted from the PDF")
    return full_text
