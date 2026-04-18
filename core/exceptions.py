"""Domain-specific errors for PDF handling and the processing pipeline."""


class BloodScopePdfError(Exception):
    """Base class for PDF constraint or format errors."""


class PdfTooLargeError(BloodScopePdfError):
    """Raised when the PDF file exceeds the configured maximum size."""


class PdfTooManyPagesError(BloodScopePdfError):
    """Raised when the PDF exceeds the configured maximum page count."""


class ScannedPdfNotSupportedError(BloodScopePdfError):
    """Raised when no extractable text is found (typical of image-only / scanned PDFs)."""
