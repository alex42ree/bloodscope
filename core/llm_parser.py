"""LLM-based structured extraction of lab report data via Claude API."""

from __future__ import annotations

import json
import logging
import os

import anthropic
from pydantic import ValidationError

from core.schemas import ExtractionResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a medical data extractor. Extract all biomarkers and patient information \
from the following lab report as JSON.

Rules:
- Decimal separators: comma (4,73) = 4.73, dot (4.73) = 4.73
- Values like "<0.2" or ">40": set value to the numeric part and value_modifier to "<" or ">"
- Capture both percentage differential values (e.g. "Neutrófilos %") and absolute values separately
- Process ALL pages of the report
- Determine the source language of the document ("es", "de", "en", or "fr")
- Determine patient sex as "male" or "female" from explicit terms in the report (e.g. Spanish Hombre/Masculino/Varón → male, Mujer/Femenino → female); use null if absent
- Extract date_of_birth and report_date as ISO format (YYYY-MM-DD) when available
- Calculate age_at_report from date_of_birth and report_date when both are available

Respond ONLY with valid JSON matching this exact schema (no markdown, no extra text):

{
  "patient": {
    "sex": "male|female|null",
    "date_of_birth": "YYYY-MM-DD or null",
    "age_at_report": integer or null,
    "report_date": "YYYY-MM-DD or null",
    "lab_name": "string or null",
    "source_language": "es|de|en|fr"
  },
  "markers": [
    {
      "original_name": "name as it appears in the document",
      "value": numeric_value,
      "value_modifier": "< or > or null",
      "unit": "original unit as in document",
      "reference_low": numeric or null,
      "reference_high": numeric or null,
      "flagged": true/false
    }
  ]
}
"""

MAX_RETRIES = 2


def parse_lab_report(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ExtractionResult:
    """Send extracted PDF text to Claude and return structured extraction.

    Parameters
    ----------
    text:
        Raw text extracted from the lab report PDF.
    model:
        Claude model to use. Defaults to env ``CLAUDE_MODEL``.
    api_key:
        Anthropic API key. Defaults to env ``ANTHROPIC_API_KEY``.

    Returns
    -------
    ExtractionResult
        Validated Pydantic model of the extraction.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed after retries.
    """
    resolved_model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        logger.info("LLM extraction attempt %d/%d", attempt, MAX_RETRIES + 1)

        message = client.messages.create(
            model=resolved_model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )

        raw = message.content[0].text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("Attempt %d: invalid JSON from LLM: %s", attempt, exc)
            continue

        try:
            return ExtractionResult.model_validate(data)
        except ValidationError as exc:
            last_error = exc
            logger.warning("Attempt %d: Pydantic validation failed: %s", attempt, exc)
            continue

    raise ValueError(f"Failed to get valid extraction after {MAX_RETRIES + 1} attempts: {last_error}")
