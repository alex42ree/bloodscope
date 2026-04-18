"""Tests for core.llm_parser with mocked Anthropic API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm_parser import parse_lab_report
from core.schemas import ExtractionResult


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock Anthropic message response."""
    block = MagicMock()
    block.text = content
    msg = MagicMock()
    msg.content = [block]
    return msg


VALID_JSON = json.dumps({
    "patient": {
        "sex": "male",
        "source_language": "es",
        "date_of_birth": "1985-03-15",
        "age_at_report": 39,
        "report_date": "2024-06-01",
        "lab_name": "Eurofins",
    },
    "markers": [
        {
            "original_name": "Hemoglobina",
            "value": 14.5,
            "value_modifier": None,
            "unit": "g/dL",
            "reference_low": 13.0,
            "reference_high": 17.5,
            "flagged": False,
        },
        {
            "original_name": "Glucosa",
            "value": 99,
            "value_modifier": None,
            "unit": "mg/dL",
            "reference_low": 70,
            "reference_high": 110,
            "flagged": False,
        },
    ],
})


class TestParsing:
    @patch("core.llm_parser.anthropic.Anthropic")
    def test_successful_parse(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.messages.create.return_value = _make_mock_response(VALID_JSON)

        result = parse_lab_report("sample lab text", api_key="test-key")

        assert isinstance(result, ExtractionResult)
        assert result.patient.sex == "male"
        assert result.patient.source_language == "es"
        assert len(result.markers) == 2
        assert result.markers[0].original_name == "Hemoglobina"

    @patch("core.llm_parser.anthropic.Anthropic")
    def test_strips_markdown_fence(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        fenced = f"```json\n{VALID_JSON}\n```"
        client.messages.create.return_value = _make_mock_response(fenced)

        result = parse_lab_report("sample lab text", api_key="test-key")
        assert isinstance(result, ExtractionResult)

    @patch("core.llm_parser.anthropic.Anthropic")
    def test_retry_on_invalid_json(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.messages.create.side_effect = [
            _make_mock_response("this is not json"),
            _make_mock_response(VALID_JSON),
        ]

        result = parse_lab_report("sample lab text", api_key="test-key")
        assert isinstance(result, ExtractionResult)
        assert client.messages.create.call_count == 2

    @patch("core.llm_parser.anthropic.Anthropic")
    def test_retry_on_validation_error(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        bad_json = json.dumps({"patient": {"sex": "invalid"}, "markers": []})
        client.messages.create.side_effect = [
            _make_mock_response(bad_json),
            _make_mock_response(VALID_JSON),
        ]

        result = parse_lab_report("sample lab text", api_key="test-key")
        assert isinstance(result, ExtractionResult)

    @patch("core.llm_parser.anthropic.Anthropic")
    def test_all_retries_exhausted(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.messages.create.return_value = _make_mock_response("not json")

        with pytest.raises(ValueError, match="Failed to get valid extraction"):
            parse_lab_report("sample lab text", api_key="test-key")
