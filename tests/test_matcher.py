"""Tests for core.matcher."""

from __future__ import annotations

from pathlib import Path

from core.matcher import BiomarkerMatcher, normalize


class TestNormalize:
    def test_lowercase(self):
        assert normalize("Hemoglobin") == "hemoglobin"

    def test_strip_accents(self):
        assert normalize("Hämoglobin") == "hamoglobin"
        assert normalize("hémoglobine") == "hemoglobine"
        assert normalize("Hematíes") == "hematies"

    def test_whitespace_collapse(self):
        assert normalize("  colesterol   total  ") == "colesterol total"

    def test_combined(self):
        assert normalize("  Glóbulos  Rojos  ") == "globulos rojos"

    def test_empty_string(self):
        assert normalize("") == ""


class TestBiomarkerMatcher:
    def test_exact_match_es(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("hemoglobina", "es") == "hemoglobin"

    def test_exact_match_de(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("Hämoglobin", "de") == "hemoglobin"

    def test_exact_match_en(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("Total Cholesterol", "en") == "total_cholesterol"

    def test_exact_match_fr(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("hémoglobine", "fr") == "hemoglobin"

    def test_case_insensitive(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("HEMOGLOBINA", "es") == "hemoglobin"

    def test_cross_language_fallback(self, translations_file: Path):
        """If language-specific match fails, all languages are tried."""
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("hemoglobina", "fr") == "hemoglobin"

    def test_fuzzy_match(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        result = matcher.match("hemoglobine", "en")
        assert result == "hemoglobin"

    def test_no_match(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("xyznonexistent", "en") is None

    def test_missing_file(self, tmp_path: Path):
        matcher = BiomarkerMatcher(tmp_path / "nonexistent.json")
        assert matcher.match("hemoglobin", "en") is None

    def test_multi_variant(self, translations_file: Path):
        """German cholesterol has two variants."""
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("Gesamtcholesterin", "de") == "total_cholesterol"
        assert matcher.match("Cholesterin gesamt", "de") == "total_cholesterol"

    def test_abbreviation_match(self, translations_file: Path):
        matcher = BiomarkerMatcher(translations_file)
        assert matcher.match("Hgb", "en") == "hemoglobin"
        assert matcher.match("Hb", "en") == "hemoglobin"
