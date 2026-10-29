"""Tests for the deterministic advice layer.

The important one is ``test_every_quote_is_verbatim``: it fails the moment a
sentence in an advice card stops matching the knowledge file it cites, which is
how this project keeps medical wording away from improvisation.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnnclassifier import advice_logic as logic  # noqa: E402
from cnnclassifier.advice_logic import (  # noqa: E402
    DISCLAIMER,
    FALLBACK_TEXT,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    build_advice,
    mode_for,
    pneumonia_probability,
)

KNOWLEDGE = REPO_ROOT / "knowledge"

#: A score inside the "uncertain" band, wherever the calibrated thresholds are.
MID_BAND = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2


def _file_text(name: str) -> str:
    raw = (KNOWLEDGE / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", raw)


class ModeTests(unittest.TestCase):
    def test_high_confidence_is_pneumonia(self) -> None:
        self.assertEqual(mode_for(0.99), "likely_pneumonia")
        self.assertEqual(mode_for(HIGH_CONFIDENCE), "likely_pneumonia")

    def test_middle_band_is_uncertain(self) -> None:
        self.assertEqual(mode_for(MID_BAND), "uncertain")
        self.assertEqual(mode_for(LOW_CONFIDENCE), "uncertain")

    def test_low_confidence_is_normal(self) -> None:
        self.assertEqual(mode_for(0.10), "likely_normal")
        self.assertEqual(mode_for(0.0), "likely_normal")

    def test_custom_thresholds(self) -> None:
        self.assertEqual(mode_for(0.60, high=0.5, low=0.4), "likely_pneumonia")
        self.assertEqual(mode_for(0.45, high=0.8, low=0.4), "uncertain")

    def test_bad_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mode_for(0.5, high=0.3, low=0.7)


class PredictionParsingTests(unittest.TestCase):
    API_RESULT = [
        {
            "image": "Pneumonia",
            "confidence": 0.841,
            "probabilities": {"Normal": 0.159, "Pneumonia": 0.841},
        }
    ]

    def test_accepts_api_list(self) -> None:
        self.assertAlmostEqual(pneumonia_probability(self.API_RESULT), 0.841)

    def test_accepts_single_dict(self) -> None:
        self.assertAlmostEqual(pneumonia_probability(self.API_RESULT[0]), 0.841)

    def test_accepts_bare_float(self) -> None:
        self.assertAlmostEqual(pneumonia_probability(0.3), 0.3)

    def test_accepts_confidence_without_probabilities(self) -> None:
        normal = {"image": "Normal", "confidence": 0.9}
        self.assertAlmostEqual(pneumonia_probability(normal), 0.1)

    def test_rejects_garbage(self) -> None:
        for bad in ([], {}, {"image": "Pneumonia"}, "0.9", [None]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    pneumonia_probability(bad)

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            pneumonia_probability(1.5)


class AdviceContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cards = [build_advice(probability) for probability in (0.95, MID_BAND, 0.05)]

    def test_every_mode_is_covered(self) -> None:
        self.assertEqual(
            [card.mode for card in self.cards],
            ["likely_pneumonia", "uncertain", "likely_normal"],
        )

    def test_every_quote_is_verbatim(self) -> None:
        for card in self.cards:
            for quote in (*card.quotes, *card.actions, *card.red_flags):
                with self.subTest(mode=card.mode, text=quote.text[:40]):
                    haystack = _file_text(quote.file)
                    needle = re.sub(r"\s+", " ", quote.text)
                    self.assertIn(needle, haystack)

    def test_quotes_have_no_unquoted_medical_claims(self) -> None:
        # Every quote must be a quotation or a bullet copied from the source.
        for card in self.cards:
            for quote in card.quotes:
                self.assertFalse(
                    quote.text.endswith(".") and quote.file == "",
                    quote.text,
                )

    def test_citations_exist(self) -> None:
        for card in self.cards:
            for name in card.citations:
                with self.subTest(mode=card.mode, file=name):
                    self.assertTrue((KNOWLEDGE / name).is_file(), name)

    def test_sources_match_cited_files(self) -> None:
        for card in self.cards:
            declared = set()
            for name in card.citations:
                text = (KNOWLEDGE / name).read_text(encoding="utf-8")
                declared.update(re.findall(r"^SOURCE:\s*(\S+)", text, re.MULTILINE))
            with self.subTest(mode=card.mode):
                self.assertEqual(set(card.sources), declared)

    def test_red_flags_are_always_present(self) -> None:
        for card in self.cards:
            with self.subTest(mode=card.mode):
                self.assertEqual(len(card.red_flags), 6)

    def test_disclaimer_is_present(self) -> None:
        for card in self.cards:
            self.assertEqual(card.disclaimer, DISCLAIMER)
            self.assertIn("not a medical device", card.disclaimer)


class FallbackTests(unittest.TestCase):
    def test_advice_renders_without_llm(self) -> None:
        payload = build_advice(0.9).as_dict()
        self.assertEqual(payload["explanation"], FALLBACK_TEXT)
        self.assertEqual(payload["explanation_source"], "fallback")
        self.assertTrue(payload["quotes"])
        self.assertTrue(payload["sources"])

    def test_llm_explanation_replaces_fallback(self) -> None:
        payload = build_advice(0.9).as_dict(explanation="  quoted text  ")
        self.assertEqual(payload["explanation"], "quoted text")
        self.assertEqual(payload["explanation_source"], "llm")

    def test_blank_explanation_falls_back(self) -> None:
        payload = build_advice(0.9).as_dict(explanation="   ")
        self.assertEqual(payload["explanation_source"], "fallback")

    def test_json_ready(self) -> None:
        import json

        payload = build_advice(MID_BAND).as_dict()
        round_tripped = json.loads(json.dumps(payload))
        self.assertEqual(round_tripped["mode"], "uncertain")
        self.assertIn("red_flags", round_tripped)

    def test_every_mode_has_text_without_llm(self) -> None:
        for mode in logic.MODES:
            with self.subTest(mode=mode):
                advice = logic._ADVICE[mode]
                self.assertTrue(advice.headline.strip())
                self.assertTrue(advice.summary.strip())
                self.assertTrue(advice.actions)


if __name__ == "__main__":
    unittest.main()
