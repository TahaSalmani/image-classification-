"""Tests for the /advice request path.

``advice_service`` is deliberately free of Flask and TensorFlow, so the whole
endpoint — validation, mode selection, retrieval, fallback — is tested here
without a server, a model file or an API key.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnnclassifier import llm  # noqa: E402
from cnnclassifier import advice_service as service  # noqa: E402
from cnnclassifier.advice_logic import (  # noqa: E402
    FALLBACK_TEXT,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
)

#: A score inside the "uncertain" band, wherever the calibrated thresholds are.
MID_BAND = (LOW_CONFIDENCE + HIGH_CONFIDENCE) / 2

GENERATED = {
    "summary": "The model scored this image high for pneumonia. Speak to a clinician today.",
    "what_to_do": ["Contact 111 or your GP."],
    "warning_signs": ["You're struggling to breathe."],
    "sources": [1],
    "source_urls": ["https://www.nhs.uk/conditions/pneumonia/"],
}


class ModeTests(unittest.TestCase):
    def test_high_score_selects_pneumonia_mode(self) -> None:
        payload = service.advice_for(0.93, explain=False)
        self.assertEqual(payload["mode"], "likely_pneumonia")

    def test_middle_score_selects_uncertain_mode(self) -> None:
        payload = service.advice_for(MID_BAND, explain=False)
        self.assertEqual(payload["mode"], "uncertain")

    def test_low_score_selects_normal_mode(self) -> None:
        payload = service.advice_for(0.08, explain=False)
        self.assertEqual(payload["mode"], "likely_normal")

    def test_probability_field_is_echoed(self) -> None:
        payload = service.advice_for(0.615, explain=False)
        self.assertAlmostEqual(payload["pneumonia_probability"], 0.615, places=6)


class PayloadTests(unittest.TestCase):
    def test_payload_is_json_serialisable(self) -> None:
        for probability in (0.93, 0.55, 0.08):
            payload = service.advice_for(probability, explain=True)
            self.assertIsInstance(json.dumps(payload), str)

    def test_deterministic_card_survives_without_llm(self) -> None:
        with mock.patch.object(llm, "available", return_value=False):
            payload = service.advice_for(0.93)
        self.assertEqual(payload["explanation_source"], "fallback")
        self.assertEqual(payload["explanation"], FALLBACK_TEXT)
        self.assertNotIn("explanation_details", payload)
        self.assertTrue(payload["quotes"])
        self.assertTrue(payload["red_flags"])

    def test_llm_summary_is_used_when_it_validates(self) -> None:
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(
            llm, "generate_advice", return_value=dict(GENERATED)
        ) as generate:
            payload = service.advice_for(0.93)

        generate.assert_called_once()
        mode, probability, hits = generate.call_args.args
        self.assertEqual(mode, "likely_pneumonia")
        self.assertAlmostEqual(probability, 0.93, places=6)
        self.assertLessEqual(len(hits), service.TOP_K)

        self.assertEqual(payload["explanation"], GENERATED["summary"])
        self.assertEqual(payload["explanation_source"], "llm")
        self.assertEqual(
            payload["explanation_details"]["warning_signs"],
            GENERATED["warning_signs"],
        )

    def test_llm_none_falls_back(self) -> None:
        with mock.patch.object(llm, "available", return_value=True), mock.patch.object(
            llm, "generate_advice", return_value=None
        ):
            payload = service.advice_for(0.55)
        self.assertEqual(payload["explanation_source"], "fallback")
        self.assertEqual(payload["explanation"], FALLBACK_TEXT)

    def test_explain_false_skips_the_model_call(self) -> None:
        with mock.patch.object(llm, "generate_advice") as generate:
            payload = service.advice_for(0.93, explain=False)
        generate.assert_not_called()
        self.assertEqual(payload["explanation_source"], "fallback")

    def test_passages_can_be_omitted(self) -> None:
        payload = service.advice_for(0.55, explain=False, include_passages=False)
        self.assertNotIn("passages", payload)

    def test_passages_carry_source_urls(self) -> None:
        payload = service.advice_for(0.93, explain=False)
        self.assertTrue(payload["passages"])
        for passage in payload["passages"]:
            self.assertTrue(passage["text"].strip())
            self.assertTrue(passage["sources"])
            self.assertTrue(passage["sources"][0].startswith("https://"))
            self.assertIn("score", passage)

    def test_retrieval_failure_does_not_break_the_card(self) -> None:
        with mock.patch.object(
            service, "get_retriever", side_effect=RuntimeError("no knowledge dir")
        ):
            payload = service.advice_for(0.93)
        self.assertEqual(payload["mode"], "likely_pneumonia")
        self.assertEqual(payload["passages"], [])
        self.assertEqual(payload["explanation_source"], "fallback")

    def test_retrieval_failure_also_skips_the_model_call(self) -> None:
        with mock.patch.object(
            service, "get_retriever", side_effect=RuntimeError("boom")
        ), mock.patch.object(llm, "available", return_value=True), mock.patch.object(
            llm, "generate_advice"
        ) as generate:
            service.advice_for(0.93)
        generate.assert_not_called()

    def test_unknown_mode_returns_no_passages(self) -> None:
        self.assertEqual(service.passages_for("not-a-mode"), [])


class RequestValidationTests(unittest.TestCase):
    def test_non_object_body_is_rejected(self) -> None:
        for body in (None, [], "0.5", 0.5, ["pneumonia_probability"]):
            payload, status = service.handle_advice_request(body)
            self.assertEqual(status, 400, body)
            self.assertIn("error", payload)

    def test_missing_field_is_rejected(self) -> None:
        payload, status = service.handle_advice_request({"probability": 0.5})
        self.assertEqual(status, 400)
        self.assertIn("pneumonia_probability", payload["error"])

    def test_non_numeric_values_are_rejected(self) -> None:
        for value in ("0.9", None, True, False, [], {}):
            payload, status = service.handle_advice_request(
                {"pneumonia_probability": value}
            )
            self.assertEqual(status, 400, value)

    def test_out_of_range_values_are_rejected(self) -> None:
        for value in (-0.01, 1.01, 55, -1):
            payload, status = service.handle_advice_request(
                {"pneumonia_probability": value}
            )
            self.assertEqual(status, 400, value)

    def test_non_finite_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            payload, status = service.handle_advice_request(
                {"pneumonia_probability": value}
            )
            self.assertEqual(status, 400, value)

    def test_valid_bodies_return_200(self) -> None:
        for value in (0, 0.0, 0.5, 1, 1.0):
            body, status = service.handle_advice_request(
                {"pneumonia_probability": value}
            )
            self.assertEqual(status, 200, value)
            self.assertIn("mode", body)

    def test_unexpected_failure_becomes_500_without_details(self) -> None:
        with mock.patch.object(
            service, "advice_for", side_effect=RuntimeError("secret internals")
        ):
            payload, status = service.handle_advice_request(
                {"pneumonia_probability": 0.5}
            )
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "advice is temporarily unavailable"})

    def test_route_contract_is_a_tuple(self) -> None:
        result = service.handle_advice_request({"pneumonia_probability": 0.5})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
