"""Tests for the calibration layer.

The file ``artifacts/calibration.json`` is the only place the temperature and
the band thresholds are written down; these tests pin the behaviour that keeps
prediction, the advice card and the page in agreement.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnnclassifier import calibration  # noqa: E402
from cnnclassifier.calibration import (  # noqa: E402
    Calibration,
    apply_temperature,
    load,
)

SHIPPED = REPO_ROOT / "artifacts" / "calibration.json"


class ApplyTemperatureTest(unittest.TestCase):
    def test_temperature_one_is_the_identity(self):
        raw = [0.93, 0.07]
        np.testing.assert_allclose(apply_temperature(raw, 1.0), raw)

    def test_result_is_a_probability_vector(self):
        scores = apply_temperature([0.93, 0.07], 1.588)
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=12)
        self.assertTrue((scores > 0).all())

    def test_softening_pulls_scores_towards_the_middle(self):
        scores = apply_temperature([0.93, 0.07], 1.588)
        self.assertLess(scores[0], 0.93)
        self.assertGreater(scores[1], 0.07)

    def test_order_is_preserved(self):
        raw = np.array([0.6, 0.3, 0.1])
        scores = apply_temperature(raw, 2.5)
        self.assertEqual(list(np.argsort(raw)), list(np.argsort(scores)))

    def test_matches_softmax_of_scaled_logits(self):
        raw = np.array([0.8, 0.15, 0.05])
        expected = np.exp(np.log(raw) / 3.0)
        expected = expected / expected.sum()
        np.testing.assert_allclose(apply_temperature(raw, 3.0), expected, rtol=1e-12)

    def test_saturated_scores_stay_finite(self):
        scores = apply_temperature([1.0, 0.0], 0.5)
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=12)


class LoadTest(unittest.TestCase):
    def write(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(payload if isinstance(payload, str) else json.dumps(payload))
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_valid_file_is_used(self):
        path = self.write(
            {"temperature": 2.0, "high_threshold": 0.9, "low_threshold": 0.5}
        )
        result = load(path)
        self.assertEqual(result.temperature, 2.0)
        self.assertEqual(result.high_threshold, 0.9)
        self.assertEqual(result.low_threshold, 0.5)
        self.assertEqual(result.source, path)

    def test_nested_thresholds_from_the_kaggle_cell(self):
        """The Kaggle calibration cell writes thresholds as a nested object."""
        path = self.write(
            {
                "temperature": 1.588,
                "thresholds": {"low": 0.59, "high": 0.89},
                "metrics": {"ece_before": 0.065, "ece_after": 0.064},
                "fitted_on": "official test split, 5-fold grouped CV by patient id",
                "n_images": 624,
            }
        )
        result = load(path)
        self.assertEqual(result.temperature, 1.588)
        self.assertEqual(result.high_threshold, 0.89)
        self.assertEqual(result.low_threshold, 0.59)

    def test_missing_file_falls_back(self):
        result = load(str(REPO_ROOT / "artifacts" / "no-such-file.json"))
        self.assertEqual(result, Calibration())
        self.assertEqual(result.temperature, 1.0)

    def test_broken_json_falls_back(self):
        self.assertEqual(load(self.write("not json")), Calibration())

    def test_missing_key_falls_back(self):
        self.assertEqual(load(self.write({"temperature": 1.5})), Calibration())

    def test_non_positive_temperature_falls_back(self):
        for value in (0, -1.5, "abc"):
            payload = {
                "temperature": value,
                "high_threshold": 0.9,
                "low_threshold": 0.5,
            }
            self.assertEqual(load(self.write(payload)), Calibration(), value)

    def test_crossed_thresholds_fall_back(self):
        payload = {"temperature": 1.5, "high_threshold": 0.4, "low_threshold": 0.8}
        self.assertEqual(load(self.write(payload)), Calibration())

    def test_calibration_apply_uses_its_temperature(self):
        result = Calibration(temperature=2.0, high_threshold=0.9, low_threshold=0.5)
        np.testing.assert_allclose(
            result.apply([0.9, 0.1]), apply_temperature([0.9, 0.1], 2.0)
        )


class ShippedFileTest(unittest.TestCase):
    def test_shipped_file_parses(self):
        result = load(str(SHIPPED))
        self.assertNotEqual(result.source, "defaults")
        self.assertGreater(result.temperature, 0)
        self.assertLessEqual(result.low_threshold, result.high_threshold)

    def test_bands_agree_with_the_argmax_verdict(self):
        """A NORMAL badge must never sit next to "Pneumonia 55%"."""
        result = load(str(SHIPPED))
        self.assertLessEqual(result.low_threshold, 0.5)
        self.assertGreaterEqual(result.high_threshold, 0.5)

    def test_advice_thresholds_come_from_the_calibration_module(self):
        self.assertAlmostEqual(
            calibration.HIGH_THRESHOLD, calibration.CALIBRATION.high_threshold
        )
        self.assertAlmostEqual(
            calibration.LOW_THRESHOLD, calibration.CALIBRATION.low_threshold
        )


if __name__ == "__main__":
    unittest.main()
