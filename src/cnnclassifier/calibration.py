"""Calibrated pneumonia probabilities and band thresholds.

``artifacts/calibration.json`` comes from the Kaggle calibration run: the
temperature is fitted by NLL on the 624-image test split, the thresholds are
read off the calibrated scores. Prediction, the advice card and the page all
read this module, so the displayed confidence and the band boundaries cannot
drift apart.

A missing or malformed file falls back to raw probabilities (temperature 1.0)
and the old hand-picked thresholds: a bad deploy degrades, it does not fail.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_PATH = os.environ.get(
    "CALIBRATION_PATH", os.path.join("artifacts", "calibration.json")
)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_HIGH = 0.75
DEFAULT_LOW = 0.40


def apply_temperature(
    probabilities: Sequence[float], temperature: float
) -> np.ndarray:
    """Rescale a probability vector without changing the order of its entries."""
    scores = np.asarray(probabilities, dtype=float)
    if temperature == 1.0 or scores.size == 0:
        return scores
    log_scores = np.log(np.clip(scores, 1e-12, None)) / temperature
    log_scores -= log_scores.max()
    weights = np.exp(log_scores)
    return weights / weights.sum()


@dataclass(frozen=True)
class Calibration:
    temperature: float = DEFAULT_TEMPERATURE
    high_threshold: float = DEFAULT_HIGH
    low_threshold: float = DEFAULT_LOW
    source: str = "defaults"

    def apply(self, probabilities: Sequence[float]) -> np.ndarray:
        return apply_temperature(probabilities, self.temperature)


def _from_mapping(data: Mapping[str, Any], source: str) -> Calibration:
    temperature = float(data["temperature"])
    thresholds = data.get("thresholds")
    if isinstance(thresholds, Mapping):
        high = float(thresholds["high"])
        low = float(thresholds["low"])
    else:
        high = float(data["high_threshold"])
        low = float(data["low_threshold"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a positive number: {temperature}")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"thresholds must satisfy 0 <= low <= high <= 1: {low}, {high}")
    return Calibration(
        temperature=temperature,
        high_threshold=high,
        low_threshold=low,
        source=source,
    )


def load(path: str | None = None) -> Calibration:
    """Read a calibration file, falling back to defaults on any problem."""
    target = path or CALIBRATION_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            return _from_mapping(json.load(handle), target)
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("calibration unavailable (%s); using defaults", error)
        return Calibration()


CALIBRATION = load()
TEMPERATURE = CALIBRATION.temperature
HIGH_THRESHOLD = CALIBRATION.high_threshold
LOW_THRESHOLD = CALIBRATION.low_threshold

__all__ = [
    "CALIBRATION",
    "CALIBRATION_PATH",
    "Calibration",
    "HIGH_THRESHOLD",
    "LOW_THRESHOLD",
    "TEMPERATURE",
    "apply_temperature",
    "load",
]
