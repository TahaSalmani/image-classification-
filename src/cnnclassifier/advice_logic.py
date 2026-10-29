"""Deterministic advice layer for the pneumonia demo.

Every sentence that could be read as medical guidance is quoted verbatim from
``knowledge/``; this module contributes only the wording that describes what the
model did. No LLM is involved, so the page still works when the model API is
missing, rate-limited or broken — that is the whole point of the fixed modes.

Three modes, chosen from the model's calibrated pneumonia probability:

``likely_pneumonia``  p >= 0.52
``uncertain``         0.38 <= p < 0.52
``likely_normal``     p < 0.38

The thresholds live in ``artifacts/calibration.json`` and are read through
``cnnclassifier.calibration``, so the verdict badge, this card and the model
output all use the same numbers. On the 624-image test split they put 98% of
pneumonia cases in the top band, which is right 90% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cnnclassifier.calibration import HIGH_THRESHOLD, LOW_THRESHOLD

HIGH_CONFIDENCE = HIGH_THRESHOLD
LOW_CONFIDENCE = LOW_THRESHOLD

DISCLAIMER = (
    "Research demo, not a medical device. This output is a model score on one "
    "image and must not be used for clinical decisions."
)

#: Shown whenever the optional language-model explanation is unavailable. The
#: rest of the page is deterministic, so nothing else breaks.
FALLBACK_TEXT = (
    "A written explanation is unavailable right now. This card and the source "
    "passages below do not depend on it."
)


@dataclass(frozen=True)
class Quote:
    """A sentence copied from a knowledge file."""

    text: str
    file: str

    def as_dict(self) -> dict:
        return {"text": self.text, "file": self.file}


@dataclass(frozen=True)
class Advice:
    mode: str
    label: str
    headline: str
    summary: str
    quotes: tuple[Quote, ...] = ()
    actions: tuple[Quote, ...] = ()
    red_flags: tuple[Quote, ...] = ()
    citations: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    disclaimer: str = DISCLAIMER

    def as_dict(self, explanation: str | None = None) -> dict:
        """JSON-ready form. ``explanation`` is the optional LLM text."""
        text = explanation.strip() if explanation else ""
        return {
            "mode": self.mode,
            "label": self.label,
            "headline": self.headline,
            "summary": self.summary,
            "quotes": [quote.as_dict() for quote in self.quotes],
            "actions": [quote.as_dict() for quote in self.actions],
            "red_flags": [quote.as_dict() for quote in self.red_flags],
            "citations": list(self.citations),
            "sources": list(self.sources),
            "disclaimer": self.disclaimer,
            "explanation": text or FALLBACK_TEXT,
            "explanation_source": "llm" if text else "fallback",
        }


NHS_EMERGENCY = "02-emergency-warning-signs-nhs.md"
NHS_URGENT = "03-when-to-get-urgent-advice-nhs.md"
MEDLINEPLUS_ATYPICAL = (
    "10-atypical-symptoms-infants-older-adults-medlineplus.md"
)
MEDLINEPLUS_DIAGNOSIS = "11-diagnosis-medlineplus.md"
NOT_DEFINITIVE = "12-normal-scan-not-definitive.md"

NHS_URL = "https://www.nhs.uk/conditions/pneumonia/"
MEDLINEPLUS_URL = "https://medlineplus.gov/pneumonia.html"

RED_FLAGS = (
    Quote(
        "you're struggling to breathe – you're choking, gasping and unable to speak",
        NHS_EMERGENCY,
    ),
    Quote("you have pale, blue or blotchy skin, lips or tongue", NHS_EMERGENCY),
    Quote("you're coughing up blood", NHS_EMERGENCY),
    Quote(
        "you suddenly feel confused – for example, you do not know where you are",
        NHS_EMERGENCY,
    ),
    Quote("you cannot wake your baby or they feel floppy", NHS_EMERGENCY),
    Quote(
        "Do not drive to A&E. Ask someone to drive you or call 999 and ask for an ambulance.",
        NHS_EMERGENCY,
    ),
)


def _urgent_actions(*items: str) -> tuple[Quote, ...]:
    """Urgent-care advice from the NHS page, quoted one bullet at a time."""
    return tuple(Quote(item, NHS_URGENT) for item in items)


_ADVICE: Mapping[str, Advice] = {
    "likely_pneumonia": Advice(
        mode="likely_pneumonia",
        label="Likely pneumonia",
        headline="This scan looks like pneumonia to the model.",
        summary=(
            "The pneumonia score is above the high-confidence threshold. Treat "
            "it as a reason to seek care today, not as a diagnosis: the model "
            "was trained on a small public dataset and has never been checked "
            "against radiologist reports."
        ),
        quotes=(
            Quote("The symptoms of pneumonia can range from mild to severe.", MEDLINEPLUS_ATYPICAL),
            Quote(
                "You may need to be treated in a hospital if your symptoms are severe or if you are at risk for complications.",
                NOT_DEFINITIVE,
            ),
        ),
        actions=(
            Quote(
                "Urgent advice — ask for an urgent GP appointment or get help from NHS 111 if:",
                NHS_URGENT,
            ),
            *_urgent_actions(
                "you're coughing up yellow or green mucus (phlegm)",
                "you're feeling short of breath",
                "You can call 111 or get help from 111 online.",
            ),
        ),
        red_flags=RED_FLAGS,
        citations=(NHS_EMERGENCY, NHS_URGENT, MEDLINEPLUS_ATYPICAL, NOT_DEFINITIVE),
        sources=(NHS_URL, MEDLINEPLUS_URL),
    ),
    "uncertain": Advice(
        mode="uncertain",
        label="Not conclusive",
        headline="The model cannot call this scan either way.",
        summary=(
            "The score sits between the two thresholds, so it should carry no "
            "weight in either direction. Uncertainty is a normal output of this "
            "model, not a milder version of a positive result."
        ),
        quotes=(
            Quote("Sometimes pneumonia can be hard to diagnose.", MEDLINEPLUS_DIAGNOSIS),
            Quote(
                "it can cause some of the same symptoms as a cold or the flu",
                MEDLINEPLUS_DIAGNOSIS,
            ),
            Quote("The symptoms can vary for different groups", NOT_DEFINITIVE),
        ),
        actions=(
            Quote(
                "Urgent advice — ask for an urgent GP appointment or get help from NHS 111 if:",
                NHS_URGENT,
            ),
            *_urgent_actions(
                "you have chest pain that comes and goes, or happens when breathing or coughing",
                "you're feeling short of breath",
                "You can call 111 or get help from 111 online.",
            ),
        ),
        red_flags=RED_FLAGS,
        citations=(NHS_EMERGENCY, NHS_URGENT, MEDLINEPLUS_DIAGNOSIS, NOT_DEFINITIVE),
        sources=(NHS_URL, MEDLINEPLUS_URL),
    ),
    "likely_normal": Advice(
        mode="likely_normal",
        label="No pneumonia pattern detected",
        headline="The model did not find a pneumonia pattern in this scan.",
        summary=(
            "A negative score is not a clean bill of health. It means the model "
            "did not see what it was trained to look for, and a normal-looking "
            "X-ray does not by itself rule pneumonia out."
        ),
        quotes=(
            Quote("Pneumonia is not confirmed or excluded by a single test.", NOT_DEFINITIVE),
            Quote(
                "Newborns and infants may not show any signs of the infection",
                NOT_DEFINITIVE,
            ),
            Quote("may have fewer and milder symptoms", MEDLINEPLUS_ATYPICAL),
        ),
        actions=(
            Quote(
                "Urgent advice — ask for an urgent GP appointment or get help from NHS 111 if:",
                NHS_URGENT,
            ),
            *_urgent_actions(
                "you've had a cough for up to 3 weeks",
                "you're coughing up yellow or green mucus (phlegm)",
                "You can call 111 or get help from 111 online.",
            ),
        ),
        red_flags=RED_FLAGS,
        citations=(NHS_EMERGENCY, NHS_URGENT, MEDLINEPLUS_ATYPICAL, NOT_DEFINITIVE),
        sources=(NHS_URL, MEDLINEPLUS_URL),
    ),
}

MODES = tuple(_ADVICE)


def pneumonia_probability(prediction: Any) -> float:
    """Read the pneumonia probability out of a ``/predict`` result.

    Accepts the list the API returns, a single result dict, or a bare float.
    """
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        probability = float(prediction)
    else:
        if isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes)):
            if not prediction:
                raise ValueError("empty prediction")
            prediction = prediction[0]
        if not isinstance(prediction, Mapping):
            raise ValueError("prediction must be a dict, a list of dicts or a float")

        probabilities = prediction.get("probabilities")
        if isinstance(probabilities, Mapping) and "Pneumonia" in probabilities:
            probability = float(probabilities["Pneumonia"])
        elif "confidence" in prediction and prediction.get("image") in {
            "Pneumonia",
            "Normal",
        }:
            confidence = float(prediction["confidence"])
            probability = confidence if prediction["image"] == "Pneumonia" else 1 - confidence
        else:
            raise ValueError("prediction has no pneumonia probability")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability out of range: {probability}")
    return probability


def mode_for(
    probability: float,
    *,
    high: float = HIGH_CONFIDENCE,
    low: float = LOW_CONFIDENCE,
) -> str:
    """Map a pneumonia probability to one of the three fixed modes."""
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
    if probability >= high:
        return "likely_pneumonia"
    if probability >= low:
        return "uncertain"
    return "likely_normal"


def build_advice(
    prediction: Any,
    *,
    high: float = HIGH_CONFIDENCE,
    low: float = LOW_CONFIDENCE,
) -> Advice:
    """Deterministic advice for a prediction. Works without any network access."""
    probability = pneumonia_probability(prediction)
    return _ADVICE[mode_for(probability, high=high, low=low)]


def fallback_notice() -> str:
    """Fixed text for when the optional LLM explanation is unavailable."""
    return FALLBACK_TEXT


def cited_files(advice: Advice) -> list[str]:
    """Knowledge files a rendered advice card depends on."""
    return sorted(set(advice.citations))


__all__ = [
    "Advice",
    "DISCLAIMER",
    "FALLBACK_TEXT",
    "HIGH_CONFIDENCE",
    "LOW_CONFIDENCE",
    "MODES",
    "Quote",
    "build_advice",
    "cited_files",
    "fallback_notice",
    "mode_for",
    "pneumonia_probability",
]
