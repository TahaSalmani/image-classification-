"""Glue between a model score, the knowledge base and the optional LLM.

Kept free of Flask and TensorFlow so the whole request path can be tested
without GPU, network or API key. ``app.py`` only translates HTTP to
:func:`handle_advice_request` and back.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

from . import llm
from .advice_logic import build_advice, pneumonia_probability
from .rag.retriever import Hit, get_retriever

logger = logging.getLogger(__name__)

#: What to retrieve for the explanation of each mode. These are keyword queries
#: for TF-IDF, not user questions, so they are deliberately blunt.
MODE_QUERIES: Mapping[str, str] = {
    "likely_pneumonia": (
        "urgent advice cough short of breath emergency warning signs "
        "choking gasping unable to speak hospital"
    ),
    "uncertain": "pneumonia hard to diagnose tests chest xray blood tests",
    "likely_normal": "pneumonia missed normal xray mild symptoms older adults infants",
}

TOP_K = 3


def passages_for(mode: str) -> list[Hit]:
    """Retrieve support passages for a mode; never raises."""
    query = MODE_QUERIES.get(mode)
    if not query:
        return []
    try:
        return get_retriever().search(query, k=TOP_K, min_score=0.0)
    except Exception as error:  # missing knowledge dir, broken install
        logger.warning("retrieval unavailable: %s", type(error).__name__)
        return []


def advice_for(
    prediction: Any,
    *,
    explain: bool = True,
    include_passages: bool = True,
) -> dict:
    """Full advice payload for a prediction.

    The deterministic card is always returned. The explanation field becomes LLM
    text only when the model layer is configured and its output validates;
    otherwise it carries the fixed fallback sentence.
    """
    probability = pneumonia_probability(prediction)
    card = build_advice(prediction)
    hits = passages_for(card.mode)

    explanation = None
    details = None
    if explain and hits and llm.available():
        generated = llm.generate_advice(card.mode, probability, hits)
        if generated:
            explanation = generated["summary"]
            details = {
                "what_to_do": generated.get("what_to_do", []),
                "warning_signs": generated.get("warning_signs", []),
                "source_urls": generated.get("source_urls", []),
            }

    payload = {
        "pneumonia_probability": round(probability, 6),
        **card.as_dict(explanation=explanation),
    }
    if details:
        payload["explanation_details"] = details
    if include_passages:
        payload["passages"] = [hit.as_dict() for hit in hits]
    return payload


def handle_advice_request(body: Any) -> tuple[dict, int]:
    """Validate a ``POST /advice`` body and build the response.

    Returns ``(payload, status_code)`` so the route stays three lines and the
    validation is unit-tested without Flask.
    """
    if not isinstance(body, Mapping):
        return {"error": "request body must be a JSON object"}, 400
    if "pneumonia_probability" not in body:
        return {
            "error": "request body must contain a 'pneumonia_probability' field"
        }, 400

    value = body["pneumonia_probability"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {"error": "'pneumonia_probability' must be a number in [0, 1]"}, 400

    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return {"error": "'pneumonia_probability' must be a number in [0, 1]"}, 400

    try:
        return advice_for(probability), 200
    except ValueError as error:
        return {"error": str(error)}, 400
    except Exception as error:  # never leak a traceback to the browser
        logger.exception("advice failed: %s", error)
        return {"error": "advice is temporarily unavailable"}, 500


__all__ = ["MODE_QUERIES", "TOP_K", "advice_for", "handle_advice_request", "passages_for"]
