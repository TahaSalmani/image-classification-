"""Optional language-model layer: rewrites the fixed advice card, nothing more.

The card from :mod:`cnnclassifier.advice_logic` is the product. This module only
asks Claude for a short plain-language explanation of the same facts, and it is
built so that its absence is boring:

* the SDK is optional — without ``anthropic`` installed or without
  ``ANTHROPIC_API_KEY`` set, the module still imports and returns ``None``
* the client is created lazily, so a missing key can never break ``app.py``
* every call has a short timeout and no retries, because gunicorn runs a single
  worker on Render and a stalled request would freeze the whole site
* every failure — HTTP, JSON, schema, empty sources — collapses to ``None`` and
  the caller shows the fixed fallback text instead

Nothing generated here is trusted as medical content: the prompt restricts the
model to the supplied passages, and the passages themselves are the sourced
files from ``knowledge/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Iterable, Mapping, Sequence

try:  # the dependency is serving-only; the rest of the app must not require it
    import anthropic
except Exception:  # pragma: no cover - exercised by the "SDK missing" path
    anthropic = None

logger = logging.getLogger(__name__)

MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
TIMEOUT_SECONDS = 10.0
MAX_TOKENS = 600

#: Hard limits that keep a misbehaving completion from reaching the page.
MAX_SUMMARY_CHARS = 800
MAX_ITEM_CHARS = 240
MAX_ITEMS = 5
MAX_WARNING_SIGNS = 6

SYSTEM_PROMPT = """You write short, plain-language health information for a demo website about chest X-rays.

Rules:
- You are NOT a doctor. Never diagnose, and never present the model's output as a diagnosis or as certain.
- Use ONLY the context passages provided. If they do not cover something, say so in "summary" instead of guessing.
- Never invent drug names, doses or treatments. You may repeat self-care advice that appears in the context, quoted as it is written.
- Always tell the reader to speak to a qualified clinician.
- Put any severe warning signs that the context mentions into "warning_signs", unsoftened.
- Keep "summary" under 90 words.
- "sources" must list the passage numbers you actually used, e.g. [1, 3].

Reply with JSON only, no markdown fence:
{"summary": str, "what_to_do": [str], "warning_signs": [str], "sources": [int]}"""

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")

_CLIENT: Any = None
_CLIENT_READY = False
_CACHE: dict[str, dict] = {}


def available() -> bool:
    """True when the SDK is installed and a key is configured."""
    return anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client_or_none() -> Any:
    global _CLIENT, _CLIENT_READY
    if _CLIENT_READY:
        return _CLIENT
    _CLIENT_READY = True
    _CLIENT = None
    if anthropic is None:
        logger.info("anthropic SDK not installed; LLM explanation disabled")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set; LLM explanation disabled")
        return None
    try:
        _CLIENT = anthropic.Anthropic(
            api_key=api_key,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
        )
    except Exception as error:  # bad key format, proxy config, etc.
        logger.warning("could not create LLM client: %s", error)
        _CLIENT = None
    return _CLIENT


def reset_client() -> None:
    """Forget the cached client (tests, or after changing the environment)."""
    global _CLIENT, _CLIENT_READY
    _CLIENT = None
    _CLIENT_READY = False


def clear_cache() -> None:
    _CACHE.clear()


def _clean(value: Any) -> str:
    """Strip HTML and collapse whitespace; the UI may render this as HTML."""
    text = _TAG_RE.sub(" ", str(value))
    return _WS_RE.sub(" ", text).strip()


def _clean_items(values: Any, limit: int) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return []
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = _clean(value)
        if text and len(text) <= MAX_ITEM_CHARS:
            items.append(text)
        if len(items) == limit:
            break
    return items


def _passage_source(passage: Any) -> str:
    if isinstance(passage, Mapping):
        sources = passage.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)) and sources:
            return str(sources[0])
        if passage.get("source"):
            return str(passage["source"])
        return ""
    sources = getattr(passage, "sources", ())
    return str(sources[0]) if sources else ""


def _passage_text(passage: Any) -> str:
    if isinstance(passage, Mapping):
        return str(passage.get("text", ""))
    return str(getattr(passage, "text", ""))


def build_context(passages: Iterable[Any]) -> tuple[str, list[str]]:
    """Numbered context block plus the source URL of each passage.

    Numbering only counts passages that carry text, so passage ``n`` in the
    prompt is always ``sources[n - 1]`` in the returned list.
    """
    lines = []
    sources: list[str] = []
    for passage in passages:
        text = _clean(_passage_text(passage))
        if not text:
            continue
        url = _passage_source(passage)
        sources.append(url)
        index = len(sources)
        label = f" ({url})" if url else ""
        lines.append(f"[{index}]{label}\n{text}")
    return "\n\n".join(lines), sources


def _extract_json(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None


def _validate(payload: Any, source_urls: list[str]) -> dict | None:
    """Reject anything that is not the exact shape the page can render."""
    if not isinstance(payload, Mapping):
        return None

    summary = _clean(payload.get("summary", ""))
    if not summary or len(summary) > MAX_SUMMARY_CHARS:
        return None

    raw_sources = payload.get("sources")
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        return None
    indices: list[int] = []
    for value in raw_sources:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 1 <= value <= len(source_urls) and value not in indices:
            indices.append(value)
    if not indices:
        return None

    return {
        "summary": summary,
        "what_to_do": _clean_items(payload.get("what_to_do"), MAX_ITEMS),
        "warning_signs": _clean_items(payload.get("warning_signs"), MAX_WARNING_SIGNS),
        "sources": indices,
        "source_urls": [source_urls[index - 1] for index in indices if source_urls[index - 1]],
    }


def _response_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def generate_advice(
    mode: str,
    probability: float,
    passages: Iterable[Any],
    *,
    client: Any = None,
    use_cache: bool = True,
) -> dict | None:
    """Return ``{"summary", "what_to_do", "warning_signs", "sources"}`` or ``None``.

    Never raises: a missing key, a timeout, malformed JSON, an out-of-range
    source index or an unusable schema all return ``None`` so the caller can
    show :data:`cnnclassifier.advice_logic.FALLBACK_TEXT`.
    """
    context, source_urls = build_context(passages)
    if not context or not source_urls:
        return None

    cache_key = hashlib.sha1(f"{mode}|{context}".encode("utf-8")).hexdigest()
    if use_cache and cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    client = client if client is not None else _client_or_none()
    if client is None:
        return None

    user_prompt = (
        f"Model result: {mode} (pneumonia probability {probability:.0%}).\n\n"
        f"Context passages:\n{context}\n\nWrite the JSON now."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as error:  # network, auth, rate limit, timeout
        logger.warning("LLM request failed: %s", type(error).__name__)
        return None

    payload = _extract_json(_response_text(response))
    advice = _validate(payload, source_urls)
    if advice is None:
        logger.warning("LLM response rejected by schema validation")
        return None

    if use_cache:
        _CACHE[cache_key] = dict(advice)
    return advice


__all__ = [
    "MAX_TOKENS",
    "MODEL",
    "SYSTEM_PROMPT",
    "TIMEOUT_SECONDS",
    "available",
    "build_context",
    "clear_cache",
    "generate_advice",
    "reset_client",
]
