"""Tests for the optional LLM layer, with a fake client and no network.

The point of these tests is the failure paths: a missing SDK, a missing key, a
timeout, malformed JSON or an out-of-range citation must all produce ``None``
instead of an exception, so the page can fall back to its fixed text.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnnclassifier import llm  # noqa: E402

PASSAGES = [
    {
        "text": "Immediate action required — call 999 or go to A&E if: you're coughing up blood",
        "sources": ["https://www.nhs.uk/conditions/pneumonia/"],
        "file": "02-emergency-warning-signs-nhs.md",
    },
    {
        "text": "You may need to be treated in a hospital if your symptoms are severe.",
        "sources": ["https://medlineplus.gov/pneumonia.html"],
        "file": "12-normal-scan-not-definitive.md",
    },
]

GOOD = {
    "summary": "The model flagged this image. Only a clinician can confirm pneumonia.",
    "what_to_do": ["Speak to a clinician today."],
    "warning_signs": ["Coughing up blood"],
    "sources": [1, 2],
}


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, outer: "FakeClient") -> None:
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls += 1
        self.outer.last_kwargs = kwargs
        if self.outer.error is not None:
            raise self.outer.error
        return _Response(self.outer.text)


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text if text is not None else json.dumps(GOOD)
        self.error = error
        self.calls = 0
        self.last_kwargs: dict = {}
        self.messages = _Messages(self)


class GenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        llm.clear_cache()

    def test_returns_parsed_advice(self) -> None:
        client = FakeClient()
        advice = llm.generate_advice("likely_pneumonia", 0.93, PASSAGES, client=client)
        self.assertIsNotNone(advice)
        self.assertEqual(advice["summary"], GOOD["summary"])
        self.assertEqual(advice["what_to_do"], GOOD["what_to_do"])
        self.assertEqual(advice["sources"], [1, 2])
        self.assertEqual(
            advice["source_urls"],
            ["https://www.nhs.uk/conditions/pneumonia/", "https://medlineplus.gov/pneumonia.html"],
        )
        self.assertEqual(client.calls, 1)

    def test_prompt_carries_context_and_rules(self) -> None:
        client = FakeClient()
        llm.generate_advice("uncertain", 0.55, PASSAGES, client=client)
        kwargs = client.last_kwargs
        self.assertEqual(kwargs["model"], llm.MODEL)
        self.assertIn("Use ONLY the context passages", kwargs["system"])
        user = kwargs["messages"][0]["content"]
        self.assertIn("[1]", user)
        self.assertIn("coughing up blood", user)
        self.assertIn("55%", user)

    def test_markdown_fence_is_tolerated(self) -> None:
        fenced = "```json\n" + json.dumps(GOOD) + "\n```"
        advice = llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(fenced))
        self.assertIsNotNone(advice)

    def test_cache_prevents_second_call(self) -> None:
        client = FakeClient()
        first = llm.generate_advice("uncertain", 0.5, PASSAGES, client=client)
        second = llm.generate_advice("uncertain", 0.62, PASSAGES, client=client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)

    def test_cache_can_be_bypassed(self) -> None:
        client = FakeClient()
        llm.generate_advice("uncertain", 0.5, PASSAGES, client=client)
        llm.generate_advice("uncertain", 0.5, PASSAGES, client=client, use_cache=False)
        self.assertEqual(client.calls, 2)

    def test_html_is_stripped(self) -> None:
        payload = dict(GOOD, summary="<script>alert(1)</script>Plain text")
        advice = llm.generate_advice(
            "uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload))
        )
        self.assertEqual(advice["summary"], "alert(1) Plain text")

    def test_over_long_summary_is_rejected(self) -> None:
        payload = dict(GOOD, summary="word " * 400)
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload)))
        )

    def test_non_string_items_are_dropped(self) -> None:
        payload = dict(GOOD, what_to_do=[1, None, "Real item", {"a": 1}])
        advice = llm.generate_advice(
            "uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload))
        )
        self.assertEqual(advice["what_to_do"], ["Real item"])


class FailurePathTests(unittest.TestCase):
    def setUp(self) -> None:
        llm.clear_cache()

    def test_bad_json_returns_none(self) -> None:
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient("not json"))
        )

    def test_empty_response_returns_none(self) -> None:
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(""))
        )

    def test_out_of_range_sources_are_rejected(self) -> None:
        payload = dict(GOOD, sources=[7])
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload)))
        )

    def test_missing_sources_are_rejected(self) -> None:
        payload = {key: value for key, value in GOOD.items() if key != "sources"}
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload)))
        )

    def test_string_sources_are_rejected(self) -> None:
        payload = dict(GOOD, sources=["1", "2"])
        self.assertIsNone(
            llm.generate_advice("uncertain", 0.5, PASSAGES, client=FakeClient(json.dumps(payload)))
        )

    def test_api_error_returns_none(self) -> None:
        client = FakeClient(error=TimeoutError("timed out"))
        self.assertIsNone(llm.generate_advice("uncertain", 0.5, PASSAGES, client=client))

    def test_no_passages_returns_none(self) -> None:
        self.assertIsNone(llm.generate_advice("uncertain", 0.5, [], client=FakeClient()))

    def test_missing_sdk_and_key_returns_none(self) -> None:
        # Neither the SDK nor the key exists in this environment: the call must
        # fail closed without importing anything or raising.
        llm.reset_client()
        self.assertIsNone(llm.generate_advice("likely_normal", 0.05, PASSAGES))

    def test_available_reflects_configuration(self) -> None:
        if llm.anthropic is None:
            self.assertFalse(llm.available())
        else:  # pragma: no cover - only when the serving extras are installed
            self.assertFalse(llm.available())  # no key in the test environment


class ContextTests(unittest.TestCase):
    def test_numbering_skips_empty_passages(self) -> None:
        context, sources = llm.build_context(
            [{"text": "", "sources": ["https://a"]}, PASSAGES[0], PASSAGES[1]]
        )
        self.assertIn("[1]", context)
        self.assertIn("[2]", context)
        self.assertNotIn("[3]", context)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0], "https://www.nhs.uk/conditions/pneumonia/")

    def test_chunk_objects_are_accepted(self) -> None:
        from cnnclassifier.rag.chunker import load_knowledge_base

        chunks = load_knowledge_base(REPO_ROOT / "knowledge")[:3]
        context, sources = llm.build_context(chunks)
        self.assertEqual(len(sources), 3)
        self.assertIn("https://", sources[0])
        self.assertIn("[3]", context)


if __name__ == "__main__":
    unittest.main()
