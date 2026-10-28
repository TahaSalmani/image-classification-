"""Checks for the RAG knowledge base, chunker and TF-IDF retriever.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnnclassifier.rag import chunker, retriever  # noqa: E402
from cnnclassifier.rag.chunker import load_knowledge_base, parse_file  # noqa: E402
from cnnclassifier.rag.retriever import KnowledgeRetriever, normalize  # noqa: E402

KNOWLEDGE = REPO_ROOT / "knowledge"

#: (question, files that legitimately answer it). Checked against the top two
#: hits so a near-miss still counts; the point is to catch regressions, not to
#: pretend TF-IDF is a reranker. Several questions have more than one honest
#: answer in this corpus, hence the tuples.
RETRIEVAL_CASES = [
    ("coughing up blood", ("02-emergency-warning-signs-nhs.md",)),
    ("you cannot wake your baby", ("02-emergency-warning-signs-nhs.md",)),
    ("when to call 999", ("02-emergency-warning-signs-nhs.md",)),
    ("cough for three weeks", ("03-when-to-get-urgent-advice-nhs.md",)),
    ("can I take cough medicine", ("04-self-care-at-home-nhs.md",)),
    ("how long until I feel better", ("04-self-care-at-home-nhs.md",)),
    ("do I need antibiotics", ("05-treatment-and-hospital-care-nhs.md",)),
    ("when should I go to hospital", ("05-treatment-and-hospital-care-nhs.md",)),
    ("what causes pneumonia", ("06-causes-nhs.md", "09-symptoms-cdc.md")),
    ("which vaccines prevent pneumonia", ("07-prevention-cdc.md",)),
    ("who is at higher risk", ("08-risk-factors-cdc.md",)),
    ("symptoms of pneumonia", ("01-symptoms-nhs.md", "09-symptoms-cdc.md")),
    (
        "symptoms in older adults",
        ("10-atypical-symptoms-infants-older-adults-medlineplus.md",),
    ),
    ("how is pneumonia diagnosed", ("11-diagnosis-medlineplus.md",)),
    (
        "does a normal chest x-ray rule out pneumonia",
        ("12-normal-scan-not-definitive.md",),
    ),
]


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = sorted(KNOWLEDGE.glob("*.md"))
        cls.chunks = load_knowledge_base(KNOWLEDGE)

    def test_knowledge_base_size(self) -> None:
        self.assertGreaterEqual(len(self.files), 8)
        self.assertLessEqual(len(self.files), 15)

    def test_every_file_starts_with_source(self) -> None:
        for path in self.files:
            with self.subTest(file=path.name):
                first = path.read_text(encoding="utf-8").strip().splitlines()[0]
                self.assertTrue(
                    first.upper().startswith("SOURCE:"),
                    f"{path.name} starts with {first!r}",
                )

    def test_every_file_has_body(self) -> None:
        for path in self.files:
            with self.subTest(file=path.name):
                self.assertGreater(
                    path.stat().st_size, 0, f"{path.name} is empty"
                )

    def test_sources_are_urls(self) -> None:
        for chunk in self.chunks:
            with self.subTest(chunk=chunk.chunk_id):
                self.assertTrue(chunk.sources)
                for url in chunk.sources:
                    self.assertTrue(url.startswith("https://"), url)

    def test_chunks_stay_short(self) -> None:
        for chunk in self.chunks:
            with self.subTest(chunk=chunk.chunk_id):
                # One dense list block may overflow the target slightly; a
                # chunk double the target means the packer is broken.
                self.assertLessEqual(len(chunk.text), 2 * chunker.MAX_CHUNK_CHARS)

    def test_attribution_is_metadata_not_body(self) -> None:
        for chunk in self.chunks:
            with self.subTest(chunk=chunk.chunk_id):
                self.assertNotIn("Open Government Licence", chunk.text)
                self.assertNotIn("public domain", chunk.text)

    def test_missing_source_line_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.md"
            bad.write_text("Pneumonia is a lung infection.\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_file(bad)


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = KnowledgeRetriever(load_knowledge_base(KNOWLEDGE))

    def test_hits_are_ranked_and_scored(self) -> None:
        hits = self.retriever.search("coughing up blood", k=3)
        self.assertTrue(hits)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_min_score_filters(self) -> None:
        hits = self.retriever.search("coughing up blood", k=3, min_score=0.9)
        self.assertEqual(hits, [])

    def test_empty_query_returns_nothing(self) -> None:
        for query in ("", "   ", None):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query), [])

    def test_retrieval_cases(self) -> None:
        for question, expected in RETRIEVAL_CASES:
            with self.subTest(question=question):
                top = [hit.chunk.file for hit in self.retriever.search(question, k=2)]
                self.assertTrue(
                    set(top) & set(expected),
                    f"{question!r} returned {top}, expected one of {expected}",
                )

    def test_xray_spellings_collapse(self) -> None:
        self.assertEqual(normalize("Chest X-Ray"), "chest xray")
        self.assertEqual(normalize("chest x ray"), "chest xray")
        hits = self.retriever.search("is a normal xray result definitive", k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.file, "12-normal-scan-not-definitive.md")

    def test_retriever_needs_passages(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeRetriever([])

    def test_singleton_is_cached(self) -> None:
        retriever.reset_retriever()
        first = retriever.get_retriever(KNOWLEDGE)
        second = retriever.get_retriever(KNOWLEDGE)
        self.assertIs(first, second)
        retriever.reset_retriever()


if __name__ == "__main__":
    unittest.main()
