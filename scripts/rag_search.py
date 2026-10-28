"""Query the sourced knowledge base from the command line.

    python scripts/rag_search.py "when should I go to hospital"
    python scripts/rag_search.py -k 5 --json "cough medicine"

Prints the ranked passages with their score and source URL so retrieval can be
inspected without starting Flask or loading TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The knowledge base uses typographic dashes and quotes; a Windows console
# defaults to a legacy code page and would print them as "?" or mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cnnclassifier.rag.retriever import get_retriever  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="+", help="question or keywords")
    parser.add_argument("-k", type=int, default=3, help="number of passages")
    parser.add_argument(
        "--min-score", type=float, default=0.05, help="similarity floor (0-1)"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    hits = get_retriever().search(query, k=args.k, min_score=args.min_score)

    if args.json:
        print(json.dumps([hit.as_dict() for hit in hits], indent=2, ensure_ascii=False))
        return 0 if hits else 1

    if not hits:
        print(f"no passage scored above {args.min_score} for: {query!r}")
        return 1

    for hit in hits:
        sources = ", ".join(hit.chunk.sources)
        print(f"{hit.score:.3f}  {hit.chunk.file}  <- {sources}")
        print(textwrap.indent(hit.chunk.text, "       "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
