"""Preview the three fixed advice cards without starting Flask or TensorFlow.

    python scripts/advice_preview.py                 # all three modes
    python scripts/advice_preview.py -p 0.84         # one probability
    python scripts/advice_preview.py -p 0.2 --json   # machine-readable

Handy for checking wording and citations after editing advice_logic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cnnclassifier import advice_logic  # noqa: E402

SAMPLES = (("likely_pneumonia", 0.93), ("uncertain", 0.55), ("likely_normal", 0.08))


def render(payload: dict) -> None:
    print(f"[{payload['mode']}] {payload['label']}")
    print(f"  {payload['headline']}")
    print(textwrap.fill(payload["summary"], 76, initial_indent="  ", subsequent_indent="  "))
    print("  quoted:")
    for quote in payload["quotes"]:
        print(textwrap.fill(f'"{quote["text"]}" — {quote["file"]}', 76, initial_indent="    ", subsequent_indent="    "))
    print("  do:")
    for action in payload["actions"]:
        print(textwrap.fill(f"- {action['text']} ({action['file']})", 76, initial_indent="    ", subsequent_indent="    "))
    print("  emergency (always shown):")
    for flag in payload["red_flags"]:
        print(textwrap.fill(f"- {flag['text']}", 76, initial_indent="    ", subsequent_indent="    "))
    print(f"  explanation ({payload['explanation_source']}): {payload['explanation']}")
    print(f"  sources: {', '.join(payload['sources'])}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--probability", type=float, help="pneumonia probability")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    cards = (
        [advice_logic.build_advice(args.probability).as_dict()]
        if args.probability is not None
        else [advice_logic.build_advice(p).as_dict() for _, p in SAMPLES]
    )

    if args.json:
        print(json.dumps(cards, indent=2, ensure_ascii=False))
        return 0

    for card in cards:
        render(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
