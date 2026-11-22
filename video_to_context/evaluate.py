"""Score the pipeline on the hand-labelled test set.

Each `tests/expected/<name>.json` describes `tests/samples/<name>.*`:
    {"transcript": "...", "on_screen": ["..."], "no_speech": false, "rejected": null}
"""

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import jiwer
from rapidfuzz import fuzz

from .models import PipelineError
from .run import process

ROOT = Path("tests")


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w€%+]+", " ", text.casefold()).split())


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    reference = normalise(reference)
    if not reference:
        return None
    return float(jiwer.wer(reference, normalise(hypothesis) or " "))


def caption_recall(expected: list[str], found: list[str], threshold: float = 85) -> float | None:
    if not expected:
        return None
    found = [normalise(text) for text in found]
    hits = sum(
        any(fuzz.partial_ratio(normalise(caption), text) >= threshold for text in found)
        for caption in expected
    )
    return hits / len(expected)


def evaluate(samples: Path, expected: Path) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as out:
        for spec_path in sorted(expected.glob("*.json")):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            matches = [p for p in samples.glob(f"{spec_path.stem}.*") if p.is_file()]
            row = {"name": spec_path.stem}
            if not matches:
                rows.append({**row, "result": "missing sample"})
                continue
            started = time.perf_counter()
            try:
                record = process(file=matches[0], out=out, force=True)
            except PipelineError as exc:
                wanted = spec.get("rejected")
                result = "rejected ok" if exc.code == wanted else f"error {exc.code}"
                rows.append({**row, "result": result})
                continue
            transcript = " ".join(s.text for s in record.transcript.segments)
            row.update(
                result="unexpected pass" if spec.get("rejected") else record.pipeline.status,
                seconds=round(time.perf_counter() - started, 1),
                wer=word_error_rate(spec.get("transcript", ""), transcript),
                caption_recall=caption_recall(
                    spec.get("on_screen", []), [t.text for t in record.on_screen_text]
                ),
            )
            if spec.get("no_speech"):
                row["invented_words"] = len(normalise(transcript).split())
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--expected", type=Path, default=ROOT / "expected")
    args = parser.parse_args()
    rows = evaluate(args.samples, args.expected)
    print(f"{'sample':28}{'result':18}{'time':>8}{'WER':>8}{'captions':>10}{'invented':>10}")
    for r in rows:
        fmt = lambda key, pattern: pattern.format(r[key]) if r.get(key) is not None else "-"
        print(f"{r['name']:28}{r['result']:18}{fmt('seconds', '{}s'):>8}"
              f"{fmt('wer', '{:.1%}'):>8}{fmt('caption_recall', '{:.0%}'):>10}"
              f"{fmt('invented_words', '{}'):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
