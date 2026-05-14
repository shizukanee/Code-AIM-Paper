#!/usr/bin/env python3
"""Convert benchmark_mix JSONL into a response-marker-consistent SFT text dataset.

Problem this fixes:
- The existing benchmark_mix builder emits summarization/translation rows with
  "### Response:" but QA rows end with "Answer: <gold>" and *do not* contain
  the response marker. Response-only SFT collators that split on "### Response:"
  then drop QA rows or create empty-loss batches -> NaNs.

This script rewrites QA rows into the same response-marker format while leaving
other rows untouched.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Dict, Iterable, Iterator, Tuple


QA_ANSWER_RE = re.compile(r"\n\nAnswer:\s*(?P<answer>.*)\s*\Z", re.DOTALL)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    default_in_dir = root / "data" / "vi_ft" / "benchmark_mix"
    default_out_dir = root / "data" / "vi_ft" / "benchmark_mix_sftro"

    parser = argparse.ArgumentParser(description="Convert benchmark_mix into response-only-SFT-friendly text JSONL")
    parser.add_argument("--input-dir", default=str(default_in_dir))
    parser.add_argument("--output-dir", default=str(default_out_dir))
    parser.add_argument("--response-marker", default="### Response:")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def convert_row(row: Dict[str, Any], response_marker: str) -> Tuple[Dict[str, Any], bool]:
    text = row.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    # Already in expected format.
    if response_marker in text:
        return row, False

    task = str(row.get("task", "")).lower()
    if task != "question_answering":
        # Non-QA rows without marker are suspicious; keep but mark.
        out = dict(row)
        out["sftro_warning"] = "missing_response_marker_non_qa"
        return out, False

    m = QA_ANSWER_RE.search(text)
    if not m:
        out = dict(row)
        out["sftro_warning"] = "qa_missing_expected_answer_suffix"
        return out, False

    answer = m.group("answer").strip()
    prompt = text[: m.start()].rstrip()

    # Keep the original "Answer:" label for semantics, but ensure the supervised
    # region starts at response_marker.
    prompt = prompt + "\n\nAnswer:\n" + response_marker + "\n"
    new_text = prompt + answer

    out = dict(row)
    out["text"] = new_text
    out["sftro_converted"] = True
    out["sftro_response_marker"] = response_marker
    return out, True


def main() -> None:
    args = parse_args()
    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    response_marker = args.response_marker

    in_train = in_dir / "merged_train.jsonl"
    in_test = in_dir / "merged_test.jsonl"
    if not in_train.exists():
        raise FileNotFoundError(f"Missing: {in_train}")
    if not in_test.exists():
        raise FileNotFoundError(f"Missing: {in_test}")

    def convert_file(src: pathlib.Path, dst: pathlib.Path) -> Tuple[int, int, int]:
        converted = 0
        total = 0
        out_rows = []
        for row in read_jsonl(src):
            total += 1
            new_row, did = convert_row(row, response_marker)
            if did:
                converted += 1
            out_rows.append(new_row)
        if not args.dry_run:
            wrote = write_jsonl(dst, out_rows)
        else:
            wrote = total
        return total, converted, wrote

    t_total, t_conv, _ = convert_file(in_train, out_dir / "merged_train.jsonl")
    e_total, e_conv, _ = convert_file(in_test, out_dir / "merged_test.jsonl")

    report = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "response_marker": response_marker,
        "train_total": t_total,
        "train_converted": t_conv,
        "test_total": e_total,
        "test_converted": e_conv,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        (out_dir / "convert_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

