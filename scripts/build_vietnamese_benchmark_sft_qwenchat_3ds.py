#!/usr/bin/env python3
"""Build a Qwen-chat-template-ready SFT dataset (messages JSONL) for 3 datasets.

This is a thin wrapper around `build_vietnamese_benchmark_sft.py` that:
1) selects exactly: xquad, vietnews, opus100_envi
2) writes to a new output folder (so it doesn't touch old data)

The output JSONL rows contain a `messages` field:
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]

Training scripts should call `tokenizer.apply_chat_template(messages, ...)`
to render Qwen's native chat format.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import runpy
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Vietnamese benchmark SFT (3 datasets) as messages JSONL")
    p.add_argument(
        "--benchmark-root",
        default=str(ROOT.parent / "Vietnamese benchmark"),
        help="Path to the Vietnamese benchmark repo folder.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "vietnamese_benchmark_sft_qwenchat_3ds" / "training"),
        help="Where to write *_train_sft.jsonl, merged_train.jsonl, etc.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--include-test", action="store_true")
    p.add_argument(
        "--force-system",
        default="",
        help="If set, override (or inject) a system message with this content for every sample.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Load the underlying builder script and reuse its `main()` by
    # constructing argv (keeps behavior consistent, avoids duplication).
    builder_path = ROOT / "scripts" / "build_vietnamese_benchmark_sft.py"
    builder_globals = runpy.run_path(str(builder_path))
    builder_main = builder_globals.get("main")
    if not callable(builder_main):
        raise RuntimeError(f"Expected callable main() in {builder_path}")

    argv = [
        "--benchmark-root",
        args.benchmark_root,
        "--output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
        "--only-dataset",
        "xquad",
        "--only-dataset",
        "vietnews",
        "--only-dataset",
        "opus100_envi",
    ]
    if args.no_shuffle:
        argv.append("--no-shuffle")
    if args.include_test:
        argv.append("--include-test")

    old_argv = sys.argv
    try:
        sys.argv = ["build_vietnamese_benchmark_sft_qwenchat_3ds.py", *argv]
        builder_main()
    finally:
        sys.argv = old_argv

    if args.force_system:
        # Post-process: rewrite JSONL in-place to inject/override a system message.
        out_dir = pathlib.Path(args.output_dir)
        jsonl_paths = sorted(out_dir.glob("*.jsonl"))
        for path in jsonl_paths:
            if not path.name.endswith(".jsonl"):
                continue
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msgs = obj.get("messages")
                    if isinstance(msgs, list):
                        # remove existing leading system message
                        if msgs and msgs[0].get("role") == "system":
                            msgs = msgs[1:]
                        msgs = [{"role": "system", "content": args.force_system}, *msgs]
                        obj["messages"] = msgs
                    rows.append(obj)
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for obj in rows:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
