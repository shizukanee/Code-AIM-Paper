#!/usr/bin/env python3
"""Convenience launcher for VI Wikipedia continual pretraining with LoRA.

This wraps `finetune_vi_baseline.py` and supplies defaults for the
`viwiki_clean_train.jsonl` / `viwiki_clean_valid.jsonl` corpus split.

Extra arguments are forwarded to `finetune_vi_baseline.py`.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "finetune_vi_baseline.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    default_train = ROOT / "data" / "vi_corpus" / "viwiki_clean" / "viwiki_clean_train.jsonl"
    default_eval = ROOT / "data" / "vi_corpus" / "viwiki_clean" / "viwiki_clean_valid.jsonl"
    default_en_buffer = ROOT / "data" / "en_buffer" / "en_buffer.jsonl"

    parser = argparse.ArgumentParser(
        description="Launch plain LoRA continual pretraining on the VI Wikipedia corpus."
    )
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default=str(default_train))
    parser.add_argument("--eval-file", default=str(default_eval) if default_eval.exists() else None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--en-buffer-file", default=str(default_en_buffer) if default_en_buffer.exists() else None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "checkpoints" / "viwiki_cpt_lora"),
    )
    parser.add_argument("--cache-dir", default=None)
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    if not pathlib.Path(args.train_file).exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if args.eval_file and not pathlib.Path(args.eval_file).exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")
    if args.en_buffer_file and not pathlib.Path(args.en_buffer_file).exists():
        raise FileNotFoundError(f"EN buffer file not found: {args.en_buffer_file}")

    forwarded_argv = [
        str(BASE_SCRIPT),
        "--model-name-or-path",
        args.model_name_or_path,
        "--train-file",
        args.train_file,
        "--text-column",
        args.text_column,
        "--output-dir",
        args.output_dir,
        "--use-lora",
        "--lazy-tokenize",
    ]
    if args.eval_file:
        forwarded_argv.extend(["--eval-file", args.eval_file])
    if args.en_buffer_file:
        forwarded_argv.extend(["--en-buffer-file", args.en_buffer_file])
    if args.cache_dir:
        forwarded_argv.extend(["--cache-dir", args.cache_dir])
    forwarded_argv.extend(passthrough)

    print(f"[continual_pretrain_vi_lora] train={args.train_file}")
    if args.eval_file:
        print(f"[continual_pretrain_vi_lora] eval={args.eval_file}")
    if args.en_buffer_file:
        print(f"[continual_pretrain_vi_lora] en_buffer={args.en_buffer_file}")
    print(f"[continual_pretrain_vi_lora] output={args.output_dir}")

    old_argv = sys.argv
    try:
        sys.argv = forwarded_argv
        runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
