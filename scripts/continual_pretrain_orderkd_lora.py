#!/usr/bin/env python3
"""Convenience launcher for VI Wikipedia continual pretraining with OrderKD + LoRA.

This wraps `train_lm_orderkd.py` and supplies defaults for the
`viwiki_clean_train.jsonl` corpus split plus the prebuilt `vi_order_kd.jsonl`
auxiliary supervision file.

Extra arguments are forwarded to `train_lm_orderkd.py`.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "train_lm_orderkd.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    default_train = ROOT / "data" / "vi_corpus" / "viwiki_clean" / "viwiki_clean_train.jsonl"
    default_order_kd = ROOT / "outputs" / "order_kd" / "vi_order_kd.jsonl"
    default_en_buffer = ROOT / "data" / "en_buffer" / "en_buffer.jsonl"

    parser = argparse.ArgumentParser(
        description="Launch OrderKD + LoRA continual pretraining on the VI Wikipedia corpus."
    )
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default=str(default_train))
    parser.add_argument("--order-kd-jsonl", default=str(default_order_kd))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--en-buffer-file", default=str(default_en_buffer) if default_en_buffer.exists() else None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "checkpoints" / "viwiki_cpt_orderkd_lora"),
    )
    parser.add_argument("--cache-dir", default=None)
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    if not pathlib.Path(args.train_file).exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not pathlib.Path(args.order_kd_jsonl).exists():
        raise FileNotFoundError(f"Order-KD file not found: {args.order_kd_jsonl}")
    if args.en_buffer_file and not pathlib.Path(args.en_buffer_file).exists():
        raise FileNotFoundError(f"EN buffer file not found: {args.en_buffer_file}")

    forwarded_argv = [
        str(BASE_SCRIPT),
        "--model-name-or-path",
        args.model_name_or_path,
        "--vi-train-file",
        args.train_file,
        "--order-kd-jsonl",
        args.order_kd_jsonl,
        "--text-column",
        args.text_column,
        "--output-dir",
        args.output_dir,
        "--use-lora",
        "--lazy-tokenize",
    ]
    if args.en_buffer_file:
        forwarded_argv.extend(["--en-buffer-file", args.en_buffer_file])
    if args.cache_dir:
        forwarded_argv.extend(["--cache-dir", args.cache_dir])
    forwarded_argv.extend(passthrough)

    print(f"[continual_pretrain_orderkd_lora] train={args.train_file}")
    print(f"[continual_pretrain_orderkd_lora] order_kd={args.order_kd_jsonl}")
    if args.en_buffer_file:
        print(f"[continual_pretrain_orderkd_lora] en_buffer={args.en_buffer_file}")
    print(f"[continual_pretrain_orderkd_lora] output={args.output_dir}")

    old_argv = sys.argv
    try:
        sys.argv = forwarded_argv
        runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
