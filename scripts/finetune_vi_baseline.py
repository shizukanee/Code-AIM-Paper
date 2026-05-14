#!/usr/bin/env python3
"""Baseline Vietnamese adaptation for causal LMs.

This script is intended for remote training machines (GPU).
It supports:
1) Standard Vietnamese-only LM fine-tuning.
2) Optional EN replay baseline by interleaving an English buffer.
"""

from __future__ import annotations

import argparse
import inspect
import math
import pathlib

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def normalize_text_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def resolve_text_column(dataset: datasets.Dataset, preferred: str) -> str:
    if preferred in dataset.column_names:
        return preferred
    for col in dataset.column_names:
        if dataset.features[col].dtype == "string":
            return col
    raise ValueError(f"No text column found. Columns: {dataset.column_names}")


def load_text_split(path: str, text_column: str) -> datasets.Dataset:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in {".jsonl", ".json"}:
        ds = datasets.load_dataset("json", data_files={"train": path})["train"]
    elif suffix in {".txt"}:
        ds = datasets.load_dataset("text", data_files={"train": path})["train"]
        if text_column != "text" and "text" in ds.column_names:
            ds = ds.rename_column("text", text_column)
    else:
        raise ValueError(f"Unsupported file extension for text data: {suffix}")
    return ds


class TextLMCollator:
    def __init__(self, tokenizer, max_length: int, text_column: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts = [normalize_text_value(ex.get(self.text_column, "")) for ex in examples]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def estimate_total_steps(num_examples: int, args: argparse.Namespace) -> int:
    if args.max_steps > 0:
        return args.max_steps

    micro_steps_per_epoch = math.ceil(num_examples / max(1, args.per_device_train_batch_size))
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / max(1, args.gradient_accumulation_steps))
    return max(1, int(math.ceil(args.num_train_epochs * optimizer_steps_per_epoch)))


def subset_dataset(
    dataset: datasets.Dataset,
    *,
    seed: int,
    fraction: float,
    max_samples: int,
    label: str,
) -> datasets.Dataset:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"--{label}-fraction must be in (0, 1].")
    if max_samples < 0:
        raise ValueError(f"--{label}-max-samples must be >= 0.")

    target_size = len(dataset)
    if fraction < 1.0:
        target_size = max(1, int(len(dataset) * fraction))
    if max_samples > 0:
        target_size = min(target_size, max_samples)

    if target_size >= len(dataset):
        return dataset

    print(f"Subsampling {label} dataset: keeping {target_size} / {len(dataset)} examples")
    return dataset.shuffle(seed=seed).select(range(target_size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline VI fine-tuning for causal LM")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True, help="JSONL/JSON/TXT with Vietnamese text")
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--en-buffer-file",
        default=None,
        help="Optional EN replay buffer (JSONL/JSON/TXT) for replay baseline.",
    )
    parser.add_argument(
        "--en-text-column",
        default="text",
        help="Text column for EN replay data (if provided).",
    )
    parser.add_argument(
        "--en-replay-prob",
        type=float,
        default=0.0,
        help="Replay probability for EN samples in interleaving, in [0,1).",
    )
    parser.add_argument(
        "--en-buffer-max-samples",
        type=int,
        default=0,
        help="Optional cap for EN replay samples (0 means no cap).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=1.0)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lazy-tokenize", action="store_true")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report-to", default="none", help="none|wandb|tensorboard")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora-8bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names for LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_qlora_8bit:
        args.use_lora = True

    if not (0.0 <= args.en_replay_prob < 1.0):
        raise ValueError("--en-replay-prob must be in [0, 1).")
    if args.en_replay_prob > 0.0 and not args.en_buffer_file:
        raise ValueError("--en-buffer-file is required when --en-replay-prob > 0.")

    train_ds = load_text_split(args.train_file, args.text_column)
    text_column = resolve_text_column(train_ds, args.text_column)
    if text_column != args.text_column:
        train_ds = train_ds.rename_column(text_column, args.text_column)
        text_column = args.text_column
    train_ds = subset_dataset(
        train_ds,
        seed=args.seed,
        fraction=args.train_fraction,
        max_samples=args.train_max_samples,
        label="train",
    )

    eval_ds = None
    if args.eval_file:
        eval_ds = load_text_split(args.eval_file, text_column)
        eval_col = resolve_text_column(eval_ds, text_column)
        if eval_col != text_column:
            eval_ds = eval_ds.rename_column(eval_col, text_column)
        eval_ds = subset_dataset(
            eval_ds,
            seed=args.seed,
            fraction=args.eval_fraction,
            max_samples=args.eval_max_samples,
            label="eval",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: dict) -> dict:
        texts = [normalize_text_value(v) for v in batch[text_column]]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    train_tok = train_ds
    if not args.lazy_tokenize:
        remove_cols = list(train_ds.column_names)
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    if args.en_buffer_file and args.en_replay_prob > 0.0:
        en_ds = load_text_split(args.en_buffer_file, args.en_text_column)
        en_text_col = resolve_text_column(en_ds, args.en_text_column)
        if en_text_col != args.en_text_column:
            en_ds = en_ds.rename_column(en_text_col, args.en_text_column)
            en_text_col = args.en_text_column

        if en_text_col != text_column:
            if text_column in en_ds.column_names:
                # Avoid accidental column collision when user sets mismatched names.
                en_ds = en_ds.remove_columns([text_column])
            en_ds = en_ds.rename_column(en_text_col, text_column)

        if args.en_buffer_max_samples > 0 and len(en_ds) > args.en_buffer_max_samples:
            en_ds = en_ds.shuffle(seed=args.seed).select(range(args.en_buffer_max_samples))

        en_tok = en_ds
        if not args.lazy_tokenize:
            en_remove_cols = list(en_ds.column_names)
            en_tok = en_ds.map(tokenize_fn, batched=True, remove_columns=en_remove_cols)

        train_tok = datasets.interleave_datasets(
            [train_tok, en_tok],
            probabilities=[1.0 - args.en_replay_prob, args.en_replay_prob],
            seed=args.seed,
            stopping_strategy="all_exhausted",
        )
        print(
            "Using EN replay baseline: "
            f"en_replay_prob={args.en_replay_prob}, "
            f"vi_samples={len(train_ds)}, en_samples={len(en_ds)}"
        )

    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds
        if not args.lazy_tokenize:
            eval_remove = list(eval_ds.column_names)
            eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_remove)

    data_collator = (
        TextLMCollator(tokenizer=tokenizer, max_length=args.max_length, text_column=text_column)
        if args.lazy_tokenize
        else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_load_kwargs = {
        "cache_dir": args.cache_dir,
    }
    if args.use_qlora_8bit:
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["low_cpu_mem_usage"] = True
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_load_kwargs["torch_dtype"] = dtype
    else:
        model_load_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs,
    )

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:
            raise RuntimeError("PEFT is required when --use-lora is set.") from exc

        if args.use_qlora_8bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    total_steps = estimate_total_steps(len(train_tok), args)
    warmup_steps = int(total_steps * args.warmup_ratio)

    ta_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "save_strategy": "steps",
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_mode = "steps" if eval_tok is not None else "no"
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = eval_mode
    else:
        ta_kwargs["evaluation_strategy"] = eval_mode

    train_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": data_collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=== Baseline VI Fine-tuning Done ===")
    print(f"Saved model: {args.output_dir}")


if __name__ == "__main__":
    main()
