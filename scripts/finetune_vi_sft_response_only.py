#!/usr/bin/env python3
"""Response-only SFT fine-tuning for causal LMs.

This is like finetune_vi_baseline.py, but it masks the prompt portion of each
example so the loss is computed only on the assistant response tokens.

Supported dataset formats:
1) Single `text` column containing a response marker (default: "### Response:")
   Example:
     "### Instruction: ...\n\n### Response:\n<assistant text>"
2) Separate `prompt` and `response` columns (optional flags).
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
    DataCollatorForTokenClassification,
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


def estimate_total_steps(num_examples: int, args: argparse.Namespace) -> int:
    if args.max_steps > 0:
        return args.max_steps
    micro_steps_per_epoch = math.ceil(num_examples / max(1, args.per_device_train_batch_size))
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / max(1, args.gradient_accumulation_steps))
    return max(1, int(math.ceil(args.num_train_epochs * optimizer_steps_per_epoch)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response-only SFT fine-tuning for causal LM")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True, help="JSONL/JSON/TXT containing SFT examples")
    parser.add_argument("--eval-file", default=None)

    parser.add_argument("--text-column", default="text", help="Used when examples are stored in one text column.")
    parser.add_argument("--prompt-column", default="", help="Optional prompt column for prompt/response datasets.")
    parser.add_argument("--response-column", default="", help="Optional response column for prompt/response datasets.")
    parser.add_argument("--response-marker", default="### Response:", help="Split marker in the text column.")
    parser.add_argument("--add-eos", action="store_true", help="Append eos_token to the end of each example.")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--max-length", type=int, default=512)
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


def split_text_by_marker(text: str, marker: str) -> tuple[str, str]:
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"Missing response marker {marker!r}")
    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker) :]
    return prompt, response


def build_response_only_features(
    *,
    tokenizer,
    max_length: int,
    prompt_text: str,
    response_text: str,
    add_eos: bool,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    if add_eos and tokenizer.eos_token_id is not None:
        eos = [int(tokenizer.eos_token_id)]
    else:
        eos = []

    # Prefer keeping the response intact; truncate from the left side of the prompt.
    total_ids = prompt_ids + response_ids + eos
    total_len = len(total_ids)
    if total_len > max_length:
        max_resp = max_length
        if len(response_ids) + len(eos) >= max_resp:
            keep_resp = max_resp
            total_ids = (response_ids + eos)[-keep_resp:]
            prompt_len = 0
        else:
            keep_prompt = max_length - (len(response_ids) + len(eos))
            prompt_ids = prompt_ids[-keep_prompt:]
            total_ids = prompt_ids + response_ids + eos
            prompt_len = len(prompt_ids)
    else:
        prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + total_ids[prompt_len:]
    attention_mask = [1] * len(total_ids)
    return {"input_ids": total_ids, "attention_mask": attention_mask, "labels": labels}


def main() -> None:
    args = parse_args()
    if args.use_qlora_8bit:
        args.use_lora = True

    train_ds = load_text_split(args.train_file, args.text_column)
    text_column = resolve_text_column(train_ds, args.text_column)
    if text_column != args.text_column:
        train_ds = train_ds.rename_column(text_column, args.text_column)
        text_column = args.text_column

    eval_ds = None
    if args.eval_file:
        eval_ds = load_text_split(args.eval_file, text_column)
        eval_col = resolve_text_column(eval_ds, text_column)
        if eval_col != text_column:
            eval_ds = eval_ds.rename_column(eval_col, text_column)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_col = args.prompt_column.strip() or None
    response_col = args.response_column.strip() or None

    if (prompt_col is None) != (response_col is None):
        raise ValueError("--prompt-column and --response-column must be set together (or neither).")

    def encode_batch(batch: dict) -> dict:
        input_ids_list = []
        attn_list = []
        labels_list = []
        dropped = 0

        if prompt_col and response_col:
            prompts = [normalize_text_value(x) for x in batch.get(prompt_col, [])]
            responses = [normalize_text_value(x) for x in batch.get(response_col, [])]
            pairs = zip(prompts, responses)
        else:
            texts = [normalize_text_value(x) for x in batch.get(text_column, [])]
            pairs = []
            for t in texts:
                try:
                    pairs.append(split_text_by_marker(t, args.response_marker))
                except Exception:
                    pairs.append(("", ""))

        for prompt_text, response_text in pairs:
            if not prompt_text or not response_text:
                dropped += 1
                continue
            feats = build_response_only_features(
                tokenizer=tokenizer,
                max_length=args.max_length,
                prompt_text=prompt_text,
                response_text=response_text,
                add_eos=args.add_eos,
            )
            input_ids_list.append(feats["input_ids"])
            attn_list.append(feats["attention_mask"])
            labels_list.append(feats["labels"])

        if dropped:
            # Keep lengths aligned for Dataset.map; pad with 1 dummy example that will be filtered out.
            # We filter later using a separate column.
            pass

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    remove_cols = list(train_ds.column_names)
    train_tok = train_ds.map(encode_batch, batched=True, remove_columns=remove_cols)

    eval_tok = None
    if eval_ds is not None:
        eval_remove = list(eval_ds.column_names)
        eval_tok = eval_ds.map(encode_batch, batched=True, remove_columns=eval_remove)

    # Pads input_ids/attention_mask/labels; labels are padded with -100.
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_load_kwargs = {"cache_dir": args.cache_dir}
    if args.use_qlora_8bit:
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["low_cpu_mem_usage"] = True
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_load_kwargs["torch_dtype"] = dtype
    else:
        model_load_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_load_kwargs)

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

    total_steps = estimate_total_steps(len(train_tok), args)
    warmup_steps = int(args.warmup_ratio * total_steps)

    # TrainingArguments changed a bit across transformers versions; keep this script tolerant.
    training_kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": "steps" if eval_tok is not None else "no",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": args.report_to,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_total_limit": 2,
        "remove_unused_columns": False,
    }

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    accepted.discard("self")

    # Handle common rename in some versions.
    if "evaluation_strategy" not in accepted and "eval_strategy" in accepted:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")

    training_args = TrainingArguments(**{k: v for k, v in training_kwargs.items() if k in accepted})

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }
    trainer_accepted = set(inspect.signature(Trainer.__init__).parameters)
    trainer_accepted.discard("self")
    trainer = Trainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_accepted})

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=== Training Completed ===")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
