#!/usr/bin/env python3
"""Chat-template SFT (completion-only loss) for causal LMs, pure Transformers.

This trains on JSONL/JSON datasets that contain a `messages` column:
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]

We apply `tokenizer.apply_chat_template(...)` and compute loss only on the final
assistant message tokens (prompt tokens are masked with -100).

No TRL/Unsloth dependency.
"""

from __future__ import annotations

import argparse
import inspect
import math
import pathlib
from typing import Any, Dict, List, Sequence

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_split(path: str) -> datasets.Dataset:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return datasets.load_dataset("json", data_files={"train": path})["train"]
    if suffix == ".txt":
        return datasets.load_dataset("text", data_files={"train": path})["train"]
    raise ValueError(f"Unsupported dataset file extension: {suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat-template SFT with completion-only loss")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", default=None)
    p.add_argument("--messages-column", default="messages")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--report-to", default="none")

    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--use-qlora-8bit", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    return p.parse_args()


def _apply_chat_ids(tokenizer, messages: Sequence[dict], *, add_generation_prompt: bool) -> List[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template; cannot train with chat template.")

    try:
        out = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
        )
        # transformers may return list[int] or list[list[int]]
        if isinstance(out, list) and out and isinstance(out[0], int):
            return [int(x) for x in out]
        if isinstance(out, list) and out and isinstance(out[0], list):
            return [int(x) for x in out[0]]
    except TypeError:
        pass

    # Fallback: render to text and tokenize (avoid special token duplication).
    text = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt)
    enc = tokenizer(text, add_special_tokens=False)
    return [int(x) for x in enc["input_ids"]]


def _split_prompt_and_full_ids(tokenizer, messages: Sequence[dict]) -> tuple[List[int], List[int]]:
    # Train only on the last assistant message.
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each sample must end with an assistant message.")
    prompt_msgs = list(messages[:-1])
    full_msgs = list(messages)

    prompt_ids = _apply_chat_ids(tokenizer, prompt_msgs, add_generation_prompt=True)
    full_ids = _apply_chat_ids(tokenizer, full_msgs, add_generation_prompt=False)

    # Sanity: full should start with prompt.
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids, full_ids

    # Fallback: try text-prefix logic.
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    if full_text.startswith(prompt_text):
        prompt_ids2 = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids2 = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids2[: len(prompt_ids2)] == prompt_ids2:
            return [int(x) for x in prompt_ids2], [int(x) for x in full_ids2]

    raise RuntimeError("Chat template prefix mismatch; cannot reliably compute completion-only labels.")


class ChatCompletionOnlyCollator:
    def __init__(self, tokenizer, *, max_length: int, messages_column: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_column = messages_column

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        attn_list: List[List[int]] = []

        for ex in examples:
            messages = ex.get(self.messages_column)
            if not isinstance(messages, list):
                raise ValueError(f"Missing or invalid `{self.messages_column}`; expected list[dict].")

            prompt_ids, full_ids = _split_prompt_and_full_ids(self.tokenizer, messages)
            prompt_len = len(prompt_ids)

            if len(full_ids) > self.max_length:
                drop = len(full_ids) - self.max_length
                full_ids = full_ids[drop:]
                prompt_len = max(0, prompt_len - drop)

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attn_list.append([1] * len(full_ids))

        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        labels_t = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            labels_t[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        batch["labels"] = labels_t
        return batch


def _training_args(args: argparse.Namespace, *, eval_enabled: bool, warmup_steps: int) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
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
        "evaluation_strategy": "steps" if eval_enabled else "no",
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
    if "evaluation_strategy" not in accepted and "eval_strategy" in accepted:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})


def main() -> None:
    args = parse_args()
    if args.use_qlora_8bit:
        args.use_lora = True

    train_ds = load_split(args.train_file)
    eval_ds = load_split(args.eval_file) if args.eval_file else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else max(
            1,
            int(
                math.ceil(
                    args.num_train_epochs
                    * math.ceil(len(train_ds) / max(1, args.per_device_train_batch_size))
                    / max(1, args.gradient_accumulation_steps)
                )
            ),
        )
    )
    warmup_steps = int(args.warmup_ratio * total_steps)

    model_kwargs: Dict[str, Any] = {"cache_dir": args.cache_dir}
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if args.use_qlora_8bit:
        model_kwargs["device_map"] = "auto"
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:
            raise RuntimeError("PEFT is required when --use-lora is set.") from exc
        if args.use_qlora_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
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

    collator = ChatCompletionOnlyCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        messages_column=args.messages_column,
    )

    training_args = _training_args(args, eval_enabled=eval_ds is not None, warmup_steps=warmup_steps)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    accepted = set(inspect.signature(Trainer.__init__).parameters)
    accepted.discard("self")
    trainer = Trainer(**{k: v for k, v in trainer_kwargs.items() if k in accepted})

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
