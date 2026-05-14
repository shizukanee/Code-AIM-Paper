#!/usr/bin/env python3
"""Train Vietnamese LM adaptation with orderKD and optional retention KD.

Objective:
L_total = L_lm_vi + lambda_ord * L_orderKD + lambda_ret * L_ret

Assumptions:
- Vietnamese LM data: JSONL/JSON/TXT with a text column.
- order_kd_jsonl: built by scripts/build_order_kd_dataset.py.
- Optional English retention buffer: JSONL/JSON/TXT with text column.

This script is meant for remote training machines (GPU).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import datasets
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.order_head import OrderHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LM with orderKD and optional retention KD")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--vi-train-file", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--order-kd-jsonl", default=None)
    parser.add_argument("--en-buffer-file", default=None)
    parser.add_argument("--base-model-name-or-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--order-kd-fraction", type=float, default=0.0)
    parser.add_argument("--order-kd-max-samples", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--order-batch-size", type=int, default=4)
    parser.add_argument("--retention-batch-size", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--order-layer", type=int, default=-1)
    parser.add_argument("--order-proj", type=int, default=512)
    parser.add_argument("--order-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-ord", type=float, default=0.1)

    parser.add_argument("--lambda-ret", type=float, default=0.0)
    parser.add_argument("--ret-temperature", type=float, default=2.0)

    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lazy-tokenize", action="store_true")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora-8bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def normalize_text_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


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


def subset_rows(
    rows: List[dict],
    *,
    seed: int,
    fraction: float,
    max_samples: int,
    label: str,
) -> List[dict]:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"--{label}-fraction must be in (0, 1].")
    if max_samples < 0:
        raise ValueError(f"--{label}-max-samples must be >= 0.")

    target_size = len(rows)
    if fraction < 1.0:
        target_size = max(1, int(len(rows) * fraction))
    if max_samples > 0:
        target_size = min(target_size, max_samples)

    if target_size >= len(rows):
        return rows

    idxs = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = idxs[:target_size]
    idxs.sort()
    print(f"Subsampling {label} rows: keeping {target_size} / {len(rows)} examples")
    return [rows[i] for i in idxs]


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
        raise ValueError(f"Unsupported text file extension: {suffix}")
    return ds


class OrderKDDataset(Dataset):
    def __init__(self, rows: List[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class OrderKDCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("OrderKDCollator requires a fast tokenizer for word alignment.")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        token_lists = [ex["tokens"] for ex in examples]
        enc = self.tokenizer(
            token_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        pair_batch_idx = []
        dep_positions = []
        head_positions = []
        q_teacher = []

        for bidx, ex in enumerate(examples):
            word_ids = enc.word_ids(batch_index=bidx)
            first_subword = {}
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid not in first_subword:
                    first_subword[wid] = pos

            for arc in ex.get("arcs", []):
                dep_w = int(arc["dep_word"])
                head_w = int(arc["head_word"])
                if dep_w not in first_subword or head_w not in first_subword:
                    continue
                pair_batch_idx.append(bidx)
                dep_positions.append(first_subword[dep_w])
                head_positions.append(first_subword[head_w])
                q_teacher.append(float(arc["q_teacher"]))

        if pair_batch_idx:
            pair_batch_idx_t = torch.tensor(pair_batch_idx, dtype=torch.long)
            dep_positions_t = torch.tensor(dep_positions, dtype=torch.long)
            head_positions_t = torch.tensor(head_positions, dtype=torch.long)
            q_teacher_t = torch.tensor(q_teacher, dtype=torch.float32)
        else:
            pair_batch_idx_t = torch.zeros(0, dtype=torch.long)
            dep_positions_t = torch.zeros(0, dtype=torch.long)
            head_positions_t = torch.zeros(0, dtype=torch.long)
            q_teacher_t = torch.zeros(0, dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pair_batch_idx": pair_batch_idx_t,
            "dep_positions": dep_positions_t,
            "head_positions": head_positions_t,
            "q_teacher": q_teacher_t,
        }


class TextOnlyCollator:
    def __init__(self, tokenizer, max_length: int, text_column: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [normalize_text_value(ex.get(self.text_column, "")) for ex in examples]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


class TextLMCollator:
    def __init__(self, tokenizer, max_length: int, text_column: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
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


def load_order_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def masked_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, attention_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    # Shift to next-token prediction positions.
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    mask = attention_mask[:, 1:].to(student_logits.dtype)

    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1)

    denom = mask.sum().clamp(min=1.0)
    return (token_kl * mask).sum() / denom * (temperature ** 2)


def next_or_restart(loader: DataLoader, iterator: Optional[Iterable]) -> tuple[dict, Iterable]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def save_checkpoint(model, tokenizer, output_dir: pathlib.Path, name: str) -> None:
    ckpt = output_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    if hasattr(model, "order_head"):
        torch.save(model.order_head.state_dict(), ckpt / "order_head.pt")


def main() -> None:
    args = parse_args()
    if args.use_qlora_8bit:
        args.use_lora = True
    set_seed(args.seed)

    if args.lambda_ord > 0.0 and not args.order_kd_jsonl:
        raise ValueError("--order-kd-jsonl is required when --lambda-ord > 0")
    if args.lambda_ret > 0.0 and not args.en_buffer_file:
        raise ValueError("--en-buffer-file is required when --lambda-ret > 0")

    device = choose_device()
    dtype = choose_dtype(args)

    print(f"Device: {device} | Dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model.order_head = OrderHead(
        hidden_size=model.config.hidden_size,
        proj=args.order_proj,
        dropout=args.order_dropout,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if not args.use_qlora_8bit:
        model = model.to(device)
    model.order_head = model.order_head.to(device=device, dtype=dtype)

    # Vietnamese LM train dataset.
    vi_ds = load_text_split(args.vi_train_file, args.text_column)
    text_col = resolve_text_column(vi_ds, args.text_column)
    if text_col != args.text_column:
        vi_ds = vi_ds.rename_column(text_col, args.text_column)
        text_col = args.text_column
    vi_ds = subset_dataset(
        vi_ds,
        seed=args.seed,
        fraction=args.train_fraction,
        max_samples=args.train_max_samples,
        label="train",
    )

    if args.lazy_tokenize:
        lm_collator = TextLMCollator(tokenizer=tokenizer, max_length=args.max_length, text_column=text_col)
        lm_loader = DataLoader(vi_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=lm_collator)
    else:
        def tok_fn(batch: dict) -> dict:
            texts = [normalize_text_value(v) for v in batch[text_col]]
            return tokenizer(texts, truncation=True, max_length=args.max_length)

        remove_cols = list(vi_ds.column_names)
        vi_tok = vi_ds.map(tok_fn, batched=True, remove_columns=remove_cols)
        lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        lm_loader = DataLoader(vi_tok, batch_size=args.train_batch_size, shuffle=True, collate_fn=lm_collator)

    # Order-KD auxiliary dataset.
    order_loader = None
    order_iter = None
    if args.order_kd_jsonl:
        order_rows = load_order_rows(args.order_kd_jsonl)
        order_fraction = args.order_kd_fraction if args.order_kd_fraction > 0.0 else args.train_fraction
        order_rows = subset_rows(
            order_rows,
            seed=args.seed,
            fraction=order_fraction,
            max_samples=args.order_kd_max_samples,
            label="order-kd",
        )
        order_ds = OrderKDDataset(order_rows)
        order_collator = OrderKDCollator(tokenizer=tokenizer, max_length=args.max_length)
        order_loader = DataLoader(order_ds, batch_size=args.order_batch_size, shuffle=True, collate_fn=order_collator)

    # Optional retention KD dataset and frozen teacher model.
    ret_loader = None
    ret_iter = None
    base_model = None
    if args.lambda_ret > 0.0:
        base_name = args.base_model_name_or_path or args.model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=dtype,
            cache_dir=args.cache_dir,
        ).to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

        en_ds = load_text_split(args.en_buffer_file, args.text_column)
        en_col = resolve_text_column(en_ds, args.text_column)
        if en_col != args.text_column:
            en_ds = en_ds.rename_column(en_col, args.text_column)
            en_col = args.text_column
        ret_collator = TextOnlyCollator(tokenizer=tokenizer, max_length=args.max_length, text_column=en_col)
        ret_loader = DataLoader(en_ds, batch_size=args.retention_batch_size, shuffle=True, collate_fn=ret_collator)

    # Training setup.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = math.ceil(len(lm_loader) / max(1, args.gradient_accumulation_steps))
        total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    optimizer_step = 0

    loss_log = []

    for epoch in range(args.num_epochs):
        for lm_batch in lm_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            model.train()
            lm_batch = to_device(lm_batch, device)

            lm_outputs = model(**lm_batch, use_cache=False)
            lm_loss = lm_outputs.loss
            total_loss = lm_loss

            order_loss = torch.zeros((), device=device)
            if order_loader is not None and args.lambda_ord > 0.0:
                order_batch, order_iter = next_or_restart(order_loader, order_iter)
                order_batch = to_device(order_batch, device)
                if order_batch["pair_batch_idx"].numel() > 0:
                    ord_out = model(
                        input_ids=order_batch["input_ids"],
                        attention_mask=order_batch["attention_mask"],
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hs = ord_out.hidden_states
                    layer_idx = args.order_layer if args.order_layer >= 0 else (len(hs) + args.order_layer)
                    layer_idx = max(0, min(layer_idx, len(hs) - 1))
                    h = hs[layer_idx]
                    h_dep = h[order_batch["pair_batch_idx"], order_batch["dep_positions"], :]
                    h_head = h[order_batch["pair_batch_idx"], order_batch["head_positions"], :]
                    head_dtype = model.order_head.mlp[0].weight.dtype
                    if h_dep.dtype != head_dtype:
                        h_dep = h_dep.to(head_dtype)
                        h_head = h_head.to(head_dtype)
                    probs, _ = model.order_head(h_dep, h_head)
                    q_teacher = order_batch["q_teacher"].to(probs.dtype)
                    order_loss = F.binary_cross_entropy(probs, q_teacher)
                    total_loss = total_loss + args.lambda_ord * order_loss

            ret_loss = torch.zeros((), device=device)
            if ret_loader is not None and args.lambda_ret > 0.0 and base_model is not None:
                ret_batch, ret_iter = next_or_restart(ret_loader, ret_iter)
                ret_batch = to_device(ret_batch, device)
                with torch.no_grad():
                    base_logits = base_model(
                        input_ids=ret_batch["input_ids"],
                        attention_mask=ret_batch["attention_mask"],
                        use_cache=False,
                    ).logits
                student_logits = model(
                    input_ids=ret_batch["input_ids"],
                    attention_mask=ret_batch["attention_mask"],
                    use_cache=False,
                ).logits
                ret_loss = masked_kl(student_logits, base_logits, ret_batch["attention_mask"], args.ret_temperature)
                total_loss = total_loss + args.lambda_ret * ret_loss

            (total_loss / args.gradient_accumulation_steps).backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                    save_checkpoint(model, tokenizer, out_dir, f"checkpoint-{optimizer_step}")

            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                rec = {
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "lr": scheduler.get_last_lr()[0],
                    "lm_loss": float(lm_loss.detach().cpu().item()),
                    "order_loss": float(order_loss.detach().cpu().item()),
                    "ret_loss": float(ret_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                }
                loss_log.append(rec)
                print(
                    f"step={global_step} opt={optimizer_step} "
                    f"lm={rec['lm_loss']:.4f} ord={rec['order_loss']:.4f} "
                    f"ret={rec['ret_loss']:.4f} total={rec['total_loss']:.4f}"
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save final artifacts.
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.order_head.state_dict(), out_dir / "order_head.pt")

    metrics = {
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "lambda_ord": args.lambda_ord,
        "lambda_ret": args.lambda_ret,
        "ret_temperature": args.ret_temperature,
        "order_layer": args.order_layer,
        "loss_log": loss_log,
    }
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Training Completed ===")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
