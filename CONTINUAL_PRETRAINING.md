# Continual Pretraining

These two entrypoints are wrappers around the copied trainers in `order_kd/scripts/`.

They default to:

- `data/vi_corpus/viwiki_clean/viwiki_clean_train.jsonl`
- `data/vi_corpus/viwiki_clean/viwiki_clean_valid.jsonl`
- `data/en_buffer/en_buffer.jsonl`
- `outputs/order_kd/vi_order_kd.jsonl`

The expected self-contained layout is:

```text
order_kd/data/vi_corpus/viwiki_clean/
```

with:

- `viwiki_clean_train.jsonl`
- `viwiki_clean_valid.jsonl` (optional, only needed if you want eval during CPT)

## Install

Setup scripts:

```bash
bash scripts/setup_env_cuda128.sh /data/caches/uv order_kd/.venv311
```

```powershell
pwsh order_kd/scripts/setup_env_cuda128.ps1 -CacheDir "D:\Hung-Dung\caches\uv" -VenvPath "order_kd\.venv311"
```

Equivalent manual sync/install commands:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install --python .venv/bin/python -r requirements/base.txt
uv pip install --python .venv/bin/python -r requirements/torch-cu128.txt
uv pip install --python .venv/bin/python -r requirements/train-extra.txt
```

Quick sanity check after install:

```bash
.venv/bin/python -c "from transformers import Trainer; print('Trainer import ok')"
```

This bundle intentionally installs `torch` only, not `torchvision` or `torchaudio`.
They are not needed for these training scripts, and mismatched CUDA builds can break
`transformers` imports before training even starts.

If your GPU does not support BF16, replace `--bf16` with `--fp16`.

## Plain LoRA CPT

```bash
python scripts/continual_pretrain_vi_lora.py \
  --gradient-checkpointing --bf16 \
  --max-length 512 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --save-steps 200 \
  --eval-steps 200 \
  --logging-steps 20
```

## 8-bit QLoRA-style CPT

```bash
python scripts/continual_pretrain_vi_qlora8.py \
  --gradient-checkpointing --bf16 \
  --max-length 512 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --save-steps 200 \
  --eval-steps 200 \
  --logging-steps 20
```

The QLoRA 8-bit launchers pass `--lazy-tokenize` by default, so they skip the long upfront dataset `map(...)` tokenization stage.
The plain LoRA launchers now do the same.

## OrderKD + LoRA CPT

```bash
python scripts/continual_pretrain_orderkd_lora.py \
  --gradient-checkpointing --bf16 \
  --max-length 512 \
  --num-epochs 1 \
  --train-batch-size 2 \
  --order-batch-size 4 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --lambda-ord 0.1 \
  --save-steps 200 \
  --logging-steps 20
```

## OrderKD + 8-bit QLoRA-style CPT

```bash
python scripts/continual_pretrain_orderkd_qlora8.py \
  --gradient-checkpointing --bf16 \
  --max-length 512 \
  --num-epochs 1 \
  --train-batch-size 2 \
  --order-batch-size 4 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --lambda-ord 0.1 \
  --save-steps 200 \
  --logging-steps 20
```

## Retention / Replay

The self-contained bundle also includes:

- `data/en_buffer/en_buffer.jsonl`

So these work without pointing back to any outside folder:

```bash
python scripts/continual_pretrain_vi_lora.py \
  --en-replay-prob 0.1 \
  --gradient-checkpointing --bf16
```

```bash
python scripts/continual_pretrain_orderkd_lora.py \
  --lambda-ret 0.1 \
  --gradient-checkpointing --bf16
```

## Quick 5% / 10% Test Runs

For a fast smoke test before a full run:

```bash
python scripts/continual_pretrain_vi_lora.py \
  --gradient-checkpointing --bf16 \
  --train-fraction 0.05 \
  --eval-fraction 0.10 \
  --max-length 512 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16
```

```bash
python scripts/continual_pretrain_orderkd_lora.py \
  --gradient-checkpointing --bf16 \
  --train-fraction 0.10 \
  --order-kd-fraction 0.10 \
  --max-length 512 \
  --num-epochs 1 \
  --train-batch-size 2 \
  --order-batch-size 4 \
  --gradient-accumulation-steps 16
```

You can also cap by sample count instead of percentage:

```bash
python scripts/continual_pretrain_vi_lora.py --train-max-samples 20000 --eval-max-samples 1000
python scripts/continual_pretrain_orderkd_lora.py --train-max-samples 20000 --order-kd-max-samples 20000
```

## Override Paths

If your files live elsewhere, pass them explicitly:

```bash
python scripts/continual_pretrain_vi_lora.py \
  --train-file /path/to/viwiki_clean_train.jsonl \
  --eval-file /path/to/viwiki_clean_valid.jsonl
```

```bash
python scripts/continual_pretrain_orderkd_lora.py \
  --train-file /path/to/viwiki_clean_train.jsonl \
  --order-kd-jsonl /path/to/vi_order_kd.jsonl
```
