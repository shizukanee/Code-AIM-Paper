# Qwen2.5-1.5B Linux Training Commands

Run these from the `order_kd` folder on Linux.

If your GPU does not support BF16, replace `--bf16` with `--fp16`.

## 1. Ordinary LoRA continual pretraining

```bash
.venv311/bin/python scripts/continual_pretrain_vi_lora.py \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/checkpoints/qwen25_1_5b_viwiki_lora \
  --gradient-checkpointing \
  --bf16 \
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

## 2. OrderKD + LoRA continual pretraining

```bash
.venv311/bin/python scripts/continual_pretrain_orderkd_lora.py \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/checkpoints/qwen25_1_5b_viwiki_orderkd_lora \
  --gradient-checkpointing \
  --bf16 \
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

Defaults already used by these wrappers:

- `data/vi_corpus/viwiki_clean/viwiki_clean_train.jsonl`
- `data/vi_corpus/viwiki_clean/viwiki_clean_valid.jsonl` for plain LoRA eval
- `outputs/order_kd/vi_order_kd.jsonl` for OrderKD
- `data/en_buffer/en_buffer.jsonl` if you later add replay flags
