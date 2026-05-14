# Qwen2.5-1.5B-Instruct Vietnamese Benchmark

Use these PowerShell commands from the project root.

## 1. Go to the benchmark folder

```powershell
Set-Location "D:\Hung-Dung\Expriments\catastrophic_research\Vietnamese benchmark"
```

## 2. Install dependencies

```powershell
uv sync
```

Optional, if you want 4-bit local inference support:

```powershell
uv sync --group gpu
```

## 3. Download and normalize all benchmark datasets

```powershell
uv run python scripts/download_datasets.py
```

Optional, if you only want a quick subset first:

```powershell
uv run python scripts/download_datasets.py --only xquad --only mlqa --only uit_vsfc
```

## 4. Smoke test the pipeline

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.qwen25_1_5b_instruct.json --only-model-key qwen25_1_5b_instruct --only-dataset xquad --only-dataset mlqa --max-test-examples 20 --smoke-test
```

## 5. Run the full Vietnamese benchmark

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.qwen25_1_5b_instruct.json --only-model-key qwen25_1_5b_instruct
```

## 6. Run the full benchmark with native Qwen chat template

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.qwen25_1_5b_instruct.json --only-model-key qwen25_1_5b_instruct --qwen-native-chat-template
```

## 7. Run a few-shot version

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.qwen25_1_5b_instruct.json --only-model-key qwen25_1_5b_instruct --fewshot --num-fs 3
```

## 8. Run with 4-bit loading

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.qwen25_1_5b_instruct.json --only-model-key qwen25_1_5b_instruct --use-4bit
```

## 9. Chat with the model in terminal

```powershell
uv run python scripts/chat_in_terminal.py --model Qwen/Qwen2.5-1.5B-Instruct
```

Optional, if you want 4-bit chat on GPU:

```powershell
uv run python scripts/chat_in_terminal.py --model Qwen/Qwen2.5-1.5B-Instruct --use-4bit
```

Commands inside the chat session:

- `/help` show available commands
- `/reset` clear chat history
- `/exit` or `/quit` leave the session

## 10. Where results will be written

- Generations: `results/generation/qwen25_1_5b_instruct`
- Evaluation summaries: `results/evaluation/qwen25_1_5b_instruct`
- Logs: `results/logs`
- Runner summary: `results/benchmark_run_summary.json`

## 11. Sample Tasks with Answers

### Summarization (Wikilingua)
**Input:**
"Chúng sẽ không thích bị bắn nước vào người. Xịt lên đàn chim khi chúng vừa bay đến. Nếu bạn chờ cho đến khi chúng xây tổ thì đã muộn."

**Output:**
"Dùng vòi xịt nước vào chim bồ câu."

**Input:**
"Việc loại bỏ lớp sơn bóng trên móng bột rất quan trọng trong quá trình gỡ bỏ lớp bột. Dũa từng móng thật kỹ và thật đều - như vậy, việc gỡ móng bột sẽ hiệu quả hơn. Thời gian chờ 10-15 phút sẽ giúp acetone phát huy tác dụng. Cố gắng không xê dịch giấy bạc hoặc bông gòn nhiều lần trong thời gian chờ."

**Output:**
"Dũa bề mặt móng bằng dụng cụ dũa. Chờ 10-15 phút để móng tay ngấm acetone."

### Question Answering (XQuAD)
**Question:**
"Đội thủ Panthers đã thua bao nhiêu điểm?"

**Answer:**
"308"

**Question:**
"Jared Allen có bao nhiêu lần vật ngã trong sự nghiệp?"

**Answer:**
"136"

### Translation (Opus100 En-Vi)
**Input (English):**
"We are in a dive."

**Output (Vietnamese):**
"Chúng ta đang lao xuống."

**Input (English):**
"Beautiful country."

**Output (Vietnamese):**
"Một đất nước đẹp tuyệt."
