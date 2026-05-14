# Vietnamese benchmark

This folder is a standalone Vietnamese evaluation workspace for benchmarking both instruct checkpoints and LoRA adapters.

It uses the upstream `MELT` evaluator as the backend, but keeps a separate local config layer so we can:

- register the exact datasets from your table
- normalize every dataset into one local split format under `datasets/`
- run the same suite against plain instruct models and PEFT / LoRA adapters

## Included datasets

The full source registry is in [configs/datasets_manifest.json](./configs/datasets_manifest.json).

Covered benchmark ids:

- `xquad`
- `mlqa`
- `vietnews`
- `wikilingua`
- `vlsp2016`
- `uit_vsfc`
- `uit_vsmec`
- `phoatis`
- `zalo_e2eqa`
- `uit_vimmrc`
- `uit_victsd`
- `uit_vihsd`
- `mmarco_vi`
- `mrobust04_vi`
- `mlqa_mlm`
- `vsec`
- `synthetic_reasoning_natural`
- `synthetic_reasoning_abstract`
- `vietnamese_math`
- `opus100_envi`
- `phomt_envi`

## Setup

From this folder:

```powershell
uv sync
```

If you want 4-bit local inference:

```powershell
uv sync --group gpu
```

Then download and normalize the benchmark datasets:

```powershell
uv run python scripts/download_datasets.py
```

Two synthetic reasoning datasets are gated on Hugging Face. To include them,
authenticate first (for example by setting `HF_TOKEN`) and ensure your account
has accepted access:

```powershell
$env:HF_TOKEN="<your_hf_token>"
uv run python scripts/download_datasets.py --only synthetic_reasoning_natural --only synthetic_reasoning_abstract
```

If you only want a subset first:

```powershell
uv run python scripts/download_datasets.py --only xquad --only mlqa --only uit_vsfc
```

The downloader writes a report to [results/dataset_download_report.json](./results/dataset_download_report.json).

## Portable transfer (copy to another machine)

Yes, you can package this benchmark into a portable folder so the target
machine needs little to no setup.

From this folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_portable_bundle.ps1 -Clean
```

What this includes by default:

- project files (`configs`, `datasets`, `scripts`, `melt-upstream`, `results`)
- local `.venv`
- Hugging Face cache from `D:\Hung-Dung\caches\hf`
- helper runner `run_portable_benchmark.ps1`

Output folder:

- `portable_bundle`

On the new Windows machine:

```powershell
Set-Location .\portable_bundle
.\run_portable_benchmark.ps1
```

Optional: include one or more LoRA adapter folders in the bundle:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_portable_bundle.ps1 -Clean -AdapterPaths "D:\path\to\adapter1","D:\path\to\adapter2"
```

Important constraints:

- The target machine still needs compatible GPU drivers/CUDA for GPU runs.
- If your model plan still has placeholder adapter paths, update them before running LoRA evaluations.

## Model plan

Start from [configs/model_plan.example.json](./configs/model_plan.example.json).

For a ready instruct-vs-LoRA baseline, use
[configs/model_plan.qwen25_0_5b_instruct_vs_lora.json](./configs/model_plan.qwen25_0_5b_instruct_vs_lora.json).

For an instruct model, set:

- `model_name`
- `wrapper`
- `prompt_template`

For a LoRA model, also set:

- `base_model_name`
- `adapter_path`

The patched MELT loader will attach the adapter automatically during evaluation.

## Run benchmarks

Run the full plan:

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.example.json
```

Run one model on selected datasets:

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.example.json --only-model-key qwen25_3b_instruct --only-dataset xquad --only-dataset mlqa
```

Smoke-test orchestration without a full run:

```powershell
uv run python scripts/run_vietnamese_benchmark.py --plan configs/model_plan.example.json --smoke-test --max-models 1 --max-datasets 2
```

## Outputs

- Raw generations: [results/generation](./results/generation)
- Metric summaries: [results/evaluation](./results/evaluation)
- Run logs: [results/logs](./results/logs)
- Runner summary: [results/benchmark_run_summary.json](./results/benchmark_run_summary.json)

## Notes

- Some upstream sources are benchmark mirrors or curated copies of the original datasets.
- Large datasets are normalized into a small local support split plus the benchmark split needed for evaluation.
- The IR datasets are consumed from locally normalized files because the original benchmark repos use non-uniform schemas across train and test files.
