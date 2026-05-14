#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
MANIFEST_PATH = CONFIG_DIR / "datasets_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Vietnamese benchmark suite for instruct and LoRA models")
    parser.add_argument("--plan", default=str(CONFIG_DIR / "model_plan.example.json"))
    parser.add_argument("--only-model-key", action="append", default=[])
    parser.add_argument("--only-dataset", action="append", default=[])
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=0,
        help="Cap evaluated test examples per dataset (0 uses full test split).",
    )
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--num-fs", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prompt-type", default="normal", choices=["normal", "weaker", "medium"])
    parser.add_argument("--category", default="zero-shot", choices=["zero-shot", "few-shot", "robustness-aware", "fairness-aware", "chain-of-thought", "randomized-choice"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--continue-infer", action="store_true")
    parser.add_argument(
        "--save-detailed-eval",
        action="store_true",
        help=(
            "Save per-example detailed evaluation records in each model's "
            "evaluation output folder."
        ),
    )
    parser.add_argument(
        "--qwen-template-override",
        default="",
        help=(
            "Optional prompt template name to use for Qwen models only "
            "(for example: chatglm). Leave empty to keep model plan values."
        ),
    )
    parser.add_argument(
        "--qwen-native-chat-template",
        action="store_true",
        help=(
            "Use tokenizer-native chat template for Qwen models. "
            "When disabled, static prompt templates are used as before."
        ),
    )
    return parser.parse_args()


def dataset_selected(dataset_id: str, selected: list[str]) -> bool:
    return not selected or dataset_id in selected


def model_selected(model_key: str, selected: list[str]) -> bool:
    return not selected or model_key in selected


def resolve_prompt_template(model: dict[str, Any], args: argparse.Namespace) -> str:
    prompt_template = model.get("prompt_template", "llama-3")
    override = args.qwen_template_override.strip()
    model_name = str(model.get("model_name", "")).lower()

    if override and "qwen" in model_name:
        return override

    return prompt_template


def build_command(
    model: dict[str, Any],
    dataset_id: str,
    args: argparse.Namespace,
    prompt_template: str,
) -> list[str]:
    # Keep outputs separate per (model_key, dataset_id) to avoid overwriting
    # when running multiple datasets or repeating runs.
    model_output_dir = RESULTS_DIR / "generation" / model["model_key"] / dataset_id
    model_eval_dir = RESULTS_DIR / "evaluation" / model["model_key"] / dataset_id

    command = [
        sys.executable,
        "-m",
        "melt",
        "--model_name",
        model["model_name"],
        "--dataset_name",
        dataset_id,
        "--lang",
        "vi",
        "--dataset_dir",
        str(DATASET_DIR),
        "--config_dir",
        str(CONFIG_DIR),
        "--output_dir",
        str(model_output_dir),
        "--output_eval_dir",
        str(model_eval_dir),
        "--wtype",
        model.get("wrapper", "hf"),
        "--ptemplate",
        prompt_template,
        "--device",
        args.device,
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--num_fs",
        str(args.num_fs),
        "--prompt_type",
        args.prompt_type,
        "--category",
        args.category,
    ]

    if model.get("base_model_name"):
        command.extend(["--base_model_name", model["base_model_name"]])
    if model.get("adapter_path"):
        command.extend(["--adapter_path", model["adapter_path"]])
    if args.fewshot:
        command.append("--fewshot_prompting")
    if args.max_test_examples > 0:
        command.extend(["--max_test_examples", str(args.max_test_examples)])
    if args.smoke_test:
        command.append("--smoke_test")
    if args.use_4bit:
        command.append("--use_4bit")
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    if args.continue_infer:
        command.append("--continue_infer")
    if args.save_detailed_eval:
        command.append("--save_detailed_eval")
    if args.qwen_native_chat_template and "qwen" in model["model_name"].lower():
        command.append("--use_tokenizer_chat_template")

    return command


def run_with_live_logs(command: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n" + " ".join(command) + "\n\nOUTPUT:\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return process.wait()


def main() -> None:
    args = parse_args()
    plan = read_json(Path(args.plan))
    manifest = read_json(MANIFEST_PATH)

    dataset_order = [item["id"] for item in manifest["datasets"] if dataset_selected(item["id"], args.only_dataset)]
    if args.max_datasets > 0:
        dataset_order = dataset_order[: args.max_datasets]

    models = [m for m in plan["models"] if model_selected(m["model_key"], args.only_model_key)]
    if args.max_models > 0:
        models = models[: args.max_models]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"runs": []}
    total_runs = len(models) * len(dataset_order)
    print(
        f"Planned runs: {total_runs} "
        f"({len(models)} model(s) x {len(dataset_order)} dataset(s))"
    )
    print("Execution order: dataset-first (run all models per dataset).")
    if args.qwen_template_override.strip():
        print(
            "Qwen template override enabled: "
            f"{args.qwen_template_override.strip()}"
        )
    if args.qwen_native_chat_template:
        print("Qwen native tokenizer chat template enabled.")
    if args.save_detailed_eval:
        print("Detailed evaluation export enabled.")
    if args.max_test_examples > 0:
        print(f"Per-dataset test example cap enabled: {args.max_test_examples}")

    run_index = 0

    for dataset_id in dataset_order:
        print(f"\n=== Dataset {dataset_id}: running {len(models)} model(s) ===")
        for model in models:
            run_index += 1
            print(
                f"\n[{run_index}/{total_runs}] Starting "
                f"model={model['model_key']} dataset={dataset_id}"
            )
            dataset_test_file = DATASET_DIR / dataset_id / f"{dataset_id}_test.json"
            if not dataset_test_file.exists():
                print(
                    f"[{run_index}/{total_runs}] Skipped missing dataset file: "
                    f"{dataset_test_file}"
                )
                summary["runs"].append(
                    {
                        "model_key": model["model_key"],
                        "dataset_id": dataset_id,
                        "status": "skipped_missing_dataset",
                        "note": f"Missing dataset file: {dataset_test_file}"
                    }
                )
                write_json(RESULTS_DIR / "benchmark_run_summary.json", summary)
                continue

            prompt_template = resolve_prompt_template(model, args)
            command = build_command(model, dataset_id, args, prompt_template)
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            melt_src = str(PROJECT_ROOT / "melt-upstream" / "src")
            env["PYTHONPATH"] = melt_src if not existing_pythonpath else melt_src + os.pathsep + existing_pythonpath
            env["PYTHONUNBUFFERED"] = "1"

            log_path = LOGS_DIR / f"{model['model_key']}__{dataset_id}.log"
            start_time = time.time()
            return_code = run_with_live_logs(command, env, log_path)
            elapsed_sec = time.time() - start_time
            status = "completed" if return_code == 0 else "failed"
            log_rel_path = str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

            print(
                f"[{run_index}/{total_runs}] {status.upper()} "
                f"model={model['model_key']} dataset={dataset_id} "
                f"template={prompt_template} "
                f"in {elapsed_sec:.1f}s"
            )
            print(
                f"[{run_index}/{total_runs}] Log: "
                f"{log_rel_path}"
            )

            summary["runs"].append(
                {
                    "model_key": model["model_key"],
                    "dataset_id": dataset_id,
                    "status": status,
                    "return_code": return_code,
                    "log_path": log_rel_path
                }
            )
            write_json(RESULTS_DIR / "benchmark_run_summary.json", summary)

    write_json(RESULTS_DIR / "benchmark_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
