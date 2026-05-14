#!/usr/bin/env python3
"""Build response-separated SFT data from the local Vietnamese benchmark.

The existing benchmark_mix builder writes prompt and answer into one `text`
field. That is convenient for plain language modeling, but it makes it hard to
train only on the answer. This script keeps the prompt and response separate
and also emits a chat-style `messages` field for SFT tooling.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    default_benchmark_root = root.parent / "Vietnamese benchmark"
    default_output = root / "data" / "vietnamese_benchmark_sft" / "training"

    parser = argparse.ArgumentParser(description="Build Vietnamese benchmark SFT JSONL files")
    parser.add_argument("--benchmark-root", default=str(default_benchmark_root))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also write merged_test.jsonl and per-dataset *_test_sft.jsonl.",
    )
    parser.add_argument(
        "--only-dataset",
        action="append",
        default=[],
        help="Dataset id to include. Can be repeated. Defaults to every local configured dataset.",
    )
    return parser.parse_args()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def first_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        texts = value.get("text")
        if isinstance(texts, list) and texts:
            return normalize_text(texts[0])
        if isinstance(texts, str):
            return normalize_text(texts)
        for key in ("answer", "value"):
            if key in value:
                return normalize_text(value[key])
        return normalize_text(value)
    if isinstance(value, list):
        if not value:
            return ""
        return first_answer(value[0])
    return normalize_text(value)


def get_value(row: dict[str, Any], primary: str, fallbacks: Iterable[str] = ()) -> Any:
    if primary in row:
        return row[primary]
    for key in fallbacks:
        if key in row:
            return row[key]
    return None


def format_response(answer_format: str, answer: str) -> str:
    if not answer:
        return ""
    if "{}" in answer_format:
        return answer_format.replace("{}", answer, 1)
    return answer


def render_prompt(task: str, template: str, row: dict[str, Any], columns: dict[str, str]) -> str:
    if task == "question-answering":
        return template.format(
            normalize_text(get_value(row, columns.get("context", "context"))),
            normalize_text(get_value(row, columns.get("query", "question"), ("query",))),
        )
    if task in {"summarization", "translation", "language-modeling"}:
        return template.format(
            normalize_text(get_value(row, columns.get("source", "source"), ("query", "text")))
        )
    if task in {
        "sentiment-analysis",
        "text-classification",
        "toxicity-detection",
        "knowledge-openended",
        "reasoning",
        "math",
    }:
        return template.format(
            normalize_text(get_value(row, columns.get("query", "query"), ("question", "source", "text")))
        )
    if task == "knowledge-mtpchoice":
        return template.format(
            normalize_text(get_value(row, columns.get("context", "context"))),
            normalize_text(get_value(row, columns.get("query", "question"), ("query",))),
            normalize_text(get_value(row, columns.get("options", "options"))),
        )
    if task == "information-retrieval":
        passage = get_value(row, columns.get("passages", "passages"), ("passage", "text"))
        return template.format(
            normalize_text(passage),
            normalize_text(get_value(row, columns.get("query", "query"))),
        )
    raise ValueError(f"Unsupported task: {task}")


def extract_answer(task: str, row: dict[str, Any], columns: dict[str, str]) -> str:
    if task == "question-answering":
        return first_answer(get_value(row, columns.get("answer", "answers"), ("answer",)))
    if task in {"summarization", "translation", "language-modeling"}:
        return first_answer(get_value(row, columns.get("target", "target"), ("answer",)))
    return first_answer(get_value(row, columns.get("answer", "answer"), ("answer", "target", "references")))


def build_messages(system_prompt: str, prompt: str, response: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


def build_rows(
    *,
    dataset_id: str,
    split: str,
    source_path: pathlib.Path,
    dataset_cfg: dict[str, Any],
    prompt_templates: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    task = dataset_cfg["task"]
    columns = dataset_cfg.get("columns", {})
    template_idx = int(dataset_cfg.get("prompting_strategy", 0))
    task_templates = prompt_templates[task]
    template = task_templates[min(template_idx, len(task_templates) - 1)]
    prompt_template = template["prompt"]
    system_prompt = template.get("system_prompt", "")
    answer_format = template.get("answer_format", "{}")
    answer_key = template.get("answer_key", "")

    raw_rows = read_json(source_path)
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected JSON list in {source_path}")

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        prompt = render_prompt(task, prompt_template, row, columns).rstrip()
        answer = extract_answer(task, row, columns)
        response = format_response(answer_format, answer).strip()
        if not prompt or not response:
            continue

        row_id = f"{dataset_id}_{split}_{idx:06d}"
        text = f"{prompt}\n{response}"
        out.append(
            {
                "id": row_id,
                "dataset": dataset_id,
                "task": task,
                "split": split,
                "prompt": prompt,
                "response": response,
                "text": text,
                "messages": build_messages(system_prompt, prompt, response),
                "answer_key": answer_key,
                "source_file": str(source_path),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    benchmark_root = pathlib.Path(args.benchmark_root)
    output_dir = pathlib.Path(args.output_dir)
    config_dir = benchmark_root / "configs" / "vi"
    dataset_dir = benchmark_root / "datasets"

    dataset_info = read_json(config_dir / "dataset_info.json")
    prompt_config = read_json(config_dir / "prompt_template.json")["PROMPT_TEMPLATE"]
    selected = set(args.only_dataset)

    merged_train: list[dict[str, Any]] = []
    merged_test: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for dataset_id in sorted(dataset_info):
        if selected and dataset_id not in selected:
            continue
        dataset_cfg = dataset_info[dataset_id]
        task = dataset_cfg["task"]
        if task not in prompt_config:
            manifest.append({"dataset": dataset_id, "status": "skipped", "reason": f"No prompt for task {task}"})
            continue

        local_dir = dataset_dir / dataset_id
        train_path = local_dir / f"{dataset_id}_train.json"
        test_path = local_dir / f"{dataset_id}_test.json"
        if not train_path.exists():
            manifest.append({"dataset": dataset_id, "status": "skipped", "reason": f"Missing {train_path}"})
            continue

        train_rows = build_rows(
            dataset_id=dataset_id,
            split="train",
            source_path=train_path,
            dataset_cfg=dataset_cfg,
            prompt_templates=prompt_config,
        )
        train_count = write_jsonl(output_dir / f"{dataset_id}_train_sft.jsonl", train_rows)
        merged_train.extend(train_rows)

        test_count = 0
        if args.include_test and test_path.exists():
            test_rows = build_rows(
                dataset_id=dataset_id,
                split="test",
                source_path=test_path,
                dataset_cfg=dataset_cfg,
                prompt_templates=prompt_config,
            )
            test_count = write_jsonl(output_dir / f"{dataset_id}_test_sft.jsonl", test_rows)
            merged_test.extend(test_rows)

        manifest.append(
            {
                "dataset": dataset_id,
                "task": task,
                "status": "written",
                "train_rows": train_count,
                "test_rows": test_count,
            }
        )

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(merged_train)
        rng.shuffle(merged_test)

    write_jsonl(output_dir / "merged_train.jsonl", merged_train)
    if args.include_test:
        write_jsonl(output_dir / "merged_test.jsonl", merged_test)

    metadata = {
        "format": "response-separated-sft",
        "fields": ["prompt", "response", "text", "messages"],
        "note": "Use prompt/response or messages for response-only SFT; text is provided for compatibility only.",
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "include_test": args.include_test,
        "train_rows": len(merged_train),
        "test_rows": len(merged_test),
        "datasets": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote SFT data to: {output_dir}")
    print(f"Train rows: {len(merged_train)}")
    if args.include_test:
        print(f"Test rows: {len(merged_test)}")


if __name__ == "__main__":
    main()
