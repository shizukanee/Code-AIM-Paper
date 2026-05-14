#!/usr/bin/env python3
"""Build a mixed SFT-style dataset from Vietnamese benchmark tasks.

This script creates per-task train/test JSONL files and merged train/test
JSONL files (shuffled) for SFT-style finetuning.

Defaults:
- Summarization: vietnews
- QA: xquad
- Translation: opus100_envi
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    default_benchmark_root = root.parent / "Vietnamese benchmark"
    default_output = root / "data" / "vi_ft" / "benchmark_mix"

    parser = argparse.ArgumentParser(description="Build merged SFT dataset from benchmark tasks")
    parser.add_argument("--benchmark-root", default=str(default_benchmark_root))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of merged datasets")

    parser.add_argument("--summarization-dataset", default="vietnews")
    parser.add_argument("--qa-dataset", default="xquad")
    parser.add_argument("--translation-dataset", default="opus100_envi")

    parser.add_argument(
        "--summarization-prompt",
        default=(
            "### Instruction:\n"
            "Summarize the following text.\n\n"
            "### Input:\n"
            "{source}\n\n"
            "### Response:\n"
            "{target}"
        ),
    )
    parser.add_argument(
        "--qa-prompt",
        default=(
            "Read the passage below and answer the question with a short phrase or span "
            "taken directly from the passage. Do not include any other text.\n\n"
            "Passage: {context}\n\n"
            "Question: {question}\n\n"
            "Answer: {answer}"
        ),
    )
    parser.add_argument(
        "--translation-prompt",
        default=(
            "### Instruction:\n"
            "Translate the following text into Vietnamese.\n\n"
            "### Input:\n"
            "{source}\n\n"
            "### Response:\n"
            "{target}"
        ),
    )

    return parser.parse_args()


def read_json_list(path: pathlib.Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def render_prompt(template: str, **fields: str) -> str:
    try:
        return template.format(**fields)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Prompt template is missing placeholder: {missing}") from exc


def pick_first_answer(answers: Any) -> str:
    if answers is None:
        return ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if isinstance(texts, list) and texts:
            return normalize_text(texts[0])
        if isinstance(texts, str):
            return texts
    if isinstance(answers, list) and answers:
        return normalize_text(answers[0])
    if isinstance(answers, str):
        return answers
    return ""


def build_summarization(rows: List[Dict[str, Any]], template: str, dataset_name: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        source = normalize_text(row.get("source"))
        target = normalize_text(row.get("target"))
        if not source or not target:
            continue
        text = render_prompt(template, source=source, target=target)
        out.append(
            {
                "text": text,
                "task": "summarization",
                "dataset": dataset_name,
            }
        )
    return out


def build_qa(rows: List[Dict[str, Any]], template: str, dataset_name: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        context = normalize_text(row.get("context"))
        question = normalize_text(row.get("question") or row.get("query"))
        answer = pick_first_answer(row.get("answers") or row.get("answer"))
        if not context or not question or not answer:
            continue
        text = render_prompt(template, context=context, question=question, answer=answer)
        out.append(
            {
                "text": text,
                "task": "question_answering",
                "dataset": dataset_name,
            }
        )
    return out


def build_translation(rows: List[Dict[str, Any]], template: str, dataset_name: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        source = normalize_text(row.get("source"))
        target = normalize_text(row.get("target"))
        if not source or not target:
            continue
        text = render_prompt(template, source=source, target=target)
        out.append(
            {
                "text": text,
                "task": "translation",
                "dataset": dataset_name,
            }
        )
    return out


def dataset_paths(benchmark_root: pathlib.Path, name: str) -> tuple[pathlib.Path, pathlib.Path]:
    dataset_dir = benchmark_root / "datasets" / name
    train_path = dataset_dir / f"{name}_train.json"
    test_path = dataset_dir / f"{name}_test.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test files for dataset: {name}")
    return train_path, test_path


def main() -> None:
    args = parse_args()
    benchmark_root = pathlib.Path(args.benchmark_root)
    output_dir = pathlib.Path(args.output_dir)

    sum_train_path, sum_test_path = dataset_paths(benchmark_root, args.summarization_dataset)
    qa_train_path, qa_test_path = dataset_paths(benchmark_root, args.qa_dataset)
    tr_train_path, tr_test_path = dataset_paths(benchmark_root, args.translation_dataset)

    sum_train = build_summarization(read_json_list(sum_train_path), args.summarization_prompt, args.summarization_dataset)
    sum_test = build_summarization(read_json_list(sum_test_path), args.summarization_prompt, args.summarization_dataset)

    qa_train = build_qa(read_json_list(qa_train_path), args.qa_prompt, args.qa_dataset)
    qa_test = build_qa(read_json_list(qa_test_path), args.qa_prompt, args.qa_dataset)

    tr_train = build_translation(read_json_list(tr_train_path), args.translation_prompt, args.translation_dataset)
    tr_test = build_translation(read_json_list(tr_test_path), args.translation_prompt, args.translation_dataset)

    write_jsonl(output_dir / f"summarization_{args.summarization_dataset}_train.jsonl", sum_train)
    write_jsonl(output_dir / f"summarization_{args.summarization_dataset}_test.jsonl", sum_test)
    write_jsonl(output_dir / f"qa_{args.qa_dataset}_train.jsonl", qa_train)
    write_jsonl(output_dir / f"qa_{args.qa_dataset}_test.jsonl", qa_test)
    write_jsonl(output_dir / f"translation_{args.translation_dataset}_train.jsonl", tr_train)
    write_jsonl(output_dir / f"translation_{args.translation_dataset}_test.jsonl", tr_test)

    merged_train = sum_train + qa_train + tr_train
    merged_test = sum_test + qa_test + tr_test

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(merged_train)
        rng.shuffle(merged_test)

    write_jsonl(output_dir / "merged_train.jsonl", merged_train)
    write_jsonl(output_dir / "merged_test.jsonl", merged_test)

    prompt_manifest = {
        "summarization_dataset": args.summarization_dataset,
        "qa_dataset": args.qa_dataset,
        "translation_dataset": args.translation_dataset,
        "summarization_prompt": args.summarization_prompt,
        "qa_prompt": args.qa_prompt,
        "translation_prompt": args.translation_prompt,
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
    }
    write_jsonl(output_dir / "prompt_manifest.jsonl", [prompt_manifest])

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
