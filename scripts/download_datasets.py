#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
CONFIG_PATH = PROJECT_ROOT / "configs" / "vi" / "dataset_info.json"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "datasets_manifest.json"

MLQA_MIRROR_REPO = "phamvlap/vi-mlqa-v1.0"
WIKILINGUA_MIRROR_REPO = "Linhz/Wikilingua_VN"
VIETNAMESE_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def take_records(dataset: Dataset, limit: int = 0) -> list[dict[str, Any]]:
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def write_split_dataset(name: str, train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> None:
    out_dir = DATASET_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_train.json").write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{name}_test.json").write_text(
        json.dumps(test_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def sample_support(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return records
    return records[:limit]


def resolve_first(row: dict[str, Any], candidates: list[str], required: bool = True) -> Any:
    for key in candidates:
        if key in row:
            return row[key]
    if required:
        raise KeyError(f"Missing required columns. Available keys: {sorted(row.keys())}")
    return None


def update_labels(dataset_id: str, labels: list[str]) -> None:
    config = read_json(CONFIG_PATH)
    if dataset_id in config:
        config[dataset_id]["label"] = labels
        write_json(CONFIG_PATH, config)


def parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return value


def to_squad_answers(value: Any) -> dict[str, list[Any]]:
    value = parse_json_like(value)

    if isinstance(value, dict) and "text" in value:
        texts = value["text"]
        answer_starts = value.get("answer_start", [])
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(answer_starts, list):
            answer_starts = [answer_starts]
        if len(answer_starts) < len(texts):
            answer_starts = answer_starts + [-1] * (len(texts) - len(answer_starts))
        return {
            "text": [str(x) for x in texts],
            "answer_start": [int(x) if str(x).lstrip("-").isdigit() else -1 for x in answer_starts[: len(texts)]],
        }

    if isinstance(value, list):
        texts = []
        answer_starts = []
        for item in value:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
                raw_start = item.get("answer_start", -1)
                answer_starts.append(int(raw_start) if str(raw_start).lstrip("-").isdigit() else -1)
            else:
                texts.append(str(item))
                answer_starts.append(-1)
        return {"text": texts, "answer_start": answer_starts}

    return {"text": [str(value)], "answer_start": [-1]}


def to_string_list(value: Any) -> list[str]:
    value = parse_json_like(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def normalize_ir_passages(value: Any) -> list[dict[str, str]]:
    value = parse_json_like(value)

    if isinstance(value, dict):
        ids = value.get("id", value.get("ids", []))
        passages = value.get("passage", value.get("text", value.get("passages", [])))
        if isinstance(ids, list) and isinstance(passages, list):
            return [
                {"id": str(pid), "passage": str(ptext)}
                for pid, ptext in zip(ids, passages)
            ]

    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                pid = resolve_first(item, ["id", "passage_id"], required=False)
                ptext = resolve_first(item, ["passage", "text"], required=False)
                if pid is not None and ptext is not None:
                    normalized.append({"id": str(pid), "passage": str(ptext)})
        return normalized

    return []


def find_first_dataset_split(ds: Any, preferred: list[str]) -> Any:
    for key in preferred:
        if key in ds:
            return ds[key]
    first_key = next(iter(ds.keys()))
    return ds[first_key]


def build_xquad(support_limit: int, _test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset("google/xquad", "xquad.vi", split="validation")
    records = [{"context": r["context"], "question": r["question"], "answers": r["answers"]} for r in take_records(ds)]
    return sample_support(records, support_limit), records, None


def build_mlqa(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(MLQA_MIRROR_REPO)
    train_ds = find_first_dataset_split(ds, ["validation", "train"])
    test_ds = find_first_dataset_split(ds, ["test", "validation"])

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "context": resolve_first(row, ["context"]),
            "question": resolve_first(row, ["question"]),
            "answers": to_squad_answers(resolve_first(row, ["answers", "answer"])),
        }

    train_records = [convert(r) for r in take_records(train_ds, support_limit)]
    test_records = [convert(r) for r in take_records(test_ds, test_limit)]
    return train_records, test_records, None


def build_mlqa_mlm(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(MLQA_MIRROR_REPO)
    train_ds = find_first_dataset_split(ds, ["validation", "train"])
    test_ds = find_first_dataset_split(ds, ["test", "validation"])

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        answers = to_squad_answers(resolve_first(row, ["answers", "answer"]))
        answer = answers["text"][0] if answers["text"] else ""
        source = (
            "Context:\n"
            f"{resolve_first(row, ['context'])}\n\n"
            "Question:\n"
            f"{resolve_first(row, ['question'])}\n\n"
            "Masked answer: [MASKED]\n"
            "Fill the masked answer exactly."
        )
        return {"source": source, "target": answer}

    return (
        [convert(r) for r in take_records(train_ds, support_limit)],
        [convert(r) for r in take_records(test_ds, test_limit)],
        None,
    )


def build_vietnews(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset("Yuhthe/vietnews")
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        source = resolve_first(row, ["article", "document", "text"])
        target = resolve_first(row, ["abstract", "summary", "title"])
        return {"source": source, "target": target}

    return (
        [convert(r) for r in take_records(train_split, support_limit)],
        [convert(r) for r in take_records(test_split, test_limit)],
        None,
    )


def build_wikilingua(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(WIKILINGUA_MIRROR_REPO)
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        source = resolve_first(row, ["source", "document", "article", "text"])
        target = resolve_first(row, ["summary", "target", "abstract"])
        return {"source": source, "target": target}

    return (
        [convert(r) for r in take_records(train_split, support_limit)],
        [convert(r) for r in take_records(test_split, test_limit)],
        None,
    )


def build_binary_or_multiclass_text(
    repo_id: str,
    support_limit: int,
    test_limit: int,
    query_keys: list[str],
    answer_keys: list[str],
    answer_map: Callable[[Any], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(repo_id)
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        query = resolve_first(row, query_keys)
        answer = answer_map(resolve_first(row, answer_keys))
        return {"query": query, "answer": answer}

    train_records = [convert(r) for r in take_records(train_split, support_limit)]
    test_records = [convert(r) for r in take_records(test_split, test_limit)]
    labels = sorted({r["answer"] for r in train_records + test_records})
    return train_records, test_records, labels


def build_zalo(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset("ura-hcmut/zalo_e2eqa")
    train_split = ds["train"]
    test_split = ds["test"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        answer = row.get("answers", row.get("answer"))
        if isinstance(answer, list):
            answer = answer[0]
        return {"question": row["question"], "answer": str(answer)}

    return (
        [convert(r) for r in take_records(train_split, support_limit)],
        [convert(r) for r in take_records(test_split, test_limit)],
        None,
    )


def build_vimmrc(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset("ura-hcmut/ViMMRC")
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "context": row["article"],
            "question": row["question"],
            "options": json.dumps(row["options"], ensure_ascii=False),
            "answer": str(row["answer"]).strip(),
        }

    return (
        [convert(r) for r in take_records(train_split, support_limit)],
        [convert(r) for r in take_records(test_split, test_limit)],
        ["A", "B", "C", "D"],
    )


def build_vietnamese_math(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    def convert(row: dict[str, Any], fallback_type: str) -> dict[str, Any]:
        query = resolve_first(row, ["problem_vi", "problem", "query", "question"])
        answer = resolve_first(row, ["solution_vi", "solution", "short_solution", "answer"])
        type_id = resolve_first(row, ["type", "level", "subject"], required=False) or fallback_type
        return {
            "type_id": str(type_id),
            "query": str(query),
            "answer": str(answer),
        }

    for subset in VIETNAMESE_MATH_CONFIGS:
        ds = load_dataset("ura-hcmut/Vietnamese-MATH", subset)
        train_split = find_first_dataset_split(ds, ["train", "validation", "test"])
        test_split = find_first_dataset_split(ds, ["test", "validation", "train"])

        train_records.extend([convert(r, subset) for r in take_records(train_split)])
        test_records.extend([convert(r, subset) for r in take_records(test_split)])

    if test_limit > 0:
        test_records = test_records[:test_limit]

    return (
        sample_support(train_records, support_limit),
        test_records,
        None,
    )


def build_translation(repo_id: str, subset: str | None, support_limit: int, test_limit: int, source_key: str, target_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(repo_id, subset) if subset else load_dataset(repo_id)
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        if "translation" in row and isinstance(row["translation"], dict):
            source = row["translation"][source_key]
            target = row["translation"][target_key]
        else:
            source = row[source_key]
            target = row[target_key]
        return {"source": source, "target": target}

    return (
        [convert(r) for r in take_records(train_split, support_limit)],
        [convert(r) for r in take_records(test_split, test_limit)],
        None,
    )


def build_vsec(support_limit: int, test_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset("ura-hcmut/VSEC")
    test_split = find_first_dataset_split(ds, ["test", "validation", "train"])

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        source = resolve_first(row, ["text", "source", "incorrect"])
        target = resolve_first(row, ["correct", "target", "label"])
        return {"source": source, "target": target}

    test_records = [convert(r) for r in take_records(test_split, test_limit)]
    return (
        sample_support(test_records, support_limit),
        test_records,
        None,
    )


def build_ir_snapshot(repo_id: str, stem: str, local_name: str, support_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    local_repo = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.json", "*.jsonl", "*.csv", "*.txt"],
        )
    )

    candidates = list(local_repo.rglob(f"*{stem}*test*.json"))
    candidates += list(local_repo.rglob(f"*{stem}*test*.jsonl"))
    candidates += list(local_repo.rglob(f"*{stem}*test*.csv"))
    candidates += list(local_repo.rglob(f"*{stem}*.jsonl"))
    candidates += list(local_repo.rglob("*runs*.jsonl"))
    candidates += list(local_repo.rglob(f"*{stem}*.json"))
    candidates += list(local_repo.rglob(f"*{stem}*.csv"))

    if not candidates:
        raise FileNotFoundError(f"Cannot find a usable IR split file in {local_repo}")

    test_path = sorted(candidates)[0]
    suffix = test_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        test_ds = load_dataset("json", data_files={"test": str(test_path)}, split="test")
    elif suffix == ".csv":
        test_ds = load_dataset("csv", data_files={"test": str(test_path)}, split="test")
    else:
        raise ValueError(f"Unsupported IR file format for {test_path}")

    test_records = []
    support_records = []
    for row in take_records(test_ds):
        query_id = str(resolve_first(row, ["id", "query_id"]))
        query = resolve_first(row, ["query", "question"])
        passages = normalize_ir_passages(resolve_first(row, ["passages"]))
        references = to_string_list(resolve_first(row, ["references"]))
        test_records.append(
            {
                "id": query_id,
                "query": query,
                "passages": {
                    "id": [p["id"] for p in passages],
                    "passage": [p["passage"] for p in passages],
                },
                "references": references
            }
        )

        if len(support_records) < support_limit:
            ref_set = {str(x) for x in references}
            for passage in passages[:4]:
                pid = str(resolve_first(passage, ["id", "passage_id"]))
                ptext = resolve_first(passage, ["passage", "text"])
                support_records.append(
                    {
                        "query": query,
                        "passage": ptext,
                        "answer": "Yes" if pid in ref_set else "No"
                    }
                )
                if len(support_records) >= support_limit:
                    break

    return support_records, test_records, ["Yes", "No"]


def build_synthetic_reasoning(repo_id: str, support_limit: int, test_limit: int, query_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]:
    ds = load_dataset(repo_id)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    full_split = ds[split_name]

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": resolve_first(row, [query_key, "source", "problem"]),
            "answer": resolve_first(row, ["target", "answer", "response"])
        }

    records = [convert(r) for r in take_records(full_split, test_limit)]
    return sample_support(records, support_limit), records, None


BUILDERS: dict[str, Callable[[int, int], tuple[list[dict[str, Any]], list[dict[str, Any]], list[str] | None]]] = {
    "xquad": build_xquad,
    "mlqa": build_mlqa,
    "mlqa_mlm": build_mlqa_mlm,
    "vietnews": build_vietnews,
    "wikilingua": build_wikilingua,
    "vlsp2016": lambda s, t: build_binary_or_multiclass_text(
        "ura-hcmut/vlsp2016", s, t, ["Data", "text", "Sentence"], ["Class", "label", "Sentiment"], normalize_text_label
    ),
    "uit_vsfc": lambda s, t: build_binary_or_multiclass_text(
        "ura-hcmut/UIT-VSFC", s, t, ["Sentence", "sentence", "text"], ["Sentiment", "sentiment", "label"], normalize_text_label
    ),
    "uit_vsmec": lambda s, t: build_binary_or_multiclass_text(
        "ura-hcmut/UIT-VSMEC", s, t, ["Sentence", "sentence", "text"], ["Emotion", "Label", "label"], normalize_text_label
    ),
    "phoatis": lambda s, t: build_binary_or_multiclass_text(
        "ura-hcmut/PhoATIS", s, t, ["text", "Text", "sentence"], ["label", "intent", "Label"], normalize_text_label
    ),
    "zalo_e2eqa": build_zalo,
    "uit_vimmrc": build_vimmrc,
    "uit_victsd": lambda s, t: build_binary_or_multiclass_text(
        "tarudesu/ViCTSD", s, t, ["Comment", "comment", "text"], ["Toxicity"], lambda x: "toxic" if int(x) == 1 else "non_toxic"
    ),
    "uit_vihsd": lambda s, t: build_binary_or_multiclass_text(
        "ura-hcmut/UIT-ViHSD", s, t, ["free_text", "text", "comment"], ["label", "label_id"], lambda x: {0: "clean", 1: "offensive", 2: "hate"}.get(int(x), normalize_text_label(x))
    ),
    "mmarco_vi": lambda s, _t: build_ir_snapshot("ura-hcmut/Information_Retrieval", "mmarco", "mmarco_vi", s),
    "mrobust04_vi": lambda s, _t: build_ir_snapshot("ura-hcmut/mrobust", "mrobust", "mrobust04_vi", s),
    "vsec": build_vsec,
    "synthetic_reasoning_natural": lambda s, t: build_synthetic_reasoning("ura-hcmut/synthetic_reasoning_natural", s, t, "problem"),
    "synthetic_reasoning_abstract": lambda s, t: build_synthetic_reasoning("ura-hcmut/synthetic_reasoning", s, t, "source"),
    "vietnamese_math": build_vietnamese_math,
    "opus100_envi": lambda s, t: build_translation("vietgpt/opus100_envi", None, s, t, "en", "vi"),
    "phomt_envi": lambda s, t: build_translation("ura-hcmut/PhoMT", None, s, t, "en", "vi"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize Vietnamese benchmark datasets")
    parser.add_argument("--only", action="append", default=[], help="Dataset id(s) to download")
    parser.add_argument("--support-train-size", type=int, default=256, help="Max support examples to store in the local train split")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional cap on benchmark test examples; 0 means full split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(MANIFEST_PATH)
    selected_ids = args.only or [item["id"] for item in manifest["datasets"]]
    report: dict[str, Any] = {"downloaded": [], "failed": []}

    for dataset_id in selected_ids:
        builder = BUILDERS.get(dataset_id)
        if builder is None:
            report["failed"].append({"id": dataset_id, "error": "No builder registered"})
            continue

        try:
            train_records, test_records, labels = builder(args.support_train_size, args.test_limit)
            write_split_dataset(dataset_id, train_records, test_records)
            if labels:
                update_labels(dataset_id, labels)
            report["downloaded"].append(
                {
                    "id": dataset_id,
                    "train_examples": len(train_records),
                    "test_examples": len(test_records),
                    "labels": labels or []
                }
            )
        except Exception as exc:
            report["failed"].append({"id": dataset_id, "error": str(exc)})

    write_json(PROJECT_ROOT / "results" / "dataset_download_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
