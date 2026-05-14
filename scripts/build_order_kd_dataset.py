#!/usr/bin/env python3
"""Build order-KD dataset from UD CoNLL-U files.

This script computes statistical teacher targets q_vi(t) from dependency triples:
t = (UPOS_dep, UPOS_head, deprel)

Then writes sentence-level training items with token-level dependency arcs:
{
  "sent_id": "...",
  "text": "...",
  "tokens": [...],
  "arcs": [
    {
      "dep_word": int,
      "head_word": int,
      "dep_upos": "...",
      "head_upos": "...",
      "deprel": "...",
      "q_teacher": float,
      "label_left": 0|1,
      "triple_total": int
    }
  ]
}

Important: token boundaries are taken from UD token rows (FORM column),
not naive whitespace splitting.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple


Triple = Tuple[str, str, str]


@dataclass
class Token:
    tid: int
    form: str
    upos: str
    head: int
    deprel: str


def parse_conllu(paths: Sequence[str]) -> Iterator[List[Token]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            buf: List[str] = []
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    if buf:
                        sent = parse_sentence(buf)
                        if sent:
                            yield sent
                        buf = []
                else:
                    buf.append(line)
            if buf:
                sent = parse_sentence(buf)
                if sent:
                    yield sent


def parse_sentence(lines: Sequence[str]) -> List[Token]:
    out: List[Token] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            continue

        tid_raw = cols[0]
        if "-" in tid_raw or "." in tid_raw:
            continue

        head_raw = cols[6]
        if head_raw == "_":
            continue

        try:
            tid = int(tid_raw)
            head = int(head_raw)
        except ValueError:
            continue

        out.append(
            Token(
                tid=tid,
                form=cols[1],
                upos=cols[3],
                head=head,
                deprel=cols[7].split(":", 1)[0],
            )
        )
    return out


def collect_teacher_probs(sentences: Sequence[List[Token]]) -> Dict[Triple, Dict[str, float]]:
    counts: Dict[Triple, Dict[str, int]] = defaultdict(lambda: {"left": 0, "total": 0})

    for sent in sentences:
        tok_map = {t.tid: t for t in sent}
        for dep in sent:
            if dep.head == 0:
                continue
            head_tok = tok_map.get(dep.head)
            if head_tok is None:
                continue

            triple = (dep.upos, head_tok.upos, dep.deprel)
            counts[triple]["total"] += 1
            if dep.tid < dep.head:
                counts[triple]["left"] += 1

    probs: Dict[Triple, Dict[str, float]] = {}
    for triple, c in counts.items():
        total = c["total"]
        left = c["left"]
        probs[triple] = {
            "q_teacher": left / total if total else 0.0,
            "left": float(left),
            "total": float(total),
        }
    return probs


def build_order_kd_rows(
    sentences: Sequence[List[Token]],
    teacher: Dict[Triple, Dict[str, float]],
    min_triple_count: int,
) -> List[dict]:
    rows: List[dict] = []

    for idx, sent in enumerate(sentences):
        tok_map = {t.tid: t for t in sent}
        token_ids = sorted(tok_map)
        tokens = [tok_map[tid].form for tid in token_ids]

        # Map CoNLL-U token ids to contiguous 0-based word indices in this sentence.
        tid_to_word = {tid: i for i, tid in enumerate(token_ids)}

        arcs = []
        for dep in sent:
            if dep.head == 0:
                continue
            head_tok = tok_map.get(dep.head)
            if head_tok is None:
                continue

            triple = (dep.upos, head_tok.upos, dep.deprel)
            tmeta = teacher.get(triple)
            if tmeta is None:
                continue
            triple_total = int(tmeta["total"])
            if triple_total < min_triple_count:
                continue

            dep_word = tid_to_word.get(dep.tid)
            head_word = tid_to_word.get(dep.head)
            if dep_word is None or head_word is None:
                continue

            arcs.append(
                {
                    "dep_word": dep_word,
                    "head_word": head_word,
                    "dep_upos": dep.upos,
                    "head_upos": head_tok.upos,
                    "deprel": dep.deprel,
                    "q_teacher": float(tmeta["q_teacher"]),
                    "label_left": int(dep.tid < dep.head),
                    "triple_total": triple_total,
                }
            )

        if not arcs:
            continue

        rows.append(
            {
                "sent_id": f"vi_ud_{idx:06d}",
                "text": " ".join(tokens),
                "tokens": tokens,
                "arcs": arcs,
            }
        )

    return rows


def write_jsonl(path: pathlib.Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build order-KD dataset from UD CoNLL-U")
    parser.add_argument("--conllu-glob", required=True, help="Glob for Vietnamese .conllu files")
    parser.add_argument(
        "--output-jsonl",
        default="outputs/order_kd/vi_order_kd.jsonl",
        help="Output JSONL path for sentence-level order-KD samples.",
    )
    parser.add_argument(
        "--teacher-json",
        default="outputs/order_kd/vi_teacher_probs.json",
        help="Output JSON path for triple->q_teacher map.",
    )
    parser.add_argument(
        "--min-triple-count",
        type=int,
        default=5,
        help="Drop arcs from very rare triples below this pooled count.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.conllu_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.conllu_glob}")

    sentences = list(parse_conllu(paths))
    teacher = collect_teacher_probs(sentences)
    rows = build_order_kd_rows(sentences, teacher, args.min_triple_count)

    out_jsonl = pathlib.Path(args.output_jsonl)
    write_jsonl(out_jsonl, rows)

    teacher_out = []
    for (dep_upos, head_upos, deprel), meta in teacher.items():
        teacher_out.append(
            {
                "dep_upos": dep_upos,
                "head_upos": head_upos,
                "deprel": deprel,
                "q_teacher": meta["q_teacher"],
                "left": int(meta["left"]),
                "total": int(meta["total"]),
            }
        )
    teacher_out.sort(key=lambda r: (-r["total"], r["dep_upos"], r["head_upos"], r["deprel"]))

    teacher_path = pathlib.Path(args.teacher_json)
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    with open(teacher_path, "w", encoding="utf-8") as f:
        json.dump(teacher_out, f, ensure_ascii=False, indent=2)

    n_arcs = sum(len(r["arcs"]) for r in rows)
    print("=== Built Order-KD Dataset ===")
    print(f"Sentences: {len(rows)}")
    print(f"Arcs: {n_arcs}")
    print(f"Teacher triples: {len(teacher_out)}")
    print(f"JSONL: {out_jsonl}")
    print(f"Teacher: {teacher_path}")


if __name__ == "__main__":
    main()
