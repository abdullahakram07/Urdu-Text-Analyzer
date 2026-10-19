"""Command line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_tsv, split_rows, write_splits
from .model import (discover_topics, evaluate_model, load_model, predict, save_json,
                    save_model, train_model)


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(prog="urdu-analyzer", description="Urdu sentiment and topic baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create leakage-resistant train/validation/test splits")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/splits"))
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Train from prepared split JSONL files")
    train.add_argument("--splits", type=Path, default=Path("data/splits"))
    train.add_argument("--model", type=Path, default=Path("models/sentiment.joblib"))
    train.add_argument("--report", type=Path, default=Path("reports/evaluation.json"))
    train.add_argument("--topics", type=Path, default=Path("models/topics.json"))
    train.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="Predict sentiment for an Urdu string")
    infer.add_argument("text")
    infer.add_argument("--model", type=Path, default=Path("models/sentiment.joblib"))
    infer.add_argument("--topics", type=Path, default=Path("models/topics.json"))
    args = parser.parse_args()

    if args.command == "prepare":
        rows = load_tsv(args.dataset)
        splits = split_rows(rows, seed=args.seed)
        write_splits(splits, args.output)
        overlap = ""
        split_hashes = {
            name: {hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                   for row in split_rows_}
            for name, split_rows_ in splits.items()
        }
        names = list(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if split_hashes[left] & split_hashes[right]:
                    overlap = f"Duplicate leakage detected between {left} and {right}"
        if overlap:
            raise RuntimeError(overlap)
        print(json.dumps({name: len(rows_) for name, rows_ in splits.items()}, ensure_ascii=False))
    elif args.command == "train":
        splits = {name: _read_jsonl(args.splits / f"{name}.jsonl")
                  for name in ("train", "validation", "test")}
        if not splits["validation"]:
            raise SystemExit("Validation split is empty; add more labeled data before training.")
        model = train_model(splits["train"], seed=args.seed)
        report = evaluate_model(model, splits["test"])
        topics = discover_topics([row["text"] for row in splits["train"]])
        save_model(model, args.model)
        save_json(report, args.report)
        save_json(topics, args.topics)
        print(f"Saved model to {args.model} and evaluation to {args.report}")
    else:
        model = load_model(args.model)
        topics = json.loads(args.topics.read_text(encoding="utf-8")) if args.topics.exists() else []
        print(json.dumps(predict(model, args.text, topics), ensure_ascii=False, indent=2))
