"""Dataset loading and leakage-resistant split preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .preprocessing import normalize_urdu

LABEL_MAP = {"P": "positive", "N": "negative", "O": "neutral"}
VALID_LABELS = frozenset(LABEL_MAP.values())


def load_tsv(path: Path) -> list[dict[str, str]]:
    """Load a TSV with columns Tweet/Class or text/label; reject malformed rows."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Dataset must have a header row")
        fields = {name.strip().casefold(): name for name in reader.fieldnames}
        text_col = fields.get("tweet") or fields.get("text")
        label_col = fields.get("class") or fields.get("label") or fields.get("sentiment")
        if text_col is None or label_col is None:
            raise ValueError("Expected TSV columns Tweet/Class or text/label")
        for line, raw in enumerate(reader, start=2):
            text = normalize_urdu(raw.get(text_col) or "")
            label = (raw.get(label_col) or "").strip()
            label = LABEL_MAP.get(label.upper(), label.casefold())
            if not text or label not in VALID_LABELS:
                raise ValueError(f"Invalid or empty record at TSV line {line}")
            rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError("Dataset contains no records")
    return rows


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_rows(
    rows: Iterable[dict[str, str]], *, seed: int = 42, train: float = 0.8, validation: float = 0.1
) -> dict[str, list[dict[str, str]]]:
    """Make deterministic approximate stratified splits, keeping duplicate text in one split.

    Duplicate normalized texts are grouped and must have a single label. Group sizes can make
    exact class proportions impossible; all source records still occur in exactly one split.
    """
    records = list(rows)
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than 1")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        groups[_digest(normalize_urdu(row["text"]))].append(row)
    by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups.values():
        labels = {r["label"] for r in group}
        if len(labels) != 1:
            raise ValueError("Duplicate normalized text has conflicting labels; resolve before splitting")
        by_label[next(iter(labels))].append(group)

    rng = random.Random(seed)
    result = {name: [] for name in ("train", "validation", "test")}
    for label, label_groups in sorted(by_label.items()):
        rng.shuffle(label_groups)
        # Round at the group level and preserve at least a training example when possible.
        n = len(label_groups)
        n_train = max(1, int(n * train)) if n else 0
        n_val = int(n * validation)
        if n >= 3:
            n_val = max(1, n_val)
            n_train = min(n_train, n - n_val - 1)
        else:
            n_val = min(n_val, max(0, n - n_train))
        assignment = [("train", g) for g in label_groups[:n_train]]
        assignment += [("validation", g) for g in label_groups[n_train : n_train + n_val]]
        assignment += [("test", g) for g in label_groups[n_train + n_val :]]
        for split, group in assignment:
            result[split].extend(group)
    for split_rows_ in result.values():
        rng.shuffle(split_rows_)
    return result


def write_splits(splits: dict[str, list[dict[str, str]]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = directory / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
