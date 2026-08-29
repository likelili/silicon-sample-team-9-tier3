"""Reproduce the submitted Tier-2 and Tier-3 tables from public artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import authors
from .outcomes import TARGETS
from .production import aggregate
from .published import FIT_PARAMS, SPEC


def _matrix(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    id_col = "base_pid"
    columns = [x for x in fields if x != id_col]
    ids = [r[id_col] for r in rows]
    values = np.array([[float(r[c]) if r[c] else np.nan for c in columns]
                       for r in rows], dtype=float)
    return ids, values, columns


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compare(expected: list[dict], actual: list[dict], keys: list[str], value: str) -> float:
    e = {tuple(r[k] for k in keys): float(r[value]) for r in expected}
    a = {tuple(r[k] for k in keys): float(r[value]) for r in actual}
    if set(e) != set(a):
        raise SystemExit(f"coverage differs for {value}: expected {len(e)}, got {len(a)}")
    return max(abs(e[k] - a[k]) for k in e)


def run(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[2]
    artifacts = root / "artifacts"
    pids, raw, target_columns = _matrix(artifacts / "silicon_targets_raw.csv")
    human_ids, Y_full, human_columns = _matrix(artifacts / "human_wave4_anchor_matrix.csv")
    twin_ids, T_full, twin_columns = _matrix(artifacts / "twin_wave4_anchor_matrix.csv")
    if human_columns != twin_columns:
        raise SystemExit("human and synthetic anchor columns differ")
    with open(artifacts / "reference_column_map.json", encoding="utf-8") as handle:
        anchor_columns = list(json.load(handle)["column_map"])
    column_index = {name: i for i, name in enumerate(human_columns)}
    if any(name not in column_index for name in anchor_columns):
        raise SystemExit("a verified anchor is absent from the public matrices")
    selected = [column_index[name] for name in anchor_columns]
    Y_all = Y_full[:, selected]
    T_all = T_full[:, selected]
    human_index = {pid: i for i, pid in enumerate(human_ids)}
    twin_index = {pid: i for i, pid in enumerate(twin_ids)}
    if any(pid not in human_index or pid not in twin_index for pid in pids):
        raise SystemExit("a Silicon target respondent is absent from an anchor matrix")
    Y = Y_all[[human_index[pid] for pid in pids], :]
    T = T_all[[twin_index[pid] for pid in pids], :]
    expected_targets = [f"{c}||{o}" for c, o in TARGETS]
    if target_columns != expected_targets:
        raise SystemExit("Silicon target order differs from the frozen specification")

    calibrated = np.empty_like(raw)
    cache: dict = {}
    for j, (condition, outcome) in enumerate(TARGETS):
        real = np.column_stack([Y, np.full(len(pids), np.nan)])
        synthetic = np.column_stack([T, raw[:, j]])
        sc = authors.build(real, synthetic, name=f"reproduce::{condition}::{outcome}",
                           imputation_rank=SPEC["imputation_rank"],
                           min_col_std=SPEC["min_col_std"])
        pred, _, _ = authors.predict_column(
            sc, len(anchor_columns), method=SPEC["method"], fit_finite_only=True,
            imputed_cache=cache, **FIT_PARAMS)
        calibrated[:, j] = pred

    _, archived, archived_columns = _matrix(artifacts / "silicon_targets_calibrated.csv")
    matrix_error = float(np.max(np.abs(calibrated - archived)))
    if archived_columns != target_columns or matrix_error > 1e-10:
        raise SystemExit(f"calibrated matrix differs: max error {matrix_error}")

    main, moderator, tier3, _, _ = aggregate(pids, calibrated)
    t2_main = root / "predictions" / "team_9_T2_secondary-1_v1_cells_main.csv"
    t2_mod = root / "predictions" / "team_9_T2_secondary-1_v1_cells_moderator.csv"
    t3_file = root / "predictions" / "team_9_T3_secondary-1_v1.csv"
    errors = {"calibrated_matrix": matrix_error}
    if t2_main.exists():
        errors["tier2_main"] = _compare(_rows(t2_main), main,
                                        ["condition", "outcome"], "mean")
        errors["tier2_moderator"] = _compare(
            _rows(t2_mod), moderator,
            ["condition", "moderator", "moderator_level", "outcome"], "mean")
    if t3_file.exists():
        errors["tier3"] = _compare(_rows(t3_file), tier3,
                                   ["condition", "outcome"], "ate")
    if any(not math.isfinite(v) or v > 1e-10 for v in errors.values()):
        raise SystemExit(f"reproduction failed: {errors}")
    return {"status": "pass", "max_absolute_errors": errors}


if __name__ == "__main__":
    print(run())
