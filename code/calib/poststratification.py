"""Fixed 40-cell poststratification for Tier-2 and Tier-3 aggregation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter

import numpy as np

from .common import BENCH_ROOT, demographics


QUOTA_RACE_MAP = {
    "Asian / Asian American": "Asian / Asian American",
    "Black / African American": "Black / African American",
    "Hispanic / Latino": "Hispanic / Latino",
    "Other": "Other",
    "White (non-Hispanic)": "White / Caucasian",
}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_weights(pool: list[str]) -> tuple[np.ndarray, dict[str, dict], dict, list[dict]]:
    """Return one fixed poststratification weight per clean-pool respondent.

    The target distribution is the recovered gender x age x race table. Every
    respondent in a cell receives target_cell_share / observed_cell_share.
    The same respondent weight is then reused for all conditions and outcomes.
    """
    quota_path = BENCH_ROOT / "artifacts" / "quota_joint_gender_age_race.csv"
    if not quota_path.exists():
        raise SystemExit(f"poststratification: missing target grid {quota_path}")

    target: dict[tuple[str, str, str], float] = {}
    with open(quota_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            race = QUOTA_RACE_MAP.get(row["race"].strip())
            if race is None:
                raise SystemExit(f"poststratification: unmapped quota race {row['race']!r}")
            key = (row["gender"].strip(), row["age_band"].strip(), race)
            if key in target:
                raise SystemExit(f"poststratification: duplicate target cell {key}")
            target[key] = float(row["proportion"])
    if len(target) != 40:
        raise SystemExit(f"poststratification: expected 40 target cells, got {len(target)}")
    target_total = sum(target.values())
    if not np.isfinite(target_total) or abs(target_total - 1.0) > 1e-3:
        raise SystemExit(f"poststratification: target shares sum to {target_total}")
    target = {key: value / target_total for key, value in target.items()}

    demo = demographics(pool)
    if set(demo) != set(pool):
        missing = sorted(set(pool) - set(demo))
        raise SystemExit(f"poststratification: missing demographics for {missing[:5]}")
    person_cells = [(demo[pid]["gender"], demo[pid]["age_band"], demo[pid]["race"])
                    for pid in pool]
    counts = Counter(person_cells)
    unknown = sorted(set(counts) - set(target))
    absent = sorted(key for key in target if counts[key] == 0)
    if unknown:
        raise SystemExit(f"poststratification: clean-pool cells absent from target grid: {unknown}")
    if absent:
        raise SystemExit(f"poststratification: target cells have no clean-pool support: {absent}")

    n = len(pool)
    cell_weight = {key: n * share / counts[key] for key, share in target.items()}
    weights = np.array([cell_weight[key] for key in person_cells], dtype=float)
    weights *= n / float(weights.sum())
    total_weight = float(weights.sum())
    ess = total_weight ** 2 / float(np.square(weights).sum())

    cells = []
    max_error = 0.0
    for key in sorted(target):
        indices = [i for i, value in enumerate(person_cells) if value == key]
        realized = float(weights[indices].sum() / total_weight)
        error = realized - target[key]
        max_error = max(max_error, abs(error))
        cells.append({
            "gender": key[0], "age_band": key[1], "race": key[2],
            "target_proportion": target[key], "pool_n": counts[key],
            "pool_proportion": counts[key] / n,
            "per_person_weight": float(weights[indices[0]]),
            "realized_weighted_proportion": realized,
            "difference": error,
        })
    if max_error > 1e-12:
        raise SystemExit(f"poststratification: target grid not recovered ({max_error})")

    weight_rows = []
    for pid, key, weight in zip(pool, person_cells, weights):
        weight_rows.append({
            "base_pid": pid, "gender": key[0], "age_band": key[1], "race": key[2],
            "target_cell_proportion": format(target[key], ".15g"),
            "clean_pool_cell_n": counts[key],
            "poststratification_weight": format(float(weight), ".15g"),
        })

    audit = {
        "source": str(quota_path.relative_to(BENCH_ROOT)),
        "source_sha256": _sha256(quota_path),
        "formula": "target cell proportion / clean-pool cell proportion",
        "dimensions": ["gender", "age_band", "race"],
        "target_cells": len(target), "pool_n": n,
        "weight_sum": total_weight, "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "effective_sample_size": ess,
        "max_absolute_target_cell_error": max_error,
        "same_weights_for_all_conditions_and_outcomes": True,
        "cells": cells,
    }
    return weights, demo, audit, weight_rows


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) != len(weights) or not len(values):
        raise SystemExit("poststratification: invalid weighted-mean inputs")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise SystemExit("poststratification: nonfinite values or invalid weights")
    return float(np.average(values, weights=weights))
