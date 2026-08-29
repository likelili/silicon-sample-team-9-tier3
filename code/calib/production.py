"""Production SYN-DIGITS calibration and Tier-2/Tier-3 aggregation.

The frozen procedure is the published Twin-2K elastic-net specification,
applied to every Silicon target with no adaptive gate.  Selection and the
untouched Wave-4 holdout are recorded separately in ``published.py``.

The 221 synthetic target columns are reconstructed from the adopted batch
outputs recorded by the simulation pipeline.  This deliberately uses the same
source-of-record and response parsing as the Tier-1 exporter; a mandatory
parity check proves that all 13 outcomes agree with the released Tier-1 file
before any calibration is fitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import authors
from .common import (CALIB_DATA, STUDY_RUN, anchor_matrices, clean_pool,
                     read_csv_dicts, save_json)
from .outcomes import (CODENAMES, CONDITIONS, MODERATORS, OUTCOMES, RENAME,
                       TARGETS, build_outcomes, to_binary)
from .published import FIT_PARAMS, SPEC
from .poststratification import build_weights, weighted_mean


def _pipeline_imports():
    """Import the Tier-1 response parser without duplicating its constants."""
    import sys

    pipeline_dir = str(Path(__file__).resolve().parents[1])
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    from sbench.tier1_export import (_qualtrics_label_map, _word_to_number)

    return _qualtrics_label_map, _word_to_number


def _answer_value(label: str, answer: dict, word_to_number) -> float:
    """Apply the Tier-1 export's value rules and return a numeric item value."""
    raw_value = str(answer.get("answer_value", "")).strip()
    raw_label = str(answer.get("answer_label", "")).strip()
    value = (raw_label or raw_value) if label == "newsletter" else (raw_value or raw_label)
    if label == "newsletter":
        return float(to_binary(value))
    if label == "donation":
        match = re.search(r"\d+", value)
        return float(match.group(0)) if match else math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        as_word = word_to_number(value)
        return float(as_word) if as_word is not None else math.nan


def load_silicon_targets(pool: list[str]) -> tuple[np.ndarray, dict]:
    """Return the clean-pool x 221 raw Silicon outcome matrix.

    Only sessions marked ``exact`` in the adopted-source ledger are accepted.
    Each recorded source JSONL is streamed once, including repair sources.
    """
    qualtrics_label_map, word_to_number = _pipeline_imports()
    qid_to_label = qualtrics_label_map()
    pool_set = set(pool)
    completeness_path = STUDY_RUN / "audit" / "batch" / "completeness_by_session.csv"
    rows = [r for r in read_csv_dicts(completeness_path)
            if r["base_pid"] in pool_set and r["status"] == "exact"]

    meta: dict[str, tuple[str, str]] = {}
    wanted: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        canonical = CODENAMES.get(row["condition"], row["condition"])
        if canonical not in CONDITIONS:
            raise SystemExit(f"production: unknown condition {row['condition']!r}")
        meta[row["run_id"]] = (row["base_pid"], canonical)
        for stage, source in (("1", row["s1_source"]), ("2", row["s2_source"])):
            if source:
                wanted[(stage, source)].add(row["run_id"])

    expected = len(pool) * len(CONDITIONS)
    if len(meta) != expected:
        raise SystemExit(f"production: expected {expected} exact sessions, found {len(meta)}")

    items_by_run: dict[str, dict[str, float]] = defaultdict(dict)
    seen_sources = []
    for (stage, source), run_ids in sorted(wanted.items()):
        directory = (STUDY_RUN / "data" / "batch" /
                     ("round1_results" if stage == "1" else "round2_results"))
        path = directory / source
        if not path.exists():
            raise SystemExit(f"production: missing adopted source {path}")
        seen_sources.append(str(path.relative_to(STUDY_RUN)))
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                run_id = obj.get("custom_id")
                if run_id not in run_ids:
                    continue
                choice = obj["response"]["body"]["choices"][0]
                parsed = json.loads(choice["message"]["content"])
                answers = parsed.get("answers") if isinstance(parsed, dict) else parsed
                for answer in answers:
                    qid = str(answer.get("question_id"))
                    label = qid_to_label.get(qid)
                    target_name = RENAME.get(label or "")
                    if target_name and target_name not in items_by_run[run_id]:
                        items_by_run[run_id][target_name] = _answer_value(
                            label, answer, word_to_number)

    outcomes: dict[tuple[str, str], dict[str, float]] = {}
    missing_runs = []
    for run_id, key in meta.items():
        items = items_by_run.get(run_id, {})
        result = build_outcomes(items)
        if any(not np.isfinite(result[o]) for o in OUTCOMES):
            missing_runs.append({"run_id": run_id, "base_pid": key[0],
                                 "condition": key[1],
                                 "missing": [o for o in OUTCOMES
                                             if not np.isfinite(result[o])]})
        outcomes[key] = result
    if missing_runs:
        save_json(missing_runs, CALIB_DATA / "production_missing_targets.json")
        raise SystemExit(f"production: {len(missing_runs)} exact sessions have missing outcomes")

    matrix = np.array(
        [[outcomes[(pid, condition)][outcome] for condition, outcome in TARGETS]
         for pid in pool], dtype=float)
    if matrix.shape != (len(pool), len(TARGETS)) or not np.isfinite(matrix).all():
        raise SystemExit(f"production: invalid raw target matrix {matrix.shape}")

    audit = {
        "n_pool": len(pool), "n_conditions": len(CONDITIONS),
        "n_outcomes": len(OUTCOMES), "n_targets": len(TARGETS),
        "n_exact_sessions": len(meta), "matrix_shape": list(matrix.shape),
        "adopted_sources": sorted(set(seen_sources)),
        "source": str(completeness_path.relative_to(STUDY_RUN)),
    }
    return matrix, audit


def tier1_parity(pool: list[str], raw: np.ndarray) -> dict:
    """Prove the reconstructed raw outcomes match the released Tier-1 rows."""
    row_index = {pid: i for i, pid in enumerate(pool)}
    target_index = {target: j for j, target in enumerate(TARGETS)}
    path = STUDY_RUN / "data" / "tier1" / "tier1_predictions.csv"
    checked = 0
    mismatches = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = re.search(r"(pid_\d+)$", row["profile_id"])
            if not match:
                raise SystemExit(f"production: cannot recover base pid from {row['profile_id']}")
            pid, condition = match.group(1), row["condition"]
            for outcome in OUTCOMES:
                expected = float(row[outcome])
                actual = float(raw[row_index[pid], target_index[(condition, outcome)]])
                checked += 1
                if not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-10):
                    mismatches.append({"profile_id": row["profile_id"],
                                       "condition": condition, "outcome": outcome,
                                       "tier1": expected, "reconstructed": actual})
    report = {"source": str(path.relative_to(STUDY_RUN)),
              "cells_checked": checked, "mismatches": len(mismatches),
              "examples": mismatches[:20]}
    if mismatches:
        raise SystemExit(f"production: Tier-1 parity failed ({len(mismatches)} cells)")
    return report


def _write_matrix(path: Path, pool: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["base_pid"] + [f"{c}||{o}" for c, o in TARGETS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for pid, row in zip(pool, matrix):
            writer.writerow([pid] + [format(float(v), ".15g") for v in row])


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(pool: list[str], calibrated: np.ndarray):
    """Build Tier 2/3 using the full clean pool and fixed 40-cell weights."""
    weights, demo, weight_audit, weight_rows = build_weights(pool)

    main = []
    moderator = []
    unsupported_cells = []
    for j, (condition, outcome) in enumerate(TARGETS):
        vals = calibrated[:, j]
        condition_mean = weighted_mean(vals, weights)
        main.append({"condition": condition, "outcome": outcome,
                     "mean": format(condition_mean, ".15g")})
        for mod, levels in MODERATORS.items():
            for level in levels:
                cell = np.array([i for i, pid in enumerate(pool)
                                 if demo[pid][mod] == level], dtype=int)
                if not len(cell):
                    # Organizer guidance: when a demographic level has no
                    # observations, repeat the condition mean as the explicit
                    # no-moderation prediction.  Twin-2K has no respondents in
                    # the benchmark's gender="Other" level; all other 26
                    # moderator levels have support in the frozen panel.
                    cell_mean = condition_mean
                    unsupported_cells.append({"condition": condition,
                                              "moderator": mod,
                                              "moderator_level": level,
                                              "outcome": outcome,
                                              "rule": "condition mean (no moderation)"})
                else:
                    cell_mean = weighted_mean(calibrated[cell, j], weights[cell])
                moderator.append({"condition": condition, "moderator": mod,
                                  "moderator_level": level, "outcome": outcome,
                                  "mean": format(cell_mean, ".15g")})

    main_lookup = {(r["condition"], r["outcome"]): float(r["mean"]) for r in main}
    tier3 = []
    for condition in CONDITIONS:
        if condition == "control":
            continue
        for outcome in OUTCOMES:
            ate = main_lookup[(condition, outcome)] - main_lookup[("control", outcome)]
            tier3.append({"condition": condition, "outcome": outcome,
                          "ate": format(ate, ".15g")})

    weight_audit["moderator_rule"] = (
        "restrict the full clean pool to the requested one-way moderator level and "
        "renormalize the same fixed respondent weights within that subset")
    return main, moderator, tier3, weight_audit, unsupported_cells, weight_rows


def run() -> dict:
    """Fit all 221 targets, aggregate, write artifacts, and enforce invariants."""
    pool = clean_pool()
    Y, T, anchor_items = anchor_matrices(pool)
    raw, source_audit = load_silicon_targets(pool)
    parity = tier1_parity(pool, raw)

    calibrated = np.empty_like(raw)
    fit_rows = []
    imputed_cache: dict = {}
    for j, (condition, outcome) in enumerate(TARGETS):
        real = np.column_stack([Y, np.full(len(pool), np.nan)])
        synthetic = np.column_stack([T, raw[:, j]])
        sc = authors.build(real, synthetic, name=f"silicon::{condition}::{outcome}",
                           imputation_rank=SPEC["imputation_rank"],
                           min_col_std=SPEC["min_col_std"])
        prediction, train_mse, info = authors.predict_column(
            sc, len(anchor_items), method=SPEC["method"], fit_finite_only=True,
            imputed_cache=imputed_cache, **FIT_PARAMS)
        if not np.isfinite(prediction).all():
            raise SystemExit(f"production: nonfinite prediction for {condition} / {outcome}")
        calibrated[:, j] = prediction
        fit_rows.append({"condition": condition, "outcome": outcome,
                         "train_mse": format(float(train_mse), ".15g"),
                         "n_fit_rows": info.get("n_fit_rows"),
                         "raw_mean": format(float(raw[:, j].mean()), ".15g"),
                         "calibrated_mean": format(float(prediction.mean()), ".15g")})

    raw_path = CALIB_DATA / "silicon_targets_raw.csv"
    cal_path = CALIB_DATA / "silicon_targets_calibrated.csv"
    fits_path = CALIB_DATA / "silicon_fit_diagnostics.csv"
    main_path = CALIB_DATA / "team_9_T2_secondary-1_v1_cells_main.csv"
    mod_path = CALIB_DATA / "team_9_T2_secondary-1_v1_cells_moderator.csv"
    t3_path = CALIB_DATA / "team_9_T3_secondary-1_v1.csv"
    weights_path = CALIB_DATA / "poststratification_weights.csv"
    weights_report_path = CALIB_DATA / "poststratification_report.json"
    _write_matrix(raw_path, pool, raw)
    _write_matrix(cal_path, pool, calibrated)
    _write_rows(fits_path, ["condition", "outcome", "train_mse", "n_fit_rows",
                             "raw_mean", "calibrated_mean"], fit_rows)
    main, moderator, tier3, weight_audit, unsupported_cells, weight_rows = aggregate(
        pool, calibrated)
    _write_rows(main_path, ["condition", "outcome", "mean"], main)
    _write_rows(mod_path, ["condition", "moderator", "moderator_level", "outcome", "mean"],
                moderator)
    _write_rows(t3_path, ["condition", "outcome", "ate"], tier3)
    _write_rows(weights_path,
                ["base_pid", "gender", "age_band", "race", "target_cell_proportion",
                 "clean_pool_cell_n", "poststratification_weight"], weight_rows)
    save_json(weight_audit, weights_report_path)

    if len(main) != 221 or len(moderator) != 5967 or len(tier3) != 208:
        raise SystemExit("production: output row-count invariant failed")
    if len({(r["condition"], r["outcome"]) for r in main}) != 221:
        raise SystemExit("production: duplicate/missing Tier-2 main cells")
    if len({(r["condition"], r["moderator"], r["moderator_level"], r["outcome"])
            for r in moderator}) != 5967:
        raise SystemExit("production: duplicate/missing Tier-2 moderator cells")

    report = {
        "status": "pass", "method": {**SPEC, "tau": None,
                                        "source": "SYN-DIGITS Table 8 elastic-net settings",
                                        "adaptive_gate": None,
                                        "production_rule": "always calibrate"},
        "fit_pool": {"n": len(pool), "rule": "structurally exact in all 17 conditions"},
        "anchors": {"n": len(anchor_items), "source": "Twin-2K Wave 4"},
        "targets": {"n": len(TARGETS), "conditions": len(CONDITIONS),
                    "outcomes": len(OUTCOMES)},
        "source_audit": source_audit, "tier1_parity": parity,
        "aggregation": {
            "main": "full 1,921-person clean pool with fixed 40-cell poststratification weights",
            "moderators": weight_audit["moderator_rule"],
            "tier3": "poststratified intervention mean minus poststratified pooled-control mean",
            "poststratification": weight_audit,
        },
        "row_counts": {"tier2_main": len(main), "tier2_moderator": len(moderator),
                       "tier3": len(tier3)},
        "unsupported_moderator_cells": {
            "count": len(unsupported_cells),
            "rule": "organizer-prescribed no-moderation fallback: repeat condition mean",
            "cells": unsupported_cells,
        },
        "files": {str(p.name): _sha256(p) for p in
                  (raw_path, cal_path, fits_path, main_path, mod_path, t3_path,
                   weights_path, weights_report_path)},
    }
    save_json(report, CALIB_DATA / "production_report.json")
    return report
