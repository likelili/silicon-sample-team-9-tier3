"""The published SYN-DIGITS elastic-net specification, taken as externally fixed.

Nothing here is optimised. The specification is given:

    method                 elastic net
    alpha                  0.01          (Table 8, Twin-2K-500, new question)
    L1 ratio               0.3           (Table 8)
    donor imputation rank  5
    min_col_std            1.0
    donors                 all 123 Wave-4 anchors
    tau                    0.15          (the paper's adaptive-transfer threshold)
    gate                   calibrated prediction when train_mse <= tau,
                           otherwise the raw digital-twin prediction

**Missing-value handling.** Their ``evaluate_column`` mean-fills a missing
synthetic target and keeps the row in the fit. Wave-4 assigns question variants
per person, so a target column is missing for about half the panel *by design*;
mean-filling would enter ~1,000 invented observations at exactly the column mean.
This module fits on rows with a finite DT target only, and scores on rows where
the human target, the DT target and the prediction are all finite — so the
calibrated and raw arms are always compared on the same respondents. The
departure is implemented in ``authors.predict_column(fit_finite_only=True)`` and
documented at the point of change.

Three prespecified arms are compared on the tuning anchors before the holdout is
touched: raw DT, elastic net applied everywhere, and elastic net behind the
published gate.
"""

from __future__ import annotations

import math

import numpy as np

from . import authors
from .common import (
    CALIB_DATA,
    anchor_matrices,
    clean_pool,
    demographics,
    save_json,
    tier1_ids,
)
from .scoring import MIN_CELL, declared_spans, group_masks, guarded_r

SPEC = {
    "method": "elastic_net",
    "regularization_multiplier": 0.01,
    "en_l1_ratio": 0.3,
    "imputation_rank": 5,
    "min_col_std": 1.0,
    "tau": 0.15,
    "donors": "all 123 Wave-4 anchor columns",
    "source": "SYN-DIGITS Table 8 (Twin-2K-500, new-question column); tau from the "
              "paper's adaptive-transfer rule",
}
FIT_PARAMS = {"regularization_multiplier": SPEC["regularization_multiplier"],
              "en_l1_ratio": SPEC["en_l1_ratio"]}


def score(y_true, y_hat, valid, span, masks) -> dict:
    """Panel AE, subgroup RMSE and item r over an explicit valid-row mask."""
    if valid.sum() < 2 or not np.isfinite(span) or span <= 0:
        return {"panel_ae": math.nan, "subgroup_rmse": math.nan,
                "item_r": math.nan, "n_scored": int(valid.sum())}
    item_r = guarded_r(y_true[valid], y_hat[valid])
    panel = masks["panel"] & valid
    panel_ae = (abs(float(y_hat[panel].mean() - y_true[panel].mean())) / span * 100
                if panel.sum() else math.nan)
    errors = []
    for mask in masks["levels"].values():
        cell = mask & valid
        if cell.sum() >= MIN_CELL:
            errors.append(float(y_hat[cell].mean() - y_true[cell].mean()) / span * 100)
    subgroup = float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan
    return {"panel_ae": panel_ae, "subgroup_rmse": subgroup,
            "item_r": item_r, "n_scored": int(valid.sum())}


def per_target(Y, T, targets, masks, spans, sc) -> list[dict]:
    """Raw and calibrated scores per target, on a common valid-row mask."""
    rows = []
    for j in targets:
        y_true, y_dt = Y[:, j], T[:, j]
        try:
            y_cal, train_mse, info = authors.predict_column(
                sc, j, method=SPEC["method"], fit_finite_only=True, **FIT_PARAMS)
            error = ""
        except Exception as exc:
            y_cal, train_mse, info = np.full_like(y_true, np.nan), math.nan, {}
            error = f"{type(exc).__name__}: {exc}"[:140]

        # Same respondents for both arms: human, DT and prediction all finite.
        valid = np.isfinite(y_true) & np.isfinite(y_dt) & np.isfinite(y_cal)
        raw = score(y_true, y_dt, valid, spans[j], masks)
        cal = score(y_true, y_cal, valid, spans[j], masks)
        rows.append({
            "target": int(j), "train_mse": train_mse,
            "n_fit_rows": info.get("n_fit_rows"), "n_scored": raw["n_scored"],
            "raw_panel_ae": raw["panel_ae"], "cal_panel_ae": cal["panel_ae"],
            "raw_subgroup_rmse": raw["subgroup_rmse"],
            "cal_subgroup_rmse": cal["subgroup_rmse"],
            "raw_item_r": raw["item_r"], "cal_item_r": cal["item_r"],
            "gated": (not math.isnan(train_mse)) and train_mse <= SPEC["tau"],
            "error": error,
        })
    return rows


def _mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def _paired(a, b):
    diff = [x - y for x, y in zip(a, b)
            if not (math.isnan(x) or math.isnan(y))]
    n = len(diff)
    if n < 2:
        return math.nan, math.nan, math.nan, 0
    mean = sum(diff) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diff) / (n - 1))
    se = sd / math.sqrt(n)
    return mean, se, (mean / se if se else math.nan), sum(1 for d in diff if d < 0)


def arms(rows: list[dict]) -> dict:
    """The three prespecified arms, scored on the same targets."""
    ok = [r for r in rows if not r["error"] and not math.isnan(r["cal_panel_ae"])]
    raw_ae = [r["raw_panel_ae"] for r in ok]
    always_ae = [r["cal_panel_ae"] for r in ok]
    gated_ae = [r["cal_panel_ae"] if r["gated"] else r["raw_panel_ae"] for r in ok]
    raw_sg = [r["raw_subgroup_rmse"] for r in ok]
    always_sg = [r["cal_subgroup_rmse"] for r in ok]
    gated_sg = [r["cal_subgroup_rmse"] if r["gated"] else r["raw_subgroup_rmse"]
                for r in ok]
    raw_r = [r["raw_item_r"] for r in ok]
    always_r = [r["cal_item_r"] for r in ok]
    gated_r = [r["cal_item_r"] if r["gated"] else r["raw_item_r"] for r in ok]

    def arm(name, ae, sg, ir, n_cal):
        d_mean, d_se, d_t, n_better = _paired(ae, raw_ae)
        s_mean, s_se, s_t, _ = _paired(sg, raw_sg)
        return {"arm": name, "panel_ae": _mean(ae), "subgroup_rmse": _mean(sg),
                "item_r": _mean(ir), "n_targets_calibrated": n_cal,
                "paired_vs_raw_panel_ae": d_mean, "paired_se": d_se, "paired_t": d_t,
                "targets_improved": n_better, "n_targets": len(ok),
                "paired_vs_raw_subgroup": s_mean, "subgroup_t": s_t}

    return {
        "n_targets": len(ok),
        "raw": arm("raw DT", raw_ae, raw_sg, raw_r, 0),
        "always": arm("elastic net, every target", always_ae, always_sg, always_r, len(ok)),
        "gated": arm(f"elastic net, tau={SPEC['tau']} gate", gated_ae, gated_sg,
                     gated_r, sum(1 for r in ok if r["gated"])),
    }


def diagnostic(rows: list[dict]) -> dict:
    """Does LOWER train_mse predict GREATER calibration benefit?

    The gate's premise. A valid diagnostic needs a POSITIVE correlation between
    ``train_mse`` and the benefit ``cal - raw`` (benefit is negative when
    calibration helps, so higher error should mean less benefit).
    """
    from scipy.stats import pearsonr, spearmanr

    ok = [r for r in rows if not r["error"]
          and not math.isnan(r["cal_panel_ae"]) and not math.isnan(r["train_mse"])]
    mse = [r["train_mse"] for r in ok]
    benefit = [r["cal_panel_ae"] - r["raw_panel_ae"] for r in ok]
    pr, pp = pearsonr(mse, benefit)
    sr, sp = spearmanr(mse, benefit)
    median = sorted(mse)[len(mse) // 2]
    low = [b for m, b in zip(mse, benefit) if m <= median]
    high = [b for m, b in zip(mse, benefit) if m > median]
    return {
        "n": len(ok),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "low_mse_mean_benefit": _mean(low), "low_mse_helped": sum(1 for b in low if b < 0),
        "low_mse_n": len(low),
        "high_mse_mean_benefit": _mean(high), "high_mse_helped": sum(1 for b in high if b < 0),
        "high_mse_n": len(high),
        "intended_direction": bool(pr > 0),
        "note": "intended direction is a POSITIVE correlation: low train_mse should "
                "mark targets where calibration helps most",
    }


def run_tuning() -> dict:
    import random

    from .tune import SPLIT_SEED

    pool = clean_pool()
    panel, _ = tier1_ids()
    Y, T, items = anchor_matrices(pool)
    masks = group_masks(pool, panel, demographics(pool))
    spans = declared_spans(items)

    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    tune_idx = sorted(order[: len(order) // 2])

    sc = authors.build(Y, T, name="published", imputation_rank=SPEC["imputation_rank"],
                       min_col_std=SPEC["min_col_std"])
    rows = per_target(Y, T, tune_idx, masks, spans, sc)
    for r in rows:
        r["item_id"] = items[r["target"]]

    result = {
        "specification": SPEC,
        "missing_value_handling": {
            "fit": "rows with a finite DT target only",
            "score": "rows where human target, DT target and prediction are all finite",
            "departure_from_repo": "evaluate_column mean-fills a missing synthetic "
                                   "target and keeps the row; that would enter ~1,000 "
                                   "invented observations at the column mean, because "
                                   "Wave-4 variants leave a target missing for about "
                                   "half the panel by design",
        },
        "n_tuning_targets": len(rows),
        "arms": arms(rows),
        "diagnostic": diagnostic(rows),
        "per_target": rows,
    }
    save_json(result, CALIB_DATA / "published_spec_tuning.json")
    return result


def run_holdout() -> dict:
    """Evaluate the frozen always-calibrate procedure ONCE on the reserved holdout.

    No selection, no tuning, no threshold. The gate is not applied: tau = 0.15
    failed on the tuning matrices (it made panel-mean AE worse, +0.017 pp, and
    calibrated only 21 of 61 targets), and the train_mse diagnostic carried no
    usable signal (Pearson +0.034, p = 0.79; the high-error half benefited more).
    That failure is recorded, not worked around.
    """
    import random

    from .tune import SPLIT_SEED

    pool = clean_pool()
    panel, _ = tier1_ids()
    Y, T, items = anchor_matrices(pool)
    masks = group_masks(pool, panel, demographics(pool))
    spans = declared_spans(items)

    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    hold_idx = sorted(order[len(order) // 2:])

    sc = authors.build(Y, T, name="published_holdout",
                       imputation_rank=SPEC["imputation_rank"],
                       min_col_std=SPEC["min_col_std"])
    rows = per_target(Y, T, hold_idx, masks, spans, sc)
    for r in rows:
        r["item_id"] = items[r["target"]]

    ok = [r for r in rows if not r["error"] and not math.isnan(r["cal_panel_ae"])]
    ae_mean, ae_se, ae_t, ae_better = _paired(
        [r["cal_panel_ae"] for r in ok], [r["raw_panel_ae"] for r in ok])
    sg_mean, sg_se, sg_t, sg_better = _paired(
        [r["cal_subgroup_rmse"] for r in ok], [r["raw_subgroup_rmse"] for r in ok])

    # Cast NumPy comparison results to built-in bools so the report remains
    # JSON-serializable across NumPy versions.
    panel_improves = bool(ae_mean < 0)
    subgroup_not_worse = bool(sg_mean <= 0)
    result = {
        "stage": "HOLDOUT — evaluated once, frozen procedure, no selection",
        "specification": {**SPEC, "tau": None,
                          "gate": "NOT APPLIED — always-calibrate",
                          "gate_note": "the published tau=0.15 rule did not transfer "
                                       "to our Luna/Summary matrices; see "
                                       "published_spec_tuning.json"},
        "n_holdout_targets": len(ok),
        "panel_ae": {"calibrated": _mean([r["cal_panel_ae"] for r in ok]),
                     "raw_dt": _mean([r["raw_panel_ae"] for r in ok]),
                     "paired_mean": ae_mean, "paired_se": ae_se, "paired_t": ae_t,
                     "targets_improved": ae_better,
                     "ci95": [ae_mean - 1.96 * ae_se, ae_mean + 1.96 * ae_se]},
        "subgroup_rmse": {"calibrated": _mean([r["cal_subgroup_rmse"] for r in ok]),
                          "raw_dt": _mean([r["raw_subgroup_rmse"] for r in ok]),
                          "paired_mean": sg_mean, "paired_se": sg_se, "paired_t": sg_t,
                          "targets_improved": sg_better},
        "item_r_descriptive": {"calibrated": _mean([r["cal_item_r"] for r in ok]),
                               "raw_dt": _mean([r["raw_item_r"] for r in ok])},
        "gate_condition": {
            "panel_ae_improves": panel_improves,
            "subgroup_rmse_not_worse": subgroup_not_worse,
            "proceed_to_production": panel_improves and subgroup_not_worse,
        },
        "per_target": rows,
    }
    save_json(result, CALIB_DATA / "published_spec_holdout.json")
    return result
