"""Select the calibration method and its ordinary hyperparameters. Nothing else.

This step deliberately answers ONE question: which calibration model, at which
ordinary hyperparameters, produces the most accurate calibrated predictions on
the tuning anchors.

The transfer threshold is **not** part of it. Both earlier selections entangled
the two, and that was the defect: a candidate could win by gating most items
back to the uncalibrated baseline, which measures the gate rather than the
calibrator. Here every candidate is evaluated **always-calibrate** — the
calibrated prediction is used for every anchor, no gate, no fallback — so the
comparison is between calibrators and nothing else.

Consequences of dropping tau:

* ``train_mse``, transfer frequency and the gate play no part in ranking.
* Configurations that differ only in tau are the same specification, so hard SVD
  and soft SVD collapse automatically — with no gate there is nothing left for
  tau to change. (Their ``train_mse`` is identically 0 in any case, verified in
  the cached fold, which is why their gate was inert.)
* The candidate set is exactly the distinct (method, hyperparameters,
  ``min_col_std``) configurations: 116, not thousands.

Criteria, fixed before reading any result:

  **Primary** — mean normalised panel-mean absolute error of the calibrated
  predictions across the tuning anchors, in percentage points of each item's
  declared scale. Lower is better.

  **Practical ties** — anchors vary enormously in difficulty, so the SE of the
  absolute mean (~1.9 pp) is far larger than the spread between candidates and
  cannot separate them. Candidates are therefore compared to the best candidate
  **pairwise**, on the same anchors: keep candidate c when
  ``mean_i(L_c,i - L_best,i) <= SE`` of that same difference.

  **Guardrail** — subgroup RMSE across the moderator levels. A candidate whose
  mean subgroup RMSE is worse than the uncalibrated baseline's is excluded: a
  method that improves the headline mean by damaging moderator cells is not
  usable, because Tier 2 submits both.

  **Tie-breaks** — proportion of anchors improved over baseline, then subgroup
  RMSE, then model simplicity.

Reads only ``tuning_tune_fold.csv``. Writes only new files. The holdout fold is
not opened.
"""

from __future__ import annotations

import math

from .common import CALIB_DATA, save_json
from .reselect import _FAMILY_RANK, _capacity, _stats, load_fold


def build_candidates(fold: dict[str, list[dict]]) -> list[dict]:
    """One candidate per distinct configuration, evaluated always-calibrate."""
    candidates = []
    for config, rows in fold.items():
        if not rows:
            continue
        method = rows[0]["method"]
        loss = [r["cal_panel_ae"] for r in rows]
        base = [r["base_panel_ae"] for r in rows]
        sub = [(r["cal_subgroup_rmse"], r["base_subgroup_rmse"]) for r in rows
               if not math.isnan(r["cal_subgroup_rmse"])
               and not math.isnan(r["base_subgroup_rmse"])]
        delta = [l - b for l, b in zip(loss, base)]
        mean_loss, se_loss = _stats(loss)
        d_mean, d_se = _stats(delta)
        sub_mean, _ = _stats([s for s, _ in sub])
        sub_base_mean, _ = _stats([b for _, b in sub])
        candidates.append({
            "config": config, "method": method,
            "min_col_std": float(rows[0]["min_col_std"]),
            "n_anchors": len(loss),
            "loss": loss,
            "panel_ae": mean_loss, "panel_ae_se": se_loss,
            "baseline_panel_ae": _stats(base)[0],
            "paired_delta_mean": d_mean, "paired_delta_se": d_se,
            "paired_t": d_mean / d_se if d_se else math.nan,
            "prop_improved": sum(1 for d in delta if d < 0) / len(delta),
            "subgroup_rmse": sub_mean,
            "baseline_subgroup_rmse": sub_base_mean,
            "subgroup_delta": sub_mean - sub_base_mean,
            "item_r": _stats([r["cal_item_r"] for r in rows])[0],
            "baseline_item_r": _stats([r["base_item_r"] for r in rows])[0],
            "family_rank": _FAMILY_RANK.get(method, 9),
            "capacity": _capacity(method, config),
        })
    return candidates


def select(candidates: list[dict]) -> dict:
    scored = [c for c in candidates if not math.isnan(c["panel_ae"])]
    if not scored:
        raise ValueError("no candidate produced a finite panel error")

    best = min(scored, key=lambda c: c["panel_ae"])
    ties = []
    for c in scored:
        diff = [x - y for x, y in zip(c["loss"], best["loss"])]
        mean, se = _stats(diff)
        c["delta_vs_best_mean"] = mean
        c["delta_vs_best_se"] = se
        if mean <= (se if math.isfinite(se) else 0.0) + 1e-12:
            ties.append(c)

    eligible = [c for c in ties if c["subgroup_delta"] <= 0]
    pool = eligible or ties
    chosen = min(pool, key=lambda c: (
        -c["prop_improved"],
        c["subgroup_rmse"] if math.isfinite(c["subgroup_rmse"]) else math.inf,
        c["family_rank"], c["capacity"], c["config"],
    ))
    order = sorted(scored, key=lambda c: c["panel_ae"])
    return {
        "chosen": chosen,
        "best_by_primary": best["config"],
        "n_candidates": len(scored),
        "n_practical_ties": len(ties),
        "n_excluded_by_guardrail": len(ties) - len(eligible),
        "guardrail_emptied_pool": not eligible,
        "ranking_all": [{k: v for k, v in c.items() if k != "loss"} for c in order],
        "tie_set": [{k: v for k, v in c.items() if k != "loss"}
                    for c in sorted(pool, key=lambda c: (
                        -c["prop_improved"], c["subgroup_rmse"],
                        c["family_rank"], c["capacity"], c["config"]))],
    }


def run() -> dict:
    fold = load_fold(CALIB_DATA / "tuning_tune_fold.csv")
    candidates = build_candidates(fold)
    result = select(candidates)
    chosen = result["chosen"]

    report = {
        "step": "method and ordinary hyperparameter selection ONLY",
        "excluded_from_this_step": [
            "tau / transfer threshold", "transfer frequency",
            "the train-MSE gate", "the holdout fold", "production fits",
        ],
        "evaluation": "always-calibrate — the calibrated prediction is used for "
                      "every tuning anchor, with no gate and no fallback, so the "
                      "comparison is between calibrators alone",
        "tau_collapse": "with no gate, configurations differing only in tau are the "
                        "same specification; hard SVD and soft SVD collapse "
                        "automatically (their train_mse is identically 0 in any case)",
        "primary_criterion": "mean normalised panel-mean absolute error (pp of each "
                             "item's declared scale) over the tuning anchors",
        "tie_rule": "paired difference from the best candidate, kept when mean <= SE "
                    "of that difference",
        "guardrail": "exclude candidates whose mean subgroup RMSE is worse than the "
                     "uncalibrated baseline",
        "tie_breaks": ["proportion of anchors improved", "subgroup RMSE",
                       "model simplicity"],
        "source": str(CALIB_DATA / "tuning_tune_fold.csv"),
        "n_candidates": result["n_candidates"],
        "n_practical_ties": result["n_practical_ties"],
        "n_excluded_by_guardrail": result["n_excluded_by_guardrail"],
        "best_by_primary": result["best_by_primary"],
        "proposed": {
            "method": chosen["method"], "config": chosen["config"],
            "min_col_std": chosen["min_col_std"],
            "donor_policy": "the 123 Wave-4 anchor columns only",
            "tau": "NOT SELECTED IN THIS STEP",
            "panel_ae": chosen["panel_ae"],
            "baseline_panel_ae": chosen["baseline_panel_ae"],
            "paired_delta_mean": chosen["paired_delta_mean"],
            "paired_delta_se": chosen["paired_delta_se"],
            "paired_t": chosen["paired_t"],
            "prop_improved": chosen["prop_improved"],
            "subgroup_rmse": chosen["subgroup_rmse"],
            "baseline_subgroup_rmse": chosen["baseline_subgroup_rmse"],
            "subgroup_delta": chosen["subgroup_delta"],
            "item_r": chosen["item_r"], "baseline_item_r": chosen["baseline_item_r"],
        },
        "tie_set": result["tie_set"],
        "ranking_all": result["ranking_all"],
    }
    save_json(report, CALIB_DATA / "method_selection.json")
    return report
