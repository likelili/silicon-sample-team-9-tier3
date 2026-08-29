"""Select the transfer threshold tau, for an already-frozen calibration method.

The method is fixed before this step runs (neural net, hidden [16], weight decay
0.1, ``min_col_std`` 1.0, all 123 anchors as donors).  Nothing here may change
it; this step decides only the SYN-DIGITS transfer rule:

    use the calibrated prediction when ``train_mse <= tau``
    otherwise keep the uncalibrated digital-twin prediction

Separating the two decisions is the point.  When tau and the method are chosen
together, a candidate can win by gating most items back to the baseline, which
scores the gate rather than the calibrator.

Primary criterion is gated panel-mean absolute error over the tuning anchors, in
percentage points of each item's declared scale.  Subgroup RMSE across the
moderator levels is the guardrail: a tau whose gated subgroup RMSE is worse than
the uncalibrated baseline's is rejected, since Tier 2 submits both.

Practical ties use the paired difference from the best tau on the same anchors —
the SE of the absolute mean is ~1.9 pp and cannot separate thresholds that
differ by hundredths.  Among ties the preference order is: more anchors
improved, then lower subgroup RMSE, then the **simplest gate**, meaning the tau
that transfers the most items.  A gate that fires rarely is closer to
always-calibrate and carries one less moving part; if the data cannot show that
gating helps, the specification should say so rather than encode an inert rule.

Reads the cached tune fold only.  Does not open the holdout.
"""

from __future__ import annotations

import math

from .common import CALIB_DATA, save_json
from .reselect import _gated, _stats, load_fold

TAU_GRID = [round(0.01 * k, 2) for k in range(0, 101)]


def candidates(rows: list[dict]) -> list[dict]:
    """One entry per distinct gate behaviour, over the tau grid."""
    out, seen = [], set()
    for tau in TAU_GRID:
        g = _gated(rows, tau)
        signature = tuple(round(x, 12) for x in g["loss"])
        if signature in seen:          # a tau that transfers the same set is the same rule
            continue
        seen.add(signature)
        delta = [l - b for l, b in zip(g["loss"], g["base"])]
        d_mean, d_se = _stats(delta)
        sub_mean, _ = _stats(g["sub"])
        sub_base_mean, _ = _stats(g["sub_base"])
        out.append({
            "tau": tau,
            "loss": g["loss"],
            "panel_ae": _stats(g["loss"])[0],
            "baseline_panel_ae": _stats(g["base"])[0],
            "paired_delta_mean": d_mean, "paired_delta_se": d_se,
            "paired_t": d_mean / d_se if d_se else math.nan,
            "prop_improved": sum(1 for d in delta if d < 0) / len(delta),
            "subgroup_rmse": sub_mean,
            "baseline_subgroup_rmse": sub_base_mean,
            "subgroup_delta": sub_mean - sub_base_mean,
            "item_r": _stats(g["item_r"])[0],
            "transferred": g["transferred"],
            "n_anchors": len(g["loss"]),
        })
    return out


def select(cands: list[dict]) -> dict:
    scored = [c for c in cands if not math.isnan(c["panel_ae"])]
    best = min(scored, key=lambda c: c["panel_ae"])
    ties = []
    for c in scored:
        diff = [x - y for x, y in zip(c["loss"], best["loss"])]
        mean, se = _stats(diff)
        c["delta_vs_best_mean"], c["delta_vs_best_se"] = mean, se
        if mean <= (se if math.isfinite(se) else 0.0) + 1e-12:
            ties.append(c)
    eligible = [c for c in ties if c["subgroup_delta"] <= 0]
    pool = eligible or ties
    chosen = min(pool, key=lambda c: (
        -c["prop_improved"],
        c["subgroup_rmse"] if math.isfinite(c["subgroup_rmse"]) else math.inf,
        -c["transferred"],                     # simplest gate = transfers most
        c["tau"],
    ))
    return {"chosen": chosen, "best_tau": best["tau"], "n_distinct_rules": len(scored),
            "n_ties": len(ties), "n_excluded_by_guardrail": len(ties) - len(eligible),
            "guardrail_emptied_pool": not eligible,
            "all": [{k: v for k, v in c.items() if k != "loss"} for c in scored]}


def run(config: str) -> dict:
    fold = load_fold(CALIB_DATA / "tuning_tune_fold.csv")
    if config not in fold:
        raise SystemExit(f"config {config!r} not in the cached tune fold")
    rows = fold[config]
    result = select(candidates(rows))
    chosen = result["chosen"]

    mses = sorted(r["train_mse"] for r in rows)
    always = max(mses)
    report = {
        "step": "transfer threshold (tau) selection, method already frozen",
        "frozen_method": config,
        "rule": "use the calibrated prediction when train_mse <= tau, else keep the "
                "uncalibrated digital-twin prediction (SYN-DIGITS adaptive transfer)",
        "primary_criterion": "gated panel-mean absolute error on the tuning anchors",
        "guardrail": "reject any tau whose gated subgroup RMSE is worse than baseline",
        "tie_breaks": ["proportion of anchors improved", "subgroup RMSE",
                       "simplest gate (transfers most items)", "lower tau"],
        "train_mse_range": {"min": mses[0], "median": mses[len(mses) // 2],
                            "max": always,
                            "tau_for_always_calibrate": always},
        "n_distinct_gate_rules": result["n_distinct_rules"],
        "n_practical_ties": result["n_ties"],
        "n_excluded_by_guardrail": result["n_excluded_by_guardrail"],
        "best_tau_by_primary": result["best_tau"],
        "chosen": {k: v for k, v in chosen.items() if k != "loss"},
        "all_rules": result["all"],
        "source": str(CALIB_DATA / "tuning_tune_fold.csv"),
    }
    save_json(report, CALIB_DATA / "tau_selection.json")
    return report
