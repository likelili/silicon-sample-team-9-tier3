"""Re-selection of the frozen calibration specification, from the cached fold.

The first selection used a rule that did not function on this data: it compared
candidates by the standard error of the *best candidate's absolute* panel error
(1.85 pp), which is far larger than the spread between candidates, so 10,723 of
them fell inside the band and the guardrail did all the work.

This module replaces that with a paired comparison, which is the right scale.
Every candidate is scored against the SAME anchors as the uncalibrated
baseline, so the anchor-to-anchor variation that dominated the absolute SE
cancels: the paired SE is ~0.12 pp rather than 1.85 pp, roughly 16x more
discriminating.

The rule, fixed before any result is read:

  1. **Rank** candidates by the mean paired change in normalised panel-mean
     absolute error against the uncalibrated baseline: mean_i(L_c,i - L_base,i).
     More negative is better.
  2. **Practical ties** are judged against the BEST candidate, pairwise: for
     candidate c, take the per-anchor difference from the best candidate's loss,
     E_i = L_c,i - L_best,i, and keep c when mean(E) <= SE(E).  The standard
     error is of *that difference*, not of anyone's absolute error — two
     candidates that move together across anchors are correctly judged
     indistinguishable however noisy the anchors themselves are.
  3. **Exclude** any candidate whose mean gated subgroup RMSE is worse than the
     baseline's.  A specification that improves the headline mean by damaging
     the moderator cells is not acceptable, since Tier 2 submits both.
  4. **Break ties** by, in order: higher proportion of anchors improved, lower
     subgroup RMSE, then model simplicity.

**Degenerate tau.**  ``train_mse`` is identically 0 for every hard-SVD and
soft-SVD fit — hard imputation preserves observed entries by construction, so
the reconstruction cannot differ from the data where data exists.  Verified in
the cached fold: those two methods have exactly one distinct value (0.0) across
all fits, while every other method has hundreds.  For them the gate can never
fire, every tau is the same specification, and sweeping tau would enter the same
candidate 101 times and distort any tie count.  They are collapsed to a single
candidate with ``tau = None`` and the gate recorded as not applicable.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict

from .common import CALIB_DATA, save_json

TAU_GRID = [round(0.01 * k, 2) for k in range(0, 101)]
DEGENERATE_TAU_METHODS = {"mc_hard_svd", "mc_soft_svd"}

# Model simplicity, used only as the final tie-break. Lower is simpler.
# Family order reflects how much structure each method can express: a
# simplex-constrained combination is the most constrained, then penalised
# linear models, then low-rank completion, then an MLP.
_FAMILY_RANK = {
    "synthetic_control": 0, "lasso": 1, "ridge": 1, "elastic_net": 1,
    "mc_hard_svd": 2, "mc_soft_svd": 2, "mc_als": 2, "neural_net": 3,
}


def _capacity(method: str, config: str) -> float:
    """A within-family capacity proxy: lower means simpler.

    Matrix completion -> the rank.  Penalised regression -> stronger
    regularisation is simpler, so the negative log of the penalty.  Networks ->
    total hidden units.
    """
    if method in ("mc_hard_svd", "mc_soft_svd", "mc_als"):
        match = re.search(r"_r(\d+)", config)
        return float(match.group(1)) if match else math.inf
    if method in ("ridge", "lasso", "elastic_net", "synthetic_control"):
        match = re.search(r"_[al]([0-9.e+-]+)", config)
        if match:
            try:
                value = float(match.group(1))
                return -math.log10(value) if value > 0 else math.inf
            except ValueError:
                return math.inf
        return math.inf                       # a table8 config, unparsed
    if method == "neural_net":
        match = re.search(r"_h([0-9x]+)", config)
        if match:
            return float(sum(int(x) for x in match.group(1).split("x")))
    return math.inf


def load_fold(path) -> dict[str, list[dict]]:
    """config -> its per-anchor rows, from a cached fold CSV."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["error"]:
                continue
            try:
                row["train_mse"] = float(row["train_mse"])
                for key in ("cal_panel_ae", "base_panel_ae",
                            "cal_subgroup_rmse", "base_subgroup_rmse",
                            "cal_item_r", "base_item_r"):
                    row[key] = float(row[key])
            except ValueError:
                continue
            if any(math.isnan(row[k]) for k in ("cal_panel_ae", "base_panel_ae")):
                continue
            rows[row["config"]].append(row)
    return rows


def _gated(rows: list[dict], tau: float | None) -> dict:
    """Per-anchor gated losses for one (config, tau) candidate.

    ``tau=None`` means the gate does not apply: every item transfers.
    """
    loss, base, sub, sub_base, item_r, transferred = [], [], [], [], [], 0
    for row in rows:
        use = True if tau is None else row["train_mse"] <= tau
        transferred += use
        loss.append(row["cal_panel_ae"] if use else row["base_panel_ae"])
        base.append(row["base_panel_ae"])
        s = row["cal_subgroup_rmse"] if use else row["base_subgroup_rmse"]
        if not math.isnan(s) and not math.isnan(row["base_subgroup_rmse"]):
            sub.append(s)
            sub_base.append(row["base_subgroup_rmse"])
        item_r.append(row["cal_item_r"] if use else row["base_item_r"])
    return {"loss": loss, "base": base, "sub": sub, "sub_base": sub_base,
            "item_r": item_r, "transferred": transferred}


def _stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def build_candidates(fold: dict[str, list[dict]]) -> list[dict]:
    """One entry per distinct specification, with tau collapsed where degenerate."""
    candidates = []
    for config, rows in fold.items():
        if not rows:
            continue
        method = rows[0]["method"]
        degenerate = method in DEGENERATE_TAU_METHODS
        taus = [None] if degenerate else TAU_GRID
        seen_signature = set()
        for tau in taus:
            g = _gated(rows, tau)
            # Two taus that transfer the same set are the same specification;
            # keeping both would inflate the tie count with duplicates.
            signature = tuple(round(x, 12) for x in g["loss"])
            if signature in seen_signature:
                continue
            seen_signature.add(signature)
            delta = [l - b for l, b in zip(g["loss"], g["base"])]
            d_mean, d_se = _stats(delta)
            sub_mean, _ = _stats(g["sub"])
            sub_base_mean, _ = _stats(g["sub_base"])
            candidates.append({
                "name": config if degenerate else f"{config}@tau{tau:.2f}",
                "config": config, "method": method, "tau": tau,
                "min_col_std": float(rows[0]["min_col_std"]),
                "gate_applicable": not degenerate,
                "n_anchors": len(g["loss"]),
                "loss": g["loss"],
                "paired_delta_mean": d_mean, "paired_delta_se": d_se,
                "prop_improved": sum(1 for d in delta if d < 0) / len(delta),
                "subgroup_rmse": sub_mean,
                "subgroup_delta": sub_mean - sub_base_mean,
                "item_r": sum(g["item_r"]) / len(g["item_r"]),
                "transferred": g["transferred"],
                "family_rank": _FAMILY_RANK.get(method, 9),
                "capacity": _capacity(method, config),
            })
    return candidates


def select(candidates: list[dict]) -> dict:
    """Apply the rule and return the frozen specification plus its audit trail."""
    scored = [c for c in candidates if not math.isnan(c["paired_delta_mean"])]
    if not scored:
        raise ValueError("no candidate produced a finite paired delta")

    best = min(scored, key=lambda c: c["paired_delta_mean"])

    # Practical ties: paired difference FROM THE BEST, and the SE of that
    # difference — not the SE of any candidate's absolute error.
    ties = []
    for c in scored:
        diff = [x - y for x, y in zip(c["loss"], best["loss"])]
        mean, se = _stats(diff)
        c["delta_vs_best_mean"] = mean
        c["delta_vs_best_se"] = se
        if mean <= (se if math.isfinite(se) else 0.0) + 1e-12:
            ties.append(c)

    eligible = [c for c in ties if c["subgroup_delta"] <= 0]
    excluded_for_subgroup = len(ties) - len(eligible)
    pool = eligible or ties        # if the guardrail empties the pool, say so

    chosen = min(pool, key=lambda c: (
        -c["prop_improved"],
        c["subgroup_rmse"] if math.isfinite(c["subgroup_rmse"]) else math.inf,
        c["family_rank"], c["capacity"], c["name"],
    ))
    return {
        "chosen": chosen,
        "best_by_primary": best["name"],
        "best_paired_delta": best["paired_delta_mean"],
        "n_candidates": len(scored),
        "n_practical_ties": len(ties),
        "n_excluded_worsening_subgroup": excluded_for_subgroup,
        "n_eligible": len(pool),
        "guardrail_emptied_pool": not eligible,
        "rule": "rank by mean paired change in normalised panel AE vs baseline; "
                "practical ties = mean paired difference from the BEST candidate "
                "within 1 SE of that difference; exclude candidates worsening mean "
                "subgroup RMSE; break ties by proportion of anchors improved, then "
                "subgroup RMSE, then model simplicity",
        "top_candidates": [
            {k: v for k, v in c.items() if k != "loss"}
            for c in sorted(pool, key=lambda c: (
                -c["prop_improved"], c["subgroup_rmse"], c["family_rank"],
                c["capacity"], c["name"]))[:10]
        ],
    }


def run() -> dict:
    fold = load_fold(CALIB_DATA / "tuning_tune_fold.csv")
    candidates = build_candidates(fold)
    result = select(candidates)
    chosen = result["chosen"]

    print(f"candidates (tau collapsed where degenerate): {result['n_candidates']}")
    print(f"best by primary: {result['best_by_primary']} "
          f"({result['best_paired_delta']:+.4f} pp paired)")
    print(f"practical ties with best: {result['n_practical_ties']}")
    print(f"excluded for worsening subgroup RMSE: {result['n_excluded_worsening_subgroup']}")
    print(f"eligible after guardrail: {result['n_eligible']}")
    print()
    print("FROZEN:")
    for key in ("name", "method", "config", "min_col_std", "tau", "gate_applicable"):
        print(f"  {key:22s} {chosen[key]}")
    for key in ("paired_delta_mean", "paired_delta_se", "prop_improved",
                "subgroup_delta", "subgroup_rmse", "item_r"):
        print(f"  {key:22s} {chosen[key]:.4f}")
    save_json({k: v for k, v in result.items() if k != "chosen"} |
              {"chosen": {k: v for k, v in chosen.items() if k != "loss"}},
              CALIB_DATA / "reselection.json")
    return result


def evaluate_holdout(chosen: dict, n_jobs: int = 10) -> dict:
    """Run the frozen specification once on the holdout anchors.

    The holdout is touched exactly once, with one configuration. Selecting the
    minimum of ~2,000 noisy paired estimates is subject to winner's curse, so
    this number — not the tuning number — is what the specification is worth.
    """
    import random

    from joblib import Parallel, delayed

    from .common import anchor_matrices, clean_pool, demographics, tier1_ids
    from .scoring import declared_spans, group_masks
    from .tune import SPLIT_SEED, _one_target, build_grid, screen_grid

    pool = clean_pool()
    panel, _ = tier1_ids()
    Y, T, items = anchor_matrices(pool)
    masks = group_masks(pool, panel, demographics(pool))
    spans = declared_spans(items)

    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    hold_idx = sorted(order[len(order) // 2:])

    grid = [c for c in (screen_grid(True) + build_grid(True))
            if c["name"] == chosen["config"]][:1]
    if not grid:
        raise SystemExit(f"frozen config {chosen['config']!r} not found in either grid")

    batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_target)(Y, T, T, t, masks, spans, grid, [1.0, 0.1])
        for t in hold_idx)
    rows = [r for batch in batches for r in batch if not r["error"]]
    for row in rows:
        for key in ("train_mse", "cal_panel_ae", "base_panel_ae",
                    "cal_subgroup_rmse", "base_subgroup_rmse",
                    "cal_item_r", "base_item_r"):
            row[key] = float(row[key])
    rows = [r for r in rows
            if not (math.isnan(r["cal_panel_ae"]) or math.isnan(r["base_panel_ae"]))]

    g = _gated(rows, chosen["tau"])
    delta = [l - b for l, b in zip(g["loss"], g["base"])]
    d_mean, d_se = _stats(delta)
    sub_delta = [s - b for s, b in zip(g["sub"], g["sub_base"])]
    s_mean, s_se = _stats(sub_delta)
    loss_mean, _ = _stats(g["loss"])
    base_mean, _ = _stats(g["base"])

    return {
        "n_anchors": len(rows),
        "panel_ae": loss_mean,
        "baseline_panel_ae": base_mean,
        "paired_delta_mean": d_mean,
        "paired_delta_se": d_se,
        "paired_t": d_mean / d_se if d_se else math.nan,
        "paired_ci95": [d_mean - 1.96 * d_se, d_mean + 1.96 * d_se],
        "prop_improved": sum(1 for d in delta if d < 0) / len(delta),
        "improvement_pct": 100 * d_mean / base_mean if base_mean else math.nan,
        "subgroup_delta_mean": s_mean,
        "subgroup_delta_se": s_se,
        "subgroup_t": s_mean / s_se if s_se else math.nan,
        "item_r": sum(g["item_r"]) / len(g["item_r"]),
        "baseline_item_r": sum(r["base_item_r"] for r in rows) / len(rows),
        "transferred": f"{g['transferred']}/{len(rows)}",
        "rows": rows,
    }
