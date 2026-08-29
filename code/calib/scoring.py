"""Scoring for calibration selection — declared scales, all 27 moderator levels.

Two earlier versions of this file were wrong in ways worth recording, because
both failed silently:

1.  It averaged raw absolute errors across items on incompatible scales.  The
    anchor block mixes 0-100 sliders with 1-2 binary choices, so the mean was
    effectively the 20 slider items alone (measured base panel AE by family:
    attitudes 6.054, heuristics 1.331, pricing 0.040).
2.  It then normalised by the **observed** sample range.  An observed range is a
    property of whoever happened to answer; it makes the same absolute error
    score differently depending on sample spread, and a degenerate item can blow
    the score up or make it vanish.

Errors are now divided by each item's **declared** span (``calib.ranges``) and
expressed in percentage points of that span.  The single item with no declared
bound is excluded from the aggregate rather than assigned an invented scale.

Subgroup error covers **all six official moderators and all 27 levels**
(gender 3, age_band 4, race 5, education 6, income 5, party 4), not the three
quota dimensions.  Levels the pool cannot populate are reported, not hidden:
``gender = Other`` has zero twins, so 26 of 27 levels carry data.
"""

from __future__ import annotations

import math

import numpy as np

from .outcomes import MODERATORS

# Panel cells below this contribute no usable mean; they are counted and named
# rather than silently dropped.
MIN_CELL = 5


def guarded_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The authors' correlation, including their near-constant guard."""
    if len(y_true) < 2 or np.std(y_true) <= 1e-2 or np.std(y_pred) <= 1e-2:
        return 0.0
    from scipy.stats import pearsonr

    return float(pearsonr(y_true, y_pred)[0])


def declared_spans(items: list[str]) -> np.ndarray:
    """Per-item declared span, NaN where the instrument declares none."""
    from .ranges import declared_ranges

    ranges = declared_ranges()
    return np.array([(ranges.get(i, {}).get("span") or np.nan) for i in items], dtype=float)


def group_masks(pool: list[str], panel: set[str], demo: dict[str, dict]) -> dict:
    """Panel mask plus one mask per moderator level, all six moderators.

    Returns ``{"panel": mask, "levels": {"moderator=level": mask}, "sizes": {...},
    "empty": [...]}`` so the caller can report coverage instead of assuming it.
    """
    in_panel = np.array([p in panel for p in pool])
    levels, sizes, empty = {}, {}, []
    for moderator, allowed in MODERATORS.items():
        for level in allowed:
            key = f"{moderator}={level}"
            mask = in_panel & np.array(
                [p in demo and demo[p].get(moderator) == level for p in pool])
            sizes[key] = int(mask.sum())
            if mask.sum() >= MIN_CELL:
                levels[key] = mask
            else:
                empty.append(key)
    return {"panel": in_panel, "levels": levels, "sizes": sizes, "empty": empty,
            "n_levels_declared": sum(len(v) for v in MODERATORS.values()),
            "n_levels_scored": len(levels)}


def score_column(y_true: np.ndarray, y_hat: np.ndarray, span: float,
                 masks: dict) -> dict:
    """Scores for one predicted column, in pp of the item's DECLARED span.

    ``panel_ae`` and ``subgroup_rmse`` are NaN when the item has no declared
    span — the aggregate skips it rather than scoring it on an invented scale.
    ``item_r`` is still computed, since a correlation needs no scale.
    """
    observed = (~np.isnan(y_true)) & (~np.isnan(y_hat))
    if not observed.any():
        return {"item_r": math.nan, "panel_ae": math.nan,
                "subgroup_rmse": math.nan, "n_obs": 0, "n_levels": 0}

    item_r = guarded_r(y_true[observed], y_hat[observed])
    if not np.isfinite(span) or span <= 0:
        return {"item_r": item_r, "panel_ae": math.nan,
                "subgroup_rmse": math.nan, "n_obs": int(observed.sum()), "n_levels": 0}

    panel = masks["panel"] & observed
    panel_ae = (abs(float(y_hat[panel].mean() - y_true[panel].mean())) / span * 100
                if panel.sum() else math.nan)

    errors = []
    for mask in masks["levels"].values():
        cell = mask & observed
        if cell.sum() >= MIN_CELL:
            errors.append(float(y_hat[cell].mean() - y_true[cell].mean()) / span * 100)
    subgroup = float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan

    return {"item_r": item_r, "panel_ae": panel_ae, "subgroup_rmse": subgroup,
            "n_obs": int(observed.sum()), "n_levels": len(errors)}


def gated_vectors(rows: list[dict], tau: float) -> tuple[list[float], list[float], list[float], int]:
    """Per-item gated (panel_ae, subgroup_rmse, item_r) and transfer count.

    An item whose synthetic-side ``train_mse`` exceeds ``tau`` reverts to the
    uncalibrated twin column — the paper's Figure-5 adaptive-transfer rule.
    """
    ae, sg, r, transferred = [], [], [], 0
    for row in rows:
        if row.get("error"):
            continue
        mse = row.get("train_mse")
        use = mse is not None and not math.isnan(mse) and mse <= tau
        transferred += use
        ae.append(row["cal_panel_ae"] if use else row["base_panel_ae"])
        sg.append(row["cal_subgroup_rmse"] if use else row["base_subgroup_rmse"])
        r.append(row["cal_item_r"] if use else row["base_item_r"])
    return ae, sg, r, transferred


def summarise(rows: list[dict], tau: float) -> dict:
    """Mean panel error with its standard error, plus the guardrail statistic."""
    ae, sg, r, transferred = gated_vectors(rows, tau)
    ae_arr = np.array([v for v in ae if not math.isnan(v)], dtype=float)
    sg_arr = np.array([v for v in sg if not math.isnan(v)], dtype=float)
    n = ae_arr.size
    return {
        "tau": tau,
        "panel_ae": float(ae_arr.mean()) if n else math.nan,
        # SE of the mean across anchor items — the unit the 1-SE rule is in.
        "panel_ae_se": float(ae_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
        "subgroup_rmse": float(sg_arr.mean()) if sg_arr.size else math.nan,
        "item_r": float(np.nanmean(r)) if r else math.nan,
        "transferred": transferred,
        "n_scored_items": int(n),
    }


def select_within_one_se(candidates: list[dict]) -> dict:
    """The prespecified selection rule, written down before any holdout is seen.

    1. take the lowest mean panel error and its standard error;
    2. keep every candidate within ONE standard error of it — these are not
       distinguishable on the primary criterion at this sample of anchors;
    3. among those, choose the lowest subgroup RMSE across the 27 moderator
       levels.

    Ties on the guardrail break on lower panel error, then on the name, so the
    rule is deterministic and reproducible.
    """
    scored = [c for c in candidates if not math.isnan(c["panel_ae"])]
    if not scored:
        raise ValueError("no candidate produced a finite panel error")
    best = min(scored, key=lambda c: c["panel_ae"])
    se = best["panel_ae_se"]
    threshold = best["panel_ae"] + (se if math.isfinite(se) else 0.0)
    within = [c for c in scored if c["panel_ae"] <= threshold]
    chosen = min(within, key=lambda c: (
        c["subgroup_rmse"] if math.isfinite(c["subgroup_rmse"]) else math.inf,
        c["panel_ae"], c["name"]))
    return {
        "chosen": chosen,
        "best_panel_ae": best["panel_ae"],
        "panel_ae_se": se,
        "one_se_threshold": threshold,
        "n_within_one_se": len(within),
        "within_one_se": sorted(c["name"] for c in within),
        "rule": "min panel_ae; keep all within +1 SE; among those pick min "
                "subgroup_rmse over the 27 moderator levels; ties -> lower "
                "panel_ae, then name",
    }
