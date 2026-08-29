"""Hyperparameter tuning, with an honest split between tuning and reporting.

The authors published tuned hyperparameters for Twin-2K-500 (Table 8), and that
matrix IS our anchor block — so their grid is a strong starting point.  It is
not sufficient, for three reasons:

  1. They tuned for **individual-level correlation**.  We submit **group means**,
     and the first sweep showed the two disagree sharply: the matrix-completion
     methods scored worst on item r and best on panel-mean error.
  2. Their synthetic side is GPT-4.1-mini.  Ours is Luna, a different simulator
     with a different bias structure, so the twin->human map is not theirs.
  3. ``min_col_std`` interacts with regularisation strength, and we have reason
     to vary it: at the published value of 1.0, 65 of our 126 columns are
     centred but never scaled.

Protocol.  The 123 anchors are split once, by seed, into a TUNE half and a
HOLDOUT half.

  * The grid is searched on TUNE **only**, and ONE configuration is frozen
    there — method, hyperparameters, donor policy, ``min_col_std`` and gate
    threshold — by the rule in ``scoring.select_within_one_se``, which is
    written down before any result is produced.
  * The HOLDOUT half then evaluates **that single frozen choice** and nothing
    else.  It is never used to compare methods: choosing among several methods
    after seeing their holdout scores would make the holdout a second tuning
    set, and its number would stop being an out-of-sample estimate.

Every anchor still serves as a donor throughout — the split governs which
columns' scores may influence a choice, not which columns exist, and that
matches production where all 123 anchors (and only those) are donors.
"""

from __future__ import annotations

import math
import random

import numpy as np
from joblib import Parallel, delayed

from . import authors
from .common import CALIB_DATA, anchor_matrices, clean_pool, demographics, save_json, tier1_ids
from .scoring import (declared_spans, group_masks, score_column,
                      select_within_one_se, summarise)

TAU_GRID = [round(0.01 * k, 2) for k in range(0, 101)]
SPLIT_SEED = 20260828


def screen_grid(include_nn: bool) -> list[dict]:
    """One configuration per method, at the paper's Table 8 setting.

    Screening first is not just a speed trick: a 118-configuration grid spends
    most of its time on methods that a 16-configuration pass already rules out.
    Soft SVD alone is 70% of full-grid cost (5.97s per config against 0.28s for
    ridge), so refining only the methods that survive screening is where the
    time goes back.
    """
    from .common import PUBLISHED

    methods = ["ridge", "lasso", "elastic_net", "synthetic_control",
               "mc_hard_svd", "mc_soft_svd", "mc_als"]
    if include_nn:
        methods.append("neural_net")
    grid = []
    for method in methods:
        for std in (1.0, 0.1):
            grid.append({"name": f"{method}_table8|std{std}", "method": method,
                         "min_col_std": std, **PUBLISHED[method]})
    return grid


def build_grid(include_nn: bool, methods: set[str] | None = None) -> list[dict]:
    """The search grid, centred on Table 8 and widened around it."""
    grid: list[dict] = []

    def add(method: str, label: str, **params):
        if methods is not None and method not in methods:
            return
        for std in (1.0, 0.1):
            grid.append({"name": f"{label}|std{std}", "method": method,
                         "min_col_std": std, **params})

    for lam in (1, 10, 100, 1000, 5000):                     # Table 8: 100
        add("ridge", f"ridge_l{lam}", regularization_multiplier=lam)
    for alpha in (0.0001, 0.001, 0.01, 0.1):                 # Table 8: 0.001
        add("lasso", f"lasso_a{alpha}", regularization_multiplier=alpha)
    for alpha in (0.001, 0.01, 0.1, 1.0):                    # Table 8: 0.01 / 0.3
        for ratio in (0.1, 0.3, 0.5, 0.9):
            add("elastic_net", f"en_a{alpha}_r{ratio}",
                regularization_multiplier=alpha, en_l1_ratio=ratio)
    for lam in (1e-8, 1e-6, 1e-4):                           # Table 8: 1e-6
        add("synthetic_control", f"sc_l{lam:g}", regularization_multiplier=lam)
    for rank in (2, 5, 10, 20):                              # Table 8: 5
        add("mc_hard_svd", f"hsv_r{rank}", mc_rank=rank, mc_max_iter=1000, mc_tol=1e-4)
    for rank in (5, 10, 20):                                 # Table 8: 20 / 20
        for lam in (1, 5, 20):
            # soft-impute at lambda=1 barely shrinks the singular values, so the
            # iterate never settles and it runs to the 1000-iteration cap: 18-24s
            # per fit against 3-7s at lambda=20. Measured on this data, the six
            # lambda=1 configs are 50% of the entire refine grid's cost while
            # sitting furthest from the published setting (lambda=20). Excluded,
            # and recorded here rather than silently dropped.
            if lam > 1:
                add("mc_soft_svd", f"ssv_r{rank}_l{lam}", mc_rank=rank, mc_lambda=lam,
                    mc_max_iter=1000, mc_tol=1e-4)
            add("mc_als", f"als_r{rank}_l{lam}", mc_rank=rank, mc_lambda=lam)
    if include_nn:
        for hidden in ([8], [16], [8, 8]):                   # Table 8: [8], wd 0.05
            for wd in (0.001, 0.05, 0.1):
                add("neural_net", f"nn_h{'x'.join(map(str, hidden))}_wd{wd}",
                    nn_hidden_dims=hidden, nn_weight_decay=wd, nn_epochs=200)
    return grid


def _one_target(Y, T, T_ref, target, masks, spans, grid, stds) -> list[dict]:
    """Every configuration for one held-out anchor column.

    Runs in a separate process (see ``run``), so torch's own thread pool has to
    be pinned here too — the env vars in ``calib/__init__`` cover BLAS but not
    torch's intra-op threads.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    y_true = Y[:, target]
    base = score_column(y_true, T_ref[:, target], spans[target], masks)

    cache: dict = {}
    scs = {std: authors.build(Y, T, name="tune", imputation_rank=5, min_col_std=std)
           for std in stds}
    rows = []
    for config in grid:
        sc = scs[config["min_col_std"]]
        params = {k: v for k, v in config.items()
                  if k not in ("name", "method", "min_col_std")}
        try:
            y_hat, train_mse, _ = authors.predict_column(
                sc, target, method=config["method"], imputed_cache=cache, **params)
            scored = score_column(y_true, y_hat, spans[target], masks)
            error = ""
        except Exception as exc:
            scored = {"item_r": math.nan, "panel_ae": math.nan,
                      "subgroup_rmse": math.nan, "n_obs": 0}
            train_mse, error = math.nan, f"{type(exc).__name__}: {exc}"[:140]
        rows.append({
            "target": target, "config": config["name"], "method": config["method"],
            "min_col_std": config["min_col_std"], "train_mse": train_mse,
            **{f"cal_{k}": v for k, v in scored.items()},
            **{f"base_{k}": v for k, v in base.items()},
            "error": error,
        })
    return rows


def run(n_jobs: int = 10, include_nn: bool = True, refine_top: int = 3,
        progress: bool = True,
        refine_exclude: frozenset = frozenset({"mc_soft_svd"})) -> dict:
    """Screen, refine, freeze ONE configuration on tune anchors, then hold out.

    ``n_jobs`` defaults to all cores.  This machine is 4 performance + 6
    efficiency cores, so workers are not interchangeable; with many more tasks
    than workers joblib's dynamic dispatch absorbs that, which is why the task
    list is anchor columns rather than one chunk per worker.
    """
    import time

    pool = clean_pool()
    panel, _ = tier1_ids()
    Y, T, items = anchor_matrices(pool)
    masks = group_masks(pool, panel, demographics(pool))
    spans = declared_spans(items)
    stds = [1.0, 0.1]

    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    tune_idx = sorted(order[: len(order) // 2])
    hold_idx = sorted(order[len(order) // 2:])

    if progress:
        print(f"anchors: {len(items)} | declared span available for "
              f"{int(np.isfinite(spans).sum())}", flush=True)
        print(f"moderator levels: {masks['n_levels_scored']} scored of "
              f"{masks['n_levels_declared']} declared"
              f"{'  (empty: ' + ', '.join(masks['empty']) + ')' if masks['empty'] else ''}",
              flush=True)
        print(f"split: {len(tune_idx)} tune / {len(hold_idx)} holdout "
              f"(seed {SPLIT_SEED})", flush=True)

    def sweep(targets, grid, label):
        started = time.time()
        if progress:
            print(f"[{label}] {len(grid)} configs x {len(targets)} anchors "
                  f"on {n_jobs} workers", flush=True)
        batches = Parallel(n_jobs=n_jobs, backend="loky",
                           verbose=10 if progress else 0)(
            delayed(_one_target)(Y, T, T, t, masks, spans, grid, stds) for t in targets)
        rows = [row for batch in batches for row in batch]
        if progress:
            print(f"[{label}] {len(rows)} rows in {time.time() - started:.0f}s", flush=True)
        return rows

    def candidates(rows, grid):
        """Every (config, tau) pair scored on the tuning anchors."""
        out = []
        for config in grid:
            subset = [r for r in rows if r["config"] == config["name"]]
            if not subset:
                continue
            for tau in TAU_GRID:
                entry = summarise(subset, tau)
                entry["name"] = f"{config['name']}@tau{tau:.2f}"
                entry["config"] = config["name"]
                entry["method"] = config["method"]
                entry["min_col_std"] = config["min_col_std"]
                out.append(entry)
        return out

    # --- stage 1: screen every method at its Table 8 setting ---------------
    s_grid = screen_grid(include_nn)
    s_rows = sweep(tune_idx, s_grid, "screen")
    s_cand = candidates(s_rows, s_grid)
    by_method: dict[str, float] = {}
    for entry in s_cand:
        best = by_method.get(entry["method"])
        if best is None or entry["panel_ae"] < best:
            by_method[entry["method"]] = entry["panel_ae"]
    ordered = sorted(by_method, key=lambda m: by_method[m])

    # Refinement is bounded by COST, not by screening rank. Measured on this
    # data, one soft-impute fit costs 5.2s against 0.01s for elastic net, and
    # soft SVD alone is 56% of the full grid; every other method's whole
    # hyperparameter sweep is cheap. Cutting by rank would have dropped elastic
    # net (32 configs, 0.2s per anchor) for no saving at all.
    #
    # A method excluded here is NOT excluded from selection: its screened
    # Table 8 configuration stays in the candidate pool with the full tau sweep,
    # so it can still be frozen — it just does not get hyperparameters tuned.
    survivors = [m for m in ordered if m not in refine_exclude]
    if progress:
        print(f"[screen] method order: {', '.join(ordered)}", flush=True)
        print(f"[screen] refining {len(survivors)} method(s): {', '.join(survivors)}",
              flush=True)
        if refine_exclude:
            print(f"[screen] not refined (cost): {', '.join(sorted(refine_exclude))} "
                  f"— screened settings remain selectable", flush=True)

    # --- stage 2: full grid for the survivors ------------------------------
    r_grid = build_grid(include_nn, set(survivors))
    r_rows = sweep(tune_idx, r_grid, "refine")
    all_cand = s_cand + candidates(r_rows, r_grid)

    # --- stage 3: FREEZE one configuration, on tune anchors only -----------
    selection = select_within_one_se(all_cand)
    frozen = selection["chosen"]
    if progress:
        print(f"[freeze] {frozen['name']}  panel_ae {frozen['panel_ae']:.4f} pp "
              f"(best {selection['best_panel_ae']:.4f} +/- {selection['panel_ae_se']:.4f} SE; "
              f"{selection['n_within_one_se']} configs within 1 SE)", flush=True)

    # --- stage 4: the holdout sees ONLY the frozen configuration -----------
    frozen_grid = [c for c in (s_grid + r_grid) if c["name"] == frozen["config"]][:1]
    h_rows = sweep(hold_idx, frozen_grid, "holdout")
    holdout = summarise(h_rows, frozen["tau"])
    base_ae = float(np.nanmean([r["base_panel_ae"] for r in h_rows]))
    base_sg = float(np.nanmean([r["base_subgroup_rmse"] for r in h_rows]))
    base_r = float(np.nanmean([r["base_item_r"] for r in h_rows]))

    out = {
        "protocol": "grid searched on tune anchors; ONE configuration frozen there by "
                    "the prespecified 1-SE rule; holdout evaluates only that choice",
        "split_seed": SPLIT_SEED,
        "tune_anchors": len(tune_idx),
        "holdout_anchors": len(hold_idx),
        "anchors_with_declared_span": int(np.isfinite(spans).sum()),
        "moderator_levels_scored": masks["n_levels_scored"],
        "moderator_levels_declared": masks["n_levels_declared"],
        "moderator_levels_empty": masks["empty"],
        "moderator_level_sizes": masks["sizes"],
        "screen_configs": len(s_grid),
        "refine_configs": len(r_grid),
        "method_order_on_tune": ordered,
        "refined_methods": survivors,
        "not_refined_for_cost": sorted(refine_exclude),
        "selection_rule": selection["rule"],
        "selection": {k: v for k, v in selection.items() if k != "chosen"},
        "frozen": {
            "config": frozen["config"], "method": frozen["method"],
            "min_col_std": frozen["min_col_std"], "tau": frozen["tau"],
            "donor_policy": "the 123 Wave-4 anchor columns only",
            "tune_panel_ae": frozen["panel_ae"],
            "tune_subgroup_rmse": frozen["subgroup_rmse"],
            "tune_item_r": frozen["item_r"],
            "tune_transferred": f"{frozen['transferred']}/{frozen['n_scored_items']}",
        },
        "holdout": {
            "panel_ae": holdout["panel_ae"], "panel_ae_se": holdout["panel_ae_se"],
            "subgroup_rmse": holdout["subgroup_rmse"], "item_r": holdout["item_r"],
            "transferred": f"{holdout['transferred']}/{holdout['n_scored_items']}",
            "baseline_panel_ae": base_ae, "baseline_subgroup_rmse": base_sg,
            "baseline_item_r": base_r,
            "improvement_pct": (100 * (holdout["panel_ae"] - base_ae) / base_ae
                                if base_ae else float("nan")),
        },
    }
    CALIB_DATA.mkdir(parents=True, exist_ok=True)
    save_json(out, CALIB_DATA / "tuning_report.json")

    import csv

    cols = ["target", "config", "method", "min_col_std", "train_mse",
            "cal_item_r", "cal_panel_ae", "cal_subgroup_rmse",
            "base_item_r", "base_panel_ae", "base_subgroup_rmse",
            "n_obs", "n_levels", "error"]
    for name, rows in (("tuning_tune_fold.csv", s_rows + r_rows),
                       ("tuning_holdout_fold.csv", h_rows)):
        with open(CALIB_DATA / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return out
