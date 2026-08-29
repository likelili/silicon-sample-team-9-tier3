"""Prediction adapter over the authors' ``SyntheticControl``.

The published class is built for *evaluation*: ``evaluate_column`` fits on the
synthetic side, predicts the human column, scores it against the known truth —
and returns the metrics and the fitted model, **but never the predictions**.
Our production targets have no human truth to score against, so the
predictions are the whole point.

Two paths, chosen by how much of the authors' code already does the job:

* **Matrix-completion methods** (``mc_soft_svd`` / ``mc_hard_svd`` / ``mc_als``)
  — the private ``_matrix_completion_predict_column`` already returns
  ``(y_pred_real, train_mse, fitted_model)`` and NaNs the real target out
  before completing, so it is production-ready verbatim.  We call it directly.

* **Regression methods** (``ridge`` / ``lasso`` / ``elastic_net`` /
  ``synthetic_control``) — ``evaluate_column`` computes the prediction inline
  and discards it.  ``predict_column`` below replays that path step for step
  using the authors' own building blocks (``hard_impute_svd``, their
  normalisation, their fit functions), returning what they compute internally.
  ``verify_against_evaluate`` proves the replay is faithful: for a column whose
  human values ARE known, the adapter's prediction must reproduce the
  correlation ``evaluate_column`` reports for the same configuration.

``train_mse`` is passed through untouched — it is the statistic the paper's
adaptive-transfer rule gates on (their Figure 5; threshold sweep in the
published notebook).
"""

from __future__ import annotations

from typing import Any

from .common import wire_authors

wire_authors()

import numpy as np                                            # noqa: E402
from synthetic_control import SyntheticControl                # noqa: E402

_REGRESSION = {"ridge", "lasso", "elastic_net", "synthetic_control", "neural_net"}
_COMPLETION = {"mc_soft_svd", "mc_hard_svd", "mc_als"}


def build(real: np.ndarray, synthetic: np.ndarray, *, name: str,
          imputation_rank: int, min_col_std: float) -> SyntheticControl:
    return SyntheticControl(
        real, synthetic, dataset_name=name,
        imputation_rank=imputation_rank, min_col_std=min_col_std,
    )


def predict_column(
    sc: SyntheticControl,
    target: int,
    *,
    method: str,
    donor_mask: np.ndarray | None = None,
    imputed_cache: dict | None = None,
    fit_finite_only: bool = True,
    **params: Any,
) -> tuple[np.ndarray, float, dict]:
    """Return ``(y_pred_real, train_mse, info)`` for one target column.

    ``imputed_cache`` may be shared across calls: donor imputation depends only
    on (target, donor set) — not on the method or ``min_col_std`` — so ridge and
    elastic net over the same donors reuse one imputation instead of paying for
    it twice.
    """
    if method in _COMPLETION:
        y_pred, train_mse, fitted = sc._matrix_completion_predict_column(
            target_col_index=target,
            method=method,
            mc_rank=params.get("mc_rank"),
            mc_max_iter=params.get("mc_max_iter", 1000),
            mc_tol=params.get("mc_tol", 1e-4),
            mc_lambda=params.get("mc_lambda", 1.0),
            verbose=False,
        )
        return np.asarray(y_pred, dtype=float), float(train_mse), {"fitted": fitted}

    if method not in _REGRESSION:
        raise ValueError(f"unsupported method {method!r}")

    n_cols = sc.real.shape[1]
    donors = np.ones(n_cols, dtype=bool)
    donors[target] = False
    if donor_mask is not None:
        donors &= donor_mask
    donor_idx = np.where(donors)[0]
    if donor_idx.size == 0:
        raise ValueError("no donor columns after masking")

    cache_key = (target, tuple(donor_idx))
    if imputed_cache is not None and cache_key in imputed_cache:
        real_imp, syn_imp = imputed_cache[cache_key]
    else:
        real_imp = SyntheticControl.hard_impute_svd(
            sc.real[:, donor_idx], rank=sc.imputation_rank)
        syn_imp = SyntheticControl.hard_impute_svd(
            sc.synthetic[:, donor_idx], rank=sc.imputation_rank)
        if imputed_cache is not None:
            imputed_cache[cache_key] = (real_imp, syn_imp)

    def normalise(matrix: np.ndarray) -> np.ndarray:
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0, ddof=0)
        stds_safe = np.where(stds < sc.min_col_std, 1.0, stds)
        return (matrix - means) / stds_safe

    X_syn = normalise(syn_imp)
    X_real = normalise(real_imp)

    syn_target = sc.synthetic[:, target]
    mean = float(np.nanmean(syn_target))
    std = float(np.nanstd(syn_target))
    if std < sc.min_col_std:
        std = 1.0
    y_syn = (syn_target - mean) / std

    # --- DOCUMENTED DEPARTURE FROM syn-digits/src/synthetic_control.py --------
    # Their evaluate_column replaces a missing synthetic target with 0.0 AFTER
    # normalisation, i.e. with the column mean, and keeps that row in the fit:
    #
    #     y_syn = np.where(np.isnan(y_syn), 0.0, y_syn)
    #
    # On their matrices that is nearly harmless. On ours it is not: Wave-4
    # assigns question variants per person, so a target column is missing for
    # roughly half the panel by design, and mean-filling would enter ~1,000
    # invented observations at exactly the column mean — shrinking the fitted
    # relationship toward zero and treating "not asked" as a substantive answer.
    #
    # ``fit_finite_only=True`` instead fits on the rows whose synthetic target is
    # finite, which is the intended estimand. Set False to reproduce their code
    # verbatim (used by verify_against_evaluate on complete targets, where the
    # two are identical because there is nothing to fill).
    if fit_finite_only:
        fit_rows = np.isfinite(y_syn)
        if fit_rows.sum() < 2:
            raise ValueError(
                f"target column has {int(fit_rows.sum())} finite synthetic value(s); "
                "cannot fit")
    else:
        fit_rows = np.ones_like(y_syn, dtype=bool)
        y_syn = np.where(np.isnan(y_syn), 0.0, y_syn)

    X_syn_fit, y_syn_fit = X_syn[fit_rows], y_syn[fit_rows]

    reg = params.get("regularization_multiplier", 1e-6)
    if method == "ridge":
        w, b, train_mse = sc._linear_regression_l2(X_syn_fit, y_syn_fit, regularization_multiplier=reg)
    elif method == "lasso":
        w, b, train_mse = sc._lasso_regression(X_syn_fit, y_syn_fit, regularization_multiplier=reg)
    elif method == "elastic_net":
        w, b, train_mse = sc._elastic_net_regression(
            X_syn_fit, y_syn_fit, regularization_multiplier=reg,
            l1_ratio=params.get("en_l1_ratio", 0.5))
    elif method == "neural_net":
        # The MLP has no closed-form weights to carry, so the authors' helper
        # trains and predicts in one call; X_real is the transfer design.
        y_pred_norm, train_mse, meta = sc._neural_net_regression_predict(
            X_syn_fit, y_syn_fit, X_real,
            nn_hidden_dims=params.get("nn_hidden_dims"),
            nn_epochs=params.get("nn_epochs", 300),
            nn_lr=params.get("nn_lr", 1e-3),
            nn_weight_decay=params.get("nn_weight_decay", 1e-6),
            nn_batch_size=params.get("nn_batch_size", 256),
            nn_patience=params.get("nn_patience", 20),
            nn_device=params.get("nn_device", "auto"),
            nn_seed=params.get("nn_seed", 42),
            verbose=False,
        )
        y_pred = np.asarray(y_pred_norm, dtype=float) * std + mean
        return y_pred, float(train_mse), {
            "nn": meta, "target_mean": mean, "target_std": std,
            "num_donors": int(donor_idx.size), "n_fit_rows": int(fit_rows.sum()),
        }
    else:  # synthetic_control — simplex-constrained, no intercept
        w, train_mse = sc._mirror_descent_simplex(
            X_syn_fit, y_syn_fit, regularization_multiplier=reg)
        b = 0.0

    y_pred = (X_real @ np.asarray(w) + b) * std + mean
    return np.asarray(y_pred, dtype=float), float(train_mse), {
        "weights": np.asarray(w), "intercept": float(b),
        "target_mean": mean, "target_std": std, "num_donors": int(donor_idx.size),
        "n_fit_rows": int(fit_rows.sum()),
    }


def verify_against_evaluate(sc: SyntheticControl, targets: list[int], *,
                            method: str, tol: float = 1e-6, **params: Any) -> list[dict]:
    """Prove the adapter reproduces ``evaluate_column`` for known columns.

    For each target the adapter's prediction is scored against the real column
    exactly as the authors score theirs (Pearson over observed cells); the two
    correlations must agree to ``tol``.  Run once per configuration before any
    production use — a drift here means the replay is no longer their method.
    """
    from scipy.stats import pearsonr

    reports = []
    for target in targets:
        theirs = sc.evaluate_column(target_col_index=target, method=method,
                                    verbose=False, **params)
        y_pred, train_mse, _ = predict_column(
            sc, target, method=method, fit_finite_only=False, **params)
        truth = sc.real[:, target]
        mask = ~np.isnan(truth)
        ours = float(pearsonr(truth[mask], y_pred[mask])[0])
        ref = float(theirs["metrics"]["correlation"])
        reports.append({
            "target": target, "method": method,
            "corr_adapter": round(ours, 6), "corr_evaluate_column": round(ref, 6),
            "train_mse_adapter": round(train_mse, 6),
            "train_mse_evaluate_column": round(float(theirs["train_mse"]), 6),
            "match": abs(ours - ref) <= tol,
        })
    return reports
