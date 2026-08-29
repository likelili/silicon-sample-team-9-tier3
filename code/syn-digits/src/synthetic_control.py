from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from causaltensor.matlib import SVD
from joblib import Parallel, delayed

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'outputs')
SYNTHETIC_CONTROL_DIR = os.path.join(OUTPUT_DIR, 'synthetic_control')

class SyntheticControl:
    """Synthetic control with hard imputation to avoid data leakage.

    Attributes:
        real (np.ndarray): Real-valued matrix of shape (n_rows, n_cols), may contain NaNs.
        synthetic (np.ndarray): Synthetic-valued matrix of shape (n_rows, n_cols), may contain NaNs.
        additional_baseline (Optional[np.ndarray]): Optional additional baseline matrix of shape (n_rows, n_cols),
                                                    used for additional baseline comparisons.
        nanmask (np.ndarray): Boolean mask indicating originally missing values in real_matrix (before any imputation).
                              True indicates a missing value.
        dataset_name (str): Name of the dataset for saving results.
        imputation_rank (Optional[int]): Rank to use for hard imputation. If None, will be selected via CV.
        imputation_rank_max (Optional[int]): Maximum rank to test during CV. If None, defaults to min(20, min(n,m)//10).
        imputation_rank_step (int): Step size for rank range during CV. Default: 5.
        imputation_holdout_fraction (float): Fraction of observed values to hold out for validation. Default: 0.2.
        min_col_std (float): Columns with std below this are treated as low-variance during normalization. Default: 1.
        results_dir (str): Directory for saving results.
        results_figures_dir (str): Directory for saving figures.

    """

    def __init__(
        self,
        real_matrix: np.ndarray,
        synthetic_matrix: np.ndarray,
        dataset_name: str,
        additional_baseline_matrix: Optional[np.ndarray] = None,
        imputation_rank: Optional[int] = None,
        imputation_rank_max: Optional[int] = None,
        imputation_rank_step: int = 5,
        imputation_holdout_fraction: float = 0.2,
        min_col_std: float = 1,
    ) -> None:
        """Initialize the synthetic control model with data and hyperparameters.

        Args:
            real_matrix: Real-valued data matrix of shape (num_rows, num_cols), may contain NaNs.
            synthetic_matrix: Synthetic data matrix of the same shape as `real_matrix`, may contain NaNs.
            additional_baseline_matrix: Optional matrix of shape (num_rows, num_cols) used as an
                                        additional baseline for evaluation. Default: None.
            dataset_name: Name of the dataset. Used for saving the results.
            imputation_rank: Rank to use for hard imputation. If None, will be selected via CV for each
                           evaluation scenario. Default: None.
            imputation_rank_max: Maximum rank to test during CV. If None, defaults to min(20, min(n, m) // 10).
                                Default: None. (More conservative than process_and_diagnostics.py default of // 5)
            imputation_rank_step: Step size for rank range during CV. Larger values = faster CV.
                                 Default: 5. (Larger than process_and_diagnostics.py default of 3)
            imputation_holdout_fraction: Fraction of observed values to hold out for validation during CV.
                                        Default: 0.2.
            min_col_std: Threshold below which columns are considered low-variance during normalization. Default: 1.

        Returns:
            None
        """
        if real_matrix.shape != synthetic_matrix.shape:
            raise ValueError("real_matrix and synthetic_matrix must have the same shape")
        if real_matrix.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")
        if additional_baseline_matrix is not None:
            if additional_baseline_matrix.shape != real_matrix.shape:
                raise ValueError("additional_baseline_matrix must have the same shape as real_matrix")
            if additional_baseline_matrix.ndim != 2:
                raise ValueError("additional_baseline_matrix must be a 2D matrix")

        self.real: np.ndarray = np.asarray(real_matrix, dtype=float)
        self.synthetic: np.ndarray = np.asarray(synthetic_matrix, dtype=float)
        self.additional_baseline: Optional[np.ndarray] = (
            np.asarray(additional_baseline_matrix, dtype=float)
            if additional_baseline_matrix is not None
            else None
        )

        # Store the nanmask - this represents originally missing values before any imputation
        self.nanmask: np.ndarray = np.isnan(self.real)
        percentage_missing = np.sum(self.nanmask) / np.prod(self.nanmask.shape)
        print(f"Data contains {percentage_missing * 100:.2f}% missing values")

        self.dataset_name = dataset_name
        self.imputation_rank = imputation_rank
        self.imputation_rank_max = imputation_rank_max
        self.imputation_rank_step = imputation_rank_step
        self.imputation_holdout_fraction = imputation_holdout_fraction

        self.results_dir = os.path.join(SYNTHETIC_CONTROL_DIR, dataset_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.results_figures_dir = os.path.join(self.results_dir, 'figures')
        if not os.path.exists(self.results_figures_dir):
            os.makedirs(self.results_figures_dir)

        self.min_col_std: float = min_col_std

    @staticmethod
    def hard_impute_svd(
        X: np.ndarray,
        rank: int = 5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> np.ndarray:
        """Perform hard imputation using SVD with rank constraint.

        Implements the hard imputation algorithm: iteratively performs low-rank SVD
        approximation and replaces missing values with the approximation, while keeping
        observed values fixed. The algorithm converges when the change between iterations
        falls below the tolerance threshold.

        Args:
            X: Input matrix with missing values (NaNs), shape (n_rows, n_cols).
            rank: Target rank for matrix completion. Must be positive and <= min(n_rows, n_cols).
                 Default: 5.
            max_iter: Maximum number of iterations. Default: 1000.
            tol: Convergence tolerance. Algorithm stops when relative Frobenius norm change
                between iterations is below this value. Default: 1e-4.
            verbose: If True, print convergence information every 10 iterations. Default: False.

        Returns:
            Completed matrix with missing values filled, same shape as X.

        Note:
            Missing values are initially filled with column means. Columns with all NaNs
            are filled with zeros.
        """
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)

        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X_filled[mask, j] = col_means[j]

        # Handle columns with all NaNs
        X_filled = np.nan_to_num(X_filled, nan=0)

        prev_X = X_filled.copy()

        for iteration in range(max_iter):
            try:
                X_filled = SVD(X_filled, min(min(X_filled.shape), rank))
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Warning: SVD did not converge at iteration {iteration}. Using previous result.")
                return prev_X if iteration > 0 else X_filled

            mask_observed = ~np.isnan(X)
            X_filled[mask_observed] = X[mask_observed]

            diff = np.linalg.norm(X_filled - prev_X, 'fro') / np.linalg.norm(prev_X, 'fro')
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Diff: {diff}")
            if diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_X = X_filled.copy()

        return X_filled

    @staticmethod
    def soft_impute_svd(
        X: np.ndarray,
        rank: int = 5,
        lambda_: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> np.ndarray:
        """Perform soft-thresholded SVD imputation (nuclear-norm regularisation).

        Iteratively computes a rank-k SVD, soft-thresholds the singular values
        by subtracting ``lambda_``, and fills missing entries with the resulting
        low-rank approximation while preserving observed values.

        Args:
            X: Input matrix with missing values (NaNs), shape (n_rows, n_cols).
            rank: Maximum rank for the truncated SVD.  Must be positive and
                  <= min(n_rows, n_cols).  Default: 5.
            lambda_: Soft-threshold parameter applied to singular values.
                     Larger values impose stronger nuclear-norm shrinkage.
                     Default: 1.0.
            max_iter: Maximum number of iterations.  Default: 1000.
            tol: Convergence tolerance on relative Frobenius norm change.
                 Default: 1e-4.
            verbose: If True, print convergence information every 10 iterations.
                     Default: False.

        Returns:
            Completed matrix with missing values filled, same shape as X.

        Note:
            Missing values are initially filled with column means.  Columns
            that are entirely NaN are filled with zeros.
        """
        mask_missing = np.isnan(X)
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)

        for j in range(X.shape[1]):
            col_nan = np.isnan(X[:, j])
            fill_val = col_means[j] if np.isfinite(col_means[j]) else 0.0
            X_filled[col_nan, j] = fill_val
        X_filled = np.nan_to_num(X_filled, nan=0.0)

        prev_X = X_filled.copy()

        for iteration in range(max_iter):
            try:
                U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
            except np.linalg.LinAlgError:
                if verbose:
                    print(
                        f"Warning: SVD did not converge at iteration {iteration}. "
                        "Using previous result."
                    )
                return prev_X if iteration > 0 else X_filled

            k = min(rank, len(s))
            s_thresh = np.maximum(s[:k] - lambda_, 0.0)
            X_filled = (U[:, :k] * s_thresh) @ Vt[:k, :]

            X_filled[~mask_missing] = X[~mask_missing]

            prev_norm = np.linalg.norm(prev_X, "fro")
            if prev_norm < 1e-12:
                diff = np.linalg.norm(X_filled - prev_X, "fro")
            else:
                diff = np.linalg.norm(X_filled - prev_X, "fro") / prev_norm
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Diff: {diff}")
            if diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_X = X_filled.copy()

        return X_filled

    @staticmethod
    def als_complete(
        X: np.ndarray,
        rank: int = 5,
        lambda_: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        """Complete a matrix using Alternating Least Squares (ALS).

        Factorises ``X ≈ U @ V.T`` where ``U`` is ``(n, rank)`` and ``V`` is
        ``(m, rank)``.  Only observed (non-NaN) entries contribute to the loss.
        Each alternating step solves a ridge-regularised least-squares problem.

        Args:
            X: Input matrix with NaN for missing entries, shape (n, m).
            rank: Number of latent factors.  Default: 5.
            lambda_: L2 regularisation strength on the factor matrices.
                     Default: 0.1.
            max_iter: Maximum number of alternating iterations.  Default: 100.
            tol: Convergence tolerance on relative change in the objective.
                 Default: 1e-4.
            verbose: Print objective every 10 iterations.  Default: False.
            random_state: Seed for the random initialisation of ``U`` and ``V``.
                          Default: 42.

        Returns:
            Completed matrix ``U @ V.T``, same shape as *X*.
        """
        n, m = X.shape
        rng = np.random.RandomState(random_state)

        U = rng.randn(n, rank) * 0.01
        V = rng.randn(m, rank) * 0.01

        obs = ~np.isnan(X)
        eye_k = lambda_ * np.eye(rank)

        prev_obj = float("inf")

        for iteration in range(max_iter):
            for i in range(n):
                obs_cols = np.where(obs[i, :])[0]
                if len(obs_cols) == 0:
                    continue
                V_sub = V[obs_cols, :]
                x_sub = X[i, obs_cols]
                U[i, :] = np.linalg.solve(V_sub.T @ V_sub + eye_k, V_sub.T @ x_sub)

            for j in range(m):
                obs_rows = np.where(obs[:, j])[0]
                if len(obs_rows) == 0:
                    continue
                U_sub = U[obs_rows, :]
                x_sub = X[obs_rows, j]
                V[j, :] = np.linalg.solve(U_sub.T @ U_sub + eye_k, U_sub.T @ x_sub)

            X_approx = U @ V.T
            residuals = X_approx[obs] - X[obs]
            obj = float(np.sum(residuals**2) + lambda_ * (np.sum(U**2) + np.sum(V**2)))

            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
            if verbose and iteration % 10 == 0:
                print(f"ALS iter {iteration + 1}, obj={obj:.6f}, rel_change={rel_change:.2e}")
            if rel_change < tol:
                if verbose:
                    print(f"ALS converged after {iteration + 1} iterations")
                break
            prev_obj = obj

        return U @ V.T

    @staticmethod
    def select_optimal_rank_cv(
        X: np.ndarray,
        max_rank: Optional[int] = None,
        rank_step: int = 3,
        holdout_fraction: float = 0.2,
        verbose: bool = False,
        random_state: Optional[int] = None
    ) -> Tuple[int, float]:
        """Select optimal rank using cross-validation with holdout validation.

        Uses a holdout validation approach to select the optimal rank for matrix completion.
        The method tests a range of ranks and selects the one with the lowest validation error.

        Args:
            X: Matrix for completion with missing values (NaNs), shape (n_rows, n_cols).
            max_rank: Maximum rank to test. If None, defaults to min(20, min(n, m) // 10).
                     If outside [1, min(n, m)], will be clamped to valid range. Default: None.
            rank_step: Step size for rank range (e.g., 3 means testing ranks 1, 4, 7, ...).
                      Default: 3.
            holdout_fraction: Fraction of observed values to hold out for validation.
                            Must be in (0, 1). Default: 0.2.
            verbose: If True, print progress for each rank tested. Default: False.
            random_state: Random seed for holdout mask generation. If None, uses current np.random state.
                         Should be different for each parallel task for diverse CV splits. Default: None.

        Returns:
            Tuple of (optimal_rank, best_error):
                - optimal_rank: The rank with the lowest validation error (int)
                - best_error: The validation error for the optimal rank (float)

        Note:
            Uses random holdout, so results may vary between runs. For reproducibility,
            set numpy random seed before calling.
        """
        n, m = X.shape
        if max_rank is None:
            # Default: more conservative (smaller) than process_and_diagnostics.py (which uses // 5)
            max_rank = min(20, min(n, m) // 10)
            if max_rank < 1:
                max_rank = min(5, min(n, m))  # At least try up to rank 5
        elif max_rank > min(n, m) or max_rank < 1:
            print(f"max_rank is outside of allowed range [1, min(n, m)], setting max_rank to min(n, m)")
            max_rank = min(n, m)

        rank_range = list(range(1, max_rank + 1, rank_step))

        observed_mask = ~np.isnan(X)

        # Generate holdout mask with optional random state for reproducibility
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            holdout_mask = rng.rand(n, m) < holdout_fraction
        else:
            holdout_mask = np.random.rand(n, m) < holdout_fraction
        validation_mask = observed_mask & holdout_mask

        X_observed = X.copy()
        X_observed[validation_mask] = np.nan

        best_error = np.inf
        best_rank = None

        errors = []
        for rank in rank_range:
            try:
                Y_completed = SyntheticControl.hard_impute_svd(X_observed, rank=rank, verbose=False)

                denom = np.linalg.norm(X[validation_mask])
                if denom == 0:
                    # If all validation values are zero, fallback to absolute error
                    error = np.linalg.norm(Y_completed[validation_mask] - X[validation_mask])
                else:
                    error = np.linalg.norm(Y_completed[validation_mask] - X[validation_mask]) / denom
                errors.append(error)

                if error < best_error:
                    best_error = error
                    best_rank = rank

                if verbose:
                    print(f"Rank: {rank}, Error: {error:.6f}")
            except (np.linalg.LinAlgError, ValueError):
                if verbose:
                    print(f"Rank: {rank}, Skipped (SVD error)")
                errors.append(np.inf)
                continue

        # Fallback if all ranks failed
        if best_rank is None:
            best_rank = 3 
            best_error = np.inf
            if verbose:
                print(f"Warning: All ranks failed CV. Using fallback rank={best_rank}")

        if verbose and best_error != np.inf:
            print(f"\nResults:")
            print(f"  Optimal rank: {best_rank}")
            print(f"  Best error: {best_error:.6f}")
            finite_errors = [e for e in errors if e != np.inf]
            if len(finite_errors) > 1:
                print(f"  Error reduction: {(max(finite_errors) - best_error)/max(finite_errors)*100:.1f}%")

        return best_rank, best_error

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if len(y_true) < 2:
            return 0.0
        if np.std(y_true) <= 1e-2 or np.std(y_pred) <= 1e-2:
            return 0.0
        return float(pearsonr(y_true, y_pred)[0])

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R^2 score."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a bounded accuracy proxy based on absolute error and value range."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if np.max(y_true) > np.min(y_true):
            return float(1 - np.mean(np.abs(y_true - y_pred)) / (np.max(y_true) - np.min(y_true)))
        else:
            return float(np.nan)

    @staticmethod
    def _wasserstein_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Wasserstein distance normalized by the range of y_true."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        try:
            wass = wasserstein_distance(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
        except (ValueError, ZeroDivisionError):
            wass = np.nan
        return float(wass)

    @staticmethod
    def fisher_z_average(correlations: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Average correlations using Fisher z-transformation.

        Args:
            correlations: Array of correlation values.
            weights: Optional weights for weighted averaging (e.g., sample sizes). Default: None.

        Returns:
            Average correlation value (transformed back from z-space).
        """
        mask = ~np.isnan(correlations)
        clean_corrs = correlations[mask]

        if len(clean_corrs) == 0:
            return np.nan

        if weights is not None:
            clean_weights = weights[mask]
            if len(clean_weights) != len(clean_corrs):
                clean_weights = None  # Fall back to unweighted if mismatch
        else:
            clean_weights = None

        # Clamp correlations to valid range to avoid arctanh issues
        clean_corrs = np.clip(clean_corrs, -0.9999, 0.9999)

        z_values = np.arctanh(clean_corrs)

        if clean_weights is not None and len(clean_weights) == len(z_values):
            avg_z = np.average(z_values, weights=clean_weights)
        else:
            avg_z = np.mean(z_values)

        avg_r = np.tanh(avg_z)

        return avg_r
    

    @staticmethod
    def _std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute standard deviation ratio (std_pred / std_true)."""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        if std_true > 0:
            return float(std_pred / std_true)
        else:
            return float(np.nan)


    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute a comprehensive set of scalar metrics comparing predictions to ground truth.

        Args:
            y_true: Ground-truth vector or array-like.
            y_pred: Predicted vector or array-like.
            mask: Optional boolean mask. If provided, only values where mask is False are used.
                  True indicates a missing value that should be excluded.

        Returns:
            Dictionary mapping metric names to float values.
        """
        if mask is not None:
            mask = np.asarray(mask, dtype=bool).reshape(y_true.shape)
            valid_mask = ~mask
            if valid_mask.sum() == 0:
                return {
                    "mse": float(np.nan),
                    "correlation": float(np.nan),
                    "r2": float(np.nan),
                    "accuracy": float(np.nan),
                    "wasserstein_distance": float(np.nan),
                    "std_ratio": float(np.nan),
                }
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

        return {
            "mse": SyntheticControl._mse(y_true, y_pred),
            "correlation": SyntheticControl._correlation(y_true, y_pred),
            "r2": SyntheticControl._r2(y_true, y_pred),
            "accuracy": SyntheticControl._accuracy(y_true, y_pred),
            "wasserstein_distance": SyntheticControl._wasserstein_distance(y_true, y_pred),
            "std_ratio": SyntheticControl._std_ratio(y_true, y_pred),
        }

    @staticmethod
    def _denormalize_column(vec_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Denormalize a normalized vector using column mean and std.

        Args:
            vec_norm: Normalized vector.
            mean: Mean to add back.
            std: Standard deviation to scale by.

        Returns:
            Denormalized vector.
        """
        return vec_norm * std + mean

    def _normalize_columns(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Column-wise zero-mean unit-variance normalisation.

        Returns:
            (X_normalised, col_means, col_stds_safe) where low-variance
            columns have their std replaced by 1.0 and all-NaN column
            means are replaced by 0.0.
        """
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        stds_safe = stds.copy()
        stds_safe[stds < self.min_col_std] = 1.0
        means = np.where(np.isnan(means), 0.0, means)
        X_norm = (X - means) / stds_safe
        return X_norm, means, stds_safe

    def _mirror_descent_simplex(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization_multiplier: float = 1e-6,
        learning_rate: float = 0.01,
        max_iter: int = 3000,
        tol: float = 1e-9,
        eps: float = 1e-18,
        adaptive_lr: bool = True,
        lr_decay: float = 0.9,
        min_lr_ratio: float = 0.01,
        lr_patience: int = 10,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """Solve a simplex-constrained L2-regularized regression via mirror descent.

        Objective: min_w ||X w - y||^2 + λ||w||^2  s.t.  w ∈ Δ (probability simplex).

        Args:
            X: Design matrix of shape (num_samples, num_features).
            y: Target vector of shape (num_samples,).
            regularization_multiplier: L2 coefficient for weight shrinkage. Default: 1e-6.
            learning_rate: Initial learning rate. Default: 0.01.
            max_iter: Maximum mirror-descent iterations. Default: 3000.
            tol: Convergence tolerance on objective value. Default: 1e-9.
            eps: Lower bound for positive weights on simplex. Default: 1e-18.
            adaptive_lr: If True, use AdaGrad-like per-coordinate scaling. Default: True.
            lr_decay: Multiplicative decay when no improvement. Default: 0.9.
            min_lr_ratio: Minimum learning rate ratio relative to initial value. Default: 0.01.
            lr_patience: Iterations without improvement before decay. Default: 10.
            verbose: If True, print optimization progress. Default: False.

        Returns:
            Tuple of (weights on the simplex, training MSE).
        """
        _, d = X.shape
        if d == 0:
            return np.zeros(0, dtype=float), float("nan")

        lam = regularization_multiplier
        eta = learning_rate

        # Initialize uniformly on simplex
        w = np.full(d, 1.0 / d, dtype=float)

        Xt = X.T

        if adaptive_lr:
            grad_squared_sum = np.zeros(d, dtype=float)
            min_lr = eta * min_lr_ratio
            no_improvement_count = 0
            obj = np.inf
            best_obj = np.inf

        for i in range(max_iter):
            Xw = X @ w
            resid = Xw - y
            grad = 2.0 * (Xt @ resid) + 2.0 * lam * w

            if adaptive_lr:
                grad_squared_sum += grad * grad
                adaptive_eta = eta / (1.0 + np.sqrt(grad_squared_sum))

                if obj < best_obj:
                    best_obj = obj
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= lr_patience:
                        eta *= lr_decay
                        eta = max(eta, min_lr)
                        no_improvement_count = 0
                        if verbose:
                            print(f"Reducing learning rate to {eta:.4f}")

                w *= np.exp(-adaptive_eta * grad)
            else:
                w *= np.exp(-eta * grad)

            w = np.maximum(w, eps)
            w /= float(np.sum(w))

            obj = float(resid @ resid + lam * (w @ w))
            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Objective: {obj:.4f}, LR: {eta:.4f}")
            if obj < tol:
                if verbose:
                    print(f"Converged after {i + 1} iterations")
                break

        train_mse = self._mse(y, X @ w)
        return w, train_mse

    def _linear_regression_l2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization_multiplier: float = 1e-6,
    ) -> Tuple[np.ndarray, float, float]:
        """Fit ridge regression and return weights, intercept, and training MSE.

        Args:
            X: Design matrix of shape (num_samples, num_features).
            y: Target vector of shape (num_samples,).
            regularization_multiplier: L2 coefficient for weight shrinkage. Default: 1e-6.

        Returns:
            Tuple of (weights vector, intercept scalar, training MSE float).
        """
        if regularization_multiplier == 0:
            fit = LinearRegression(fit_intercept=True).fit(X, y)
            fitted_weights, fitted_intercept = fit.coef_, fit.intercept_
        else:
            fit = Ridge(alpha=regularization_multiplier).fit(X, y)
            fitted_weights, fitted_intercept = fit.coef_, fit.intercept_
        train_mse = self._mse(y, X @ fitted_weights + fitted_intercept)
        return fitted_weights, fitted_intercept, train_mse

    def _lasso_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization_multiplier: float = 1e-6,
    ) -> Tuple[np.ndarray, float, float]:
        """Fit Lasso (L1) regression and return weights, intercept, and training MSE."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fit = Lasso(
                alpha=max(regularization_multiplier, 1e-12),
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4,
            ).fit(X, y)
        w, b = fit.coef_, fit.intercept_
        train_mse = self._mse(y, X @ w + b)
        return w, float(b), train_mse

    def _elastic_net_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization_multiplier: float = 1e-6,
        l1_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, float, float]:
        """Fit Elastic Net regression and return weights, intercept, and training MSE."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fit = ElasticNet(
                alpha=max(regularization_multiplier, 1e-12),
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4,
            ).fit(X, y)
        w, b = fit.coef_, fit.intercept_
        train_mse = self._mse(y, X @ w + b)
        return w, float(b), train_mse

    def _neural_net_regression_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_epochs: int = 300,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-6,
        nn_batch_size: int = 256,
        nn_patience: int = 20,
        nn_device: str = "auto",
        nn_seed: int = 42,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train a small MLP regressor and predict on eval features."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "method='neural_net' requires PyTorch. Install it with `pip install torch`."
            ) from exc

        hidden_dims = [32] if nn_hidden_dims is None else list(nn_hidden_dims)
        if len(hidden_dims) == 0 or any(h <= 0 for h in hidden_dims):
            raise ValueError("nn_hidden_dims must be a non-empty list of positive integers")
        if nn_epochs <= 0:
            raise ValueError("nn_epochs must be > 0")
        if nn_lr <= 0:
            raise ValueError("nn_lr must be > 0")
        if nn_batch_size <= 0:
            raise ValueError("nn_batch_size must be > 0")
        if nn_patience <= 0:
            raise ValueError("nn_patience must be > 0")

        if nn_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("nn_device must be one of {'auto', 'cpu', 'cuda'}")

        torch.manual_seed(nn_seed)
        np.random.seed(nn_seed)

        use_cuda = torch.cuda.is_available()
        if nn_device == "auto":
            resolved_device = "cuda" if use_cuda else "cpu"
        elif nn_device == "cuda":
            resolved_device = "cuda" if use_cuda else "cpu"
            if not use_cuda and verbose:
                print("CUDA requested for neural net, but no GPU is available. Falling back to CPU.")
        else:
            resolved_device = "cpu"

        device = torch.device(resolved_device)
        input_dim = X_train.shape[1]

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=nn_lr, weight_decay=nn_weight_decay)
        criterion = nn.MSELoss()

        X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=min(nn_batch_size, X_train_tensor.shape[0]),
            shuffle=True,
        )

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_improvement = 0
        epochs_trained = 0

        for epoch in range(nn_epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                batch_size = xb.shape[0]
                epoch_loss_sum += float(loss.item()) * batch_size
                epoch_count += batch_size

            epoch_loss = epoch_loss_sum / max(epoch_count, 1)
            epochs_trained = epoch + 1
            if verbose and (epoch % 50 == 0 or epoch == nn_epochs - 1):
                print(f"Epoch {epoch + 1}/{nn_epochs}, train loss: {epoch_loss:.6f}")

            if epoch_loss + 1e-12 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= nn_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor.to(device)).cpu().numpy().reshape(-1)
            eval_pred = model(torch.as_tensor(X_eval, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

        train_mse = self._mse(y_train, train_pred)
        fitted_model = {
            "method": "neural_net",
            "predictor": {
                "state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
                "architecture": {
                    "input_dim": int(input_dim),
                    "hidden_dims": [int(h) for h in hidden_dims],
                    "output_dim": 1,
                    "activation": "relu",
                },
                "dtype": "float32",
            },
            "metadata": {
                "device": resolved_device,
                "epochs_trained": int(epochs_trained),
                "epochs_max": int(nn_epochs),
                "batch_size": int(min(nn_batch_size, X_train_tensor.shape[0])),
                "learning_rate": float(nn_lr),
                "weight_decay": float(nn_weight_decay),
                "patience": int(nn_patience),
                "seed": int(nn_seed),
            },
        }
        return eval_pred, train_mse, fitted_model

    # ------------------------------------------------------------------
    # Matrix-completion helpers
    # ------------------------------------------------------------------

    def _matrix_completion_predict_column(
        self,
        target_col_index: int,
        method: str,
        mc_rank: Optional[int],
        mc_max_iter: int,
        mc_tol: float,
        mc_lambda: float,
        verbose: bool,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Predict a target column via matrix completion on vstack([real, synthetic]).

        Real and synthetic halves are normalised **separately** before stacking.
        Predictions are denormalised using synthetic column statistics.

        Returns:
            (y_pred_real, train_mse, fitted_model) where train_mse is the
            reconstruction MSE on all observed entries in the stacked matrix.
        """
        n_rows, _ = self.real.shape

        real_norm, real_means, real_stds = self._normalize_columns(self.real)
        syn_norm, syn_means, syn_stds = self._normalize_columns(self.synthetic)

        S_norm = np.vstack([real_norm, syn_norm])
        S_norm[:n_rows, target_col_index] = np.nan

        if mc_rank is None:
            rank, _ = SyntheticControl.select_optimal_rank_cv(
                S_norm,
                max_rank=self.imputation_rank_max,
                rank_step=self.imputation_rank_step,
                holdout_fraction=self.imputation_holdout_fraction,
                verbose=verbose,
                random_state=target_col_index,
            )
            if verbose:
                print(f"Selected MC rank: {rank} for column {target_col_index}")
        else:
            rank = mc_rank

        if method == "mc_hard_svd":
            S_completed = SyntheticControl.hard_impute_svd(
                S_norm, rank=rank, max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
            )
        elif method == "mc_soft_svd":
            S_completed = SyntheticControl.soft_impute_svd(
                S_norm, rank=rank, lambda_=mc_lambda,
                max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
            )
        elif method == "mc_als":
            S_completed = SyntheticControl.als_complete(
                S_norm, rank=rank, lambda_=mc_lambda,
                max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
                random_state=target_col_index,
            )
        else:
            raise ValueError(f"Unknown MC method: {method}")

        j = target_col_index
        y_pred = S_completed[:n_rows, j] * syn_stds[j] + syn_means[j]

        obs_mask = ~np.isnan(S_norm)
        if obs_mask.sum() > 0:
            train_mse = float(np.mean(
                (S_norm[obs_mask] - S_completed[obs_mask]) ** 2
            ))
        else:
            train_mse = float("nan")

        fitted_model: Dict[str, Any] = {
            "method": method,
            "predictor": {"rank_used": int(rank), "completed_target": y_pred.copy()},
            "metadata": {
                "mc_rank": int(rank),
                "mc_max_iter": int(mc_max_iter),
                "mc_tol": float(mc_tol),
                "stacked_shape": [2 * n_rows, self.real.shape[1]],
            },
        }
        if method in ("mc_soft_svd", "mc_als"):
            fitted_model["metadata"]["mc_lambda"] = float(mc_lambda)

        return y_pred, train_mse, fitted_model

    def _matrix_completion_predict_row(
        self,
        target_row_index: int,
        method: str,
        mc_rank: Optional[int],
        mc_max_iter: int,
        mc_tol: float,
        mc_lambda: float,
        verbose: bool,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Predict a target row via matrix completion on vstack([real, synthetic]).

        Real and synthetic halves are normalised **separately** before stacking.
        Predictions are denormalised using synthetic column statistics.

        Returns:
            (y_pred_real, train_mse, fitted_model) where train_mse is the
            reconstruction MSE on all observed entries in the stacked matrix.
        """
        n_rows, _ = self.real.shape

        real_norm, real_means, real_stds = self._normalize_columns(self.real)
        syn_norm, syn_means, syn_stds = self._normalize_columns(self.synthetic)

        S_norm = np.vstack([real_norm, syn_norm])
        S_norm[target_row_index, :] = np.nan

        if mc_rank is None:
            rank, _ = SyntheticControl.select_optimal_rank_cv(
                S_norm,
                max_rank=self.imputation_rank_max,
                rank_step=self.imputation_rank_step,
                holdout_fraction=self.imputation_holdout_fraction,
                verbose=verbose,
                random_state=target_row_index + 10000,
            )
            if verbose:
                print(f"Selected MC rank: {rank} for row {target_row_index}")
        else:
            rank = mc_rank

        if method == "mc_hard_svd":
            S_completed = SyntheticControl.hard_impute_svd(
                S_norm, rank=rank, max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
            )
        elif method == "mc_soft_svd":
            S_completed = SyntheticControl.soft_impute_svd(
                S_norm, rank=rank, lambda_=mc_lambda,
                max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
            )
        elif method == "mc_als":
            S_completed = SyntheticControl.als_complete(
                S_norm, rank=rank, lambda_=mc_lambda,
                max_iter=mc_max_iter, tol=mc_tol, verbose=verbose,
                random_state=target_row_index + 10000,
            )
        else:
            raise ValueError(f"Unknown MC method: {method}")

        y_pred = S_completed[target_row_index, :] * syn_stds + syn_means

        obs_mask = ~np.isnan(S_norm)
        if obs_mask.sum() > 0:
            train_mse = float(np.mean(
                (S_norm[obs_mask] - S_completed[obs_mask]) ** 2
            ))
        else:
            train_mse = float("nan")

        fitted_model: Dict[str, Any] = {
            "method": method,
            "predictor": {"rank_used": int(rank), "completed_target": y_pred.copy()},
            "metadata": {
                "mc_rank": int(rank),
                "mc_max_iter": int(mc_max_iter),
                "mc_tol": float(mc_tol),
                "stacked_shape": [2 * n_rows, self.real.shape[1]],
            },
        }
        if method in ("mc_soft_svd", "mc_als"):
            fitted_model["metadata"]["mc_lambda"] = float(mc_lambda)

        return y_pred, train_mse, fitted_model

    # ------------------------------------------------------------------
    # Synthetic-prior matrix completion
    # ------------------------------------------------------------------

    def _synthetic_prior_predict_column(
        self,
        target_col_index: int,
        mc_rank: Optional[int],
        mc_max_iter: int,
        mc_tol: float,
        verbose: bool,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Complete the real matrix with synthetic values as initialisation.

        1. Normalise real and synthetic separately.
        2. In the normalised real matrix, mask the target column.
        3. Fill all NaN positions with corresponding normalised synthetic values.
        4. Run iterative hard-SVD completion (observed real values are restored
           each iteration; synthetic-initialised positions are updated by SVD).
        5. Denormalise the completed target column with synthetic stats.
        """
        n_rows, n_cols = self.real.shape
        j = target_col_index

        real_norm, _, _ = self._normalize_columns(self.real)
        syn_norm, syn_means, syn_stds = self._normalize_columns(self.synthetic)

        M = real_norm.copy()
        M[:, j] = np.nan
        mask_missing = np.isnan(M)

        syn_fill = np.nan_to_num(syn_norm, nan=0.0)
        M_filled = M.copy()
        M_filled[mask_missing] = syn_fill[mask_missing]

        if mc_rank is None:
            rank, _ = SyntheticControl.select_optimal_rank_cv(
                M,
                max_rank=self.imputation_rank_max,
                rank_step=self.imputation_rank_step,
                holdout_fraction=self.imputation_holdout_fraction,
                verbose=verbose,
                random_state=j,
            )
            if verbose:
                print(f"Selected synthetic-prior rank: {rank} for column {j}")
        else:
            rank = mc_rank

        observed_vals = M_filled[~mask_missing].copy()
        prev_X = M_filled.copy()
        for iteration in range(mc_max_iter):
            try:
                M_filled = SVD(M_filled, min(min(M_filled.shape), rank))
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Warning: SVD did not converge at iteration {iteration}.")
                M_filled = prev_X
                break
            M_filled[~mask_missing] = observed_vals
            diff_norm = np.linalg.norm(prev_X, "fro")
            diff = (
                np.linalg.norm(M_filled - prev_X, "fro") / diff_norm
                if diff_norm > 1e-12
                else np.linalg.norm(M_filled - prev_X, "fro")
            )
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Diff: {diff}")
            if diff < mc_tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            prev_X = M_filled.copy()

        y_pred = M_filled[:, j] * syn_stds[j] + syn_means[j]

        syn_target = self.synthetic[:, j]
        syn_obs = ~np.isnan(syn_target)
        train_mse = (
            float(np.mean((y_pred[syn_obs] - syn_target[syn_obs]) ** 2))
            if syn_obs.sum() > 0
            else float("nan")
        )

        fitted_model: Dict[str, Any] = {
            "method": "mc_synthetic_prior",
            "predictor": {"rank_used": int(rank), "completed_target": y_pred.copy()},
            "metadata": {
                "mc_rank": int(rank),
                "mc_max_iter": int(mc_max_iter),
                "mc_tol": float(mc_tol),
                "matrix_shape": list(self.real.shape),
            },
        }
        return y_pred, train_mse, fitted_model

    def _synthetic_prior_predict_row(
        self,
        target_row_index: int,
        mc_rank: Optional[int],
        mc_max_iter: int,
        mc_tol: float,
        verbose: bool,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Row-wise analogue of ``_synthetic_prior_predict_column``."""
        n_rows, n_cols = self.real.shape
        i = target_row_index

        real_norm, _, _ = self._normalize_columns(self.real)
        syn_norm, syn_means, syn_stds = self._normalize_columns(self.synthetic)

        M = real_norm.copy()
        M[i, :] = np.nan
        mask_missing = np.isnan(M)

        syn_fill = np.nan_to_num(syn_norm, nan=0.0)
        M_filled = M.copy()
        M_filled[mask_missing] = syn_fill[mask_missing]

        if mc_rank is None:
            rank, _ = SyntheticControl.select_optimal_rank_cv(
                M,
                max_rank=self.imputation_rank_max,
                rank_step=self.imputation_rank_step,
                holdout_fraction=self.imputation_holdout_fraction,
                verbose=verbose,
                random_state=i + 10000,
            )
            if verbose:
                print(f"Selected synthetic-prior rank: {rank} for row {i}")
        else:
            rank = mc_rank

        observed_vals = M_filled[~mask_missing].copy()
        prev_X = M_filled.copy()
        for iteration in range(mc_max_iter):
            try:
                M_filled = SVD(M_filled, min(min(M_filled.shape), rank))
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Warning: SVD did not converge at iteration {iteration}.")
                M_filled = prev_X
                break
            M_filled[~mask_missing] = observed_vals
            diff_norm = np.linalg.norm(prev_X, "fro")
            diff = (
                np.linalg.norm(M_filled - prev_X, "fro") / diff_norm
                if diff_norm > 1e-12
                else np.linalg.norm(M_filled - prev_X, "fro")
            )
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Diff: {diff}")
            if diff < mc_tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            prev_X = M_filled.copy()

        y_pred = M_filled[i, :] * syn_stds + syn_means

        syn_target = self.synthetic[i, :]
        syn_obs = ~np.isnan(syn_target)
        train_mse = (
            float(np.mean((y_pred[syn_obs] - syn_target[syn_obs]) ** 2))
            if syn_obs.sum() > 0
            else float("nan")
        )

        fitted_model: Dict[str, Any] = {
            "method": "mc_synthetic_prior",
            "predictor": {"rank_used": int(rank), "completed_target": y_pred.copy()},
            "metadata": {
                "mc_rank": int(rank),
                "mc_max_iter": int(mc_max_iter),
                "mc_tol": float(mc_tol),
                "matrix_shape": list(self.real.shape),
            },
        }
        return y_pred, train_mse, fitted_model

    def evaluate_column(
        self,
        target_col_index: int,
        donor_mask: Optional[np.ndarray] = None,
        method: str = "ridge",
        regularization_multiplier: float = 1e-6,
        en_l1_ratio: float = 0.5,
        md_learning_rate: float = 0.01,
        md_max_iter: int = 3000,
        md_tol: float = 1e-9,
        md_eps: float = 1e-18,
        md_adaptive_lr: bool = True,
        md_lr_decay: float = 0.9,
        md_min_lr_ratio: float = 0.01,
        md_lr_patience: int = 10,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_epochs: int = 300,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-6,
        nn_batch_size: int = 256,
        nn_patience: int = 20,
        nn_device: str = "auto",
        nn_seed: int = 42,
        si_rank: Optional[int] = None,
        mc_rank: Optional[int] = None,
        mc_max_iter: int = 1000,
        mc_tol: float = 1e-4,
        mc_lambda: float = 1.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Predict a target column using synthetic control or matrix completion.

        Three method families are supported:

        *Regression (impute-regress-transfer)*: ``"synthetic_control"``,
        ``"ridge"``, ``"lasso"``, ``"elastic_net"``,
        ``"neural_net"``.

        *Synthetic intervention*: ``"synthetic_intervention"``.  PCR on
        the SVD-reduced donor space with ridge regression.

        *Matrix completion*: ``"mc_hard_svd"``, ``"mc_soft_svd"``,
        ``"mc_als"``, ``"mc_synthetic_prior"``.

        Args:
            target_col_index: Index of the column to predict (0-indexed).
            donor_mask: Optional boolean mask of shape (n_cols,) denoting allowable donor columns.
                       Ignored for matrix-completion methods.
            method: One of ``"synthetic_control"``, ``"ridge"``,
                    ``"lasso"``, ``"elastic_net"``, ``"neural_net"``,
                    ``"synthetic_intervention"``, ``"mc_hard_svd"``,
                    ``"mc_soft_svd"``, ``"mc_als"``, or ``"mc_synthetic_prior"``.
            regularization_multiplier: Regularization coefficient. Default: 1e-6.
            en_l1_ratio: L1/L2 mixing for method="elastic_net" (1 = pure L1). Default: 0.5.
            md_learning_rate: Learning rate for method="synthetic_control". Default: 0.01.
            md_max_iter: Maximum iterations for method="synthetic_control". Default: 3000.
            md_tol: Convergence tolerance for method="synthetic_control". Default: 1e-9.
            md_eps: Lower bound for mirror-descent weights. Default: 1e-18.
            md_adaptive_lr: Adaptive LR in mirror descent. Default: True.
            md_lr_decay: LR decay factor in mirror descent. Default: 0.9.
            md_min_lr_ratio: Minimum LR ratio in mirror descent. Default: 0.01.
            md_lr_patience: Patience before mirror-descent LR decay. Default: 10.
            nn_hidden_dims: Hidden layer widths for method="neural_net". Default: [32].
            nn_epochs: Max epochs for method="neural_net". Default: 300.
            nn_lr: Learning rate for method="neural_net". Default: 1e-3.
            nn_weight_decay: Weight decay for method="neural_net". Default: 1e-6.
            nn_batch_size: Batch size for method="neural_net". Default: 256.
            nn_patience: Early-stopping patience for method="neural_net". Default: 20.
            nn_device: Device for method="neural_net". Default: "auto".
            nn_seed: Random seed for method="neural_net". Default: 42.
            si_rank: Number of SVD components for method="synthetic_intervention".
                     None keeps all components. Default: None.
            mc_rank: Rank for MC methods. None selects via CV. Default: None.
            mc_max_iter: Max iterations for MC methods. Default: 1000.
            mc_tol: Convergence tolerance for MC methods. Default: 1e-4.
            mc_lambda: Soft-threshold / regularisation strength for
                       method="mc_soft_svd" or method="mc_als". Default: 1.0.
            verbose: Print diagnostics. Default: False.

        Returns:
            Dictionary containing evaluation results and fitted model.
        """
        if regularization_multiplier < 0:
            regularization_multiplier = 0

        _, n_cols = self.real.shape
        if target_col_index < 0 or target_col_index >= n_cols:
            raise IndexError("target_col_index out of bounds")

        real_target = self.real[:, target_col_index]
        synthetic_target = self.synthetic[:, target_col_index]

        if method.startswith("mc_"):
            # ----- Matrix completion path -----
            if donor_mask is not None and verbose:
                print("Note: donor_mask is ignored for matrix completion methods.")

            if method == "mc_synthetic_prior":
                y_pred_real, train_mse, fitted_model = self._synthetic_prior_predict_column(
                    target_col_index, mc_rank, mc_max_iter, mc_tol, verbose,
                )
            else:
                y_pred_real, train_mse, fitted_model = self._matrix_completion_predict_column(
                    target_col_index, method, mc_rank, mc_max_iter, mc_tol, mc_lambda, verbose,
                )

            synthetic_target_mean = np.nanmean(synthetic_target)
            synthetic_target_std = np.nanstd(synthetic_target)
            if synthetic_target_std < self.min_col_std:
                synthetic_target_std = 1.0
            y_pred_real_norm = (y_pred_real - synthetic_target_mean) / synthetic_target_std
            num_donors = n_cols - 1

        else:
            # ----- Regression path (impute-regress-transfer) -----
            donors = np.ones(n_cols, dtype=bool)
            donors[target_col_index] = False
            if donor_mask is not None:
                if donor_mask.shape != (n_cols,):
                    raise ValueError("donor_mask must have shape (num_columns,)")
                donors = donors & donor_mask
            donor_idx = np.where(donors)[0]
            if donor_idx.size == 0:
                raise ValueError("No donor columns available after masking")

            real_donors = self.real[:, donor_idx]
            synthetic_donors = self.synthetic[:, donor_idx]

            if self.imputation_rank is None:
                rank, _ = SyntheticControl.select_optimal_rank_cv(
                    synthetic_donors,
                    max_rank=self.imputation_rank_max,
                    rank_step=self.imputation_rank_step,
                    holdout_fraction=self.imputation_holdout_fraction,
                    verbose=verbose,
                    random_state=target_col_index,
                )
                if verbose:
                    print(f"Selected imputation rank: {rank} for column {target_col_index}")
            else:
                rank = self.imputation_rank

            real_donors_imputed = SyntheticControl.hard_impute_svd(real_donors, rank=rank, verbose=verbose)
            synthetic_donors_imputed = SyntheticControl.hard_impute_svd(synthetic_donors, rank=rank, verbose=verbose)

            syn_means = synthetic_donors_imputed.mean(axis=0)
            syn_stds = synthetic_donors_imputed.std(axis=0, ddof=0)
            syn_lowvar = syn_stds < self.min_col_std
            syn_stds_safe = syn_stds.copy()
            syn_stds_safe[syn_lowvar] = 1.0

            real_means = real_donors_imputed.mean(axis=0)
            real_stds = real_donors_imputed.std(axis=0, ddof=0)
            real_lowvar = real_stds < self.min_col_std
            real_stds_safe = real_stds.copy()
            real_stds_safe[real_lowvar] = 1.0

            synthetic_donors_normalized = (synthetic_donors_imputed - syn_means) / syn_stds_safe
            real_donors_normalized = (real_donors_imputed - real_means) / real_stds_safe

            synthetic_target_mean = np.nanmean(synthetic_target)
            synthetic_target_std = np.nanstd(synthetic_target)
            if synthetic_target_std < self.min_col_std:
                synthetic_target_std = 1.0
            synthetic_target_normalized = (synthetic_target - synthetic_target_mean) / synthetic_target_std

            mask_target = np.isnan(synthetic_target_normalized)
            synthetic_target_normalized_filled = synthetic_target_normalized.copy()
            synthetic_target_normalized_filled[mask_target] = 0.0

            X_syn = synthetic_donors_normalized
            y_syn = synthetic_target_normalized_filled

            if method == "synthetic_control":
                w, train_mse = self._mirror_descent_simplex(
                    X_syn, y_syn,
                    regularization_multiplier=regularization_multiplier,
                    learning_rate=md_learning_rate,
                    max_iter=md_max_iter,
                    tol=md_tol,
                    eps=md_eps,
                    adaptive_lr=md_adaptive_lr,
                    lr_decay=md_lr_decay,
                    min_lr_ratio=md_min_lr_ratio,
                    lr_patience=md_lr_patience,
                    verbose=verbose,
                )
                b = 0
                y_pred_real_norm = real_donors_normalized @ w + b
                fitted_model = {
                    "method": "synthetic_control",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "learning_rate": float(md_learning_rate),
                        "max_iter": int(md_max_iter),
                        "tol": float(md_tol),
                        "eps": float(md_eps),
                        "adaptive_lr": bool(md_adaptive_lr),
                        "lr_decay": float(md_lr_decay),
                        "min_lr_ratio": float(md_min_lr_ratio),
                        "lr_patience": int(md_lr_patience),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "ridge":
                w, b, train_mse = self._linear_regression_l2(
                    X_syn, y_syn, regularization_multiplier=regularization_multiplier,
                )
                y_pred_real_norm = real_donors_normalized @ w + b
                fitted_model = {
                    "method": "ridge",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "lasso":
                w, b, train_mse = self._lasso_regression(
                    X_syn, y_syn, regularization_multiplier,
                )
                y_pred_real_norm = real_donors_normalized @ w + b
                fitted_model = {
                    "method": "lasso",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "elastic_net":
                w, b, train_mse = self._elastic_net_regression(
                    X_syn, y_syn, regularization_multiplier, en_l1_ratio,
                )
                y_pred_real_norm = real_donors_normalized @ w + b
                fitted_model = {
                    "method": "elastic_net",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "l1_ratio": float(en_l1_ratio),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "neural_net":
                y_pred_real_norm, train_mse, fitted_model = self._neural_net_regression_predict(
                    X_train=X_syn,
                    y_train=y_syn,
                    X_eval=real_donors_normalized,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    verbose=verbose,
                )
                fitted_model["metadata"].update({
                    "num_donors": int(donor_idx.size),
                    "donor_indices": donor_idx.astype(int),
                })
            elif method == "synthetic_intervention":
                U, s, Vt = np.linalg.svd(X_syn, full_matrices=False)
                k = len(s) if si_rank is None else min(si_rank, len(s))
                V_k = Vt[:k, :].T
                X_syn_r = X_syn @ V_k
                X_real_r = real_donors_normalized @ V_k
                w_r, b, train_mse = self._linear_regression_l2(
                    X_syn_r, y_syn, regularization_multiplier,
                )
                y_pred_real_norm = X_real_r @ w_r + b
                fitted_model = {
                    "method": "synthetic_intervention",
                    "predictor": {
                        "weights": w_r,
                        "intercept": float(b),
                        "projection_matrix": V_k,
                    },
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "si_rank": int(k),
                        "singular_values": s[:k].tolist(),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            else:
                raise ValueError(f"Unknown method: {method}")

            y_pred_real = y_pred_real_norm * synthetic_target_std + synthetic_target_mean
            num_donors = int(donor_idx.size)

        # ------ Shared evaluation ------
        y_baseline = synthetic_target.copy()
        y_additional_baseline = None
        if self.additional_baseline is not None:
            y_additional_baseline = self.additional_baseline[:, target_col_index].copy()
        col_mask = self.nanmask[:, target_col_index]

        metrics = self._metrics(real_target, y_pred_real, mask=col_mask)
        baseline_metrics = self._metrics(real_target, y_baseline, mask=col_mask)
        if y_additional_baseline is not None:
            additional_baseline_metrics = self._metrics(real_target, y_additional_baseline, mask=col_mask)

        real_target_normalized = (real_target - synthetic_target_mean) / synthetic_target_std
        y_baseline_normalized = (y_baseline - synthetic_target_mean) / synthetic_target_std
        if y_additional_baseline is not None:
            y_additional_baseline_normalized = (y_additional_baseline - synthetic_target_mean) / synthetic_target_std

        if col_mask is not None:
            valid_mask = ~col_mask
            if valid_mask.sum() > 0:
                corr_normalized = self._correlation(real_target_normalized[valid_mask], y_pred_real_norm[valid_mask])
                corr_baseline_normalized = self._correlation(real_target_normalized[valid_mask], y_baseline_normalized[valid_mask])
                if y_additional_baseline is not None:
                    corr_additional_baseline_normalized = self._correlation(
                        real_target_normalized[valid_mask],
                        y_additional_baseline_normalized[valid_mask],
                    )
            else:
                corr_normalized = float(np.nan)
                corr_baseline_normalized = float(np.nan)
                if y_additional_baseline is not None:
                    corr_additional_baseline_normalized = float(np.nan)
        else:
            corr_normalized = self._correlation(real_target_normalized, y_pred_real_norm)
            corr_baseline_normalized = self._correlation(real_target_normalized, y_baseline_normalized)
            if y_additional_baseline is not None:
                corr_additional_baseline_normalized = self._correlation(
                    real_target_normalized,
                    y_additional_baseline_normalized,
                )

        if verbose:
            print(f"\nNum donors: {num_donors}")
            metric_names = ["mse", "correlation", "r2", "accuracy", "wasserstein_distance", "std_ratio"]
            if y_additional_baseline is not None:
                for metric_name in metric_names:
                    print(
                        f"{metric_name}: {metrics[metric_name]:.4f}, "
                        f"Baseline {metric_name}: {baseline_metrics[metric_name]:.4f}, "
                        f"Additional Baseline {metric_name}: {additional_baseline_metrics[metric_name]:.4f}"
                    )
                print(
                    f"corr_normalized: {corr_normalized:.4f}, "
                    f"Baseline corr_normalized: {corr_baseline_normalized:.4f}, "
                    f"Additional Baseline corr_normalized: {corr_additional_baseline_normalized:.4f}"
                )
            else:
                for metric_name in metric_names:
                    print(
                        f"{metric_name}: {metrics[metric_name]:.4f}, "
                        f"Baseline {metric_name}: {baseline_metrics[metric_name]:.4f}"
                    )
                print(f"corr_normalized: {corr_normalized:.4f}, Baseline corr_normalized: {corr_baseline_normalized:.4f}")

        result = {
            "fitted_model": fitted_model,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "corr_normalized": corr_normalized,
            "corr_baseline_normalized": corr_baseline_normalized,
            "num_donors": num_donors,
            "train_mse": train_mse,
        }
        if y_additional_baseline is not None:
            result["additional_baseline_metrics"] = additional_baseline_metrics
            result["corr_additional_baseline_normalized"] = corr_additional_baseline_normalized
        return result

    def evaluate_all_columns(
        self,
        donor_mask: Optional[np.ndarray] = None,
        method: str = "ridge",
        regularization_multiplier: float = 1e-6,
        en_l1_ratio: float = 0.5,
        md_learning_rate: float = 0.01,
        md_max_iter: int = 3000,
        md_tol: float = 1e-9,
        md_eps: float = 1e-18,
        md_adaptive_lr: bool = True,
        md_lr_decay: float = 0.9,
        md_min_lr_ratio: float = 0.01,
        md_lr_patience: int = 10,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_epochs: int = 300,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-6,
        nn_batch_size: int = 256,
        nn_patience: int = 20,
        nn_device: str = "auto",
        nn_seed: int = 42,
        si_rank: Optional[int] = None,
        mc_rank: Optional[int] = None,
        mc_max_iter: int = 1000,
        mc_tol: float = 1e-4,
        mc_lambda: float = 1.0,
        train_mse_thresholds: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate all columns. See ``evaluate_column`` for method details.

        Args:
            donor_mask: Optional boolean mask denoting allowable donor columns. Default: None.
            method: See ``evaluate_column``.
            regularization_multiplier: Regularization coefficient. Default: 1e-6.
            en_l1_ratio: L1/L2 mixing for method="elastic_net". Default: 0.5.
            si_rank: SVD components for method="synthetic_intervention". Default: None.
            mc_rank: Rank for MC methods. Default: None.
            mc_max_iter: Max iterations for MC methods. Default: 1000.
            mc_tol: Convergence tolerance for MC methods. Default: 1e-4.
            mc_lambda: Soft-threshold / regularisation strength for
                       method="mc_soft_svd" or method="mc_als". Default: 1.0.
            train_mse_thresholds: Train MSE thresholds for adaptive evaluation.
            n_jobs: Number of parallel jobs. Default: 1.
            verbose: Print progress. Default: False.

        Returns:
            Dictionary containing per-column metrics and aggregates.
        """
        if regularization_multiplier < 0:
            regularization_multiplier = 0

        n_cols = self.real.shape[1]
        method_tag = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in method)
        if method == "neural_net":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "method='neural_net' requires PyTorch. Install it with `pip install torch`."
                ) from exc

            resolved_device = nn_device
            if nn_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            elif nn_device == "cuda" and not torch.cuda.is_available():
                resolved_device = "cpu"

            if resolved_device == "cuda" and n_jobs != 1:
                if verbose:
                    print("Using n_jobs=1 for neural_net on GPU to avoid contention.")
                n_jobs = 1

        print_every = max(10, 10**int(np.log10(n_cols))) if n_cols > 0 else 10

        if n_jobs == 1:
            results_list = []
            for j in range(n_cols):
                res = self.evaluate_column(
                    j,
                    donor_mask=donor_mask,
                    method=method,
                    regularization_multiplier=regularization_multiplier,
                    en_l1_ratio=en_l1_ratio,
                    md_learning_rate=md_learning_rate,
                    md_max_iter=md_max_iter,
                    md_tol=md_tol,
                    md_eps=md_eps,
                    md_adaptive_lr=md_adaptive_lr,
                    md_lr_decay=md_lr_decay,
                    md_min_lr_ratio=md_min_lr_ratio,
                    md_lr_patience=md_lr_patience,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    si_rank=si_rank,
                    mc_rank=mc_rank,
                    mc_max_iter=mc_max_iter,
                    mc_tol=mc_tol,
                    mc_lambda=mc_lambda,
                )
                results_list.append(res)
                if verbose and j % print_every == 0:
                    print(f"Evaluated column {j} of {n_cols}")
        else:
            if verbose:
                print(f"Evaluating {n_cols} columns in parallel using {n_jobs} jobs...")
            results_list = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(self.evaluate_column)(
                    j,
                    donor_mask=donor_mask,
                    method=method,
                    regularization_multiplier=regularization_multiplier,
                    en_l1_ratio=en_l1_ratio,
                    md_learning_rate=md_learning_rate,
                    md_max_iter=md_max_iter,
                    md_tol=md_tol,
                    md_eps=md_eps,
                    md_adaptive_lr=md_adaptive_lr,
                    md_lr_decay=md_lr_decay,
                    md_min_lr_ratio=md_min_lr_ratio,
                    md_lr_patience=md_lr_patience,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    si_rank=si_rank,
                    mc_rank=mc_rank,
                    mc_max_iter=mc_max_iter,
                    mc_tol=mc_tol,
                    mc_lambda=mc_lambda,
                )
                for j in range(n_cols)
            )

        metrics: List[Dict[str, float]] = [res["metrics"] for res in results_list]
        baseline_metrics: List[Dict[str, float]] = [res["baseline_metrics"] for res in results_list]
        fitted_models: List[Dict[str, Any]] = [res["fitted_model"] for res in results_list]
        has_additional_baseline = bool(results_list) and ("additional_baseline_metrics" in results_list[0])
        if has_additional_baseline:
            additional_baseline_metrics: List[Dict[str, float]] = [res["additional_baseline_metrics"] for res in results_list]
        train_mses: List[float] = [res["train_mse"] for res in results_list]
        corr_normalized: List[float] = [res["corr_normalized"] for res in results_list]
        corr_baseline_normalized: List[float] = [res["corr_baseline_normalized"] for res in results_list]
        if has_additional_baseline:
            corr_additional_baseline_normalized: List[float] = [
                res["corr_additional_baseline_normalized"] for res in results_list
            ]

        if donor_mask is None:
            num_donors = n_cols - 1
        else:
            if donor_mask.shape != (n_cols,):
                raise ValueError("donor_mask must have shape (num_columns,)")
            num_donors = np.sum(donor_mask) - 1

        train_mse_thresholds_metrics = {}
        for thresh in train_mse_thresholds:
            mixed_metrics = []
            for j in range(n_cols):
                if train_mses[j] > thresh:
                    mixed_metrics.append(baseline_metrics[j])
                else:
                    mixed_metrics.append(metrics[j])
            train_mse_thresholds_metrics[thresh] = mixed_metrics

        if verbose:
            print(f"\nNum donors: {num_donors}")
            mse_arr = np.array([m["mse"] for m in metrics], dtype=float)
            base_mse_arr = np.array([m["mse"] for m in baseline_metrics], dtype=float)
            corr_arr = np.array([m["correlation"] for m in metrics], dtype=float)
            base_corr_arr = np.array([m["correlation"] for m in baseline_metrics], dtype=float)
            r2_arr = np.array([m["r2"] for m in metrics], dtype=float)
            base_r2_arr = np.array([m["r2"] for m in baseline_metrics], dtype=float)
            acc_arr = np.array([m["accuracy"] for m in metrics], dtype=float)
            base_acc_arr = np.array([m["accuracy"] for m in baseline_metrics], dtype=float)
            wass_arr = np.array([m["wasserstein_distance"] for m in metrics], dtype=float)
            base_wass_arr = np.array([m["wasserstein_distance"] for m in baseline_metrics], dtype=float)
            std_ratio_arr = np.array([m["std_ratio"] for m in metrics], dtype=float)
            base_std_ratio_arr = np.array([m["std_ratio"] for m in baseline_metrics], dtype=float)
            corr_normalized_arr = np.array(corr_normalized, dtype=float)
            corr_baseline_normalized_arr = np.array(corr_baseline_normalized, dtype=float)
            if has_additional_baseline:
                add_mse_arr = np.array([m["mse"] for m in additional_baseline_metrics], dtype=float)
                add_corr_arr = np.array([m["correlation"] for m in additional_baseline_metrics], dtype=float)
                add_r2_arr = np.array([m["r2"] for m in additional_baseline_metrics], dtype=float)
                add_acc_arr = np.array([m["accuracy"] for m in additional_baseline_metrics], dtype=float)
                add_wass_arr = np.array([m["wasserstein_distance"] for m in additional_baseline_metrics], dtype=float)
                add_std_ratio_arr = np.array([m["std_ratio"] for m in additional_baseline_metrics], dtype=float)
                corr_additional_baseline_normalized_arr = np.array(corr_additional_baseline_normalized, dtype=float)

            def sem(arr):
                """Compute standard error of the mean."""
                n = len(arr)
                if n <= 1:
                    return np.nan
                return np.std(arr, ddof=1) / np.sqrt(n)

            if has_additional_baseline:
                print(
                    f"MSE mean: {float(np.mean(mse_arr)):.4f} ± {float(sem(mse_arr)):.4f}, "
                    f"Baseline MSE mean: {float(np.mean(base_mse_arr)):.4f} ± {float(sem(base_mse_arr)):.4f}, "
                    f"Additional Baseline MSE mean: {float(np.mean(add_mse_arr)):.4f} ± {float(sem(add_mse_arr)):.4f}"
                )
                print(
                    f"Corr mean: {float(np.mean(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, "
                    f"Baseline Corr mean: {float(np.mean(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}, "
                    f"Additional Baseline Corr mean: {float(np.mean(add_corr_arr)):.4f} ± {float(sem(add_corr_arr)):.4f}"
                )
                print(
                    f"Corr (Normalized) mean: {float(np.mean(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, "
                    f"Baseline Corr (Normalized) mean: {float(np.mean(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}, "
                    f"Additional Baseline Corr (Normalized) mean: {float(np.mean(corr_additional_baseline_normalized_arr)):.4f} ± {float(sem(corr_additional_baseline_normalized_arr)):.4f}"
                )
                print(
                    f"Corr mean (Fisher's z): {float(self.fisher_z_average(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, "
                    f"Baseline Corr mean: {float(self.fisher_z_average(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}, "
                    f"Additional Baseline Corr mean: {float(self.fisher_z_average(add_corr_arr)):.4f} ± {float(sem(add_corr_arr)):.4f}"
                )
                print(
                    f"Corr (Normalized, Fisher's z) mean: {float(self.fisher_z_average(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, "
                    f"Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}, "
                    f"Additional Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_additional_baseline_normalized_arr)):.4f} ± {float(sem(corr_additional_baseline_normalized_arr)):.4f}"
                )
                print(
                    f"R2 mean: {float(np.mean(r2_arr)):.4f} ± {float(sem(r2_arr)):.4f}, "
                    f"Baseline R2 mean: {float(np.mean(base_r2_arr)):.4f} ± {float(sem(base_r2_arr)):.4f}, "
                    f"Additional Baseline R2 mean: {float(np.mean(add_r2_arr)):.4f} ± {float(sem(add_r2_arr)):.4f}"
                )
                print(
                    f"Accuracy mean: {float(np.mean(acc_arr)):.4f} ± {float(sem(acc_arr)):.4f}, "
                    f"Baseline Accuracy mean: {float(np.mean(base_acc_arr)):.4f} ± {float(sem(base_acc_arr)):.4f}, "
                    f"Additional Baseline Accuracy mean: {float(np.mean(add_acc_arr)):.4f} ± {float(sem(add_acc_arr)):.4f}"
                )
                print(
                    f"Wasserstein Distance mean: {float(np.mean(wass_arr)):.4f} ± {float(sem(wass_arr)):.4f}, "
                    f"Baseline Wasserstein Distance mean: {float(np.mean(base_wass_arr)):.4f} ± {float(sem(base_wass_arr)):.4f}, "
                    f"Additional Baseline Wasserstein Distance mean: {float(np.mean(add_wass_arr)):.4f} ± {float(sem(add_wass_arr)):.4f}"
                )
                print(
                    f"Std Ratio mean: {float(np.mean(std_ratio_arr)):.4f} ± {float(sem(std_ratio_arr)):.4f}, "
                    f"Baseline Std Ratio mean: {float(np.mean(base_std_ratio_arr)):.4f} ± {float(sem(base_std_ratio_arr)):.4f}, "
                    f"Additional Baseline Std Ratio mean: {float(np.mean(add_std_ratio_arr)):.4f} ± {float(sem(add_std_ratio_arr)):.4f}"
                )
            else:
                print(f"MSE mean: {float(np.mean(mse_arr)):.4f} ± {float(sem(mse_arr)):.4f}, Baseline MSE mean: {float(np.mean(base_mse_arr)):.4f} ± {float(sem(base_mse_arr)):.4f}")
                print(f"Corr mean: {float(np.mean(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, Baseline Corr mean: {float(np.mean(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}")
                print(f"Corr (Normalized) mean: {float(np.mean(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, Baseline Corr (Normalized) mean: {float(np.mean(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}")
                print(f"Corr mean (Fisher's z): {float(self.fisher_z_average(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, Baseline Corr mean: {float(self.fisher_z_average(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}")
                print(f"Corr (Normalized, Fisher's z) mean: {float(self.fisher_z_average(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}")
                print(f"R2 mean: {float(np.mean(r2_arr)):.4f} ± {float(sem(r2_arr)):.4f}, Baseline R2 mean: {float(np.mean(base_r2_arr)):.4f} ± {float(sem(base_r2_arr)):.4f}")
                print(f"Accuracy mean: {float(np.mean(acc_arr)):.4f} ± {float(sem(acc_arr)):.4f}, Baseline Accuracy mean: {float(np.mean(base_acc_arr)):.4f} ± {float(sem(base_acc_arr)):.4f}")
                print(f"Wasserstein Distance mean: {float(np.mean(wass_arr)):.4f} ± {float(sem(wass_arr)):.4f}, Baseline Wasserstein Distance mean: {float(np.mean(base_wass_arr)):.4f} ± {float(sem(base_wass_arr)):.4f}")
                print(f"Std Ratio mean: {float(np.mean(std_ratio_arr)):.4f} ± {float(sem(std_ratio_arr)):.4f}, Baseline Std Ratio mean: {float(np.mean(base_std_ratio_arr)):.4f} ± {float(sem(base_std_ratio_arr)):.4f}")

            # Plot correlation gain vs train MSE
            plt.figure(figsize=(8, 6))
            plt.scatter(train_mses, corr_arr - base_corr_arr, alpha=0.5)
            plt.xlabel("Train MSE", fontsize=16)
            plt.ylabel("Correlation Gain", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_figures_dir, f'columns_correlation_gain_vs_train_mse_{method_tag}.pdf'))
            plt.show()

            for q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                q_train_mses = np.quantile(train_mses, q)
                q_corr_arr = np.mean(corr_arr[np.where(train_mses <= q_train_mses)[0]])
                q_base_corr_arr = np.mean(base_corr_arr[np.where(train_mses < q_train_mses)[0]])
                if has_additional_baseline:
                    q_add_corr_arr = np.mean(add_corr_arr[np.where(train_mses < q_train_mses)[0]])
                    print(
                        f"Train MSE {q*100}% quantile: {float(q_train_mses):.4f}, "
                        f"Corr mean: {float(q_corr_arr):.4f}, "
                        f"Baseline Corr mean: {float(q_base_corr_arr):.4f}, "
                        f"Additional Baseline Corr mean: {float(q_add_corr_arr):.4f}"
                    )
                else:
                    print(f"Train MSE {q*100}% quantile: {float(q_train_mses):.4f}, Corr mean: {float(q_corr_arr):.4f}, Baseline Corr mean: {float(q_base_corr_arr):.4f}")

            print("\nAdaptive correlation vs Train MSE Threshold:")
            corr_thresh_list = []
            for thresh in train_mse_thresholds:
                corr_thresh = np.mean(np.array([m["correlation"] for m in train_mse_thresholds_metrics[thresh]]))
                corr_thresh_list.append(corr_thresh)

            plt.figure(figsize=(8, 6))
            plt.scatter(train_mse_thresholds, corr_thresh_list, alpha=0.5, label="Adaptive correlation")
            plt.axhline(y=np.mean(corr_arr), color="red", linestyle="--", label="Full synthetic control correlation")
            plt.axhline(y=np.mean(base_corr_arr), color="blue", linestyle="--", label="Full baseline correlation")
            if has_additional_baseline:
                plt.axhline(y=np.mean(add_corr_arr), color="green", linestyle="--", label="Full additional baseline correlation")
            plt.xlabel("Train MSE Threshold", fontsize=16)
            plt.ylabel("Correlation Mean", fontsize=16)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.results_figures_dir,
                    f'columns_adaptive_correlation_vs_train_mse_threshold_{method_tag}.pdf',
                )
            )
            plt.show()
            print(f"Maximum adaptive correlation: {float(np.max(corr_thresh_list)):.4f} achieved at train MSE threshold: {train_mse_thresholds[np.argmax(corr_thresh_list)]}")

        result = {
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "fitted_models": fitted_models,
            "train_mses": train_mses,
            "corr_normalized": corr_normalized,
            "corr_baseline_normalized": corr_baseline_normalized,
            "num_donors": num_donors,
            "train_mse_thresholds_metrics": train_mse_thresholds_metrics
        }
        if has_additional_baseline:
            result["additional_baseline_metrics"] = additional_baseline_metrics
            result["corr_additional_baseline_normalized"] = corr_additional_baseline_normalized
        return result

    def evaluate_row(
        self,
        target_row_index: int,
        donor_mask: Optional[np.ndarray] = None,
        method: str = "ridge",
        regularization_multiplier: float = 1e-6,
        en_l1_ratio: float = 0.5,
        md_learning_rate: float = 0.01,
        md_max_iter: int = 3000,
        md_tol: float = 1e-9,
        md_eps: float = 1e-18,
        md_adaptive_lr: bool = True,
        md_lr_decay: float = 0.9,
        md_min_lr_ratio: float = 0.01,
        md_lr_patience: int = 10,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_epochs: int = 300,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-6,
        nn_batch_size: int = 256,
        nn_patience: int = 20,
        nn_device: str = "auto",
        nn_seed: int = 42,
        si_rank: Optional[int] = None,
        mc_rank: Optional[int] = None,
        mc_max_iter: int = 1000,
        mc_tol: float = 1e-4,
        mc_lambda: float = 1.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Predict a target row using synthetic control or matrix completion.

        See ``evaluate_column`` for the full list of supported methods.

        Args:
            target_row_index: Index of the row to predict (0-indexed).
            donor_mask: Optional boolean mask denoting allowable donor rows.
                       Ignored for matrix-completion methods.
            method: See ``evaluate_column``.
            regularization_multiplier: Regularization coefficient. Default: 1e-6.
            en_l1_ratio: L1/L2 mixing for method="elastic_net". Default: 0.5.
            si_rank: SVD components for method="synthetic_intervention". Default: None.
            mc_rank: Rank for MC methods. Default: None.
            mc_max_iter: Max iterations for MC methods. Default: 1000.
            mc_tol: Convergence tolerance for MC methods. Default: 1e-4.
            mc_lambda: Soft-threshold / regularisation strength for
                       method="mc_soft_svd" or method="mc_als". Default: 1.0.
            verbose: Print diagnostics. Default: False.

        Returns:
            Dictionary containing evaluation results and fitted model.
        """
        if regularization_multiplier < 0:
            regularization_multiplier = 0

        n_rows, n_cols = self.real.shape
        if target_row_index < 0 or target_row_index >= n_rows:
            raise IndexError("target_row_index out of bounds")

        real_target = self.real[target_row_index, :]
        synthetic_target = self.synthetic[target_row_index, :]

        if method.startswith("mc_"):
            # ----- Matrix completion path -----
            if donor_mask is not None and verbose:
                print("Note: donor_mask is ignored for matrix completion methods.")

            if method == "mc_synthetic_prior":
                y_pred_real, train_mse, fitted_model = self._synthetic_prior_predict_row(
                    target_row_index, mc_rank, mc_max_iter, mc_tol, verbose,
                )
            else:
                y_pred_real, train_mse, fitted_model = self._matrix_completion_predict_row(
                    target_row_index, method, mc_rank, mc_max_iter, mc_tol, mc_lambda, verbose,
                )

            syn_means = np.nanmean(self.synthetic, axis=0)
            syn_stds = np.nanstd(self.synthetic, axis=0)
            syn_lowvar = syn_stds < self.min_col_std
            syn_stds_safe = syn_stds.copy()
            syn_stds_safe[syn_lowvar] = 1.0
            y_pred_real_norm = (y_pred_real - syn_means) / syn_stds_safe
            num_donors = n_rows - 1

        else:
            # ----- Regression path (impute-regress-transfer) -----
            donors = np.ones(n_rows, dtype=bool)
            donors[target_row_index] = False
            if donor_mask is not None:
                if donor_mask.shape != (n_rows,):
                    raise ValueError("donor_mask must have shape (num_rows,)")
                donors = donors & donor_mask
            donor_idx = np.where(donors)[0]
            if donor_idx.size == 0:
                raise ValueError("No donor rows available after masking")

            real_donors = self.real[donor_idx, :]
            synthetic_donors = self.synthetic[donor_idx, :]

            if self.imputation_rank is None:
                rank, _ = SyntheticControl.select_optimal_rank_cv(
                    synthetic_donors,
                    max_rank=self.imputation_rank_max,
                    rank_step=self.imputation_rank_step,
                    holdout_fraction=self.imputation_holdout_fraction,
                    verbose=verbose,
                    random_state=target_row_index,
                )
                if verbose:
                    print(f"Selected imputation rank: {rank} for row {target_row_index}")
            else:
                rank = self.imputation_rank

            real_donors_imputed = SyntheticControl.hard_impute_svd(real_donors, rank=rank, verbose=verbose)
            synthetic_donors_imputed = SyntheticControl.hard_impute_svd(synthetic_donors, rank=rank, verbose=verbose)

            syn_means = synthetic_donors_imputed.mean(axis=0)
            syn_stds = synthetic_donors_imputed.std(axis=0, ddof=0)
            syn_lowvar = syn_stds < self.min_col_std
            syn_stds_safe = syn_stds.copy()
            syn_stds_safe[syn_lowvar] = 1.0

            real_means = real_donors_imputed.mean(axis=0)
            real_stds = real_donors_imputed.std(axis=0, ddof=0)
            real_lowvar = real_stds < self.min_col_std
            real_stds_safe = real_stds.copy()
            real_stds_safe[real_lowvar] = 1.0

            synthetic_donors_normalized = (synthetic_donors_imputed - syn_means) / syn_stds_safe
            real_donors_normalized = (real_donors_imputed - real_means) / real_stds_safe

            synthetic_target_normalized = (synthetic_target - syn_means) / syn_stds_safe
            synthetic_target_normalized_filled = synthetic_target_normalized.copy()
            mask_target = np.isnan(synthetic_target_normalized)
            synthetic_target_normalized_filled[mask_target] = 0.0

            X_syn = synthetic_donors_normalized.T
            y_syn = synthetic_target_normalized_filled

            if method == "synthetic_control":
                w, train_mse = self._mirror_descent_simplex(
                    X_syn, y_syn,
                    regularization_multiplier=regularization_multiplier,
                    learning_rate=md_learning_rate,
                    max_iter=md_max_iter,
                    tol=md_tol,
                    eps=md_eps,
                    adaptive_lr=md_adaptive_lr,
                    lr_decay=md_lr_decay,
                    min_lr_ratio=md_min_lr_ratio,
                    lr_patience=md_lr_patience,
                    verbose=verbose,
                )
                b = 0
                y_pred_real_norm = real_donors_normalized.T @ w + b
                fitted_model = {
                    "method": "synthetic_control",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "learning_rate": float(md_learning_rate),
                        "max_iter": int(md_max_iter),
                        "tol": float(md_tol),
                        "eps": float(md_eps),
                        "adaptive_lr": bool(md_adaptive_lr),
                        "lr_decay": float(md_lr_decay),
                        "min_lr_ratio": float(md_min_lr_ratio),
                        "lr_patience": int(md_lr_patience),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "ridge":
                w, b, train_mse = self._linear_regression_l2(
                    X_syn, y_syn, regularization_multiplier=regularization_multiplier,
                )
                y_pred_real_norm = real_donors_normalized.T @ w + b
                fitted_model = {
                    "method": "ridge",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "lasso":
                w, b, train_mse = self._lasso_regression(
                    X_syn, y_syn, regularization_multiplier,
                )
                y_pred_real_norm = real_donors_normalized.T @ w + b
                fitted_model = {
                    "method": "lasso",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "elastic_net":
                w, b, train_mse = self._elastic_net_regression(
                    X_syn, y_syn, regularization_multiplier, en_l1_ratio,
                )
                y_pred_real_norm = real_donors_normalized.T @ w + b
                fitted_model = {
                    "method": "elastic_net",
                    "predictor": {"weights": w, "intercept": float(b)},
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "l1_ratio": float(en_l1_ratio),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            elif method == "neural_net":
                y_pred_real_norm, train_mse, fitted_model = self._neural_net_regression_predict(
                    X_train=X_syn,
                    y_train=y_syn,
                    X_eval=real_donors_normalized.T,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    verbose=verbose,
                )
                fitted_model["metadata"].update({
                    "num_donors": int(donor_idx.size),
                    "donor_indices": donor_idx.astype(int),
                })
            elif method == "synthetic_intervention":
                U, s, Vt = np.linalg.svd(X_syn, full_matrices=False)
                k = len(s) if si_rank is None else min(si_rank, len(s))
                V_k = Vt[:k, :].T
                X_syn_r = X_syn @ V_k
                X_real_r = real_donors_normalized.T @ V_k
                w_r, b, train_mse = self._linear_regression_l2(
                    X_syn_r, y_syn, regularization_multiplier,
                )
                y_pred_real_norm = X_real_r @ w_r + b
                fitted_model = {
                    "method": "synthetic_intervention",
                    "predictor": {
                        "weights": w_r,
                        "intercept": float(b),
                        "projection_matrix": V_k,
                    },
                    "metadata": {
                        "regularization_multiplier": float(regularization_multiplier),
                        "si_rank": int(k),
                        "singular_values": s[:k].tolist(),
                        "num_donors": int(donor_idx.size),
                        "donor_indices": donor_idx.astype(int),
                    },
                }
            else:
                raise ValueError(f"Unknown method: {method}")

            y_pred_real = y_pred_real_norm * syn_stds_safe + syn_means
            num_donors = int(donor_idx.size)

        # ------ Shared evaluation ------
        y_baseline = synthetic_target.copy()
        y_additional_baseline = None
        if self.additional_baseline is not None:
            y_additional_baseline = self.additional_baseline[target_row_index, :].copy()

        row_mask = self.nanmask[target_row_index, :]

        metrics = self._metrics(real_target, y_pred_real, mask=row_mask)
        baseline_metrics = self._metrics(real_target, y_baseline, mask=row_mask)
        if y_additional_baseline is not None:
            additional_baseline_metrics = self._metrics(real_target, y_additional_baseline, mask=row_mask)

        real_target_normalized = (real_target - syn_means) / syn_stds_safe
        y_baseline_normalized = (y_baseline - syn_means) / syn_stds_safe
        if y_additional_baseline is not None:
            y_additional_baseline_normalized = (y_additional_baseline - syn_means) / syn_stds_safe

        if row_mask is not None:
            valid_mask = ~row_mask
            if valid_mask.sum() > 0:
                corr_normalized = self._correlation(real_target_normalized[valid_mask], y_pred_real_norm[valid_mask])
                corr_baseline_normalized = self._correlation(real_target_normalized[valid_mask], y_baseline_normalized[valid_mask])
                if y_additional_baseline is not None:
                    corr_additional_baseline_normalized = self._correlation(
                        real_target_normalized[valid_mask],
                        y_additional_baseline_normalized[valid_mask],
                    )
            else:
                corr_normalized = float(np.nan)
                corr_baseline_normalized = float(np.nan)
                if y_additional_baseline is not None:
                    corr_additional_baseline_normalized = float(np.nan)
        else:
            corr_normalized = self._correlation(real_target_normalized, y_pred_real_norm)
            corr_baseline_normalized = self._correlation(real_target_normalized, y_baseline_normalized)
            if y_additional_baseline is not None:
                corr_additional_baseline_normalized = self._correlation(
                    real_target_normalized,
                    y_additional_baseline_normalized,
                )

        if verbose:
            print(f"\nNum donors: {num_donors}")
            metric_names = ["mse", "correlation", "r2", "accuracy", "wasserstein_distance", "std_ratio"]
            if y_additional_baseline is not None:
                for metric_name in metric_names:
                    print(
                        f"{metric_name}: {metrics[metric_name]:.4f}, "
                        f"Baseline {metric_name}: {baseline_metrics[metric_name]:.4f}, "
                        f"Additional Baseline {metric_name}: {additional_baseline_metrics[metric_name]:.4f}"
                    )
                print(
                    f"corr_normalized: {corr_normalized:.4f}, "
                    f"Baseline corr_normalized: {corr_baseline_normalized:.4f}, "
                    f"Additional Baseline corr_normalized: {corr_additional_baseline_normalized:.4f}"
                )
            else:
                for metric_name in metric_names:
                    print(
                        f"{metric_name}: {metrics[metric_name]:.4f}, "
                        f"Baseline {metric_name}: {baseline_metrics[metric_name]:.4f}"
                    )
                print(f"corr_normalized: {corr_normalized:.4f}, Baseline corr_normalized: {corr_baseline_normalized:.4f}")

        result = {
            "fitted_model": fitted_model,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "corr_normalized": corr_normalized,
            "corr_baseline_normalized": corr_baseline_normalized,
            "num_donors": num_donors,
            "train_mse": train_mse,
        }
        if y_additional_baseline is not None:
            result["additional_baseline_metrics"] = additional_baseline_metrics
            result["corr_additional_baseline_normalized"] = corr_additional_baseline_normalized
        return result

    def evaluate_all_rows(
        self,
        donor_mask: Optional[np.ndarray] = None,
        method: str = "ridge",
        regularization_multiplier: float = 1e-6,
        en_l1_ratio: float = 0.5,
        md_learning_rate: float = 0.01,
        md_max_iter: int = 3000,
        md_tol: float = 1e-9,
        md_eps: float = 1e-18,
        md_adaptive_lr: bool = True,
        md_lr_decay: float = 0.9,
        md_min_lr_ratio: float = 0.01,
        md_lr_patience: int = 10,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_epochs: int = 300,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-6,
        nn_batch_size: int = 256,
        nn_patience: int = 20,
        nn_device: str = "auto",
        nn_seed: int = 42,
        si_rank: Optional[int] = None,
        mc_rank: Optional[int] = None,
        mc_max_iter: int = 1000,
        mc_tol: float = 1e-4,
        mc_lambda: float = 1.0,
        train_mse_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate all rows. See ``evaluate_row`` / ``evaluate_column`` for method details.

        Args:
            donor_mask: Optional boolean mask denoting allowable donor rows. Default: None.
            method: See ``evaluate_column``.
            regularization_multiplier: Regularization coefficient. Default: 1e-6.
            en_l1_ratio: L1/L2 mixing for method="elastic_net". Default: 0.5.
            si_rank: SVD components for method="synthetic_intervention". Default: None.
            mc_rank: Rank for MC methods. Default: None.
            mc_max_iter: Max iterations for MC methods. Default: 1000.
            mc_tol: Convergence tolerance for MC methods. Default: 1e-4.
            mc_lambda: Soft-threshold / regularisation strength for
                       method="mc_soft_svd" or method="mc_als". Default: 1.0.
            train_mse_thresholds: Train MSE thresholds for adaptive evaluation.
            n_jobs: Number of parallel jobs. Default: 1.
            verbose: Print progress. Default: False.

        Returns:
            Dictionary containing per-row metrics and aggregates.
        """
        if regularization_multiplier < 0:
            regularization_multiplier = 0

        n_rows = self.real.shape[0]
        method_tag = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in method)
        if method == "neural_net":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "method='neural_net' requires PyTorch. Install it with `pip install torch`."
                ) from exc

            resolved_device = nn_device
            if nn_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            elif nn_device == "cuda" and not torch.cuda.is_available():
                resolved_device = "cpu"

            if resolved_device == "cuda" and n_jobs != 1:
                if verbose:
                    print("Using n_jobs=1 for neural_net on GPU to avoid contention.")
                n_jobs = 1

        print_every = max(10, 5**int(np.log10(n_rows))) if n_rows > 0 else 10

        if n_jobs == 1:
            results_list = []
            for i in range(n_rows):
                res = self.evaluate_row(
                    i,
                    donor_mask=donor_mask,
                    method=method,
                    regularization_multiplier=regularization_multiplier,
                    en_l1_ratio=en_l1_ratio,
                    md_learning_rate=md_learning_rate,
                    md_max_iter=md_max_iter,
                    md_tol=md_tol,
                    md_eps=md_eps,
                    md_adaptive_lr=md_adaptive_lr,
                    md_lr_decay=md_lr_decay,
                    md_min_lr_ratio=md_min_lr_ratio,
                    md_lr_patience=md_lr_patience,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    si_rank=si_rank,
                    mc_rank=mc_rank,
                    mc_max_iter=mc_max_iter,
                    mc_tol=mc_tol,
                    mc_lambda=mc_lambda,
                )
                results_list.append(res)
                if verbose and i % print_every == 0:
                    print(f"Evaluated row {i} of {n_rows}")
        else:
            if verbose:
                print(f"Evaluating {n_rows} rows in parallel using {n_jobs} jobs...")
            results_list = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(self.evaluate_row)(
                    i,
                    donor_mask=donor_mask,
                    method=method,
                    regularization_multiplier=regularization_multiplier,
                    en_l1_ratio=en_l1_ratio,
                    md_learning_rate=md_learning_rate,
                    md_max_iter=md_max_iter,
                    md_tol=md_tol,
                    md_eps=md_eps,
                    md_adaptive_lr=md_adaptive_lr,
                    md_lr_decay=md_lr_decay,
                    md_min_lr_ratio=md_min_lr_ratio,
                    md_lr_patience=md_lr_patience,
                    nn_hidden_dims=nn_hidden_dims,
                    nn_epochs=nn_epochs,
                    nn_lr=nn_lr,
                    nn_weight_decay=nn_weight_decay,
                    nn_batch_size=nn_batch_size,
                    nn_patience=nn_patience,
                    nn_device=nn_device,
                    nn_seed=nn_seed,
                    si_rank=si_rank,
                    mc_rank=mc_rank,
                    mc_max_iter=mc_max_iter,
                    mc_tol=mc_tol,
                    mc_lambda=mc_lambda,
                )
                for i in range(n_rows)
            )

        metrics: List[Dict[str, float]] = [res["metrics"] for res in results_list]
        baseline_metrics: List[Dict[str, float]] = [res["baseline_metrics"] for res in results_list]
        fitted_models: List[Dict[str, Any]] = [res["fitted_model"] for res in results_list]
        has_additional_baseline = bool(results_list) and ("additional_baseline_metrics" in results_list[0])
        if has_additional_baseline:
            additional_baseline_metrics: List[Dict[str, float]] = [res["additional_baseline_metrics"] for res in results_list]
        train_mses: List[float] = [res["train_mse"] for res in results_list]
        corr_normalized: List[float] = [res["corr_normalized"] for res in results_list]
        corr_baseline_normalized: List[float] = [res["corr_baseline_normalized"] for res in results_list]
        if has_additional_baseline:
            corr_additional_baseline_normalized: List[float] = [
                res["corr_additional_baseline_normalized"] for res in results_list
            ]

        if donor_mask is None:
            num_donors = n_rows - 1
        else:
            if donor_mask.shape != (n_rows,):
                raise ValueError("donor_mask must have shape (num_rows,)")
            num_donors = np.sum(donor_mask) - 1

        train_mse_thresholds_metrics = {}
        for thresh in train_mse_thresholds:
            mixed_metrics = []
            for i in range(n_rows):
                if train_mses[i] > thresh:
                    mixed_metrics.append(baseline_metrics[i])
                else:
                    mixed_metrics.append(metrics[i])
            train_mse_thresholds_metrics[thresh] = mixed_metrics

        if verbose:
            print(f"\nNum donors: {num_donors}")
            mse_arr = np.array([m["mse"] for m in metrics], dtype=float)
            base_mse_arr = np.array([m["mse"] for m in baseline_metrics], dtype=float)
            corr_arr = np.array([m["correlation"] for m in metrics], dtype=float)
            base_corr_arr = np.array([m["correlation"] for m in baseline_metrics], dtype=float)
            corr_normalized_arr = np.array(corr_normalized, dtype=float)
            corr_baseline_normalized_arr = np.array(corr_baseline_normalized, dtype=float)
            r2_arr = np.array([m["r2"] for m in metrics], dtype=float)
            base_r2_arr = np.array([m["r2"] for m in baseline_metrics], dtype=float)
            acc_arr = np.array([m["accuracy"] for m in metrics], dtype=float)
            base_acc_arr = np.array([m["accuracy"] for m in baseline_metrics], dtype=float)
            wass_arr = np.array([m["wasserstein_distance"] for m in metrics], dtype=float)
            base_wass_arr = np.array([m["wasserstein_distance"] for m in baseline_metrics], dtype=float)
            std_ratio_arr = np.array([m["std_ratio"] for m in metrics], dtype=float)
            base_std_ratio_arr = np.array([m["std_ratio"] for m in baseline_metrics], dtype=float)
            if has_additional_baseline:
                add_mse_arr = np.array([m["mse"] for m in additional_baseline_metrics], dtype=float)
                add_corr_arr = np.array([m["correlation"] for m in additional_baseline_metrics], dtype=float)
                add_r2_arr = np.array([m["r2"] for m in additional_baseline_metrics], dtype=float)
                add_acc_arr = np.array([m["accuracy"] for m in additional_baseline_metrics], dtype=float)
                add_wass_arr = np.array([m["wasserstein_distance"] for m in additional_baseline_metrics], dtype=float)
                add_std_ratio_arr = np.array([m["std_ratio"] for m in additional_baseline_metrics], dtype=float)
                corr_additional_baseline_normalized_arr = np.array(corr_additional_baseline_normalized, dtype=float)

            def sem(arr):
                """Compute standard error of the mean."""
                n = len(arr)
                if n <= 1:
                    return np.nan
                return np.std(arr, ddof=1) / np.sqrt(n)

            if has_additional_baseline:
                print(
                    f"MSE mean: {float(np.mean(mse_arr)):.4f} ± {float(sem(mse_arr)):.4f}, "
                    f"Baseline MSE mean: {float(np.mean(base_mse_arr)):.4f} ± {float(sem(base_mse_arr)):.4f}, "
                    f"Additional Baseline MSE mean: {float(np.mean(add_mse_arr)):.4f} ± {float(sem(add_mse_arr)):.4f}"
                )
                print(
                    f"Corr mean: {float(np.mean(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, "
                    f"Baseline Corr mean: {float(np.mean(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}, "
                    f"Additional Baseline Corr mean: {float(np.mean(add_corr_arr)):.4f} ± {float(sem(add_corr_arr)):.4f}"
                )
                print(
                    f"Corr (Normalized) mean: {float(np.mean(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, "
                    f"Baseline Corr (Normalized) mean: {float(np.mean(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}, "
                    f"Additional Baseline Corr (Normalized) mean: {float(np.mean(corr_additional_baseline_normalized_arr)):.4f} ± {float(sem(corr_additional_baseline_normalized_arr)):.4f}"
                )
                print(
                    f"Corr mean (Fisher's z): {float(self.fisher_z_average(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, "
                    f"Baseline Corr mean: {float(self.fisher_z_average(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}, "
                    f"Additional Baseline Corr mean: {float(self.fisher_z_average(add_corr_arr)):.4f} ± {float(sem(add_corr_arr)):.4f}"
                )
                print(
                    f"Corr (Normalized, Fisher's z) mean: {float(self.fisher_z_average(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, "
                    f"Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}, "
                    f"Additional Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_additional_baseline_normalized_arr)):.4f} ± {float(sem(corr_additional_baseline_normalized_arr)):.4f}"
                )
                print(
                    f"R2 mean: {float(np.mean(r2_arr)):.4f} ± {float(sem(r2_arr)):.4f}, "
                    f"Baseline R2 mean: {float(np.mean(base_r2_arr)):.4f} ± {float(sem(base_r2_arr)):.4f}, "
                    f"Additional Baseline R2 mean: {float(np.mean(add_r2_arr)):.4f} ± {float(sem(add_r2_arr)):.4f}"
                )
                print(
                    f"Accuracy mean: {float(np.mean(acc_arr)):.4f} ± {float(sem(acc_arr)):.4f}, "
                    f"Baseline Accuracy mean: {float(np.mean(base_acc_arr)):.4f} ± {float(sem(base_acc_arr)):.4f}, "
                    f"Additional Baseline Accuracy mean: {float(np.mean(add_acc_arr)):.4f} ± {float(sem(add_acc_arr)):.4f}"
                )
                print(
                    f"Wasserstein Distance mean: {float(np.mean(wass_arr)):.4f} ± {float(sem(wass_arr)):.4f}, "
                    f"Baseline Wasserstein Distance mean: {float(np.mean(base_wass_arr)):.4f} ± {float(sem(base_wass_arr)):.4f}, "
                    f"Additional Baseline Wasserstein Distance mean: {float(np.mean(add_wass_arr)):.4f} ± {float(sem(add_wass_arr)):.4f}"
                )
                print(
                    f"Std Ratio mean: {float(np.mean(std_ratio_arr)):.4f} ± {float(sem(std_ratio_arr)):.4f}, "
                    f"Baseline Std Ratio mean: {float(np.mean(base_std_ratio_arr)):.4f} ± {float(sem(base_std_ratio_arr)):.4f}, "
                    f"Additional Baseline Std Ratio mean: {float(np.mean(add_std_ratio_arr)):.4f} ± {float(sem(add_std_ratio_arr)):.4f}"
                )
            else:
                print(f"MSE mean: {float(np.mean(mse_arr)):.4f} ± {float(sem(mse_arr)):.4f}, Baseline MSE mean: {float(np.mean(base_mse_arr)):.4f} ± {float(sem(base_mse_arr)):.4f}")
                print(f"Corr mean: {float(np.mean(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, Baseline Corr mean: {float(np.mean(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}")
                print(f"Corr (Normalized) mean: {float(np.mean(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, Baseline Corr (Normalized) mean: {float(np.mean(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}")
                print(f"Corr mean (Fisher's z): {float(self.fisher_z_average(corr_arr)):.4f} ± {float(sem(corr_arr)):.4f}, Baseline Corr mean: {float(self.fisher_z_average(base_corr_arr)):.4f} ± {float(sem(base_corr_arr)):.4f}")
                print(f"Corr (Normalized, Fisher's z) mean: {float(self.fisher_z_average(corr_normalized_arr)):.4f} ± {float(sem(corr_normalized_arr)):.4f}, Baseline Corr (Normalized) mean: {float(self.fisher_z_average(corr_baseline_normalized_arr)):.4f} ± {float(sem(corr_baseline_normalized_arr)):.4f}")
                print(f"R2 mean: {float(np.mean(r2_arr)):.4f} ± {float(sem(r2_arr)):.4f}, Baseline R2 mean: {float(np.mean(base_r2_arr)):.4f} ± {float(sem(base_r2_arr)):.4f}")
                print(f"Accuracy mean: {float(np.mean(acc_arr)):.4f} ± {float(sem(acc_arr)):.4f}, Baseline Accuracy mean: {float(np.mean(base_acc_arr)):.4f} ± {float(sem(base_acc_arr)):.4f}")
                print(f"Wasserstein Distance mean: {float(np.mean(wass_arr)):.4f} ± {float(sem(wass_arr)):.4f}, Baseline Wasserstein Distance mean: {float(np.mean(base_wass_arr)):.4f} ± {float(sem(base_wass_arr)):.4f}")
                print(f"Std Ratio mean: {float(np.mean(std_ratio_arr)):.4f} ± {float(sem(std_ratio_arr)):.4f}, Baseline Std Ratio mean: {float(np.mean(base_std_ratio_arr)):.4f} ± {float(sem(base_std_ratio_arr)):.4f}")

            # Plot correlation gain vs train MSE
            plt.figure(figsize=(8, 6))
            plt.scatter(train_mses, corr_arr - base_corr_arr, alpha=0.5)
            plt.xlabel("Train MSE", fontsize=16)
            plt.ylabel("Correlation Gain", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_figures_dir, f'rows_correlation_gain_vs_train_mse_{method_tag}.pdf'))
            plt.show()

            for q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                q_train_mses = np.quantile(train_mses, q)
                q_corr_arr = np.mean(corr_arr[np.where(train_mses <= q_train_mses)[0]])
                q_base_corr_arr = np.mean(base_corr_arr[np.where(train_mses < q_train_mses)[0]])
                if has_additional_baseline:
                    q_add_corr_arr = np.mean(add_corr_arr[np.where(train_mses < q_train_mses)[0]])
                    print(
                        f"Train MSE {q*100}% quantile: {float(q_train_mses):.4f}, "
                        f"Corr mean: {float(q_corr_arr):.4f}, "
                        f"Baseline Corr mean: {float(q_base_corr_arr):.4f}, "
                        f"Additional Baseline Corr mean: {float(q_add_corr_arr):.4f}"
                    )
                else:
                    print(f"Train MSE {q*100}% quantile: {float(q_train_mses):.4f}, Corr mean: {float(q_corr_arr):.4f}, Baseline Corr mean: {float(q_base_corr_arr):.4f}")

            print("\nAdaptive correlation vs Train MSE Threshold:")
            corr_thresh_list = []
            for thresh in train_mse_thresholds:
                corr_thresh = np.mean(np.array([m["correlation"] for m in train_mse_thresholds_metrics[thresh]]))
                corr_thresh_list.append(corr_thresh)

            plt.figure(figsize=(8, 6))
            plt.scatter(train_mse_thresholds, corr_thresh_list, alpha=0.5, label="Adaptive correlation")
            plt.axhline(y=np.mean(corr_arr), color="red", linestyle="--", label="Full synthetic control correlation")
            plt.axhline(y=np.mean(base_corr_arr), color="blue", linestyle="--", label="Full baseline correlation")
            if has_additional_baseline:
                plt.axhline(y=np.mean(add_corr_arr), color="green", linestyle="--", label="Full additional baseline correlation")
            plt.xlabel("Train MSE Threshold", fontsize=16)
            plt.ylabel("Correlation Mean", fontsize=16)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.results_figures_dir,
                    f'rows_adaptive_correlation_vs_train_mse_threshold_{method_tag}.pdf',
                )
            )
            plt.show()
            print(f"Maximum adaptive correlation: {float(np.max(corr_thresh_list)):.4f} achieved at train MSE threshold: {train_mse_thresholds[np.argmax(corr_thresh_list)]}")

        result = {
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "fitted_models": fitted_models,
            "train_mses": train_mses,
            "corr_normalized": corr_normalized,
            "corr_baseline_normalized": corr_baseline_normalized,
            "num_donors": num_donors,
            "train_mse_thresholds_metrics": train_mse_thresholds_metrics
        }
        if has_additional_baseline:
            result["additional_baseline_metrics"] = additional_baseline_metrics
            result["corr_additional_baseline_normalized"] = corr_additional_baseline_normalized
        return result
