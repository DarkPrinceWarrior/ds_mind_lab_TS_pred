"""Extended metrics module with additional performance measures."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

EPSILON = 1e-9


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + EPSILON)))


def adjusted_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    """Calculate adjusted R².
    
    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    where n is sample size and p is number of features
    """
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    
    if n <= n_features + 1:
        return r2
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    return float(adj_r2)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency.
    
    NSE = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²
    
    NSE ranges from -∞ to 1:
    - 1 = perfect match
    - 0 = model predictions as good as mean
    - <0 = worse than mean
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (numerator / (denominator + EPSILON)))


def kge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[tuple[float, float, float]] = None,
) -> float:
    """Calculate Kling-Gupta Efficiency.
    
    KGE = 1 - sqrt[(r-1)² + (α-1)² + (β-1)²]
    
    where:
    - r = correlation coefficient
    - α = std(y_pred) / std(y_true) (variability ratio)
    - β = mean(y_pred) / mean(y_true) (bias ratio)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Optional weights for (r, α, β) components, defaults to (1, 1, 1)
    
    Returns:
        KGE value (ranges from -∞ to 1, 1 is perfect)
    """
    if weights is None:
        weights = (1.0, 1.0, 1.0)
    
    # Correlation
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(r):
        r = 0.0
    
    # Variability ratio
    std_pred = np.std(y_pred, ddof=1)
    std_true = np.std(y_true, ddof=1)
    alpha = std_pred / (std_true + EPSILON)
    
    # Bias ratio
    mean_pred = np.mean(y_pred)
    mean_true = np.mean(y_true)
    beta = mean_pred / (mean_true + EPSILON)
    
    # KGE calculation
    ed = np.sqrt(
        weights[0] * (r - 1) ** 2
        + weights[1] * (alpha - 1) ** 2
        + weights[2] * (beta - 1) ** 2
    )
    
    return float(1 - ed)


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Percent Bias (PBIAS).
    
    PBIAS = 100 * Σ(y_true - y_pred) / Σ(y_true)
    
    Optimal value is 0:
    - Positive: model underestimates
    - Negative: model overestimates
    """
    numerator = np.sum(y_true - y_pred)
    denominator = np.sum(y_true)
    return float(100.0 * numerator / (denominator + EPSILON))


def index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Willmott's Index of Agreement (d).
    
    d = 1 - Σ(y_true - y_pred)² / Σ(|y_pred - mean(y_true)| + |y_true - mean(y_true)|)²
    
    Ranges from 0 to 1, where 1 indicates perfect agreement.
    """
    mean_obs = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(
        (np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2
    )
    return float(1 - (numerator / (denominator + EPSILON)))


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Bias Error (MBE)."""
    return float(np.mean(y_pred - y_true))


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
) -> float:
    """Calculate Skill Score relative to baseline.
    
    SS = 1 - MSE(model) / MSE(baseline)
    
    Args:
        y_true: True values
        y_pred: Model predictions
        y_baseline: Baseline predictions (e.g., persistence, climatology)
    
    Returns:
        Skill score (>0 means better than baseline)
    """
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_baseline = np.mean((y_true - y_baseline) ** 2)
    return float(1 - (mse_model / (mse_baseline + EPSILON)))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    """Calculate comprehensive set of metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_insample: In-sample values for MASE calculation
        n_features: Number of features for adjusted R²
    
    Returns:
        Dictionary with all calculated metrics
    """
    # Basic error metrics
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # Percentage errors
    abs_true = np.abs(y_true)
    wmape = float(np.sum(np.abs(errors)) / (np.sum(abs_true) + EPSILON) * 100.0)
    
    denom = np.where(abs_true > EPSILON, abs_true, np.nan)
    mape = float(np.nanmean(np.abs(errors) / denom) * 100.0)
    
    smape = float(
        np.mean(2.0 * np.abs(errors) / (abs_true + np.abs(y_pred) + EPSILON)) * 100.0
    )
    
    # MASE
    mase = None
    if y_insample is not None and len(y_insample) > 1:
        diffs = np.abs(y_insample[1:] - y_insample[:-1])
        scale = float(np.mean(diffs))
        if np.isfinite(scale) and scale > EPSILON:
            mase = mae / scale
    
    # Statistical metrics
    r2 = r_squared(y_true, y_pred)
    nse_val = nse(y_true, y_pred)
    kge_val = kge(y_true, y_pred)
    pbias_val = pbias(y_true, y_pred)
    ioa = index_of_agreement(y_true, y_pred)
    mbe = mean_bias_error(y_true, y_pred)
    
    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    
    # Adjusted R² if features provided
    adj_r2 = None
    if n_features is not None and n_features > 0:
        adj_r2 = adjusted_r_squared(y_true, y_pred, n_features)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wmape": wmape,
        "mase": mase,
        "r2": r2,
        "adjusted_r2": adj_r2,
        "nse": nse_val,
        "kge": kge_val,
        "pbias": pbias_val,
        "index_of_agreement": ioa,
        "mean_bias_error": mbe,
        "correlation": corr,
    }
    
    # Filter out None and non-finite values
    return {
        k: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        for k, v in metrics.items()
    }


def calculate_metrics_by_horizon(
    merged: pd.DataFrame,
    horizon_steps: int,
) -> Dict[int, Dict[str, float]]:
    """Calculate metrics separately for each forecast horizon step.
    
    Args:
        merged: DataFrame with 'unique_id', 'ds', 'y', 'y_hat'
        horizon_steps: Number of horizon steps
    
    Returns:
        Dictionary mapping horizon step to metrics
    """
    if "time_idx" not in merged.columns:
        merged = merged.copy()
        merged["time_idx"] = merged.groupby("unique_id").cumcount()
    
    horizon_metrics = {}
    
    for step in range(1, horizon_steps + 1):
        # Get data for this horizon step
        step_data = merged[merged.groupby("unique_id")["time_idx"].transform("max") >= step - 1]
        step_data = step_data.groupby("unique_id").nth(step - 1).reset_index()
        
        if len(step_data) < 2:
            continue
        
        y_true = step_data["y"].to_numpy()
        y_pred = step_data["y_hat"].to_numpy()
        
        metrics = calculate_all_metrics(y_true, y_pred)
        horizon_metrics[step] = metrics
    
    return horizon_metrics


def print_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Metrics Summary",
) -> None:
    """Print formatted metrics summary.
    
    Args:
        metrics: Dictionary of metric name to value
        title: Title for the summary
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    # Group metrics
    error_metrics = ["mae", "rmse", "mape", "smape", "wmape", "mase"]
    stat_metrics = ["r2", "adjusted_r2", "nse", "kge", "correlation"]
    other_metrics = ["pbias", "index_of_agreement", "mean_bias_error"]
    
    def _print_group(names, label):
        logger.info(f"\n{label}:")
        for name in names:
            if name in metrics and metrics[name] is not None:
                value = metrics[name]
                if abs(value) < 0.01:
                    logger.info(f"  {name:25s}: {value:.6f}")
                else:
                    logger.info(f"  {name:25s}: {value:.4f}")
    
    _print_group(error_metrics, "Error Metrics")
    _print_group(stat_metrics, "Statistical Metrics")
    _print_group(other_metrics, "Other Metrics")
    
    logger.info("=" * 60)
