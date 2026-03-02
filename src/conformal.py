from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPSILON = 1e-9


def _resolve_method(method: str) -> str:
    normalized = (method or "").strip().lower()
    aliases = {
        "split": "icp",
        "split_cp": "icp",
        "scp": "icp",
        "wcp": "wcp_exp",
        "wcp_exp": "wcp_exp",
        "wcp_exponential": "wcp_exp",
        "wcp_lin": "wcp_linear",
        "wcp_linear": "wcp_linear",
        "icp": "icp",
    }
    return aliases.get(normalized, "wcp_exp")


def _finite_sample_quantile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    vals = np.sort(values.astype(float))
    q = float(np.clip(quantile, 0.0, 1.0))
    rank = int(np.ceil(q * len(vals))) - 1
    rank = int(np.clip(rank, 0, len(vals) - 1))
    return float(vals[rank])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    sorter = np.argsort(values)
    vals = values[sorter].astype(float)
    w = np.maximum(weights[sorter].astype(float), 0.0)
    w_sum = float(w.sum())
    if w_sum <= EPSILON:
        return _finite_sample_quantile(vals, quantile)
    cdf = np.cumsum(w) / w_sum
    q = float(np.clip(quantile, 0.0, 1.0))
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def _build_weights(n: int, method: str, decay: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if method == "wcp_linear":
        return np.arange(1, n + 1, dtype=float)
    if method == "wcp_exp":
        d = float(np.clip(decay, 1e-6, 1.0))
        return np.power(d, np.arange(n - 1, -1, -1, dtype=float))
    return np.ones(n, dtype=float)


def _compute_eps(errors: np.ndarray, alpha: float, method: str, decay: float) -> float:
    if errors.size == 0:
        return float("nan")
    q = float(np.clip(1.0 - alpha, 0.0, 1.0))
    if method == "icp":
        # Split conformal finite-sample quantile: ceil((n+1)*(1-alpha))/n.
        level = np.ceil((len(errors) + 1) * q) / max(len(errors), 1)
        level = float(np.clip(level, 0.0, 1.0))
        return _finite_sample_quantile(errors, level)
    weights = _build_weights(len(errors), method, decay)
    return _weighted_quantile(errors, weights, q)


def compute_conformal_profile(
    residuals_df: pd.DataFrame,
    horizon: int,
    alpha: float = 0.1,
    method: str = "wcp_exp",
    per_horizon: bool = True,
    exp_decay: float = 0.97,
    min_samples: int = 30,
) -> Optional[Dict[str, Any]]:
    """Build conformal calibration profile from out-of-sample residuals.

    Expected columns: `abs_err`, optionally `h_step`, `ds`, `calib_order`.
    """
    if residuals_df is None or residuals_df.empty or "abs_err" not in residuals_df.columns:
        logger.warning("Conformal calibration skipped: empty residual pool.")
        return None

    method_resolved = _resolve_method(method)
    alpha = float(np.clip(alpha, 1e-6, 1 - 1e-6))
    min_samples = int(max(min_samples, 1))
    horizon = int(max(horizon, 1))

    work = residuals_df.copy()
    work["abs_err"] = pd.to_numeric(work["abs_err"], errors="coerce")
    work = work[work["abs_err"].notna()].copy()
    if work.empty:
        logger.warning("Conformal calibration skipped: no finite residuals.")
        return None

    if "calib_order" in work.columns:
        work = work.sort_values("calib_order")
    elif "ds" in work.columns:
        work = work.sort_values("ds")
    else:
        work = work.reset_index(drop=True)
    work = work.reset_index(drop=True)

    if "h_step" in work.columns:
        work["h_step"] = pd.to_numeric(work["h_step"], errors="coerce")
    else:
        work["h_step"] = np.nan

    all_errors = work["abs_err"].to_numpy(dtype=float)
    global_eps = _compute_eps(all_errors, alpha=alpha, method=method_resolved, decay=exp_decay)

    eps_by_h: Dict[str, float] = {}
    samples_by_h: Dict[str, int] = {}
    source_by_h: Dict[str, str] = {}

    for h in range(1, horizon + 1):
        h_mask = work["h_step"] == h
        h_errors = work.loc[h_mask, "abs_err"].to_numpy(dtype=float)
        h_count = int(h_errors.size)
        samples_by_h[str(h)] = h_count

        if per_horizon and h_count >= min_samples:
            eps = _compute_eps(h_errors, alpha=alpha, method=method_resolved, decay=exp_decay)
            source_by_h[str(h)] = "per_horizon"
        else:
            eps = global_eps
            source_by_h[str(h)] = "global_fallback"
        eps_by_h[str(h)] = float(eps)

    profile: Dict[str, Any] = {
        "enabled": True,
        "method": method_resolved,
        "alpha": alpha,
        "target_coverage": float(1.0 - alpha),
        "per_horizon": bool(per_horizon),
        "exp_decay": float(exp_decay),
        "min_samples": min_samples,
        "global_eps": float(global_eps),
        "global_samples": int(len(all_errors)),
        "eps_by_horizon": eps_by_h,
        "samples_by_horizon": samples_by_h,
        "source_by_horizon": source_by_h,
    }
    return profile


def apply_conformal_intervals(
    pred_df: pd.DataFrame,
    profile: Optional[Dict[str, Any]],
    *,
    horizon: Optional[int] = None,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df
    if not profile or not profile.get("enabled", False):
        return pred_df.copy()
    if "y_hat" not in pred_df.columns:
        raise ValueError("Prediction DataFrame must contain 'y_hat' for conformal intervals.")

    out = pred_df.copy()
    if "unique_id" in out.columns and "ds" in out.columns:
        out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        h_step = out.groupby("unique_id").cumcount() + 1
    else:
        h_step = pd.Series(np.arange(1, len(out) + 1), index=out.index)
    if horizon is not None:
        h_step = h_step.clip(upper=int(max(horizon, 1)))
    out["h_step"] = h_step.astype(int)

    eps_map_raw = profile.get("eps_by_horizon", {})
    eps_map = {int(k): float(v) for k, v in eps_map_raw.items()}
    src_map_raw = profile.get("source_by_horizon", {})
    src_map = {int(k): str(v) for k, v in src_map_raw.items()}
    global_eps = float(profile.get("global_eps", 0.0))

    out["cp_eps"] = out["h_step"].map(eps_map).fillna(global_eps).astype(float)
    out["cp_source"] = out["h_step"].map(src_map).fillna("global_fallback")
    out["cp_lo"] = out["y_hat"].astype(float) - out["cp_eps"]
    out["cp_hi"] = out["y_hat"].astype(float) + out["cp_eps"]
    out["cp_alpha"] = float(profile.get("alpha", np.nan))
    out["cp_method"] = str(profile.get("method", "icp"))
    return out
