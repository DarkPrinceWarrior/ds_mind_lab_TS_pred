from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0
FREQ_ALIAS = {
    "D": "D",
    "DAILY": "D",
    "1D": "D",
    "W": "W",
    "WEEKLY": "W",
    "7D": "W",
    "M": "M",
    "MS": "M",
    "MONTHLY": "M",
}
PERIOD_LENGTH_DAYS = {
    "D": 1.0,
    "W": 7.0,
    "M": 30.0,
}
DEFAULT_PHYSICS = {
    "k_md": 150.0,
    "phi": 0.18,
    "mu_cP": 1.2,
    "ct_1overMPa": 8e-4,
}
DEFAULT_TAU_BOUND_MULTIPLIER = 2.0
DEFAULT_FRONT_BUFFER_DAYS = 30.0


@dataclass
class LagBounds:
    lower: int
    upper: int
    t_diff_days: float
    t_front_days: float


@dataclass
class LagSelection:
    lag: int
    score: float
    overlap: int
    correlations: Dict[int, float]


def _normalize_freq(freq: str) -> str:
    key = freq.upper().strip()
    if key not in FREQ_ALIAS:
        raise ValueError(f"Unsupported frequency '{freq}' for lag estimation.")
    return FREQ_ALIAS[key]


def _period_length_days(freq: str) -> float:
    base = _normalize_freq(freq)
    return PERIOD_LENGTH_DAYS[base]


def estimate_tau_window(
    distance_m: float,
    freq: str,
    physics_estimates: Optional[Dict[str, float]] = None,
    tau_bound_multiplier: float = DEFAULT_TAU_BOUND_MULTIPLIER,
) -> LagBounds:
    if distance_m <= 0 or not np.isfinite(distance_m):
        raise ValueError(f"Distance must be positive and finite, got {distance_m}.")
    base_freq = _normalize_freq(freq)
    period_days = _period_length_days(base_freq)
    physics = dict(DEFAULT_PHYSICS)
    if physics_estimates:
        physics.update({k: float(v) for k, v in physics_estimates.items() if v is not None})
    k_md = max(physics.get("k_md", DEFAULT_PHYSICS["k_md"]), 1e-6)
    phi = min(max(physics.get("phi", DEFAULT_PHYSICS["phi"]), 1e-4), 0.5)
    mu_cP = max(physics.get("mu_cP", DEFAULT_PHYSICS["mu_cP"]), 1e-4)
    ct = max(physics.get("ct_1overMPa", DEFAULT_PHYSICS["ct_1overMPa"]), 1e-6)
    k_m2 = k_md * 9.869233e-16
    mu_pa_s = mu_cP * 1e-3
    ct_1over_pa = ct / 1e6
    alpha = k_m2 / (phi * mu_pa_s * ct_1over_pa)
    alpha = max(alpha, 1e-16)
    t_diff_seconds = (distance_m ** 2) / (4.0 * alpha)
    t_diff_days = t_diff_seconds / SECONDS_PER_DAY
    front_velocity = physics_estimates.get("v_front_m_per_period") if physics_estimates else None
    if front_velocity and np.isfinite(front_velocity) and front_velocity > 0:
        t_front_periods = distance_m / front_velocity
        t_front_days = t_front_periods * period_days
    else:
        multiplier = max(float(tau_bound_multiplier), 1.5)
        buffered = t_diff_days + DEFAULT_FRONT_BUFFER_DAYS
        t_front_days = max(multiplier * t_diff_days, buffered)
    lower_periods = max(int(math.ceil(t_diff_days / period_days)), 1)
    upper_periods = max(int(math.ceil(t_front_days / period_days)), lower_periods)
    return LagBounds(lower=lower_periods, upper=upper_periods, t_diff_days=t_diff_days, t_front_days=t_front_days)


def _ridge_correlation(x: np.ndarray, y: np.ndarray, ridge_alpha: float) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = math.sqrt(np.nanmean(x ** 2) + ridge_alpha) * math.sqrt(np.nanmean(y ** 2) + ridge_alpha)
    if denom == 0:
        return 0.0
    return float(np.nanmean(x * y) / denom)


def causal_xcorr_best_lag(
    x: pd.Series,
    y: pd.Series,
    candidate_lags: Sequence[int],
    min_overlap: int = 6,
    ridge_alpha: float = 1e-6,
) -> LagSelection:
    if not candidate_lags:
        raise ValueError("Candidate lags list must not be empty.")
    corrs: Dict[int, float] = {}
    best_lag = int(candidate_lags[0])
    best_score = -np.inf
    best_overlap = 0
    x_clean = x.astype(float).replace([np.inf, -np.inf], np.nan)
    y_clean = y.astype(float).replace([np.inf, -np.inf], np.nan)
    for lag in sorted(set(int(l) for l in candidate_lags)):
        if lag < 0:
            continue
        shifted = x_clean.shift(lag)
        aligned = pd.concat([shifted, y_clean], axis=1, keys=["x", "y"]).dropna()
        overlap = len(aligned)
        if overlap < max(min_overlap, 1):
            corrs[lag] = float("nan")
            continue
        corr = _ridge_correlation(aligned["x"].to_numpy(), aligned["y"].to_numpy(), ridge_alpha=ridge_alpha)
        corrs[lag] = corr
        if corr > best_score:
            best_score = corr
            best_lag = lag
            best_overlap = overlap
    if not np.isfinite(best_score):
        best_score = 0.0
    return LagSelection(lag=int(best_lag), score=float(best_score), overlap=int(best_overlap), correlations=corrs)


def crm_exp_filter(
    series: pd.Series,
    tau: float,
    delta: float = 1.0,
    init: Optional[float] = None,
) -> pd.Series:
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError(f"CRM time constant must be positive and finite, got {tau}.")
    decay = math.exp(-float(delta) / float(tau))
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    out = np.zeros_like(values)
    if values.size == 0:
        return series.astype(float)
    prev_output = float(init) if init is not None else float(values[0])
    prev_input = float(values[0])
    out[0] = prev_output
    for idx in range(1, len(values)):
        prev_output = prev_output * decay + (1.0 - decay) * prev_input
        out[idx] = prev_output
        prev_input = float(values[idx - 1])
    return pd.Series(out, index=series.index, name=series.name)
