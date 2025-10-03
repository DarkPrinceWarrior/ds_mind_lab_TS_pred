from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window
except ImportError:  # pragma: no cover
    from utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window

logger = logging.getLogger(__name__)


@dataclass
class PairLagSummary:
    prod_id: str
    inj_id: str
    distance_m: float
    weight: float
    lag: int
    tau: float
    corr: float
    overlap: int
    lower_bound: int
    upper_bound: int
    t_diff_days: float
    t_front_days: float


SUMMARY_COLUMNS = [field.name for field in fields(PairLagSummary)]


def _normalize_well_id(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(int(value))
    raise ValueError(f"Unsupported well identifier type: {type(value)!r}")


def _build_distance_lookup(
    coords: pd.DataFrame,
    prod_wells: Iterable[str],
    inj_wells: Iterable[str],
    distances: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[str, str], float]:
    coords_idx = coords.set_index("well")
    lookup: Dict[Tuple[str, str], float] = {}
    distance_matrix: Optional[pd.DataFrame] = None
    if distances is not None and not distances.empty:
        normalized_index = [_normalize_well_id(idx) for idx in distances.index]
        normalized_cols = [_normalize_well_id(col) for col in distances.columns]
        distance_matrix = distances.copy()
        distance_matrix.index = normalized_index
        distance_matrix.columns = normalized_cols
    for prod in prod_wells:
        for inj in inj_wells:
            value = np.nan
            if distance_matrix is not None:
                if prod in distance_matrix.index and inj in distance_matrix.columns:
                    value = distance_matrix.at[prod, inj]
                elif inj in distance_matrix.index and prod in distance_matrix.columns:
                    value = distance_matrix.at[inj, prod]
            if pd.notna(value):
                lookup[(prod, inj)] = float(value)
                continue
            if prod not in coords_idx.index:
                raise ValueError(f"Missing coordinates for producer {prod}")
            if inj not in coords_idx.index:
                raise ValueError(f"Missing coordinates for injector {inj}")
            prod_vec = coords_idx.loc[prod, ["x", "y", "z"]].to_numpy(dtype=float)
            inj_vec = coords_idx.loc[inj, ["x", "y", "z"]].to_numpy(dtype=float)
            lookup[(prod, inj)] = float(np.linalg.norm(prod_vec - inj_vec))
    return lookup


def _select_topk_injectors(
    prod_wells: List[str],
    inj_wells: List[str],
    distance_lookup: Dict[Tuple[str, str], float],
    topK: int,
    kernel_p: float,
) -> Dict[str, List[Tuple[str, float]]]:
    weights: Dict[str, List[Tuple[str, float]]] = {}
    for prod in prod_wells:
        candidates: List[Tuple[str, float, float]] = []
        for inj in inj_wells:
            dist = distance_lookup.get((prod, inj), np.inf)
            if not np.isfinite(dist) or dist <= 0:
                weight = 1.0
            else:
                weight = 1.0 / max(dist, 1e-6) ** max(kernel_p, 1.0)
            candidates.append((inj, weight, dist))
        candidates.sort(key=lambda item: item[2])
        selected = candidates[: max(topK, 1)] if topK > 0 else []
        total_weight = sum(weight for _, weight, _ in selected)
        if total_weight <= 0 and selected:
            normalized = [(inj, 1.0 / len(selected)) for inj, _, _ in selected]
        else:
            normalized = [(inj, weight / total_weight) for inj, weight, _ in selected]
        weights[prod] = normalized
    return weights


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            result[col] = 0.0
    return result


def build_injection_lag_features(
    prod_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    coords: pd.DataFrame,
    freq: str,
    train_cutoff: pd.Timestamp,
    *,
    distances: Optional[pd.DataFrame] = None,
    physics_estimates: Optional[Dict[str, float]] = None,
    topK: int = 5,
    kernel_p: float = 2.0,
    use_crm: bool = True,
    tau_bound_multiplier: float = 2.0,
    min_overlap: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if prod_df.empty:
        raise ValueError("Producer dataframe is empty.")
    prod_df = prod_df.copy()
    prod_df["well"] = prod_df["well"].map(_normalize_well_id)
    inj_df = inj_df.copy()
    inj_df["well"] = inj_df["well"].map(_normalize_well_id)
    coords = coords.copy()
    coords["well"] = coords["well"].map(_normalize_well_id)
    prod_wells = sorted(prod_df["well"].unique())
    inj_wells = sorted(inj_df["well"].unique())
    if not inj_wells:
        logger.warning("No injector wells provided; returning zero injection features.")
        date_index = pd.Index(sorted(prod_df["ds"].unique()), name="ds")
        zero = pd.DataFrame(
            {
                "ds": np.repeat(date_index.values, len(prod_wells)),
                "well": np.tile(prod_wells, len(date_index)),
                "inj_wwir_lag_weighted": 0.0,
                "inj_wwir_crm_weighted": 0.0,
                "inj_wwit_diff_lag_weighted": 0.0,
            }
        )
        return zero, pd.DataFrame(columns=SUMMARY_COLUMNS)
    distance_lookup = _build_distance_lookup(coords, prod_wells, inj_wells, distances=distances)
    weight_map = _select_topk_injectors(prod_wells, inj_wells, distance_lookup, topK=topK, kernel_p=kernel_p)
    prod_df = _ensure_columns(prod_df, ["wlpr"])
    inj_df = _ensure_columns(inj_df, ["wwir", "wwit", "wwit_diff"])
    full_index = pd.Index(sorted(prod_df["ds"].unique()), name="ds")
    inj_rate = (
        inj_df.pivot(index="ds", columns="well", values="wwir")
        .reindex(full_index)
        .fillna(0.0)
    )
    inj_wwit_diff = (
        inj_df.pivot(index="ds", columns="well", values="wwit_diff")
        .reindex(full_index)
        .fillna(0.0)
    )
    prod_target = (
        prod_df.pivot(index="ds", columns="well", values="wlpr")
        .reindex(full_index)
        .ffill()
        .fillna(0.0)
    )
    train_mask = full_index <= train_cutoff
    summaries: List[PairLagSummary] = []
    lag_cache: Dict[Tuple[str, str], PairLagSummary] = {}
    for prod in prod_wells:
        prod_series_full = prod_target[prod]
        prod_series_train = prod_series_full.loc[train_mask]
        for inj, weight in weight_map.get(prod, []):
            distance = distance_lookup.get((prod, inj), np.inf)
            bounds = estimate_tau_window(
                distance_m=distance,
                freq=freq,
                physics_estimates=physics_estimates,
                tau_bound_multiplier=tau_bound_multiplier,
            )
            candidates = list(range(bounds.lower, bounds.upper + 1))
            inj_series_rate = inj_rate[inj]
            inj_series_train = inj_series_rate.loc[train_mask]
            selection = causal_xcorr_best_lag(
                inj_series_train,
                prod_series_train,
                candidate_lags=candidates,
                min_overlap=min_overlap,
            )
            tau = max(selection.lag, bounds.lower)
            summary = PairLagSummary(
                prod_id=prod,
                inj_id=inj,
                distance_m=float(distance),
                weight=float(weight),
                lag=int(selection.lag),
                tau=float(tau),
                corr=float(selection.score),
                overlap=int(selection.overlap),
                lower_bound=int(bounds.lower),
                upper_bound=int(bounds.upper),
                t_diff_days=float(bounds.t_diff_days),
                t_front_days=float(bounds.t_front_days),
            )
            summaries.append(summary)
            lag_cache[(prod, inj)] = summary
    lagged_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    lagged_wwit_diff = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    crm_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells) if use_crm else None
    for prod in prod_wells:
        for inj, weight in weight_map.get(prod, []):
            summary = lag_cache.get((prod, inj))
            if summary is None:
                continue
            lagged_rate_series = inj_rate[inj].shift(summary.lag).fillna(0.0)
            lagged_diff_series = inj_wwit_diff[inj].shift(summary.lag).fillna(0.0)
            lagged_rate[prod] += weight * lagged_rate_series
            lagged_wwit_diff[prod] += weight * lagged_diff_series
            if use_crm and crm_rate is not None:
                crm_series = crm_exp_filter(inj_rate[inj], tau=summary.tau, delta=1.0)
                crm_rate[prod] += weight * crm_series.fillna(0.0)
    feature_frames: List[pd.DataFrame] = []
    for prod in prod_wells:
        frame = pd.DataFrame(
            {
                "ds": full_index,
                "well": prod,
                "inj_wwir_lag_weighted": lagged_rate[prod].to_numpy(),
                "inj_wwit_diff_lag_weighted": lagged_wwit_diff[prod].to_numpy(),
            }
        )
        if use_crm and crm_rate is not None:
            frame["inj_wwir_crm_weighted"] = crm_rate[prod].to_numpy()
        else:
            frame["inj_wwir_crm_weighted"] = 0.0
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    features = features.fillna(0.0)
    summary_df = pd.DataFrame([asdict(summary) for summary in summaries])
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["prod_id", "lag", "inj_id"])
    return features, summary_df
