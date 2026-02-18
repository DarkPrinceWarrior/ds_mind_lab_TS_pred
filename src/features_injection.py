from __future__ import annotations

import logging
import itertools
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window
except ImportError:  # pragma: no cover
    from utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window

logger = logging.getLogger(__name__)

EPSILON_DISTANCE = 1e-6


@dataclass
class PairLagSummary:
    prod_id: str
    inj_id: str
    distance_m: float
    metric_distance_m: float
    direction_factor: float
    direction_cosine: float
    kernel_type: str
    kernel_params: Dict[str, float]
    kernel_score: float
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
    *,
    anisotropy: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    coords_idx = coords.set_index("well")
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    distance_matrix: Optional[pd.DataFrame] = None
    if distances is not None and not distances.empty:
        normalized_index = [_normalize_well_id(idx) for idx in distances.index]
        normalized_cols = [_normalize_well_id(col) for col in distances.columns]
        distance_matrix = distances.copy()
        distance_matrix.index = normalized_index
        distance_matrix.columns = normalized_cols
    anisotropy_inv = _prepare_anisotropy_matrix(anisotropy)
    for prod in prod_wells:
        if prod not in coords_idx.index:
            raise ValueError(f"Missing coordinates for producer {prod}")
        prod_vec = coords_idx.loc[prod, ["x", "y", "z"]].to_numpy(dtype=float)
        for inj in inj_wells:
            if inj not in coords_idx.index:
                raise ValueError(f"Missing coordinates for injector {inj}")
            inj_vec = coords_idx.loc[inj, ["x", "y", "z"]].to_numpy(dtype=float)
            delta = (prod_vec - inj_vec).astype(float)
            euclidean = float(np.linalg.norm(delta))
            metric_value = np.nan
            if distance_matrix is not None:
                if prod in distance_matrix.index and inj in distance_matrix.columns:
                    metric_value = distance_matrix.at[prod, inj]
                elif inj in distance_matrix.index and prod in distance_matrix.columns:
                    metric_value = distance_matrix.at[inj, prod]
            if pd.isna(metric_value):
                metric_value = _anisotropic_distance(delta, anisotropy_inv)
            lookup[(prod, inj)] = {
                "metric_distance": float(metric_value),
                "euclidean_distance": float(euclidean),
                "delta": delta,
            }
    return lookup


def _select_topk_neighbors(
    prod_wells: List[str],
    inj_wells: List[str],
    distance_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    topK: int,
) -> Dict[str, List[Dict[str, Any]]]:
    neighbors: Dict[str, List[Dict[str, Any]]] = {}
    limit = max(int(topK), 0)
    for prod in prod_wells:
        candidates: List[Dict[str, Any]] = []
        for inj in inj_wells:
            info = distance_lookup.get((prod, inj))
            if info is None:
                continue
            entry = {
                "inj": inj,
                "metric_distance": float(info["metric_distance"]),
                "euclidean_distance": float(info["euclidean_distance"]),
                "delta": np.array(info["delta"], dtype=float),
            }
            candidates.append(entry)
        candidates.sort(key=lambda item: item["metric_distance"])
        if limit > 0:
            candidates = candidates[:limit]
        neighbors[prod] = candidates
    return neighbors


def _kernel_value(distance: float, kernel_type: str, params: Dict[str, float]) -> float:
    if pd.isna(distance):
        return 0.0
    dist = float(distance)
    if dist < 0:
        dist = abs(dist)
    if not np.isfinite(dist):
        abs_dist = np.inf
    else:
        abs_dist = max(dist, EPSILON_DISTANCE)
    ktype = kernel_type.lower()
    if ktype in {"idw", "inverse_distance"}:
        power = max(float(params.get("p", 2.0)), 0.0)
        if not np.isfinite(abs_dist):
            return 0.0
        return float(1.0 / (abs_dist ** max(power, EPSILON_DISTANCE)))
    if ktype in {"exponential", "exp"}:
        scale = max(float(params.get("scale", 500.0)), EPSILON_DISTANCE)
        if not np.isfinite(scale):
            scale = 500.0
        return float(np.exp(-abs_dist / scale))
    if ktype in {"gaussian", "rbf"}:
        scale = max(float(params.get("scale", 500.0)), EPSILON_DISTANCE)
        if not np.isfinite(scale):
            scale = 500.0
        return float(np.exp(-0.5 * (abs_dist / scale) ** 2))
    if ktype in {"matern", "matern"}:
        scale = max(float(params.get("scale", 500.0)), EPSILON_DISTANCE)
        if not np.isfinite(scale):
            scale = 500.0
        nu = float(params.get("nu", 1.5))
        return float(_matern_kernel(abs_dist, scale, nu))
    if ktype in {"rational_quadratic", "rq"}:
        scale = max(float(params.get("scale", 500.0)), EPSILON_DISTANCE)
        alpha = max(float(params.get("alpha", 1.0)), EPSILON_DISTANCE)
        if not np.isfinite(scale):
            scale = 500.0
        return float((1.0 + (abs_dist ** 2) / (2.0 * alpha * scale ** 2)) ** (-alpha))
    if ktype == "spherical":
        range_ = max(float(params.get("range", 500.0)), EPSILON_DISTANCE)
        if not np.isfinite(range_):
            range_ = 500.0
        if not np.isfinite(abs_dist):
            return 0.0
        ratio = abs_dist / range_
        if ratio >= 1.0:
            return 0.0
        return float(1.0 - 1.5 * ratio + 0.5 * ratio ** 3)
    raise ValueError(f"Unsupported kernel type '{kernel_type}'.")


def _matern_kernel(distance: float, scale: float, nu: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    if distance <= 0.0:
        return 1.0
    scale = max(scale, EPSILON_DISTANCE)
    nu = float(nu)
    if np.isclose(nu, 0.5):
        return float(np.exp(-distance / scale))
    if np.isclose(nu, 1.5):
        factor = np.sqrt(3.0) * distance / scale
        return float((1.0 + factor) * np.exp(-factor))
    if np.isclose(nu, 2.5):
        factor = np.sqrt(5.0) * distance / scale
        return float((1.0 + factor + (factor ** 2) / 3.0) * np.exp(-factor))
    raise ValueError(
        "Matern kernel currently supports nu in {0.5, 1.5, 2.5}. "
        "Provide one of these values or extend the implementation."
    )


def _prepare_anisotropy_matrix(anisotropy: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not anisotropy:
        return None
    if isinstance(anisotropy, dict):
        if "inverse" in anisotropy:
            matrix_inv = np.asarray(anisotropy["inverse"], dtype=float)
        elif "matrix" in anisotropy:
            matrix = np.asarray(anisotropy["matrix"], dtype=float)
            if matrix.shape != (3, 3):
                raise ValueError("Anisotropy matrix must be 3x3.")
            matrix_inv = np.linalg.inv(matrix)
        elif "scale" in anisotropy:
            scale_map = anisotropy["scale"]
            if not isinstance(scale_map, dict):
                raise ValueError("Anisotropy 'scale' must be a mapping of axis -> value.")
            scales = [float(scale_map.get(axis, 1.0)) for axis in ("x", "y", "z")]
            if any(scale <= 0 for scale in scales):
                raise ValueError("Anisotropy scale values must be positive.")
            matrix_inv = np.diag([1.0 / (scale ** 2) for scale in scales])
        else:
            raise ValueError(
                "Anisotropy configuration must define 'matrix', 'inverse', or 'scale'."
            )
    else:
        matrix_inv = np.asarray(anisotropy, dtype=float)
    if matrix_inv.shape != (3, 3):
        raise ValueError("Anisotropy inverse matrix must be 3x3.")
    eigvals = np.linalg.eigvals(matrix_inv)
    if not np.all(eigvals > 0):
        raise ValueError("Anisotropy inverse matrix must be positive definite.")
    return matrix_inv


def _anisotropic_distance(delta: np.ndarray, matrix_inv: Optional[np.ndarray]) -> float:
    if matrix_inv is None:
        return float(np.linalg.norm(delta))
    value = float(delta.T @ matrix_inv @ delta)
    value = max(value, 0.0)
    return float(np.sqrt(value))


def _directional_components(delta: np.ndarray, bias: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not bias:
        return 1.0, float("nan")
    vector = np.asarray(bias.get("vector"), dtype=float)
    if vector.shape != (3,):
        return 1.0, float("nan")
    vector_norm = float(np.linalg.norm(vector))
    delta_norm = float(np.linalg.norm(delta))
    if vector_norm <= 0.0 or delta_norm <= 0.0:
        return 1.0, float("nan")
    raw_cos = float(np.clip(np.dot(delta, vector) / (delta_norm * vector_norm), -1.0, 1.0))
    mode = str(bias.get("mode", "forward")).lower()
    if mode == "absolute":
        base = abs(raw_cos)
    elif mode == "signed":
        base = 0.5 * (raw_cos + 1.0)
    else:  # "forward" or fallback
        base = max(raw_cos, 0.0)
    base = float(np.clip(base, 0.0, 1.0))
    kappa = float(bias.get("kappa", 1.0))
    if kappa < 0.0:
        kappa = 0.0
    if kappa == 0.0:
        modifier = 1.0
    else:
        modifier = base ** kappa
    floor = float(bias.get("floor", 0.0))
    floor = float(np.clip(floor, 0.0, 1.0))
    factor = floor + (1.0 - floor) * modifier
    return float(max(factor, 0.0)), raw_cos


def _compute_kernel_weights(
    neighbors: Dict[str, List[Dict[str, Any]]],
    kernel_type: str,
    params: Dict[str, float],
    directional_bias: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[Tuple[str, str], Dict[str, float]]]:
    weights: Dict[str, List[Tuple[str, float]]] = {}
    direction_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    for prod, entries in neighbors.items():
        raw_weights: List[Tuple[str, float]] = []
        for entry in entries:
            inj = entry["inj"]
            distance = entry["metric_distance"]
            value = max(float(_kernel_value(distance, kernel_type, params)), 0.0)
            factor, cosine = _directional_components(entry["delta"], directional_bias)
            value *= factor
            raw_weights.append((inj, value))
            direction_metrics[(prod, inj)] = {
                "factor": float(factor),
                "cosine": float(cosine) if np.isfinite(cosine) else float("nan"),
            }
        total = sum(value for _, value in raw_weights)
        if total <= 0.0 and raw_weights:
            normalized = [(inj, 1.0 / len(raw_weights)) for inj, _ in raw_weights]
        elif total > 0.0:
            normalized = [(inj, value / total) for inj, value in raw_weights]
        else:
            normalized = []
        weights[prod] = normalized
    return weights, direction_metrics


def _expand_param_grid(
    base_params: Dict[str, float],
    param_grid: Optional[Dict[str, Iterable[float]]],
) -> List[Dict[str, float]]:
    if not param_grid:
        return [dict(base_params)]
    keys: List[str] = []
    values: List[List[float]] = []
    for key, candidates in param_grid.items():
        candidate_list = [float(value) for value in candidates if value is not None]
        if not candidate_list:
            continue
        keys.append(key)
        values.append(candidate_list)
    if not keys:
        return [dict(base_params)]
    combos: List[Dict[str, float]] = []
    for combination in itertools.product(*values):
        candidate = dict(base_params)
        for key, value in zip(keys, combination):
            candidate[key] = float(value)
        combos.append(candidate)
    if not combos:
        combos.append(dict(base_params))
    if not any(candidate == base_params for candidate in combos):
        combos.append(dict(base_params))
    return combos


def _score_kernel_candidate(
    weight_map: Dict[str, List[Tuple[str, float]]],
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
    inj_rate: pd.DataFrame,
    prod_target: pd.DataFrame,
    train_mask: np.ndarray,
) -> float:
    if not weight_map:
        return 0.0
    if isinstance(train_mask, (pd.Series, pd.Index)):
        train_index = train_mask
    else:
        train_index = inj_rate.index[train_mask]
    if len(train_index) == 0:
        return 0.0
    score = 0.0
    for prod, weights in weight_map.items():
        if not weights:
            continue
        agg = pd.Series(0.0, index=inj_rate.index, dtype=float)
        for inj, weight in weights:
            info = pair_details.get((prod, inj))
            if info is None:
                continue
            lagged = inj_rate[inj].shift(int(info["lag"])).fillna(0.0)
            agg = agg.add(weight * lagged, fill_value=0.0)
        prod_series = prod_target[prod]
        agg_train = agg.loc[train_index]
        prod_train = prod_series.loc[train_index]
        if agg_train.empty or prod_train.empty:
            continue
        agg_std = float(agg_train.std())
        prod_std = float(prod_train.std())
        if not np.isfinite(agg_std) or agg_std <= 0.0:
            continue
        if not np.isfinite(prod_std) or prod_std <= 0.0:
            continue
        corr = agg_train.corr(prod_train)
        if np.isfinite(corr):
            score += abs(float(corr))
    return score


def _maybe_calibrate_kernel(
    neighbors: Dict[str, List[Dict[str, Any]]],
    kernel_type: str,
    base_params: Dict[str, float],
    calibrate: bool,
    param_grid: Optional[Dict[str, Iterable[float]]],
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
    inj_rate: pd.DataFrame,
    prod_target: pd.DataFrame,
    train_mask: np.ndarray,
    directional_bias: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, float],
    Dict[str, List[Tuple[str, float]]],
    Dict[Tuple[str, str], Dict[str, float]],
    float,
]:
    base_params = {key: float(value) for key, value in base_params.items() if value is not None}
    base_weights, base_direction = _compute_kernel_weights(
        neighbors,
        kernel_type,
        base_params,
        directional_bias=directional_bias,
    )
    base_score = _score_kernel_candidate(base_weights, pair_details, inj_rate, prod_target, train_mask)
    if not calibrate or not param_grid:
        return base_params, base_weights, base_direction, float(base_score)
    candidates = _expand_param_grid(base_params, param_grid)
    best_params = base_params
    best_weights = base_weights
    best_direction = base_direction
    best_score = float(base_score)
    for params in candidates:
        if params == base_params:
            continue
        weights, direction_metrics = _compute_kernel_weights(
            neighbors,
            kernel_type,
            params,
            directional_bias=directional_bias,
        )
        score = _score_kernel_candidate(weights, pair_details, inj_rate, prod_target, train_mask)
        if score > best_score + 1e-9:
            best_score = float(score)
            best_params = params
            best_weights = weights
            best_direction = direction_metrics
    return best_params, best_weights, best_direction, float(best_score)


def _build_weighted_matrices(
    prod_wells: List[str],
    weight_map: Dict[str, List[Tuple[str, float]]],
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
    inj_rate: pd.DataFrame,
    inj_wwit_diff: pd.DataFrame,
    use_crm: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    full_index = inj_rate.index
    lagged_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    lagged_wwit_diff = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    crm_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells) if use_crm else None
    # Per-injector superposition: track top-N individual contributions
    top_n_contributions: Dict[str, List[pd.Series]] = {prod: [] for prod in prod_wells}
    for prod in prod_wells:
        for inj, weight in weight_map.get(prod, []):
            info = pair_details.get((prod, inj))
            if info is None:
                continue
            lag_value = int(info["lag"])
            lagged_rate_series = inj_rate[inj].shift(lag_value).fillna(0.0)
            contribution = weight * lagged_rate_series
            lagged_rate[prod] += contribution
            top_n_contributions[prod].append(contribution)
            lagged_diff_series = inj_wwit_diff[inj].shift(lag_value).fillna(0.0)
            lagged_wwit_diff[prod] += weight * lagged_diff_series
            if use_crm and crm_rate is not None:
                crm_series = crm_exp_filter(inj_rate[inj], tau=float(info["tau"]), delta=1.0).fillna(0.0)
                crm_rate[prod] += weight * crm_series
    # Build top-N individual contribution DataFrames
    n_top = min(3, max((len(v) for v in top_n_contributions.values()), default=0))
    top_n_frames: Dict[int, pd.DataFrame] = {}
    for rank in range(n_top):
        col_data = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
        for prod in prod_wells:
            contribs = top_n_contributions[prod]
            if rank < len(contribs):
                col_data[prod] = contribs[rank]
        top_n_frames[rank] = col_data
    # Multi-scale fixed-lag injection features
    fixed_lags = {1: "1m", 3: "3m", 6: "6m"}
    multiscale: Dict[str, pd.DataFrame] = {}
    for lag_val, suffix in fixed_lags.items():
        ms_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
        for prod in prod_wells:
            for inj, weight in weight_map.get(prod, []):
                if pair_details.get((prod, inj)) is None:
                    continue
                ms_rate[prod] += weight * inj_rate[inj].shift(lag_val).fillna(0.0)
        multiscale[suffix] = ms_rate
    return lagged_rate, lagged_wwit_diff, crm_rate, top_n_frames, multiscale


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
    kernel_type: str = "idw",
    kernel_p: float = 2.0,
    kernel_params: Optional[Dict[str, float]] = None,
    calibrate_kernel: bool = False,
    kernel_param_grid: Optional[Dict[str, Iterable[float]]] = None,
    kernel_candidates: Optional[Iterable[Dict[str, Any]]] = None,
    anisotropy: Optional[Dict[str, Any]] = None,
    directional_bias: Optional[Dict[str, Any]] = None,
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
    distance_lookup = _build_distance_lookup(
        coords,
        prod_wells,
        inj_wells,
        distances=distances,
        anisotropy=anisotropy,
    )
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
    neighbors = _select_topk_neighbors(prod_wells, inj_wells, distance_lookup, topK=topK)
    pair_details: Dict[Tuple[str, str], Dict[str, float]] = {}
    for prod in prod_wells:
        prod_series_full = prod_target[prod]
        prod_series_train = prod_series_full.loc[train_mask]
        for entry in neighbors.get(prod, []):
            inj = entry["inj"]
            metric_distance = entry["metric_distance"]
            bounds = estimate_tau_window(
                distance_m=metric_distance,
                freq=freq,
                physics_estimates=physics_estimates,
                tau_bound_multiplier=tau_bound_multiplier,
            )
            candidates = list(range(bounds.lower, bounds.upper + 1))
            if not candidates:
                continue
            inj_series_rate = inj_rate[inj]
            inj_series_train = inj_series_rate.loc[train_mask]
            selection = causal_xcorr_best_lag(
                inj_series_train,
                prod_series_train,
                candidate_lags=candidates,
                min_overlap=min_overlap,
            )
            tau = max(selection.lag, bounds.lower)
            pair_details[(prod, inj)] = {
                "distance_m": float(entry["euclidean_distance"]),
                "metric_distance_m": float(metric_distance),
                "lag": int(selection.lag),
                "tau": float(tau),
                "corr": float(selection.score),
                "overlap": int(selection.overlap),
                "lower_bound": int(bounds.lower),
                "upper_bound": int(bounds.upper),
                "t_diff_days": float(bounds.t_diff_days),
                "t_front_days": float(bounds.t_front_days),
            }
    base_params = dict(kernel_params or {})
    effective_kernel = kernel_type or "idw"
    if effective_kernel.lower() in {"idw", "inverse_distance"}:
        base_params.setdefault("p", float(kernel_p))
    base_grid = None
    if kernel_param_grid:
        base_grid = {key: [float(value) for value in values if value is not None] for key, values in kernel_param_grid.items() if values}
        if not base_grid:
            base_grid = None
    candidate_definitions: List[Dict[str, Any]] = []
    if kernel_candidates:
        for candidate in kernel_candidates:
            if candidate is None:
                continue
            if not isinstance(candidate, dict):
                logger.warning("Skipping kernel candidate %r: expected mapping.", candidate)
                continue
            candidate_definitions.append(dict(candidate))
    if not candidate_definitions:
        candidate_definitions.append(
            {
                "type": effective_kernel,
                "params": base_params,
                "param_grid": base_grid,
                "calibrate": calibrate_kernel,
                "directional_bias": directional_bias,
            }
        )
    kernel_results: List[Dict[str, Any]] = []
    for candidate in candidate_definitions:
        cand_type = str(candidate.get("type", effective_kernel or "idw")).strip() or "idw"
        cand_params = dict(base_params)
        for key, value in dict(candidate.get("params", {})).items():
            if value is None:
                continue
            try:
                cand_params[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Kernel '%s': parameter %s=%r is not numeric; skipping candidate.", cand_type, key, value)
                cand_params = None
                break
        if cand_params is None:
            continue
        cand_grid_input = candidate.get("param_grid", base_grid)
        cand_grid = None
        if cand_grid_input:
            cand_grid = {
                key: [float(v) for v in values if v is not None]
                for key, values in dict(cand_grid_input).items()
                if values
            }
            if not cand_grid:
                cand_grid = None
        cand_bias = candidate.get("directional_bias", directional_bias)
        cand_calibrate = bool(candidate.get("calibrate", calibrate_kernel if cand_grid else False))
        try:
            best_params, weight_map_candidate, direction_metrics_candidate, score = _maybe_calibrate_kernel(
                neighbors,
                cand_type,
                cand_params,
                cand_calibrate,
                cand_grid,
                pair_details,
                inj_rate,
                prod_target,
                train_mask,
                directional_bias=cand_bias,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Kernel '%s' failed during calibration: %s", cand_type, exc)
            continue
        kernel_results.append(
            {
                "kernel_type": cand_type,
                "params": best_params,
                "score": float(score),
                "weight_map": weight_map_candidate,
                "direction_metrics": direction_metrics_candidate,
                "directional_bias": cand_bias,
            }
        )
    if not kernel_results:
        raise ValueError("No valid kernel candidates could be evaluated for injection feature weighting.")
    def _score_key(item: Dict[str, Any]) -> float:
        score = float(item.get("score", float("-inf")))
        return score if np.isfinite(score) else float("-inf")
    kernel_results.sort(key=_score_key, reverse=True)
    best_result = kernel_results[0]
    best_kernel_type = best_result["kernel_type"]
    best_params = dict(best_result["params"])
    weight_map = best_result["weight_map"]
    direction_metrics = best_result["direction_metrics"]
    best_score = float(best_result["score"])
    summary_log = []
    for result in kernel_results:
        params_repr = ", ".join(
            f"{key}={value:.4g}" if isinstance(value, (int, float)) else f"{key}={value}"
            for key, value in sorted(result["params"].items())
        )
        summary_log.append(f"{result['kernel_type']} (score={result['score']:.4f}, params={{{params_repr}}})")
    logger.info("Evaluated injection kernels: %s", "; ".join(summary_log))
    logger.info(
        "Selected kernel '%s' for injection features (score=%.4f, params=%s)",
        best_kernel_type,
        best_score,
        best_params,
    )
    # Correlation-weighted adjustment: downweight pairs with low/negative
    # cross-correlation to suppress spurious connections (e.g. faults).
    for prod in prod_wells:
        entries = weight_map.get(prod, [])
        if not entries:
            continue
        adjusted = []
        for inj, w in entries:
            info = pair_details.get((prod, inj))
            corr_val = max(float(info["corr"]), 0.0) if info else 0.0
            adjusted.append((inj, w * corr_val))
        total = sum(aw for _, aw in adjusted)
        if total > 0:
            adjusted = [(inj, aw / total) for inj, aw in adjusted]
        weight_map[prod] = adjusted
    lagged_rate, lagged_wwit_diff, crm_rate, top_n_frames, multiscale = _build_weighted_matrices(
        prod_wells,
        weight_map,
        pair_details,
        inj_rate,
        inj_wwit_diff,
        use_crm,
    )
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
        for rank, rank_df in top_n_frames.items():
            frame[f"inj_top{rank + 1}_contribution"] = rank_df[prod].to_numpy()
        for suffix, ms_df in multiscale.items():
            frame[f"inj_wwir_lag_{suffix}"] = ms_df[prod].to_numpy()
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    features = features.fillna(0.0)
    params_serial = {key: float(value) for key, value in best_params.items()}
    summaries: List[PairLagSummary] = []
    for prod in prod_wells:
        for inj, weight in weight_map.get(prod, []):
            info = pair_details.get((prod, inj))
            if info is None:
                continue
            direction_info = direction_metrics.get((prod, inj), {})
            summaries.append(
                PairLagSummary(
                    prod_id=prod,
                    inj_id=inj,
                    distance_m=float(info.get("distance_m", float("nan"))),
                    metric_distance_m=float(info.get("metric_distance_m", info.get("distance_m", float("nan")))),
                    direction_factor=float(direction_info.get("factor", 1.0)),
                    direction_cosine=float(direction_info.get("cosine", float("nan"))),
                    kernel_type=best_kernel_type,
                    kernel_params=dict(params_serial),
                    kernel_score=float(best_score),
                    weight=float(weight),
                    lag=int(info["lag"]),
                    tau=float(info["tau"]),
                    corr=float(info["corr"]),
                    overlap=int(info["overlap"]),
                    lower_bound=int(info["lower_bound"]),
                    upper_bound=int(info["upper_bound"]),
                    t_diff_days=float(info["t_diff_days"]),
                    t_front_days=float(info["t_front_days"]),
                )
            )
    summary_df = pd.DataFrame([asdict(summary) for summary in summaries])
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["prod_id", "lag", "inj_id"])
        summary_df = summary_df.reset_index(drop=True)
    return features, summary_df
