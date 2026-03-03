from __future__ import annotations

import logging
import itertools
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet

try:
    from .utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window
except ImportError:  # pragma: no cover
    from utils_lag import causal_xcorr_best_lag, crm_exp_filter, estimate_tau_window

logger = logging.getLogger(__name__)

EPSILON_DISTANCE = 1e-6
ATTENTION_METHOD_CAUSAL_STAGE_GEO = "causal_stage_geo"
CAUSAL_STAGE_GEO_BASE_MIX = np.asarray([0.70, 0.30], dtype=float)


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
    granger_score: float
    mutual_info: float
    psd_corr: float
    conditional_mi: float
    causal_score: float
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
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[int, pd.DataFrame], Dict[str, pd.DataFrame]]:
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


def _safe_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if x.ndim != 1 or y.ndim != 1:
        return float("nan")
    if len(x) != len(y) or len(x) < 8:
        return float("nan")
    if np.nanstd(x) <= 1e-9 or np.nanstd(y) <= 1e-9:
        return 0.0
    try:
        mi_raw = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
    except Exception:
        return float("nan")
    scale = np.log(float(len(x)) + 1.0)
    if not np.isfinite(mi_raw) or scale <= 0.0:
        return float("nan")
    return float(np.clip(mi_raw / scale, 0.0, 1.0))


def _granger_proxy_score(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 10 or len(y) < 10:
        return float("nan")
    y_lag = y[:-1]
    y_cur = y[1:]
    x_cur = x[1:]
    if np.nanstd(y_lag) <= 1e-9 or np.nanstd(y_cur) <= 1e-9:
        return 0.0

    X_base = np.column_stack([np.ones_like(y_lag), y_lag])
    X_full = np.column_stack([np.ones_like(y_lag), y_lag, x_cur])
    try:
        coef_base, *_ = np.linalg.lstsq(X_base, y_cur, rcond=None)
        coef_full, *_ = np.linalg.lstsq(X_full, y_cur, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    err_base = y_cur - (X_base @ coef_base)
    err_full = y_cur - (X_full @ coef_full)
    mse_base = float(np.mean(err_base ** 2))
    mse_full = float(np.mean(err_full ** 2))
    if not np.isfinite(mse_base) or mse_base <= 1e-12 or not np.isfinite(mse_full):
        return float("nan")
    improvement = (mse_base - mse_full) / mse_base
    return float(np.clip(improvement, 0.0, 1.0))


def _psd_correlation_score(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 16 or len(y) < 16:
        return float("nan")
    nperseg = int(min(32, len(x), len(y)))
    if nperseg < 8:
        return float("nan")
    try:
        _, pxx = welch(x, nperseg=nperseg, scaling="density")
        _, pyy = welch(y, nperseg=nperseg, scaling="density")
    except Exception:
        return float("nan")
    if len(pxx) != len(pyy) or len(pxx) < 3:
        return float("nan")
    lx = np.log(np.clip(pxx, 1e-12, None))
    ly = np.log(np.clip(pyy, 1e-12, None))
    corr = np.corrcoef(lx, ly)[0, 1]
    if not np.isfinite(corr):
        return float("nan")
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))


def _residualize_on_condition(
    values: np.ndarray,
    cond: np.ndarray,
) -> np.ndarray:
    if len(values) != len(cond):
        return values
    if np.nanstd(cond) <= 1e-9:
        return values - float(np.nanmean(values))
    design = np.column_stack([np.ones_like(cond), cond])
    try:
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError:
        return values - float(np.nanmean(values))
    return values - (design @ coef)


def _conditional_mi_proxy(
    x: np.ndarray,
    y: np.ndarray,
    cond: Optional[np.ndarray],
) -> float:
    if len(x) < 10 or len(y) < 10:
        return float("nan")
    if cond is None or len(cond) != len(x):
        cond = y[:-1] if len(y) > 1 else np.zeros_like(y)
        cond = np.concatenate([[cond[0]], cond]) if len(cond) < len(y) else cond
        cond = np.asarray(cond, dtype=float)[:len(x)]
    x_res = _residualize_on_condition(x, cond)
    y_res = _residualize_on_condition(y, cond)
    return _safe_mutual_information(x_res, y_res)


def _compute_causal_pairwise_features(
    inj_lagged: pd.Series,
    prod_target: pd.Series,
    cond_target: Optional[pd.Series] = None,
    *,
    min_samples: int = 12,
) -> Dict[str, float]:
    frame = pd.DataFrame({"x": inj_lagged, "y": prod_target})
    if cond_target is not None:
        frame["c"] = cond_target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < min_samples:
        return {
            "granger_score": float("nan"),
            "mutual_info": float("nan"),
            "psd_corr": float("nan"),
            "conditional_mi": float("nan"),
            "causal_score": float("nan"),
        }
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    c = frame["c"].to_numpy(dtype=float) if "c" in frame.columns else None

    granger_score = _granger_proxy_score(x, y)
    mutual_info = _safe_mutual_information(x, y)
    psd_corr = _psd_correlation_score(x, y)
    conditional_mi = _conditional_mi_proxy(x, y, c)

    weighted = []
    parts = [
        (granger_score, 0.35),
        (mutual_info, 0.30),
        (psd_corr, 0.20),
        (conditional_mi, 0.15),
    ]
    for value, weight in parts:
        if np.isfinite(value):
            weighted.append((float(value), float(weight)))
    if weighted:
        weight_sum = sum(weight for _, weight in weighted)
        causal_score = sum(value * weight for value, weight in weighted) / max(weight_sum, 1e-9)
        causal_score = float(np.clip(causal_score, 0.0, 1.0))
    else:
        causal_score = float("nan")

    return {
        "granger_score": float(granger_score) if np.isfinite(granger_score) else float("nan"),
        "mutual_info": float(mutual_info) if np.isfinite(mutual_info) else float("nan"),
        "psd_corr": float(psd_corr) if np.isfinite(psd_corr) else float("nan"),
        "conditional_mi": float(conditional_mi) if np.isfinite(conditional_mi) else float("nan"),
        "causal_score": float(causal_score) if np.isfinite(causal_score) else float("nan"),
    }


def _normalize_simplex(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(arr), 1.0 / max(len(arr), 1))
    return arr / total


def _fit_attention_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prior: np.ndarray,
    *,
    max_iter: int = 500,
    prior_strength: float = 0.2,
    l1_ratio: float = 0.7,
) -> np.ndarray:
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X_train.shape}")
    n_samples, n_features = X_train.shape
    if n_features <= 1:
        return np.ones(n_features, dtype=float)
    if n_samples < max(12, 2 * n_features):
        return _normalize_simplex(prior)
    if y_train.ndim != 1 or len(y_train) != n_samples:
        raise ValueError("y_train must be 1D and aligned with X_train rows.")

    prior = _normalize_simplex(prior)
    finite_mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    if finite_mask.sum() < max(12, 2 * n_features):
        return prior
    X = X_train[finite_mask]
    y = y_train[finite_mask]
    if np.nanstd(y) <= 1e-9:
        return prior

    max_iter = max(int(max_iter), 200)
    guidance = float(np.clip(prior_strength, 0.0, 0.95))
    l1_ratio = float(np.clip(l1_ratio, 0.0, 1.0))
    alpha_reg = max(1e-5, 1e-3 * (1.0 - guidance))

    model = ElasticNet(
        alpha=alpha_reg,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        positive=True,
        max_iter=max_iter,
        tol=1e-4,
        selection="cyclic",
        random_state=42,
    )
    try:
        model.fit(X, y)
        learned = np.asarray(model.coef_, dtype=float)
    except Exception:
        learned = prior.copy()
    learned = _normalize_simplex(learned)
    blended = _normalize_simplex((1.0 - guidance) * learned + guidance * prior)
    return blended


def _build_regime_labels(
    weighted_rate: np.ndarray,
    weighted_diff: np.ndarray,
    train_mask: np.ndarray,
    *,
    n_bins: int = 3,
    diff_quantile: float = 0.8,
) -> np.ndarray:
    n_samples = len(weighted_rate)
    if n_samples == 0:
        return np.zeros(0, dtype=int)
    train_mask_arr = np.asarray(train_mask, dtype=bool)
    valid_train = train_mask_arr & np.isfinite(weighted_rate) & np.isfinite(weighted_diff)
    if valid_train.sum() < 5:
        return np.zeros(n_samples, dtype=int)

    n_bins = max(int(n_bins), 2)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    thresholds = np.quantile(weighted_rate[valid_train], quantiles)
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = thresholds[np.isfinite(thresholds)]
    thresholds = np.unique(thresholds)

    base = np.digitize(weighted_rate, thresholds, right=False).astype(int)
    base = np.where(np.isfinite(weighted_rate), base, 0)

    q = float(np.clip(diff_quantile, 0.55, 0.95))
    q_hi = np.quantile(weighted_diff[valid_train], q)
    q_lo = np.quantile(weighted_diff[valid_train], 1.0 - q)
    regimes = base.copy()
    if np.isfinite(q_hi) and np.isfinite(q_lo) and q_hi > q_lo:
        up_label = int(base.max()) + 1
        down_label = up_label + 1
        regimes = np.where(weighted_diff >= q_hi, up_label, regimes)
        regimes = np.where(weighted_diff <= q_lo, down_label, regimes)

    train_labels = np.unique(regimes[train_mask_arr])
    if train_labels.size == 0:
        return np.zeros(n_samples, dtype=int)
    label_map = {int(label): idx for idx, label in enumerate(sorted(int(v) for v in train_labels))}
    fallback = int(np.bincount(regimes[train_mask_arr].astype(int)).argmax())
    mapped = np.array([label_map.get(int(label), label_map.get(fallback, 0)) for label in regimes], dtype=int)
    return mapped


def _fit_regime_attention_weights(
    X_rate: np.ndarray,
    X_diff: np.ndarray,
    X_crm: Optional[np.ndarray],
    y_full: np.ndarray,
    train_mask: np.ndarray,
    prior: np.ndarray,
    *,
    max_iter: int = 500,
    prior_strength: float = 0.2,
    smooth_strength: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    if X_rate.ndim != 2 or X_diff.ndim != 2:
        raise ValueError("Regime attention expects 2D X_rate/X_diff matrices.")
    n_samples, n_features = X_rate.shape
    if n_features <= 1:
        alpha = np.ones((n_samples, n_features), dtype=float)
        return alpha, np.zeros(n_samples, dtype=int)
    if X_diff.shape != (n_samples, n_features):
        raise ValueError("X_diff shape must match X_rate shape.")
    if X_crm is not None and X_crm.shape != (n_samples, n_features):
        raise ValueError("X_crm shape must match X_rate shape when provided.")
    if y_full.ndim != 1 or len(y_full) != n_samples:
        raise ValueError("y_full must be 1D and aligned with feature rows.")

    train_mask_arr = np.asarray(train_mask, dtype=bool)
    if train_mask_arr.shape[0] != n_samples:
        raise ValueError("train_mask length must match feature rows.")
    prior = _normalize_simplex(prior)

    weighted_rate = X_rate @ prior
    weighted_diff = X_diff @ prior
    regime_labels = _build_regime_labels(weighted_rate, weighted_diff, train_mask_arr)
    unique_regimes = sorted(int(v) for v in np.unique(regime_labels[train_mask_arr]))
    if not unique_regimes:
        unique_regimes = [0]

    alpha_by_regime: Dict[int, np.ndarray] = {}
    default_regime = int(pd.Series(regime_labels[train_mask_arr]).value_counts().index[0]) if train_mask_arr.any() else 0
    for regime_id in unique_regimes:
        regime_mask = train_mask_arr & (regime_labels == regime_id)
        if regime_mask.sum() < max(12, 2 * n_features):
            alpha_by_regime[regime_id] = prior.copy()
            continue
        blocks: List[np.ndarray] = [X_rate[regime_mask], X_diff[regime_mask]]
        if X_crm is not None:
            blocks.append(X_crm[regime_mask])
        X_train = np.vstack(blocks)
        y_regime = y_full[regime_mask]
        y_train = np.concatenate([y_regime for _ in blocks])
        alpha_by_regime[regime_id] = _fit_attention_weights(
            X_train,
            y_train,
            prior,
            max_iter=max_iter,
            prior_strength=prior_strength,
        )

    alpha_matrix = np.zeros((n_samples, n_features), dtype=float)
    default_alpha = alpha_by_regime.get(default_regime, prior.copy())
    for idx in range(n_samples):
        alpha_matrix[idx] = alpha_by_regime.get(int(regime_labels[idx]), default_alpha)

    # Smooth regime transitions to avoid abrupt monthly jumps in alpha(t).
    smooth = float(np.clip(smooth_strength, 0.0, 0.95))
    if smooth > 0.0 and n_samples > 1:
        for idx in range(1, n_samples):
            alpha_matrix[idx] = (1.0 - smooth) * alpha_matrix[idx] + smooth * alpha_matrix[idx - 1]
            alpha_matrix[idx] = _normalize_simplex(alpha_matrix[idx])
    return alpha_matrix, regime_labels


def _build_static_pair_prior(
    prod_id: str,
    inj_ids: List[str],
    prior: np.ndarray,
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
) -> np.ndarray:
    if len(inj_ids) == 0:
        return prior

    distances: List[float] = []
    corrs: List[float] = []
    lags: List[float] = []
    taus: List[float] = []
    for inj in inj_ids:
        info = pair_details.get((prod_id, inj), {})
        distances.append(float(info.get("metric_distance_m", info.get("distance_m", np.nan))))
        corrs.append(float(info.get("corr", 0.0)))
        lags.append(float(info.get("lag", np.nan)))
        taus.append(float(info.get("tau", np.nan)))

    dist_arr = np.asarray(distances, dtype=float)
    corr_arr = np.asarray(corrs, dtype=float)
    lag_arr = np.asarray(lags, dtype=float)
    tau_arr = np.asarray(taus, dtype=float)

    finite_dist = np.isfinite(dist_arr)
    if finite_dist.any():
        dist_med = float(np.nanmedian(dist_arr[finite_dist]))
        dist_med = max(dist_med, EPSILON_DISTANCE)
        dist_score = np.exp(-np.where(finite_dist, dist_arr, dist_med) / dist_med)
    else:
        dist_score = np.ones(len(inj_ids), dtype=float)
    corr_score = np.clip(np.where(np.isfinite(corr_arr), corr_arr, 0.0), 0.0, None)
    lag_score = 1.0 / (1.0 + np.where(np.isfinite(lag_arr), np.maximum(lag_arr, 0.0), 0.0))
    tau_score = 1.0 / (1.0 + np.where(np.isfinite(tau_arr), np.maximum(tau_arr, 0.0), 0.0))

    # Physics-guided static prior: distance + positive correlation + lag/tau penalties + kernel prior.
    combined = (
        0.45 * _normalize_simplex(prior)
        + 0.25 * _normalize_simplex(corr_score)
        + 0.20 * _normalize_simplex(dist_score)
        + 0.05 * _normalize_simplex(lag_score)
        + 0.05 * _normalize_simplex(tau_score)
    )
    return _normalize_simplex(combined)


def _build_geo_conditioned_prior(
    prod_id: str,
    inj_ids: List[str],
    prior: np.ndarray,
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
    prod_geo: Optional[pd.DataFrame],
    inj_geo: Optional[pd.DataFrame],
) -> np.ndarray:
    base = _build_static_pair_prior(prod_id, inj_ids, prior, pair_details)
    if prod_geo is None or inj_geo is None or prod_geo.empty or inj_geo.empty or prod_id not in prod_geo.index:
        return base

    prod_row = prod_geo.loc[prod_id]
    shared_cols = [col for col in prod_geo.columns if col in inj_geo.columns]
    if not shared_cols:
        return base

    similarities: List[float] = []
    for inj in inj_ids:
        if inj not in inj_geo.index:
            similarities.append(np.nan)
            continue
        inj_row = inj_geo.loc[inj]
        col_scores: List[float] = []
        for col in shared_cols:
            p_val = float(prod_row.get(col, np.nan))
            i_val = float(inj_row.get(col, np.nan))
            if not np.isfinite(p_val) or not np.isfinite(i_val):
                continue
            denom = max(abs(p_val) + abs(i_val), 1e-6)
            col_scores.append(float(np.exp(-abs(p_val - i_val) / denom)))
        similarities.append(float(np.mean(col_scores)) if col_scores else np.nan)

    sim_arr = np.asarray(similarities, dtype=float)
    if not np.isfinite(sim_arr).any():
        return base
    sim_arr = np.where(np.isfinite(sim_arr), np.clip(sim_arr, 0.0, None), 0.0)
    sim_norm = _normalize_simplex(sim_arr)
    return _normalize_simplex(0.7 * base + 0.3 * sim_norm)


def _build_stage_adaptive_weights(
    weighted_rate: np.ndarray,
    weighted_diff: np.ndarray,
    train_mask: np.ndarray,
    base_mix: np.ndarray,
    *,
    smooth_strength: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(weighted_rate)
    if n_samples == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int)
    train_mask_arr = np.asarray(train_mask, dtype=bool)
    valid_train = train_mask_arr & np.isfinite(weighted_rate) & np.isfinite(weighted_diff)
    if valid_train.sum() < 5:
        mix = np.tile(_normalize_simplex(base_mix), (n_samples, 1))
        return mix, np.zeros(n_samples, dtype=int)

    q1, q2 = np.quantile(weighted_rate[valid_train], [0.33, 0.66])
    stage_labels = np.zeros(n_samples, dtype=int)
    stage_labels = np.where(weighted_rate >= q2, 2, stage_labels)
    stage_labels = np.where((weighted_rate >= q1) & (weighted_rate < q2), 1, stage_labels)

    # [regime, static] priors by development stage.
    stage_priors = np.asarray(
        [
            [0.55, 0.45],  # early / low-injection stage
            [0.70, 0.30],  # mid stage
            [0.80, 0.20],  # high-activity stage
        ],
        dtype=float,
    )
    mix = stage_priors[np.clip(stage_labels, 0, 2)]

    diff_scale = float(np.quantile(np.abs(weighted_diff[valid_train]), 0.9))
    diff_scale = diff_scale if np.isfinite(diff_scale) and diff_scale > 1e-8 else 1.0
    transition = np.clip(np.abs(weighted_diff) / diff_scale, 0.0, 1.0)

    # During sharp transitions increase regime share and reduce static share.
    mix[:, 0] += 0.15 * transition
    mix[:, 1] -= 0.15 * transition
    mix = np.clip(mix, 1e-4, None)

    base_mix = _normalize_simplex(base_mix)
    mix = mix * base_mix.reshape(1, -1)
    mix = np.apply_along_axis(_normalize_simplex, 1, mix)

    smooth = float(np.clip(smooth_strength, 0.0, 0.95))
    if smooth > 0.0 and len(mix) > 1:
        for idx in range(1, len(mix)):
            mix[idx] = (1.0 - smooth) * mix[idx] + smooth * mix[idx - 1]
            mix[idx] = _normalize_simplex(mix[idx])
    return mix, stage_labels


def _build_attention_matrices(
    prod_wells: List[str],
    weight_map: Dict[str, List[Tuple[str, float]]],
    pair_details: Dict[Tuple[str, str], Dict[str, float]],
    inj_rate: pd.DataFrame,
    inj_wwit_diff: pd.DataFrame,
    prod_target: pd.DataFrame,
    train_mask: np.ndarray,
    use_crm: bool,
    prod_geo: Optional[pd.DataFrame] = None,
    inj_geo: Optional[pd.DataFrame] = None,
    *,
    target_mode: str = "delta",
    steps: int = 300,
    prior_strength: float = 0.2,
    smooth_strength: float = 0.05,
    future_anchor_strength: float = 0.25,
    geo_condition_strength: float = 0.35,
    stage_adaptive: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, List[Tuple[str, float]]], pd.DataFrame]:
    full_index = inj_rate.index
    train_mask_arr = np.asarray(train_mask, dtype=bool)
    if train_mask_arr.shape[0] != len(full_index):
        raise ValueError("train_mask length must match the full timeline index.")

    attn_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    attn_diff = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    attn_crm = pd.DataFrame(0.0, index=full_index, columns=prod_wells) if use_crm else None
    attn_weight_map: Dict[str, List[Tuple[str, float]]] = {}
    alpha_timeseries_frames: List[pd.DataFrame] = []

    for prod in prod_wells:
        entries = weight_map.get(prod, [])
        if not entries:
            attn_weight_map[prod] = []
            continue

        inj_ids: List[str] = []
        prior_weights: List[float] = []
        rate_blocks: List[np.ndarray] = []
        diff_blocks: List[np.ndarray] = []
        crm_blocks: List[np.ndarray] = []
        for inj, kernel_weight in entries:
            info = pair_details.get((prod, inj))
            if info is None:
                continue
            lag_value = int(info["lag"])
            rate_blocks.append(inj_rate[inj].shift(lag_value).fillna(0.0).to_numpy(dtype=float))
            diff_blocks.append(inj_wwit_diff[inj].shift(lag_value).fillna(0.0).to_numpy(dtype=float))
            if use_crm:
                crm_blocks.append(
                    crm_exp_filter(inj_rate[inj], tau=float(info["tau"]), delta=1.0).fillna(0.0).to_numpy(dtype=float)
                )
            inj_ids.append(inj)
            prior_weights.append(float(max(kernel_weight, 0.0)))
        if not inj_ids:
            attn_weight_map[prod] = []
            continue

        X_rate = np.column_stack(rate_blocks)
        X_diff = np.column_stack(diff_blocks)
        X_crm = np.column_stack(crm_blocks) if use_crm and crm_blocks else None

        y_full = prod_target[prod].to_numpy(dtype=float)
        if str(target_mode).lower() in {"delta", "diff", "dwlpr"}:
            y_full = np.diff(y_full, prepend=y_full[0])

        prior = _normalize_simplex(np.asarray(prior_weights, dtype=float))
        geo_prior = _build_geo_conditioned_prior(prod, inj_ids, prior, pair_details, prod_geo, inj_geo)
        geo_strength = float(np.clip(geo_condition_strength, 0.0, 1.0))
        effective_prior = _normalize_simplex((1.0 - geo_strength) * prior + geo_strength * geo_prior)
        alpha_regime, regime_labels = _fit_regime_attention_weights(
            X_rate,
            X_diff,
            X_crm,
            y_full,
            train_mask_arr,
            effective_prior,
            max_iter=steps,
            prior_strength=prior_strength,
            smooth_strength=smooth_strength,
        )
        alpha_static = _build_static_pair_prior(prod, inj_ids, effective_prior, pair_details)
        alpha_static = _normalize_simplex((1.0 - geo_strength) * alpha_static + geo_strength * geo_prior)

        mix_weights = _normalize_simplex(CAUSAL_STAGE_GEO_BASE_MIX)
        if stage_adaptive:
            weighted_rate = X_rate @ effective_prior
            weighted_diff = X_diff @ effective_prior
            mix_matrix, stage_labels = _build_stage_adaptive_weights(
                weighted_rate,
                weighted_diff,
                train_mask_arr,
                mix_weights,
                smooth_strength=smooth_strength,
            )
        else:
            mix_matrix = np.tile(mix_weights, (len(X_rate), 1))
            stage_labels = np.zeros(len(X_rate), dtype=int)

        alpha_matrix = (
            mix_matrix[:, 0:1] * alpha_regime
            + mix_matrix[:, 1:2] * alpha_static.reshape(1, -1)
        )
        for idx in range(alpha_matrix.shape[0]):
            alpha_matrix[idx] = _normalize_simplex(alpha_matrix[idx])

        # Keep future alpha(t) close to the latest train state to stabilize rollout.
        anchor = float(np.clip(future_anchor_strength, 0.0, 0.95))
        future_idx = np.flatnonzero(~train_mask_arr)
        train_idx = np.flatnonzero(train_mask_arr)
        if anchor > 0.0 and future_idx.size > 0 and train_idx.size > 0:
            train_tail = train_idx[max(0, len(train_idx) - 3):]
            anchor_alpha = _normalize_simplex(np.nanmean(alpha_matrix[train_tail], axis=0))
            for idx in future_idx:
                alpha_matrix[idx] = (1.0 - anchor) * alpha_matrix[idx] + anchor * anchor_alpha
                alpha_matrix[idx] = _normalize_simplex(alpha_matrix[idx])

        attn_rate[prod] = np.sum(X_rate * alpha_matrix, axis=1)
        attn_diff[prod] = np.sum(X_diff * alpha_matrix, axis=1)
        if use_crm and attn_crm is not None and X_crm is not None:
            attn_crm[prod] = np.sum(X_crm * alpha_matrix, axis=1)
        if train_mask_arr.any():
            alpha_summary = alpha_matrix[train_mask_arr].mean(axis=0)
        else:
            alpha_summary = alpha_matrix.mean(axis=0)
        attn_weight_map[prod] = [(inj, float(weight)) for inj, weight in zip(inj_ids, alpha_summary)]
        ds_rep = np.repeat(full_index.to_numpy(), len(inj_ids))
        inj_rep = np.tile(np.asarray(inj_ids, dtype=object), len(full_index))
        alpha_rep = alpha_matrix.reshape(-1)
        train_rep = np.repeat(train_mask_arr.astype(bool), len(inj_ids))
        regime_rep = np.repeat(np.asarray(regime_labels, dtype=int), len(inj_ids))
        stage_rep = np.repeat(np.asarray(stage_labels, dtype=int), len(inj_ids))
        prod_rep = np.repeat(np.asarray([prod], dtype=object), len(alpha_rep))
        alpha_timeseries_frames.append(
            pd.DataFrame(
                {
                    "ds": ds_rep,
                    "prod_id": prod_rep,
                    "inj_id": inj_rep,
                    "alpha": alpha_rep,
                    "is_train": train_rep,
                    "regime_id": regime_rep,
                    "stage_id": stage_rep,
                    "attention_mode": ATTENTION_METHOD_CAUSAL_STAGE_GEO,
                }
            )
        )
    alpha_timeseries = (
        pd.concat(alpha_timeseries_frames, ignore_index=True)
        if alpha_timeseries_frames
        else pd.DataFrame(
            columns=["ds", "prod_id", "inj_id", "alpha", "is_train", "regime_id", "stage_id", "attention_mode"]
        )
    )
    return attn_rate, attn_diff, attn_crm, attn_weight_map, alpha_timeseries


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            result[col] = 0.0
    return result


def _prepare_geo_stats(
    prod_df: pd.DataFrame,
    inj_df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    candidate_cols = [
        "porosity",
        "permeability",
        "thickness",
        "effective_thickness",
        "kh",
        "coord_z",
        "z",
    ]
    prod_geo_cols = [col for col in candidate_cols if col in prod_df.columns]
    inj_geo_cols = [col for col in candidate_cols if col in inj_df.columns]
    shared_cols = [col for col in prod_geo_cols if col in inj_geo_cols]
    if not shared_cols:
        return None, None
    prod_geo = (
        prod_df[["well", *shared_cols]]
        .groupby("well", as_index=True)
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    inj_geo = (
        inj_df[["well", *shared_cols]]
        .groupby("well", as_index=True)
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    if prod_geo.empty or inj_geo.empty:
        return None, None
    return prod_geo, inj_geo


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
    use_attention: bool = True,
    attention_target_mode: str = "delta",
    attention_steps: int = 300,
    attention_prior_strength: float = 0.2,
    attention_smooth_strength: float = 0.05,
    attention_future_anchor_strength: float = 0.25,
    attention_geo_condition_strength: float = 0.35,
    attention_stage_adaptive: bool = True,
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
                "inj_wwir_lag_attn": 0.0,
                "inj_wwir_crm_attn": 0.0,
                "inj_wwit_diff_lag_attn": 0.0,
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
    prod_conditioning = None
    if "wthp" in prod_df.columns:
        prod_conditioning = (
            prod_df.pivot(index="ds", columns="well", values="wthp")
            .reindex(full_index)
            .ffill()
            .bfill()
        )
    prod_geo, inj_geo = _prepare_geo_stats(prod_df, inj_df)
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
            inj_lagged_train = inj_rate[inj].shift(int(selection.lag)).loc[train_mask]
            cond_series = (
                prod_conditioning[prod].loc[train_mask]
                if prod_conditioning is not None and prod in prod_conditioning.columns
                else None
            )
            causal_metrics = _compute_causal_pairwise_features(
                inj_lagged_train,
                prod_series_train,
                cond_series,
                min_samples=max(12, min_overlap),
            )
            pair_details[(prod, inj)] = {
                "distance_m": float(entry["euclidean_distance"]),
                "metric_distance_m": float(metric_distance),
                "lag": int(selection.lag),
                "tau": float(tau),
                "corr": float(selection.score),
                "granger_score": float(causal_metrics["granger_score"]),
                "mutual_info": float(causal_metrics["mutual_info"]),
                "psd_corr": float(causal_metrics["psd_corr"]),
                "conditional_mi": float(causal_metrics["conditional_mi"]),
                "causal_score": float(causal_metrics["causal_score"]),
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
    # Causal-weighted adjustment: downweight pairs with weak directional
    # evidence to suppress spurious connections under confounding.
    for prod in prod_wells:
        entries = weight_map.get(prod, [])
        if not entries:
            continue
        adjusted = []
        for inj, w in entries:
            info = pair_details.get((prod, inj))
            corr_val = max(float(info.get("corr", 0.0)), 0.0) if info else 0.0
            causal_val = float(info.get("causal_score", np.nan)) if info else np.nan
            if not np.isfinite(causal_val):
                causal_val = corr_val
            blend = 0.6 * corr_val + 0.4 * max(causal_val, 0.0)
            adjusted.append((inj, w * blend))
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
    attn_rate = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    attn_wwit_diff = pd.DataFrame(0.0, index=full_index, columns=prod_wells)
    attn_crm = pd.DataFrame(0.0, index=full_index, columns=prod_wells) if use_crm else None
    attn_weight_map: Dict[str, List[Tuple[str, float]]] = {prod: [] for prod in prod_wells}
    alpha_timeseries = pd.DataFrame(
        columns=["ds", "prod_id", "inj_id", "alpha", "is_train", "regime_id", "stage_id", "attention_mode"]
    )
    if use_attention:
        attn_rate, attn_wwit_diff, attn_crm, attn_weight_map, alpha_timeseries = _build_attention_matrices(
            prod_wells,
            weight_map,
            pair_details,
            inj_rate,
            inj_wwit_diff,
            prod_target,
            train_mask,
            use_crm,
            prod_geo=prod_geo,
            inj_geo=inj_geo,
            target_mode=attention_target_mode,
            steps=attention_steps,
            prior_strength=attention_prior_strength,
            smooth_strength=attention_smooth_strength,
            future_anchor_strength=attention_future_anchor_strength,
            geo_condition_strength=attention_geo_condition_strength,
            stage_adaptive=attention_stage_adaptive,
        )
    feature_frames: List[pd.DataFrame] = []
    for prod in prod_wells:
        frame = pd.DataFrame(
            {
                "ds": full_index,
                "well": prod,
                "inj_wwir_lag_weighted": lagged_rate[prod].to_numpy(),
                "inj_wwit_diff_lag_weighted": lagged_wwit_diff[prod].to_numpy(),
                "inj_wwir_lag_attn": attn_rate[prod].to_numpy(),
                "inj_wwit_diff_lag_attn": attn_wwit_diff[prod].to_numpy(),
            }
        )
        if use_crm and crm_rate is not None:
            frame["inj_wwir_crm_weighted"] = crm_rate[prod].to_numpy()
        else:
            frame["inj_wwir_crm_weighted"] = 0.0
        if use_crm and attn_crm is not None:
            frame["inj_wwir_crm_attn"] = attn_crm[prod].to_numpy()
        else:
            frame["inj_wwir_crm_attn"] = 0.0
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
                    granger_score=float(info.get("granger_score", float("nan"))),
                    mutual_info=float(info.get("mutual_info", float("nan"))),
                    psd_corr=float(info.get("psd_corr", float("nan"))),
                    conditional_mi=float(info.get("conditional_mi", float("nan"))),
                    causal_score=float(info.get("causal_score", float("nan"))),
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
        if use_attention and attn_weight_map:
            pair_keys = list(zip(summary_df["prod_id"].astype(str), summary_df["inj_id"].astype(str)))
            attn_lookup = {
                (prod_id, inj_id): float(weight)
                for prod_id, entries in attn_weight_map.items()
                for inj_id, weight in entries
            }
            summary_df["attn_alpha"] = [attn_lookup.get((prod_id, inj_id), np.nan) for prod_id, inj_id in pair_keys]
            summary_df["attn_minus_kernel"] = summary_df["attn_alpha"] - summary_df["weight"]
            if (
                use_attention
                and not alpha_timeseries.empty
                and {"prod_id", "inj_id", "alpha", "is_train"} <= set(alpha_timeseries.columns)
            ):
                dyn = alpha_timeseries.copy()
                dyn["prod_id"] = dyn["prod_id"].astype(str)
                dyn["inj_id"] = dyn["inj_id"].astype(str)
                dyn_train = dyn[dyn["is_train"] == True]  # noqa: E712
                mean_train = {
                    (prod_id, inj_id): float(alpha)
                    for (prod_id, inj_id), alpha in dyn_train.groupby(["prod_id", "inj_id"])["alpha"].mean().items()
                }
                if not dyn_train.empty:
                    last_train = dyn_train.sort_values("ds").groupby(["prod_id", "inj_id"])["alpha"].last()
                    last_train_lookup = {(prod_id, inj_id): float(alpha) for (prod_id, inj_id), alpha in last_train.items()}
                else:
                    last_train_lookup = {}
                last_full = dyn.sort_values("ds").groupby(["prod_id", "inj_id"])["alpha"].last()
                last_full_lookup = {(prod_id, inj_id): float(alpha) for (prod_id, inj_id), alpha in last_full.items()}
                summary_df["attn_alpha_train_mean"] = [mean_train.get((prod_id, inj_id), np.nan) for prod_id, inj_id in pair_keys]
                summary_df["attn_alpha_train_last"] = [last_train_lookup.get((prod_id, inj_id), np.nan) for prod_id, inj_id in pair_keys]
                summary_df["attn_alpha_full_last"] = [last_full_lookup.get((prod_id, inj_id), np.nan) for prod_id, inj_id in pair_keys]
                summary_df["attn_alpha"] = summary_df["attn_alpha_train_mean"].where(
                    summary_df["attn_alpha_train_mean"].notna(),
                    summary_df["attn_alpha"],
                )
                summary_df["attn_minus_kernel"] = summary_df["attn_alpha"] - summary_df["weight"]
    summary_df.attrs["attention_mode"] = ATTENTION_METHOD_CAUSAL_STAGE_GEO
    if use_attention and not alpha_timeseries.empty:
        summary_df.attrs["attention_alpha_timeseries"] = alpha_timeseries
    return features, summary_df
