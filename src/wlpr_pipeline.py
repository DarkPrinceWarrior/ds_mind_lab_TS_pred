from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import PipelineConfig
    from .features_injection import build_injection_lag_features
    from .features_graph import build_graph_features
    from .data_validation import WellDataValidator
    from .metrics_extended import calculate_all_metrics
    from .metrics_reservoir import compute_all_reservoir_metrics
    from .logging_config import log_execution_time
    from .caching import CacheManager
    from .data_preprocessing_advanced import PhysicsAwarePreprocessor
    from .features_advanced import (
        create_fourier_features,
        create_spatial_features,
        create_time_series_embeddings,
    )
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from features_injection import build_injection_lag_features
    from features_graph import build_graph_features
    from data_validation import WellDataValidator
    from metrics_extended import calculate_all_metrics
    from metrics_reservoir import compute_all_reservoir_metrics
    from logging_config import log_execution_time
    from caching import CacheManager
    from data_preprocessing_advanced import PhysicsAwarePreprocessor
    from features_advanced import (
        create_fourier_features,
        create_spatial_features,
        create_time_series_embeddings,
    )

EPSILON = 1e-6
logger = logging.getLogger(__name__)

_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        _cache = CacheManager(cache_dir=Path(".cache"), enabled=True)
    return _cache


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@log_execution_time(logger)
def load_raw_data(path: Path, validate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns={"DATA": "date", "TYPE": "type"})
    df.columns = [col.lower() for col in df.columns]
    df = df.dropna(how="all")
    df = df[df["date"].notna() & df["well"].notna()]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df[df["date"].notna()]
    df["well"] = df["well"].astype(float).astype(int).astype(str)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.upper()
    unnamed = [col for col in df.columns if col.startswith("unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")
    numeric_cols = [col for col in df.columns if col not in {"date", "type", "well"}]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(["well", "date"])
    df = df.drop_duplicates(["well", "date"])
    logger.info("Loaded %d rows for %d wells", len(df), df["well"].nunique())

    try:
        logger.info("Applying physics-aware preprocessing")
        preprocessor = PhysicsAwarePreprocessor(well_type="PROD")
        df = preprocessor.detect_structural_breaks(df, rate_col="wlpr", threshold=0.7)
        rate_cols = [col for col in ["wlpr", "womr", "wwir"] if col in df.columns]
        cumulative_cols = [col for col in ["wlpt", "womt", "wwit"] if col in df.columns]
        if rate_cols or cumulative_cols:
            df = preprocessor.physics_aware_imputation(df, rate_cols=rate_cols, cumulative_cols=cumulative_cols)
        feature_cols = [col for col in ["wlpr", "wbhp", "wwir"] if col in df.columns]
        if len(feature_cols) >= 2:
            df = preprocessor.detect_outliers_multivariate(df, feature_cols=feature_cols, contamination=0.05)
        rate_cols_smooth = [col for col in ["wlpr", "womr"] if col in df.columns]
        if rate_cols_smooth:
            df = preprocessor.smooth_rates_savgol(df, rate_cols=rate_cols_smooth, window_length=7, polyorder=2)
        logger.info("Physics-aware preprocessing completed")
    except Exception as exc:
        logger.warning("Physics-aware preprocessing failed: %s", exc)

    if validate:
        try:
            validator = WellDataValidator()
            df = validator.validate_schema(df)
            logger.info("Data validation passed")
        except Exception as exc:
            logger.warning("Data validation failed: %s", exc)

    return df


def load_coordinates(path: Path) -> pd.DataFrame:
    records: List[Tuple[str, float, float, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("--"):
                continue
            parts = line.replace("'", " ").split()
            if len(parts) < 4:
                continue
            well = parts[0].strip()
            x, y, z = map(float, parts[1:4])
            records.append((well, x, y, z))
    coords = pd.DataFrame(records, columns=["well", "x", "y", "z"])
    coords["well"] = coords["well"].astype(str).str.strip()
    logger.info("Loaded coordinates for %d wells", len(coords))
    return coords


def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)

    def _normalize(label: object) -> str:
        if pd.isna(label):
            raise ValueError("Distance matrix contains unnamed wells.")
        if isinstance(label, (int, np.integer)):
            return str(int(label))
        if isinstance(label, float) and float(label).is_integer():
            return str(int(label))
        return str(label).strip()

    df.index = df.index.map(_normalize)
    df.columns = df.columns.map(_normalize)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.where(pd.notnull(df), np.nan)
    df = df.sort_index().sort_index(axis=1)
    logger.info("Loaded distance matrix with %d wells (rows) x %d wells (columns)", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _enforce_monotonic_cumulative(df: pd.DataFrame, group_col: str, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df.groupby(group_col)[col].cummax()
    return df


def _clip_non_negative(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
    return df


def _compute_watercut(df: pd.DataFrame) -> pd.DataFrame:
    required = {"wlpr", "womr"}
    if not required.issubset(df.columns):
        return df
    total = df["wlpr"].abs().clip(lower=1e-6)
    water = (df["wlpr"] - df["womr"]).clip(lower=0.0)
    fw = (water / total).clip(0.0, 1.0).fillna(0.0)
    df["fw"] = fw
    return df


def impute_numeric(df: pd.DataFrame, group_col: str, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df.groupby(group_col)[col].transform(lambda series: series.ffill())
    df[columns] = df[columns].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Well selection and reindexing
# ---------------------------------------------------------------------------

def get_target_wells(df: pd.DataFrame, config: PipelineConfig) -> List[str]:
    counts = df.groupby("well").size()
    last_types = df.sort_values("date").groupby("well").tail(1).set_index("well")["type"]
    required = config.horizon + config.val_horizon
    selected: List[str] = []
    for well, well_type in last_types.items():
        if well_type != "PROD":
            continue
        if counts.get(well, 0) < max(required, config.min_history):
            continue
        selected.append(well)
    logger.info("Selected %d producer wells for modeling", len(selected))
    return sorted(selected)


def reindex_series(df: pd.DataFrame, wells: List[str], freq: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for well in wells:
        well_df = df[df["well"] == well].set_index("date").sort_index()
        if well_df.empty:
            continue
        idx = pd.date_range(well_df.index.min(), well_df.index.max(), freq=freq)
        well_df = well_df.reindex(idx)
        well_df["well"] = well
        frames.append(well_df.reset_index().rename(columns={"index": "ds"}))
    if not frames:
        return pd.DataFrame(columns=["ds", "well"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Injection feature helpers
# ---------------------------------------------------------------------------

def _apply_injection_lag_features(
    prod_base: pd.DataFrame,
    inj_df: pd.DataFrame,
    coords: pd.DataFrame,
    config: PipelineConfig,
    train_cutoff: pd.Timestamp,
    *,
    distances: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    injection_features, pair_summary = build_injection_lag_features(
        prod_base, inj_df, coords,
        freq=config.freq,
        train_cutoff=train_cutoff,
        distances=distances,
        physics_estimates=config.physics_estimates,
        topK=config.inj_top_k,
        kernel_type=config.inj_kernel_type,
        kernel_p=config.inj_kernel_p,
        kernel_params=config.inj_kernel_params,
        calibrate_kernel=config.inj_kernel_calibrate,
        kernel_param_grid=config.inj_kernel_param_grid,
        kernel_candidates=config.inj_kernel_candidates,
        anisotropy=config.inj_distance_anisotropy,
        directional_bias=config.inj_directional_bias,
        use_crm=config.use_crm_filter,
        tau_bound_multiplier=config.tau_bound_multiplier,
        min_overlap=config.lag_min_overlap,
    )
    kernel_metadata = None
    if not pair_summary.empty and {"kernel_type", "kernel_params", "kernel_score"} <= set(pair_summary.columns):
        kernel_metadata = {
            "kernel_type": pair_summary["kernel_type"].iloc[0],
            "kernel_params": pair_summary["kernel_params"].iloc[0],
            "kernel_score": float(pair_summary["kernel_score"].iloc[0]),
        }
        logger.info(
            "Best injection kernel: %s (score=%.4f, params=%s)",
            kernel_metadata["kernel_type"], kernel_metadata["kernel_score"], kernel_metadata["kernel_params"],
        )
        pair_summary.attrs["kernel_metadata"] = kernel_metadata
    merged = prod_base.merge(injection_features, on=["ds", "well"], how="left")
    for column in ["inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted"]:
        if column not in merged.columns:
            merged[column] = 0.0
    merged[["inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted"]] = (
        merged[["inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted"]].fillna(0.0)
    )
    return merged, pair_summary


def _finalize_prod_dataframe(prod_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = prod_df.copy()
    df["type_prod"] = (df["type"] == "PROD").astype(int)
    df["type_inj"] = (df["type"] == "INJ").astype(int)
    df["month"] = df["ds"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["time_idx"] = df.sort_values("ds").groupby("well").cumcount()
    df["unique_id"] = df["well"]
    df["y"] = df["wlpr"].astype(float)
    df = df.drop(columns=["type"], errors="ignore")
    return df


def _fill_missing_features(df: pd.DataFrame, feature_cols: set, context: str = "") -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("%sFeature '%s' not found, filling with zeros", context, col)
            df[col] = 0.0
        elif df[col].isna().any():
            nan_count = df[col].isna().sum()
            logger.debug("%sFeature '%s' has %d NaN values, filling with zeros", context, col, nan_count)
            df[col] = df[col].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def generate_walk_forward_splits(
    train_df: pd.DataFrame,
    horizon: int,
    step: int,
    folds: int,
) -> List[Dict[str, pd.DataFrame]]:
    if folds <= 0 or horizon <= 0 or step <= 0:
        return []
    if train_df.empty:
        return []
    per_well_max = train_df.groupby("unique_id")["time_idx"].max()
    if per_well_max.empty:
        return []
    max_common_idx = int(per_well_max.min())
    total_points = max_common_idx + 1
    usable_prefix = total_points - horizon - step * (folds - 1)
    if usable_prefix <= 0:
        raise ValueError(
            f"Insufficient history for rolling-origin validation: "
            f"total_points={total_points}, horizon={horizon}, step={step}, folds={folds}"
        )
    logger.info(
        "Walk-forward CV layout: total_points=%d, base_train_len=%d, horizon=%d, step=%d, folds=%d",
        total_points, usable_prefix, horizon, step, folds,
    )
    splits: List[Dict[str, pd.DataFrame]] = []
    for fold_idx in range(folds):
        train_cutoff = usable_prefix + step * fold_idx - 1
        val_start = usable_prefix + step * fold_idx
        val_end = val_start + horizon - 1
        fold_train = train_df[train_df["time_idx"] <= train_cutoff].copy().sort_values(["unique_id", "ds"])
        fold_val = train_df[(train_df["time_idx"] >= val_start) & (train_df["time_idx"] <= val_end)].copy().sort_values(["unique_id", "ds"])
        if fold_train.empty or fold_val.empty:
            logger.warning("Skipping fold %d due to empty train (%d) or val (%d)", fold_idx + 1, len(fold_train), len(fold_val))
            continue
        splits.append({
            "fold": fold_idx + 1,
            "train_cutoff": train_cutoff,
            "val_start": val_start,
            "val_end": val_end,
            "train_df": fold_train,
            "val_df": fold_val,
        })
    return splits


def run_walk_forward_validation(
    frames: Dict[str, pd.DataFrame],
    coords: pd.DataFrame,
    config: PipelineConfig,
    distances: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, object]]:
    train_df = frames["train_df"]
    static_df = frames["static_df"]
    prod_base_df = frames.get("prod_base_df")
    inj_df = frames.get("inj_df")
    if prod_base_df is None or inj_df is None:
        logger.warning("Missing base data for injection features; using cached features for CV folds.")

    if not config.cv_enabled:
        return None
    try:
        splits = generate_walk_forward_splits(train_df, horizon=config.horizon, step=config.cv_step, folds=config.cv_folds)
    except ValueError as exc:
        logger.warning("Skipping walk-forward validation: %s", exc)
        return None
    if not splits:
        logger.warning("Walk-forward validation requested but no splits were generated.")
        return None

    fold_results: List[Dict[str, object]] = []
    metric_sums: Dict[str, float] = {}
    metric_weights: Dict[str, float] = {}
    feature_cols = set(config.hist_exog + config.futr_exog)
    train_columns = list(train_df.columns)

    for split in splits:
        fold_train_raw = split["train_df"]
        fold_val_raw = split["val_df"]
        cutoff_date = fold_train_raw["ds"].max() if not fold_train_raw.empty else None
        fold_pair_summary: Optional[pd.DataFrame] = None
        if cutoff_date is None:
            logger.warning("Skipping fold %s due to empty training window.", split["fold"])
            continue

        if prod_base_df is not None and inj_df is not None:
            fold_prod, fold_pair_summary = _apply_injection_lag_features(
                prod_base_df, inj_df, coords, config, cutoff_date, distances=distances,
            )
            fold_prod = _finalize_prod_dataframe(fold_prod, config)
            fold_prod = create_spatial_features(fold_prod, coords)
            fold_prod = create_fourier_features(fold_prod, date_col="ds", n_frequencies=3)
            key_features = ["wlpr", "womr"] if all(col in fold_prod.columns for col in ["wlpr", "womr"]) else ["wlpr"]
            fold_prod = create_time_series_embeddings(
                fold_prod, feature_cols=key_features, window=12, n_components=3, train_cutoff=cutoff_date, date_col="ds",
            )
            fold_target_wells = sorted(fold_prod["well"].unique()) if "well" in fold_prod.columns else sorted(fold_prod["unique_id"].unique())
            fold_prod, _ = build_graph_features(
                fold_prod, coords, fold_target_wells, fold_pair_summary if fold_pair_summary is not None else pd.DataFrame(),
                distances=distances,
                n2v_dimensions=config.graph_n2v_dimensions,
                spectral_components=config.graph_spectral_components,
                neighbor_agg_cols=config.graph_neighbor_agg_cols,
                neighbor_k=config.graph_neighbor_k,
                seed=config.random_seed,
            )
            fold_prod = _fill_missing_features(fold_prod, feature_cols, context=f"Fold {split['fold']}: ")
            train_keys = fold_train_raw[["unique_id", "ds"]].drop_duplicates()
            val_keys = fold_val_raw[["unique_id", "ds"]].drop_duplicates()
            fold_prod = fold_prod.sort_values(["unique_id", "ds"])
            fold_train = fold_prod.merge(train_keys.assign(__flag=1), on=["unique_id", "ds"], how="inner").drop(columns="__flag")
            fold_val = fold_prod.merge(val_keys.assign(__flag=1), on=["unique_id", "ds"], how="inner").drop(columns="__flag")
            fold_train = fold_train[train_columns]
            fold_val = fold_val[train_columns]
            static_cols = ["unique_id"] + [col for col in config.static_exog if col in fold_prod.columns]
            fold_static_df = fold_prod.groupby("unique_id")[static_cols].first().reset_index(drop=True)
            for col in config.static_exog:
                if col not in fold_static_df.columns:
                    fold_static_df[col] = 0.0
        else:
            fold_train = fold_train_raw
            fold_val = fold_val_raw
            fold_static_df = static_df

        cov_cols = list(set(config.hist_exog + config.futr_exog))
        available_cols = [c for c in cov_cols if c in fold_train.columns]
        future_df = pd.concat(
            [
                fold_train[["unique_id", "ds"] + available_cols],
                fold_val[["unique_id", "ds"] + [c for c in available_cols if c in fold_val.columns]],
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["unique_id", "ds"], keep="first").sort_values(["unique_id", "ds"])
        preds = _chronos2_predict(fold_train, future_df, config)

        metrics, merged = evaluate_predictions(preds, fold_val, fold_train)
        overall_raw = metrics.get("overall", {})
        overall: Dict[str, Optional[float]] = {}
        for key, value in overall_raw.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                overall[key] = float(value)
            else:
                overall[key] = None
        rows = len(merged)
        fold_results.append({
            "fold": int(split["fold"]),
            "train_span": [1, int(split["train_cutoff"]) + 1],
            "val_span": [int(split["val_start"]) + 1, int(split["val_end"]) + 1],
            "train_rows": int(len(fold_train)),
            "rows": int(rows),
            "indices": {
                "train_end_idx": int(split["train_cutoff"]),
                "val_start_idx": int(split["val_start"]),
                "val_end_idx": int(split["val_end"]),
            },
            "metrics": overall,
            "lag_pairs": int(len(fold_pair_summary)) if isinstance(fold_pair_summary, pd.DataFrame) else None,
        })
        logger.info(
            "Fold %d validation metrics: %s",
            split["fold"],
            {k: round(v, 4) if isinstance(v, (int, float, np.floating)) else v for k, v in overall.items()},
        )
        for key, value in overall.items():
            if value is None:
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + value * rows
            metric_weights[key] = metric_weights.get(key, 0.0) + rows

    aggregate = {
        key: float(metric_sums[key] / metric_weights[key])
        for key in metric_sums
        if metric_weights.get(key, 0.0) > 0.0
    }
    if aggregate:
        logger.info("Walk-forward validation aggregate metrics: %s", {k: round(v, 4) for k, v in aggregate.items()})
    return {"folds": fold_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def prepare_model_frames(
    raw_df: pd.DataFrame,
    coords: pd.DataFrame,
    config: PipelineConfig,
    distances: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    target_wells = get_target_wells(raw_df, config)
    if not target_wells:
        raise ValueError("No producer wells satisfy the selection criteria.")
    prod_df = raw_df[raw_df["well"].isin(target_wells)].copy()
    prod_df = reindex_series(prod_df, target_wells, config.freq)
    if prod_df.empty:
        raise ValueError("Reindexed producer dataframe is empty.")
    prod_df["type"] = prod_df.groupby("well")["type"].transform(lambda s: s.ffill().bfill())
    numeric_cols = [col for col in prod_df.columns if col not in {"ds", "well", "type"}]
    prod_df = impute_numeric(prod_df, "well", numeric_cols)
    prod_df = _enforce_monotonic_cumulative(prod_df, "well", ["wlpt"])
    prod_df = _clip_non_negative(prod_df, ["wlpt_diff", "wlpr", "womr", "womt"])
    prod_df = _compute_watercut(prod_df)
    prod_df["wlpr"] = prod_df["wlpr"].fillna(0.0)

    max_dates = prod_df.groupby("well")["ds"].max()
    if max_dates.empty:
        raise ValueError("Could not compute terminal dates for producers.")
    target_end = max_dates.min()
    offset = pd.tseries.frequencies.to_offset(config.freq)
    if offset is None:
        raise ValueError(f"Unsupported frequency alias: {config.freq}")
    if config.horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    test_start = target_end - offset * max(config.horizon - 1, 0)
    train_cutoff = test_start - offset
    min_date = prod_df["ds"].min()
    if pd.isna(min_date):
        raise ValueError("Producer dataframe has no valid dates after preprocessing.")
    if train_cutoff < min_date:
        train_cutoff = min_date

    prod_base = prod_df.copy()
    inj_raw = raw_df[raw_df["type"] == "INJ"].copy()
    if inj_raw.empty:
        inj_df = pd.DataFrame(columns=["ds", "well", "wwir", "wwit", "wwit_diff"])
    else:
        inj_raw["well"] = inj_raw["well"].astype(str)
        inj_wells = sorted(inj_raw["well"].unique())
        inj_df = reindex_series(inj_raw, inj_wells, config.freq)
        inj_df = _enforce_monotonic_cumulative(inj_df, "well", ["wwit"])
        inj_df = _clip_non_negative(inj_df, ["wwir", "wwit", "wwit_diff"])

    prod_df, pair_summary = _apply_injection_lag_features(
        prod_base, inj_df, coords, config, train_cutoff, distances=distances,
    )
    kernel_metadata = pair_summary.attrs.get("kernel_metadata")
    logger.info("Prepared lagged injection features: %d pairs, cutoff=%s, test_start=%s", len(pair_summary), train_cutoff.date(), test_start.date())
    prod_df = _finalize_prod_dataframe(prod_df, config)

    logger.info("Creating advanced features (spatial, Fourier, PCA)")
    prod_df = create_spatial_features(prod_df, coords)
    prod_df = create_fourier_features(prod_df, date_col="ds", n_frequencies=3)
    key_features = ["wlpr", "womr"] if all(col in prod_df.columns for col in ["wlpr", "womr"]) else ["wlpr"]
    prod_df = create_time_series_embeddings(
        prod_df, feature_cols=key_features, window=12, n_components=3, train_cutoff=train_cutoff, date_col="ds",
    )

    well_types = raw_df.sort_values("date").groupby("well").tail(1).set_index("well")["type"].to_dict()
    prod_df, _graph_static = build_graph_features(
        prod_df, coords, target_wells, pair_summary,
        distances=distances,
        well_types=well_types,
        n2v_dimensions=config.graph_n2v_dimensions,
        spectral_components=config.graph_spectral_components,
        neighbor_agg_cols=config.graph_neighbor_agg_cols,
        neighbor_k=config.graph_neighbor_k,
        seed=config.random_seed,
    )
    logger.info("Advanced + graph features created successfully")

    feature_cols = set(config.hist_exog + config.futr_exog)
    prod_df = _fill_missing_features(prod_df, feature_cols)

    train_df = prod_df[prod_df["ds"] < test_start].copy().sort_values(["unique_id", "ds"])
    test_df = prod_df[prod_df["ds"] >= test_start].copy().sort_values(["unique_id", "ds"])
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test dataframe is empty after splitting.")
    futr_df = test_df[["unique_id", "ds"] + config.futr_exog].copy()

    static_cols = ["unique_id"] + [col for col in config.static_exog if col in prod_df.columns]
    static_df = prod_df.groupby("unique_id")[static_cols].first().reset_index(drop=True)
    for col in config.static_exog:
        if col not in static_df.columns:
            logger.warning("Static feature '%s' not found, filling with zeros", col)
            static_df[col] = 0.0

    logger.info("Prepared frames: train=%d rows, test=%d rows, future=%d rows", len(train_df), len(test_df), len(futr_df))
    return {
        "train_df": train_df,
        "test_df": test_df,
        "futr_df": futr_df,
        "static_df": static_df,
        "target_wells": target_wells,
        "test_start": test_start,
        "train_cutoff": train_cutoff,
        "injection_summary": pair_summary,
        "kernel_metadata": kernel_metadata,
        "prod_base_df": prod_base,
        "inj_df": inj_df,
    }


# ---------------------------------------------------------------------------
# Chronos-2 model
# ---------------------------------------------------------------------------

def _import_chronos2():
    try:
        from darts import TimeSeries
        from darts.models import Chronos2Model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Chronos2Model requires 'darts' and its dependencies. "
            "Install with: pip install darts transformers accelerate"
        ) from exc
    return TimeSeries, Chronos2Model


def _build_chronos2_model(config: PipelineConfig, train_length: Optional[int] = None):
    _, Chronos2Model = _import_chronos2()
    output_len = config.chronos_output_chunk_length or config.horizon
    if config.chronos_input_chunk_length:
        input_len = config.chronos_input_chunk_length
    else:
        input_len = config.input_size
    if train_length:
        max_allowed = train_length - output_len - 1
        input_len = min(input_len, max(max_allowed, 1))

    kwargs = dict(config.chronos_kwargs) if config.chronos_kwargs else {}
    if config.chronos_probabilistic:
        from darts.utils.likelihood_models import QuantileRegression
        kwargs["likelihood"] = QuantileRegression(quantiles=config.chronos_quantiles)
        logger.info(
            "Chronos-2 probabilistic mode: quantiles=%s", config.chronos_quantiles,
        )

    logger.info(
        "Chronos-2 chunk sizes: input=%d, output=%d (train_length=%s)",
        input_len, output_len, train_length,
    )
    return Chronos2Model(
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        hub_model_name=config.chronos_hub_model_name,
        hub_model_revision=config.chronos_hub_model_revision,
        local_dir=config.chronos_local_dir,
        **kwargs,
    )


def _chronos2_predict(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Generate forecasts using Chronos-2 zero-shot model.

    All wells are stacked into a single multivariate TimeSeries so that
    Chronos-2's group attention mechanism can share information across
    related production series (cross-learning).
    """
    TimeSeries, _ = _import_chronos2()
    series_ids = sorted(train_df["unique_id"].unique())
    if not series_ids:
        raise ValueError("No series available for Chronos2 forecasting.")

    past_cov_cols = [col for col in config.hist_exog if col in train_df.columns]
    future_cov_cols = [col for col in config.futr_exog if col in future_df.columns]
    use_past_cov = bool(past_cov_cols)
    use_future_cov = bool(future_cov_cols)

    all_cov_cols = list(dict.fromkeys(past_cov_cols + future_cov_cols))
    train_select = ["unique_id", "ds", "y"] + [c for c in all_cov_cols if c in train_df.columns]
    future_select = ["unique_id", "ds"] + [c for c in all_cov_cols if c in future_df.columns]

    full_df = pd.concat(
        [train_df[train_select], future_df[future_select].assign(y=np.nan)],
        ignore_index=True,
    )
    full_df = full_df.drop_duplicates(subset=["unique_id", "ds"], keep="first").sort_values(["unique_id", "ds"])

    target_pivot = full_df[full_df["unique_id"].isin(series_ids)].pivot(index="ds", columns="unique_id", values="y").sort_index()
    target_pivot.columns = [str(c) for c in target_pivot.columns]

    train_end = train_df["ds"].max()
    train_pivot = target_pivot.loc[target_pivot.index <= train_end]
    if train_pivot.empty:
        raise ValueError("Empty training window for Chronos-2.")

    target_ts = TimeSeries.from_dataframe(
        train_pivot.reset_index(), time_col="ds", value_cols=list(train_pivot.columns),
    ).astype(np.float32)

    past_cov_ts = None
    if use_past_cov:
        past_frames = []
        for uid in series_ids:
            uid_full = full_df[full_df["unique_id"] == uid].set_index("ds")[past_cov_cols]
            uid_full = uid_full.rename(columns={c: f"{uid}__{c}" for c in past_cov_cols})
            past_frames.append(uid_full)
        past_merged = pd.concat(past_frames, axis=1).sort_index().fillna(0.0)
        past_cov_ts = TimeSeries.from_dataframe(
            past_merged.reset_index(), time_col="ds", value_cols=list(past_merged.columns),
        ).astype(np.float32)

    future_cov_ts = None
    if use_future_cov:
        futr_frames = []
        for uid in series_ids:
            uid_full = full_df[full_df["unique_id"] == uid].set_index("ds")[future_cov_cols]
            uid_full = uid_full.rename(columns={c: f"{uid}__{c}" for c in future_cov_cols})
            futr_frames.append(uid_full)
        futr_merged = pd.concat(futr_frames, axis=1).sort_index().fillna(0.0)
        future_cov_ts = TimeSeries.from_dataframe(
            futr_merged.reset_index(), time_col="ds", value_cols=list(futr_merged.columns),
        ).astype(np.float32)

    train_length = len(train_pivot)
    model = _build_chronos2_model(config, train_length=train_length)
    fit_kwargs: Dict[str, Any] = {"series": target_ts, "verbose": False}
    predict_kwargs: Dict[str, Any] = {"n": config.horizon, "series": target_ts}
    if past_cov_ts is not None:
        fit_kwargs["past_covariates"] = past_cov_ts
        predict_kwargs["past_covariates"] = past_cov_ts
    if future_cov_ts is not None:
        fit_kwargs["future_covariates"] = future_cov_ts
        predict_kwargs["future_covariates"] = future_cov_ts

    if config.chronos_probabilistic:
        predict_kwargs["predict_likelihood_parameters"] = True

    model.fit(**fit_kwargs)
    preds_ts = model.predict(**predict_kwargs)

    pred_wide = preds_ts.to_dataframe().reset_index()
    time_col = pred_wide.columns[0]
    value_cols = [c for c in pred_wide.columns if c != time_col]

    pred_frames: List[pd.DataFrame] = []
    if config.chronos_probabilistic:
        median_q = 0.5
        q_suffix = f"_{median_q}"
        for uid in series_ids:
            median_col = f"{uid}{q_suffix}"
            if median_col not in pred_wide.columns:
                candidates = [c for c in pred_wide.columns if c.startswith(f"{uid}_")]
                if candidates:
                    median_col = candidates[len(candidates) // 2]
                else:
                    continue
            frame = pred_wide[[time_col, median_col]].rename(
                columns={time_col: "ds", median_col: "y_hat"},
            )
            frame["unique_id"] = uid
            q_cols = [c for c in pred_wide.columns if c.startswith(f"{uid}_")]
            for qc in q_cols:
                q_label = qc.replace(f"{uid}_", "q_")
                frame[q_label] = pred_wide[qc].values
            pred_frames.append(frame)
    else:
        for col in value_cols:
            frame = pred_wide[[time_col, col]].rename(columns={time_col: "ds", col: "y_hat"})
            frame["unique_id"] = col
            pred_frames.append(frame[["unique_id", "ds", "y_hat"]])
    return pd.concat(pred_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Training & forecasting
# ---------------------------------------------------------------------------

def train_and_forecast(frames: Dict[str, pd.DataFrame], config: PipelineConfig) -> pd.DataFrame:
    train_df = frames["train_df"]
    test_df = frames["test_df"]
    cov_cols = list(set(config.hist_exog + config.futr_exog))
    available_cols = [c for c in cov_cols if c in train_df.columns]
    future_df = pd.concat(
        [
            train_df[["unique_id", "ds"] + available_cols],
            test_df[["unique_id", "ds"] + [c for c in available_cols if c in test_df.columns]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["unique_id", "ds"], keep="first").sort_values(["unique_id", "ds"])
    preds = _chronos2_predict(train_df, future_df, config)
    logger.info("Generated %d Chronos-2 forecast rows", len(preds))
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def merge_forecast_frame(pred_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ["unique_id", "ds", "y_hat"] + [
        c for c in pred_df.columns if c.startswith("q_")
    ]
    merged = test_df[["unique_id", "ds", "y"]].merge(
        pred_df[pred_cols], on=["unique_id", "ds"], how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between predictions and test data.")
    return merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def evaluate_predictions(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    seasonal_period: int = 1,
    use_extended_metrics: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    merged = merge_forecast_frame(pred_df, test_df)

    overall = calculate_all_metrics(
        merged["y"].to_numpy(),
        merged["y_hat"].to_numpy(),
        y_insample=train_df["y"].to_numpy(),
    )

    per_well: Dict[str, Dict[str, float]] = {}
    for unique_id, group in merged.groupby("unique_id"):
        insample = train_df[train_df["unique_id"] == unique_id]["y"].to_numpy()
        per_well[str(unique_id)] = calculate_all_metrics(
            group["y"].to_numpy(), group["y_hat"].to_numpy(), y_insample=insample,
        )

    reservoir_metrics = {}
    try:
        time_idx = merged.groupby("unique_id").cumcount().to_numpy()
        reservoir_metrics = compute_all_reservoir_metrics(
            y_true=merged["y"].to_numpy(), y_pred=merged["y_hat"].to_numpy(), time_idx=time_idx,
        )
        logger.info("Computed %d reservoir-specific metrics", len([k for k, v in reservoir_metrics.items() if v is not None]))
    except Exception as exc:
        logger.warning("Could not compute reservoir metrics: %s", exc)

    metrics = {
        "overall": overall,
        "by_well": per_well,
        "observations": int(len(merged)),
        "reservoir": reservoir_metrics,
    }
    return metrics, merged
