from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from volve_config import VolveConfig

logger = logging.getLogger(__name__)

EPSILON = 1e-6

VOLVE_COL_MAP = {
    "DATEPRD": "date",
    "WELL_BORE_CODE": "well_raw",
    "WELL_TYPE": "well_type",
    "BORE_OIL_VOL": "oil_vol",
    "BORE_WAT_VOL": "wat_vol",
    "BORE_GAS_VOL": "gas_vol",
    "BORE_WI_VOL": "wi_vol",
    "ON_STREAM_HRS": "on_stream_hrs",
    "AVG_DOWNHOLE_PRESSURE": "avg_downhole_pressure",
    "AVG_DOWNHOLE_TEMPERATURE": "avg_downhole_temperature",
    "AVG_DP_TUBING": "avg_dp_tubing",
    "AVG_WHP_P": "avg_whp",
    "AVG_CHOKE_SIZE_P": "avg_choke_size",
}

WELL_SHORT_NAMES = {
    "NO 15/9-F-1 C": "F1C",
    "NO 15/9-F-11 H": "F11H",
    "NO 15/9-F-12 H": "F12H",
    "NO 15/9-F-14 H": "F14H",
    "NO 15/9-F-15 D": "F15D",
    "NO 15/9-F-4 AH": "F4AH",
    "NO 15/9-F-5 AH": "F5AH",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_volve_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Daily Production Data", engine="openpyxl")
    rename = {k: v for k, v in VOLVE_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["well"] = df["well_raw"].map(WELL_SHORT_NAMES).fillna(df["well_raw"])
    for col in ["oil_vol", "wat_vol", "gas_vol", "wi_vol",
                 "on_stream_hrs", "avg_downhole_pressure",
                 "avg_downhole_temperature", "avg_dp_tubing",
                 "avg_whp", "avg_choke_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["type"] = df["well_type"].map({"OP": "PROD", "WI": "INJ"}).fillna("PROD")
    df = df.sort_values(["well", "date"]).drop_duplicates(["well", "date"])
    logger.info("Loaded Volve daily data: %d rows, %d wells", len(df), df["well"].nunique())
    return df


def _filter_shutdown_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove days where on_stream_hrs == 0 (well shut-in / maintenance)."""
    before = len(df)
    mask = df["on_stream_hrs"] > 0
    if "oil_vol" in df.columns:
        mask = mask | (df["oil_vol"] > 0)
    df = df[mask].copy()
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d shutdown days (on_stream=0 & oil=0)", removed)
    return df


def _trim_permanent_shutdown(df: pd.DataFrame, min_tail_zeros: int = 14) -> pd.DataFrame:
    """Trim consecutive zero-production tail from each well (field abandonment)."""
    frames = []
    for well in df["well"].unique():
        date_col = "ds" if "ds" in df.columns else "date"
        wd = df[df["well"] == well].sort_values(date_col)
        if wd.empty:
            frames.append(wd)
            continue
        oil = wd["oil_vol"].values
        # Find last non-zero production day
        nonzero_idx = np.where(oil > 0)[0]
        if len(nonzero_idx) == 0:
            continue
        last_production = nonzero_idx[-1]
        tail_zeros = len(oil) - 1 - last_production
        if tail_zeros >= min_tail_zeros:
            # Keep up to a few days past last production
            cut = last_production + 3
            trimmed = len(wd) - cut
            wd = wd.iloc[:cut]
            logger.info("Trimmed %d shutdown days from tail of well %s", trimmed, well)
        frames.append(wd)
    return pd.concat(frames, ignore_index=True) if frames else df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_production_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_vol"] = df["oil_vol"].fillna(0.0).clip(lower=0.0)
    df["wat_vol"] = df["wat_vol"].fillna(0.0).clip(lower=0.0)
    df["gas_vol"] = df["gas_vol"].fillna(0.0).clip(lower=0.0)

    df["water_rate"] = df["wat_vol"]
    df["gas_rate"] = df["gas_vol"]
    liq = (df["oil_vol"] + df["wat_vol"]).clip(lower=EPSILON)
    df["watercut"] = (df["wat_vol"] / liq).clip(0.0, 1.0).fillna(0.0)
    # GOR (gas-oil ratio)
    df["gor"] = (df["gas_vol"] / df["oil_vol"].clip(lower=EPSILON)).clip(upper=1e5).fillna(0.0)

    for col in ["avg_downhole_pressure", "avg_downhole_temperature",
                 "avg_dp_tubing", "avg_whp", "avg_choke_size", "on_stream_hrs"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def _compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics and decline-rate features per well."""
    df = df.copy()
    for well in df["well"].unique():
        mask = df["well"] == well
        oil = df.loc[mask, "oil_vol"]
        df.loc[mask, "oil_ma7"] = oil.rolling(7, min_periods=1).mean()
        df.loc[mask, "oil_ma14"] = oil.rolling(14, min_periods=1).mean()
        df.loc[mask, "oil_ma30"] = oil.rolling(30, min_periods=1).mean()
        df.loc[mask, "oil_std7"] = oil.rolling(7, min_periods=2).std().fillna(0.0)
        df.loc[mask, "oil_std30"] = oil.rolling(30, min_periods=2).std().fillna(0.0)
        # Decline rate features
        df.loc[mask, "oil_diff"] = oil.diff().fillna(0.0)
        df.loc[mask, "oil_diff_ma7"] = oil.diff().rolling(7, min_periods=1).mean().fillna(0.0)
        df.loc[mask, "oil_diff_ma30"] = oil.diff().rolling(30, min_periods=1).mean().fillna(0.0)
        pct = oil.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)
        df.loc[mask, "oil_pct_change"] = pct
        # Water rolling
        if "water_rate" in df.columns:
            wat = df.loc[mask, "water_rate"]
            df.loc[mask, "water_ma7"] = wat.rolling(7, min_periods=1).mean()
        # Pressure rolling
        if "avg_whp" in df.columns:
            whp = df.loc[mask, "avg_whp"]
            df.loc[mask, "whp_ma7"] = whp.rolling(7, min_periods=1).mean()
    return df


def _compute_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year seasonality features (analogous to main pipeline fourier features)."""
    df = df.copy()
    day_of_year = df["ds"].dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["ds"].dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["ds"].dt.dayofweek / 7.0)
    return df


def _compute_productivity_index(
    df: pd.DataFrame, train_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Productivity index J(t) and drawdown -- analogous to main pipeline PI feature."""
    df = df.copy()
    pressure_col = None
    for candidate in ["avg_dp_tubing", "avg_downhole_pressure", "avg_whp"]:
        if candidate in df.columns and (df[candidate] > 0).any():
            pressure_col = candidate
            break
    if pressure_col is None:
        df["productivity_index"] = 0.0
        df["dp_drawdown"] = 0.0
        return df

    train_mask = df["ds"] <= train_cutoff
    p_ref = (
        df.loc[train_mask & (df[pressure_col] > 0)]
        .groupby("well")[pressure_col]
        .quantile(0.95)
    )
    df["_p_ref"] = df["well"].map(p_ref)
    dp = (df["_p_ref"] - df[pressure_col]).clip(lower=1.0)
    df["productivity_index"] = (df["oil_vol"] / dp).fillna(0.0)
    df["dp_drawdown"] = dp.fillna(0.0)
    df = df.drop(columns=["_p_ref"])
    logger.info("Computed productivity index from %s", pressure_col)
    return df


def _compute_field_injection(prod_df: pd.DataFrame, inj_df: pd.DataFrame) -> pd.DataFrame:
    if inj_df.empty:
        prod_df["field_inj_rate"] = 0.0
        return prod_df
    field_inj = (
        inj_df.groupby("ds")["wi_vol"]
        .sum()
        .reset_index()
        .rename(columns={"wi_vol": "field_inj_rate"})
    )
    prod_df = prod_df.merge(field_inj, on="ds", how="left")
    prod_df["field_inj_rate"] = prod_df["field_inj_rate"].fillna(0.0)
    return prod_df


# ---------------------------------------------------------------------------
# Reindex & imputation
# ---------------------------------------------------------------------------

def _reindex_daily(df: pd.DataFrame, wells: List[str]) -> pd.DataFrame:
    """Reindex to continuous daily calendar with limited gap filling."""
    frames: List[pd.DataFrame] = []
    for well in wells:
        wd = df[df["well"] == well].set_index("ds").sort_index()
        if wd.empty:
            continue
        idx = pd.date_range(wd.index.min(), wd.index.max(), freq="D")
        wd = wd.reindex(idx)
        wd["well"] = well
        # Interpolate numeric columns for short gaps (<=3 days), leave longer as-is
        numeric_cols = wd.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            wd[col] = wd[col].interpolate(method="linear", limit=3)
        wd[numeric_cols] = wd[numeric_cols].ffill(limit=3).bfill(limit=1)
        # Forward-fill non-numeric (type, etc.)
        str_cols = [c for c in wd.columns if c not in numeric_cols and c != "well"]
        for col in str_cols:
            wd[col] = wd[col].ffill(limit=3)
        frames.append(wd.reset_index().rename(columns={"index": "ds"}))
    if not frames:
        return pd.DataFrame(columns=["ds", "well"])
    result = pd.concat(frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Well selection
# ---------------------------------------------------------------------------

def get_producer_wells(df: pd.DataFrame, config: VolveConfig) -> List[str]:
    last_type = df.sort_values("date").groupby("well").tail(1).set_index("well")["type"]
    producers = last_type[last_type == "PROD"].index.tolist()
    # Count only active production days (on_stream > 0 or oil > 0)
    active = df[(df["type"] == "PROD") & ((df["on_stream_hrs"] > 0) | (df["oil_vol"] > 0))]
    counts = active.groupby("well").size()
    selected = [w for w in producers if counts.get(w, 0) >= config.min_history]
    logger.info("Selected %d producer wells: %s", len(selected), selected)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def prepare_volve_frames(
    raw_df: pd.DataFrame,
    config: VolveConfig,
) -> Dict[str, Any]:
    producer_wells = get_producer_wells(raw_df, config)
    if not producer_wells:
        raise ValueError("No producer wells meet the selection criteria.")

    prod_raw = raw_df[raw_df["well"].isin(producer_wells) & (raw_df["type"] == "PROD")].copy()
    prod_raw = prod_raw.rename(columns={"date": "ds"})

    # Clean: remove shutdown days & trim permanent shutdowns
    prod_raw = _filter_shutdown_days(prod_raw)
    prod_raw = _trim_permanent_shutdown(prod_raw, min_tail_zeros=14)

    inj_raw = raw_df[raw_df["type"] == "INJ"].copy()
    inj_raw = inj_raw.rename(columns={"date": "ds"})

    # Reindex to daily calendar with limited interpolation
    prod_df = _reindex_daily(prod_raw, producer_wells)
    # Fill remaining NaN in numeric cols
    numeric_cols = [c for c in prod_df.columns
                    if c not in {"ds", "well", "type", "well_raw", "well_type"}]
    prod_df[numeric_cols] = prod_df[numeric_cols].fillna(0.0)
    prod_df["type"] = prod_df["type"].fillna("PROD")

    # Features
    prod_df = _compute_production_features(prod_df)

    inj_daily = pd.DataFrame(columns=["ds", "well", "wi_vol"])
    if not inj_raw.empty:
        inj_wells = sorted(inj_raw["well"].unique())
        inj_daily = _reindex_daily(inj_raw, inj_wells)
        inj_daily["wi_vol"] = inj_daily["wi_vol"].fillna(0.0)
    prod_df = _compute_field_injection(prod_df, inj_daily)

    # Compute train_cutoff early for leakage-safe features
    max_dates = prod_df.groupby("well")["ds"].max()
    target_end = max_dates.min()
    test_start = target_end - pd.Timedelta(days=config.horizon - 1)
    train_cutoff = test_start - pd.Timedelta(days=1)

    prod_df = _compute_rolling_features(prod_df)
    prod_df = _compute_fourier_features(prod_df)
    prod_df = _compute_productivity_index(prod_df, train_cutoff)

    prod_df["unique_id"] = prod_df["well"]
    prod_df["y"] = prod_df["oil_vol"].astype(float)
    prod_df["time_idx"] = prod_df.sort_values("ds").groupby("well").cumcount()

    # Fill any remaining NaN in feature columns
    feature_cols = set(config.hist_exog + config.futr_exog)
    for col in feature_cols:
        if col in prod_df.columns and prod_df[col].isna().any():
            prod_df[col] = prod_df[col].fillna(0.0)

    # Train/test split
    train_df = prod_df[prod_df["ds"] < test_start].copy().sort_values(["unique_id", "ds"])
    test_df = prod_df[prod_df["ds"] >= test_start].copy().sort_values(["unique_id", "ds"])

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test dataframe is empty after splitting.")

    futr_df = test_df[["unique_id", "ds"] + [c for c in config.futr_exog if c in test_df.columns]].copy()

    logger.info(
        "Prepared Volve daily frames: train=%d, test=%d, wells=%d, "
        "test_start=%s, target_end=%s",
        len(train_df), len(test_df), len(producer_wells),
        test_start.date(), target_end.date(),
    )
    return {
        "train_df": train_df,
        "test_df": test_df,
        "futr_df": futr_df,
        "static_df": pd.DataFrame({"unique_id": producer_wells}),
        "target_wells": producer_wells,
        "test_start": test_start,
        "train_cutoff": train_cutoff,
    }


# ---------------------------------------------------------------------------
# XLinear model
# ---------------------------------------------------------------------------

def xlinear_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: VolveConfig,
) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import XLinear
    from neuralforecast.losses.pytorch import MAE, MSE, HuberLoss

    loss_map = {"mse": MSE(), "mae": MAE(), "huber": HuberLoss()}
    loss_fn = loss_map.get(config.xlinear_loss, MSE())

    futr_cols = [c for c in config.futr_exog if c in train_df.columns]
    hist_cols = [c for c in config.hist_exog if c in train_df.columns and c not in futr_cols]
    stat_cols = [c for c in config.static_exog if c in train_df.columns]
    n_series = train_df["unique_id"].nunique()

    model = XLinear(
        h=config.horizon,
        input_size=config.input_size,
        n_series=n_series,
        futr_exog_list=futr_cols if futr_cols else None,
        hist_exog_list=hist_cols if hist_cols else None,
        stat_exog_list=stat_cols if stat_cols else None,
        hidden_size=config.xlinear_hidden_size,
        temporal_ff=config.xlinear_temporal_ff,
        channel_ff=config.xlinear_channel_ff,
        temporal_dropout=config.xlinear_temporal_dropout,
        channel_dropout=config.xlinear_channel_dropout,
        embed_dropout=config.xlinear_embed_dropout,
        head_dropout=config.xlinear_head_dropout,
        loss=loss_fn,
        valid_loss=MAE(),
        max_steps=config.xlinear_max_steps,
        learning_rate=config.xlinear_learning_rate,
        num_lr_decays=config.xlinear_num_lr_decays,
        batch_size=config.xlinear_batch_size,
        windows_batch_size=config.xlinear_windows_batch_size,
        early_stop_patience_steps=config.xlinear_early_stop_patience,
        val_check_steps=config.xlinear_val_check_steps,
        use_norm=True,
        scaler_type=config.xlinear_scaler_type,
        random_seed=config.random_seed,
        accelerator="auto",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )

    nf = NeuralForecast(models=[model], freq=config.freq)

    nf_train = train_df[["unique_id", "ds", "y"] + futr_cols + hist_cols + stat_cols].copy()
    nf_train = nf_train.fillna(0.0)

    static_df = None
    if stat_cols:
        static_df = nf_train.groupby("unique_id")[["unique_id"] + stat_cols].first().reset_index(drop=True)
        nf_train = nf_train.drop(columns=stat_cols, errors="ignore")

    logger.info(
        "XLinear config: input=%d, h=%d, n_series=%d, hidden=%d, "
        "futr=%d cols, hist=%d cols, stat=%d cols",
        config.input_size, config.horizon, n_series,
        config.xlinear_hidden_size,
        len(futr_cols), len(hist_cols), len(stat_cols),
    )

    nf.fit(df=nf_train, static_df=static_df, val_size=config.val_horizon)

    futr_df = None
    if futr_cols:
        futr_df = test_df[["unique_id", "ds"] + futr_cols].copy().fillna(0.0)

    forecasts = nf.predict(futr_df=futr_df, static_df=static_df)
    forecasts = forecasts.reset_index()

    xlinear_col = [c for c in forecasts.columns if c.startswith("XLinear")]
    if xlinear_col:
        forecasts = forecasts.rename(columns={xlinear_col[0]: "y_hat"})
    elif "y_hat" not in forecasts.columns:
        val_cols = [c for c in forecasts.columns if c not in {"unique_id", "ds"}]
        if val_cols:
            forecasts = forecasts.rename(columns={val_cols[0]: "y_hat"})

    forecasts["y_hat"] = forecasts["y_hat"].clip(lower=0.0)
    return forecasts[["unique_id", "ds", "y_hat"]]


def train_and_forecast(
    frames: Dict[str, Any],
    config: VolveConfig,
) -> pd.DataFrame:
    preds = xlinear_predict(frames["train_df"], frames["test_df"], config)
    logger.info("Generated %d XLinear forecast rows", len(preds))
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pred_cols = ["unique_id", "ds", "y_hat"] + [
        c for c in pred_df.columns if c.startswith("q_") or c.startswith("cp_")
    ]
    pred_cols = [c for c in pred_cols if c in pred_df.columns]
    test_cols = ["unique_id", "ds", "y"]
    if "time_idx" in test_df.columns:
        test_cols.append("time_idx")
    merged = test_df[test_cols].merge(
        pred_df[pred_cols], on=["unique_id", "ds"], how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between predictions and test data.")
    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    y_true = merged["y"].to_numpy()
    y_hat = merged["y_hat"].to_numpy()
    y_insample = train_df["y"].to_numpy()

    overall = _compute_metrics(y_true, y_hat, y_insample)

    per_well: Dict[str, Dict[str, float]] = {}
    for uid, group in merged.groupby("unique_id"):
        insample = train_df[train_df["unique_id"] == uid]["y"].to_numpy()
        per_well[str(uid)] = _compute_metrics(
            group["y"].to_numpy(), group["y_hat"].to_numpy(), insample,
        )

    metrics = {
        "overall": overall,
        "by_well": per_well,
        "observations": int(len(merged)),
    }
    return metrics, merged


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    n = len(y_true)
    if n == 0:
        return {}
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > EPSILON else None
    mean_bias = float(np.mean(residuals))

    mask = np.abs(y_true) > EPSILON
    mape = float(np.mean(np.abs(residuals[mask] / y_true[mask])) * 100) if mask.any() else None

    if np.std(y_true) > EPSILON and np.std(y_pred) > EPSILON:
        r_val = float(np.corrcoef(y_true, y_pred)[0, 1])
        alpha = float(np.std(y_pred) / np.std(y_true))
        beta = float(np.mean(y_pred) / np.mean(y_true)) if np.abs(np.mean(y_true)) > EPSILON else 1.0
        kge = float(1 - np.sqrt((r_val - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = None

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "kge": kge,
        "mean_bias_error": mean_bias,
    }


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def generate_walk_forward_splits(
    train_df: pd.DataFrame,
    horizon: int,
    step: int,
    folds: int,
) -> List[Dict[str, Any]]:
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
        logger.warning(
            "Insufficient history for CV: total=%d, horizon=%d, step=%d, folds=%d",
            total_points, horizon, step, folds,
        )
        return []
    splits: List[Dict[str, Any]] = []
    for fold_idx in range(folds):
        train_cutoff = usable_prefix + step * fold_idx - 1
        val_start = usable_prefix + step * fold_idx
        val_end = val_start + horizon - 1
        fold_train = train_df[train_df["time_idx"] <= train_cutoff].copy()
        fold_val = train_df[
            (train_df["time_idx"] >= val_start) & (train_df["time_idx"] <= val_end)
        ].copy()
        if fold_train.empty or fold_val.empty:
            continue
        splits.append({
            "fold": fold_idx + 1,
            "train_cutoff": train_cutoff,
            "val_start": val_start,
            "val_end": val_end,
            "train_df": fold_train,
            "val_df": fold_val,
        })
    logger.info("Generated %d walk-forward CV splits (total_points=%d)", len(splits), total_points)
    return splits


def run_walk_forward_validation(
    frames: Dict[str, Any],
    config: VolveConfig,
) -> Optional[Dict[str, Any]]:
    if not config.cv_enabled:
        return None
    train_df = frames["train_df"]
    splits = generate_walk_forward_splits(
        train_df, horizon=config.horizon, step=config.cv_step, folds=config.cv_folds,
    )
    if not splits:
        logger.warning("No CV splits generated.")
        return None

    fold_results: List[Dict[str, Any]] = []
    residual_pool_frames: List[pd.DataFrame] = []
    metric_sums: Dict[str, float] = {}
    metric_weights: Dict[str, float] = {}

    for split in splits:
        fold_train = split["train_df"]
        fold_val = split["val_df"]
        try:
            preds = xlinear_predict(fold_train, fold_val, config)
        except Exception as exc:
            logger.warning("Fold %d failed: %s", split["fold"], exc)
            continue

        metrics, merged = evaluate_predictions(preds, fold_val, fold_train)
        overall = {
            k: float(v) for k, v in metrics.get("overall", {}).items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        }
        rows = len(merged)
        fold_results.append({
            "fold": int(split["fold"]),
            "rows": rows,
            "metrics": overall,
        })
        logger.info("Fold %d metrics: %s", split["fold"], {k: round(v, 4) for k, v in overall.items()})

        for key, value in overall.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * rows
            metric_weights[key] = metric_weights.get(key, 0.0) + rows

        fold_residuals = merged[["unique_id", "ds", "y", "y_hat"]].copy()
        fold_residuals["h_step"] = fold_residuals.groupby("unique_id").cumcount() + 1
        fold_residuals["abs_err"] = (fold_residuals["y"] - fold_residuals["y_hat"]).abs()
        fold_residuals["fold"] = int(split["fold"])
        fold_residuals = fold_residuals[fold_residuals["h_step"] <= config.horizon]
        if not fold_residuals.empty:
            residual_pool_frames.append(
                fold_residuals[["fold", "unique_id", "ds", "h_step", "abs_err"]]
            )

    aggregate = {
        key: float(metric_sums[key] / metric_weights[key])
        for key in metric_sums
        if metric_weights.get(key, 0.0) > 0.0
    }

    conformal_profile: Optional[Dict[str, Any]] = None
    if config.conformal_enabled and residual_pool_frames:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from conformal import compute_conformal_profile
        residual_pool = pd.concat(residual_pool_frames, ignore_index=True)
        residual_pool = residual_pool.sort_values(["ds", "fold", "unique_id"]).reset_index(drop=True)
        residual_pool["calib_order"] = np.arange(len(residual_pool))
        conformal_profile = compute_conformal_profile(
            residual_pool,
            horizon=config.horizon,
            alpha=config.conformal_alpha,
            method=config.conformal_method,
            per_horizon=config.conformal_per_horizon,
            exp_decay=config.conformal_exp_decay,
            min_samples=config.conformal_min_samples,
        )

    if aggregate:
        logger.info("CV aggregate: %s", {k: round(v, 4) for k, v in aggregate.items()})

    result: Dict[str, Any] = {"folds": fold_results, "aggregate": aggregate}
    if conformal_profile:
        result["conformal_profile"] = conformal_profile
    return result
