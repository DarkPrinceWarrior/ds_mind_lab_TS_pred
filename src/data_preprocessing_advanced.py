"""Advanced data preprocessing with physics-aware imputation and anomaly detection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, signal
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class PhysicsAwarePreprocessor:
    """Preprocessor with physics-based constraints and advanced imputation."""
    
    def __init__(
        self,
        well_type: str = "PROD",
        max_rate_change_pct: float = 0.5,
        min_rate: float = 0.0,
        max_rate: Optional[float] = None,
    ):
        self.well_type = well_type
        self.max_rate_change_pct = max_rate_change_pct
        self.min_rate = min_rate
        self.max_rate = max_rate
        
    def detect_structural_breaks(
        self,
        df: pd.DataFrame,
        rate_col: str = "wlpr",
        threshold: float = 0.7,
    ) -> pd.DataFrame:
        """Detect well shutdowns, workovers using rate discontinuities.
        
        Args:
            df: Well data grouped by well
            rate_col: Rate column to analyze
            threshold: Relative drop threshold (0-1)
        
        Returns:
            DataFrame with 'is_shutdown' and 'is_startup' flags
        """
        df = df.copy()
        df["is_shutdown"] = False
        df["is_startup"] = False
        
        for well in df["well"].unique():
            well_mask = df["well"] == well
            rates = df.loc[well_mask, rate_col].values
            
            # Detect shutdowns: rate drops > threshold
            rate_change = np.diff(rates, prepend=rates[0])
            rate_prev = np.maximum(rates - rate_change, 1e-6)
            relative_drop = -rate_change / rate_prev
            
            shutdown_mask = relative_drop > threshold
            df.loc[well_mask, "is_shutdown"] = shutdown_mask
            
            # Detect startups: rate increases after near-zero
            was_low = rates < (self.min_rate + 1.0)
            startup_mask = was_low[:-1] & (rate_change[1:] > 0)
            df.loc[well_mask, "is_startup"] = np.insert(startup_mask, 0, False)
            
        logger.info(
            "Detected %d shutdowns and %d startups",
            df["is_shutdown"].sum(),
            df["is_startup"].sum(),
        )
        return df
    
    def physics_aware_imputation(
        self,
        df: pd.DataFrame,
        rate_cols: List[str],
        cumulative_cols: List[str],
    ) -> pd.DataFrame:
        """Impute missing values respecting physical constraints.
        
        Uses:
        - Cubic spline for rates with smoothness constraint
        - Monotonic interpolation for cumulatives
        - Zero filling for injection wells during production periods
        
        Args:
            df: Well data
            rate_cols: Rate columns (WLPR, WOMR, WWIR)
            cumulative_cols: Cumulative columns (WLPT, WOMT, WWIT)
        
        Returns:
            Imputed DataFrame
        """
        df = df.copy()
        
        for well in df["well"].unique():
            well_mask = df["well"] == well
            well_data = df.loc[well_mask].copy()
            dates = pd.to_datetime(well_data["date"] if "date" in well_data.columns else well_data["ds"])
            time_numeric = (dates - dates.min()).dt.total_seconds().values
            
            # Impute rates with cubic spline
            for col in rate_cols:
                if col not in well_data.columns:
                    continue
                    
                values = well_data[col].values
                mask = ~np.isnan(values)
                
                if mask.sum() < 4:  # Need at least 4 points for cubic
                    df.loc[well_mask, col] = well_data[col].fillna(method="ffill").fillna(0.0)
                    continue
                
                # Fit cubic spline to non-missing values
                spline = interpolate.UnivariateSpline(
                    time_numeric[mask],
                    values[mask],
                    k=3,
                    s=np.sum(mask) * 0.01,  # Smoothing
                )
                
                # Interpolate
                interpolated = spline(time_numeric)
                interpolated = np.clip(interpolated, self.min_rate, None)
                
                # Fill only missing values
                values[~mask] = interpolated[~mask]
                df.loc[well_mask, col] = values
            
            # Impute cumulatives with monotonic interpolation
            for col in cumulative_cols:
                if col not in well_data.columns:
                    continue
                    
                values = well_data[col].values
                mask = ~np.isnan(values)
                
                if mask.sum() < 2:
                    df.loc[well_mask, col] = well_data[col].fillna(method="ffill").fillna(0.0)
                    continue
                
                # Monotonic interpolation
                interp_func = interpolate.interp1d(
                    time_numeric[mask],
                    values[mask],
                    kind="linear",
                    fill_value="extrapolate",
                )
                
                interpolated = interp_func(time_numeric)
                interpolated = np.maximum.accumulate(interpolated)  # Enforce monotonicity
                
                values[~mask] = interpolated[~mask]
                df.loc[well_mask, col] = values
        
        return df
    
    def detect_outliers_multivariate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.05,
    ) -> pd.DataFrame:
        """Detect multivariate outliers using Elliptic Envelope.
        
        Args:
            df: Well data
            feature_cols: Columns to use for outlier detection
            contamination: Expected outlier proportion
        
        Returns:
            DataFrame with 'is_outlier' flag
        """
        df = df.copy()
        df["is_outlier"] = False
        
        for well in df["well"].unique():
            well_mask = df["well"] == well
            well_data = df.loc[well_mask, feature_cols].copy()
            
            # Remove missing values
            valid_mask = well_data.notna().all(axis=1)
            if valid_mask.sum() < 10:  # Need sufficient data
                continue
            
            X = well_data.loc[valid_mask].values
            
            # Robust scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit elliptic envelope
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X_scaled)
            
            # Mark outliers (-1 from detector)
            outlier_indices = well_data.loc[valid_mask].index[predictions == -1]
            df.loc[outlier_indices, "is_outlier"] = True
        
        logger.info("Detected %d outliers (%.2f%%)", df["is_outlier"].sum(), 100 * df["is_outlier"].mean())
        return df
    
    def smooth_rates_savgol(
        self,
        df: pd.DataFrame,
        rate_cols: List[str],
        window_length: int = 7,
        polyorder: int = 2,
    ) -> pd.DataFrame:
        """Apply Savitzky-Golay filter to smooth noisy rate data.
        
        Args:
            df: Well data
            rate_cols: Rate columns to smooth
            window_length: Filter window (must be odd)
            polyorder: Polynomial order
        
        Returns:
            DataFrame with smoothed rates
        """
        df = df.copy()
        
        if window_length % 2 == 0:
            window_length += 1
        
        for well in df["well"].unique():
            well_mask = df["well"] == well
            
            for col in rate_cols:
                if col not in df.columns:
                    continue
                
                values = df.loc[well_mask, col].values
                
                if len(values) < window_length:
                    continue
                
                # Apply Savitzky-Golay filter
                smoothed = signal.savgol_filter(
                    values,
                    window_length=window_length,
                    polyorder=polyorder,
                    mode="nearest",
                )
                
                df.loc[well_mask, f"{col}_smooth"] = smoothed
        
        return df

    def smooth_rates_bilateral(
        self,
        df: pd.DataFrame,
        rate_cols: List[str],
        window_length: int = 7,
        sigma_space: float = 3.0,
        sigma_range: Optional[float] = None,
    ) -> pd.DataFrame:
        """Apply adaptive bilateral Gaussian filter to smooth noisy rate data.

        Unlike Savitzky-Golay, bilateral filter preserves sharp transitions
        (shutdowns, startups) while smoothing gradual noise. Each sample is
        weighted by both temporal proximity (sigma_space) and value similarity
        (sigma_range), so large jumps are not blurred.

        Args:
            df: Well data with columns [well] + rate_cols.
            rate_cols: Rate columns to smooth.
            window_length: Half-window size (full window = 2*window_length+1).
            sigma_space: Std-dev of the spatial (temporal) Gaussian kernel.
            sigma_range: Std-dev of the range (value) Gaussian kernel.
                If None, set to per-well MAD (median absolute deviation) of
                differences -- automatically adapts to signal volatility.

        Returns:
            DataFrame with ``{col}_smooth`` columns added.
        """
        df = df.copy()
        hw = window_length

        for well in df["well"].unique():
            well_mask = df["well"] == well
            for col in rate_cols:
                if col not in df.columns:
                    continue
                values = df.loc[well_mask, col].values.astype(float)
                n = len(values)
                if n < 3:
                    continue

                sr = sigma_range
                if sr is None:
                    diffs = np.abs(np.diff(values))
                    mad = float(np.median(diffs)) if len(diffs) > 0 else 1.0
                    sr = max(mad * 1.4826, 1e-6)

                smoothed = np.empty(n)
                for i in range(n):
                    lo = max(0, i - hw)
                    hi = min(n, i + hw + 1)
                    window = values[lo:hi]
                    offsets = np.arange(lo - i, hi - i)
                    w_space = np.exp(-0.5 * (offsets / sigma_space) ** 2)
                    w_range = np.exp(-0.5 * ((window - values[i]) / sr) ** 2)
                    w = w_space * w_range
                    total = w.sum()
                    smoothed[i] = (w * window).sum() / total if total > 0 else values[i]

                df.loc[well_mask, f"{col}_smooth"] = smoothed

        return df


def create_decline_features(df: pd.DataFrame, rate_col: str = "wlpr") -> pd.DataFrame:
    """Create decline curve features for better forecasting.
    
    Args:
        df: Well data
        rate_col: Rate column
    
    Returns:
        DataFrame with decline features
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        rates = df.loc[well_mask, rate_col].values
        time_idx = df.loc[well_mask, "time_idx"].values if "time_idx" in df.columns else np.arange(len(rates))
        
        # Peak rate and time since peak
        peak_idx = np.argmax(rates)
        df.loc[well_mask, f"{rate_col}_peak"] = rates[peak_idx]
        df.loc[well_mask, f"{rate_col}_time_since_peak"] = time_idx - peak_idx
        
        # Decline rate (exponential)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_ratio = rates / rates[peak_idx]
            rate_ratio = np.clip(rate_ratio, 1e-6, 1.0)
            decline_const = -np.log(rate_ratio) / np.maximum(time_idx - peak_idx, 1)
            decline_const = np.nan_to_num(decline_const, nan=0.0, posinf=0.0, neginf=0.0)
        
        df.loc[well_mask, f"{rate_col}_decline_const"] = decline_const
        
        # Normalized cumulative (EUR estimation)
        cumulative_col = rate_col.replace("r", "t")
        if cumulative_col in df.columns:
            cumulative = df.loc[well_mask, cumulative_col].values
            df.loc[well_mask, f"{rate_col}_cum_normalized"] = cumulative / (cumulative[-1] + 1e-6)
    
    return df


def add_production_stage_features(df: pd.DataFrame, rate_col: str = "wlpr") -> pd.DataFrame:
    """Classify production stage (buildup, plateau, decline).
    
    Args:
        df: Well data
        rate_col: Rate column
    
    Returns:
        DataFrame with stage features
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        rates = df.loc[well_mask, rate_col].values
        
        if len(rates) < 5:
            df.loc[well_mask, "stage_buildup"] = 0
            df.loc[well_mask, "stage_plateau"] = 0
            df.loc[well_mask, "stage_decline"] = 0
            continue
        
        # Compute rolling statistics
        window = min(5, len(rates) // 3)
        rates_smooth = pd.Series(rates).rolling(window=window, center=True).mean().fillna(method="bfill").fillna(method="ffill").values
        rate_change = np.diff(rates_smooth, prepend=rates_smooth[0])
        
        # Classify stages
        buildup = rate_change > np.std(rate_change) * 0.5
        decline = rate_change < -np.std(rate_change) * 0.5
        plateau = ~buildup & ~decline
        
        df.loc[well_mask, "stage_buildup"] = buildup.astype(int)
        df.loc[well_mask, "stage_plateau"] = plateau.astype(int)
        df.loc[well_mask, "stage_decline"] = decline.astype(int)
    
    return df
