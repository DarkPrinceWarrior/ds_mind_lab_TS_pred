"""Advanced feature engineering for well production forecasting."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def create_interaction_features(
    df: pd.DataFrame,
    base_features: List[str],
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Create interaction features between key variables.
    
    Research basis: "Automated Reservoir History Matching Framework" (2025)
    Shows interactions improve interwell connectivity modeling.
    
    Args:
        df: DataFrame with features
        base_features: Base feature columns
        interaction_pairs: Specific pairs to interact, or None for auto
    
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    if interaction_pairs is None:
        # Auto-generate important interactions for well production
        interaction_pairs = [
            ("wlpr", "wbhp"),  # Rate vs bottomhole pressure
            ("wlpr", "inj_wwir_lag_weighted"),  # Production vs injection
            ("womt", "wwit"),  # Oil vs water cumulative
            ("wbhp", "inj_wwir_lag_weighted"),  # Pressure vs injection
            ("womr", "fw"),  # Oil rate vs water cut
        ]
    
    for feat1, feat2 in interaction_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Multiplicative interaction
            df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
            
            # Ratio interaction (with safety)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df[feat1] / (df[feat2] + 1e-6)
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            df[f"{feat1}_div_{feat2}"] = ratio
    
    logger.info("Created %d interaction features", len(interaction_pairs) * 2)
    return df


def create_spatial_features(
    df: pd.DataFrame,
    coords: pd.DataFrame,
    distances: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create spatial/geological features.
    
    Research basis: "WellPINN" (2025) - spatial context improves pressure predictions
    
    Args:
        df: Well data
        coords: Well coordinates
        distances: Distance matrix (optional)
    
    Returns:
        DataFrame with spatial features
    """
    df = df.copy()
    coords_dict = coords.set_index("well")[["x", "y", "z"]].to_dict("index")
    
    # Add well depth features
    df["well_depth"] = df["well"].map(lambda w: abs(coords_dict.get(str(w), {}).get("z", 0)))
    
    # Compute field centroid
    field_x = coords["x"].mean()
    field_y = coords["y"].mean()
    
    # Distance from field center
    df["dist_from_center"] = df["well"].map(
        lambda w: np.sqrt(
            (coords_dict.get(str(w), {}).get("x", field_x) - field_x) ** 2
            + (coords_dict.get(str(w), {}).get("y", field_y) - field_y) ** 2
        )
    )
    
    # Directional features (quadrant)
    def get_quadrant(well):
        coord = coords_dict.get(str(well), {})
        dx = coord.get("x", field_x) - field_x
        dy = coord.get("y", field_y) - field_y
        
        if dx >= 0 and dy >= 0:
            return 0  # NE
        elif dx < 0 and dy >= 0:
            return 1  # NW
        elif dx < 0 and dy < 0:
            return 2  # SW
        else:
            return 3  # SE
    
    df["quadrant"] = df["well"].map(get_quadrant)
    
    # One-hot encode quadrant
    for q in range(4):
        df[f"quadrant_{q}"] = (df["quadrant"] == q).astype(int)
    
    df = df.drop(columns=["quadrant"], errors="ignore")
    
    logger.info("Created spatial features: depth, distance, quadrants")
    return df


def create_pressure_gradient_features(
    df: pd.DataFrame,
    pressure_col: str = "wbhp",
    rate_col: str = "wlpr",
) -> pd.DataFrame:
    """Create pressure gradient and productivity index features.
    
    Research basis: Physics-informed approaches emphasize pressure dynamics
    
    Args:
        df: Well data
        pressure_col: Pressure column
        rate_col: Rate column
    
    Returns:
        DataFrame with pressure features
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        
        if pressure_col not in df.columns or rate_col not in df.columns:
            continue
        
        pressure = df.loc[well_mask, pressure_col].values
        rate = df.loc[well_mask, rate_col].values
        
        # Pressure gradient (time derivative)
        dp_dt = np.gradient(pressure)
        df.loc[well_mask, f"{pressure_col}_gradient"] = dp_dt
        
        # Productivity index (rate / pressure drop)
        # Assuming reservoir pressure ~200 bar (typical)
        reservoir_p = 200.0
        drawdown = reservoir_p - pressure
        drawdown = np.maximum(drawdown, 1.0)  # Avoid division by zero
        
        pi = rate / drawdown
        df.loc[well_mask, "productivity_index"] = pi
        
        # Pressure-rate coupling
        df.loc[well_mask, "pressure_rate_product"] = pressure * rate
    
    return df


def create_time_series_embeddings(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int = 12,
    n_components: int = 3,
) -> pd.DataFrame:
    """Create compressed time series representations using PCA.
    
    Research basis: "Deep insight" (2025) - dimensionality reduction improves forecasting
    
    Args:
        df: Well data
        feature_cols: Features to embed
        window: Lookback window
        n_components: Number of PCA components
    
    Returns:
        DataFrame with embedding features
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        well_data = df.loc[well_mask, feature_cols].fillna(0.0)
        
        if len(well_data) < window + n_components:
            continue
        
        # Create windowed matrix
        windows = []
        for i in range(window, len(well_data)):
            windows.append(well_data.iloc[i-window:i].values.flatten())
        
        if not windows:
            continue
        
        X = np.array(windows)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X.shape[1]))
        embeddings = pca.fit_transform(X)
        
        # Assign back to dataframe
        indices = well_data.index[window:]
        for comp in range(embeddings.shape[1]):
            df.loc[indices, f"ts_embed_{comp}"] = embeddings[:, comp]
    
    logger.info("Created %d time series embedding features", n_components)
    return df


def create_fourier_features(
    df: pd.DataFrame,
    date_col: str = "ds",
    n_frequencies: int = 3,
) -> pd.DataFrame:
    """Create Fourier features for seasonality.
    
    Research basis: "Temporal Fusion Transformer" (2024) - frequency domain helps
    
    Args:
        df: Well data
        date_col: Date column
        n_frequencies: Number of frequency components
    
    Returns:
        DataFrame with Fourier features
    """
    df = df.copy()
    
    # Extract time index
    if date_col in df.columns:
        time_numeric = (pd.to_datetime(df[date_col]) - pd.to_datetime(df[date_col].min())).dt.total_seconds()
    elif "time_idx" in df.columns:
        time_numeric = df["time_idx"]
    else:
        logger.warning("No time column found, skipping Fourier features")
        return df
    
    # Create sin/cos features for multiple frequencies
    for freq in range(1, n_frequencies + 1):
        df[f"fourier_sin_{freq}"] = np.sin(2 * np.pi * freq * time_numeric / time_numeric.max())
        df[f"fourier_cos_{freq}"] = np.cos(2 * np.pi * freq * time_numeric / time_numeric.max())
    
    logger.info("Created %d Fourier feature pairs", n_frequencies)
    return df


def create_well_vintage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on well age/maturity.
    
    Args:
        df: Well data
    
    Returns:
        DataFrame with vintage features
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        
        # Time since first production
        df.loc[well_mask, "well_age_months"] = df.loc[well_mask, "time_idx"] if "time_idx" in df.columns else np.arange(well_mask.sum())
        
        # Well age buckets
        age = df.loc[well_mask, "well_age_months"].values if "well_age_months" in df.columns else np.arange(well_mask.sum())
        df.loc[well_mask, "is_new_well"] = (age < 12).astype(int)
        df.loc[well_mask, "is_mature_well"] = (age >= 36).astype(int)
    
    return df


def create_rolling_statistics(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: List[int] = [3, 6, 12],
) -> pd.DataFrame:
    """Create rolling statistics for key features.
    
    Research basis: "TimeMixer" (2024) - multiscale features improve accuracy
    
    Args:
        df: Well data
        feature_cols: Features to compute statistics for
        windows: Rolling window sizes
    
    Returns:
        DataFrame with rolling statistics
    """
    df = df.copy()
    
    for well in df["well"].unique():
        well_mask = df["well"] == well
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            series = df.loc[well_mask, col]
            
            for window in windows:
                if len(series) < window:
                    continue
                
                # Rolling mean
                df.loc[well_mask, f"{col}_ma{window}"] = series.rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df.loc[well_mask, f"{col}_std{window}"] = series.rolling(
                    window=window, min_periods=2
                ).std().fillna(0.0)
                
                # Rolling min/max
                df.loc[well_mask, f"{col}_min{window}"] = series.rolling(
                    window=window, min_periods=1
                ).min()
                
                df.loc[well_mask, f"{col}_max{window}"] = series.rolling(
                    window=window, min_periods=1
                ).max()
    
    return df


def create_cumulative_injection_features(
    prod_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    injection_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create cumulative injection features by well pair.
    
    Research basis: CRM models emphasize cumulative injection importance
    
    Args:
        prod_df: Producer data
        inj_df: Injector data
        injection_summary: Pair summary from build_injection_lag_features
    
    Returns:
        DataFrame with cumulative injection features
    """
    prod_df = prod_df.copy()
    
    if injection_summary.empty:
        return prod_df
    
    # Group by producer
    for prod_id in prod_df["well"].unique():
        prod_mask = prod_df["well"] == prod_id
        
        # Get influencing injectors
        pairs = injection_summary[injection_summary["prod_id"] == str(prod_id)]
        
        if pairs.empty:
            continue
        
        # Sum weighted cumulative injection
        total_weighted_cum_inj = np.zeros(prod_mask.sum())
        
        for _, pair in pairs.iterrows():
            inj_id = pair["inj_id"]
            weight = pair["weight"]
            lag = pair["lag"]
            
            inj_mask = inj_df["well"] == inj_id
            if inj_mask.sum() == 0:
                continue
            
            # Get cumulative injection
            if "wwit" in inj_df.columns:
                cum_inj = inj_df.loc[inj_mask, "wwit"].values
            else:
                cum_inj = inj_df.loc[inj_mask, "wwir"].cumsum().values
            
            # Shift by lag
            if lag > 0 and lag < len(cum_inj):
                cum_inj_lagged = np.concatenate([np.zeros(lag), cum_inj[:-lag]])
            else:
                cum_inj_lagged = cum_inj
            
            # Align lengths
            min_len = min(len(cum_inj_lagged), len(total_weighted_cum_inj))
            total_weighted_cum_inj[:min_len] += weight * cum_inj_lagged[:min_len]
        
        prod_df.loc[prod_mask, "cumulative_weighted_injection"] = total_weighted_cum_inj
    
    return prod_df
