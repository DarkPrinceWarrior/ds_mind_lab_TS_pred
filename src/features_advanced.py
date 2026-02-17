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

    df["coord_x"] = df["well"].map(lambda w: coords_dict.get(str(w), {}).get("x", 0.0))
    df["coord_y"] = df["well"].map(lambda w: coords_dict.get(str(w), {}).get("y", 0.0))
    df["coord_z"] = df["well"].map(lambda w: coords_dict.get(str(w), {}).get("z", 0.0))

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
    train_cutoff: Optional[pd.Timestamp] = None,
    date_col: str = "ds",
) -> pd.DataFrame:
    """Create compressed time series representations using PCA.
    
    Research basis: "Deep insight" (2025) - dimensionality reduction improves forecasting
    
    Args:
        df: Well data
        feature_cols: Features to embed
        window: Lookback window
        n_components: Number of PCA components
        train_cutoff: Optional cutoff date to fit PCA only on training windows
        date_col: Date column name
    
    Returns:
        DataFrame with embedding features
    """
    df = df.copy()
    
    if date_col not in df.columns:
        logger.warning("No date column '%s' found, skipping time series embeddings", date_col)
        return df

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
        
        X_train = X
        if train_cutoff is not None:
            well_dates = pd.to_datetime(df.loc[well_mask, date_col]).to_numpy()
            window_end_dates = well_dates[window:]
            cutoff = np.datetime64(train_cutoff)
            train_mask = window_end_dates <= cutoff
            X_train = X[train_mask]
            if X_train.shape[0] < max(n_components, 2):
                logger.warning(
                    "Well %s: not enough training windows for PCA (have %d), skipping embeddings",
                    well,
                    X_train.shape[0],
                )
                continue

        # Apply PCA (fit on training windows only, then transform all windows)
        pca = PCA(n_components=min(n_components, X_train.shape[1]))
        pca.fit(X_train)
        embeddings = pca.transform(X)
        
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


