"""Quick feature importance analysis for XLinear feature selection."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig
from src.wlpr_pipeline import (
    load_raw_data, load_coordinates, load_distance_matrix, prepare_model_frames
)

config = PipelineConfig(model_type="xlinear")

# Load data
raw = load_raw_data(Path("MODEL_23.09.25.csv"), validate=False)
coords = load_coordinates(Path("Distance.xlsx"))
distances = load_distance_matrix(Path("Distance.xlsx"))
frames = prepare_model_frames(raw, coords, config, distances=distances)
train_df = frames["train_df"]

print(f"\n{'='*80}")
print(f"Train shape: {train_df.shape}, Wells: {train_df['unique_id'].nunique()}")
print(f"{'='*80}")

# All feature columns
all_features = list(set(config.hist_exog + config.futr_exog))
present = [c for c in all_features if c in train_df.columns]
target = "y"

df = train_df[["unique_id", "ds", target] + present].copy()
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# === 1. Pearson correlation with target ===
print(f"\n{'='*80}")
print("1. PEARSON CORRELATION WITH TARGET (WLPR)")
print(f"{'='*80}")
corr = df[present].corrwith(df[target]).abs().sort_values(ascending=False)
for feat, val in corr.items():
    print(f"  {feat:45s} |r| = {val:.4f}")

# === 2. Mutual Information ===
from sklearn.feature_selection import mutual_info_regression
print(f"\n{'='*80}")
print("2. MUTUAL INFORMATION WITH TARGET")
print(f"{'='*80}")
X = df[present].values.astype(np.float64)
y = df[target].values.astype(np.float64)
valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
mi = mutual_info_regression(X[valid], y[valid], random_state=42, n_neighbors=5)
mi_series = pd.Series(mi, index=present).sort_values(ascending=False)
for feat, val in mi_series.items():
    print(f"  {feat:45s} MI = {val:.4f}")

# === 3. Inter-feature correlation (find redundant pairs) ===
print(f"\n{'='*80}")
print("3. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.85)")
print(f"{'='*80}")
corr_matrix = df[present].corr()
pairs = []
for i in range(len(present)):
    for j in range(i+1, len(present)):
        r = abs(corr_matrix.iloc[i, j])
        if r > 0.85:
            pairs.append((present[i], present[j], r))
pairs.sort(key=lambda x: -x[2])
for f1, f2, r in pairs:
    print(f"  {f1:35s} <-> {f2:35s} |r| = {r:.4f}")

# === 4. Per-well feature variance (find near-constant features) ===
print(f"\n{'='*80}")
print("4. FEATURES WITH NEAR-ZERO VARIANCE (std < 0.01 of mean)")
print(f"{'='*80}")
for feat in present:
    std = df[feat].std()
    mean = abs(df[feat].mean()) + 1e-10
    cv = std / mean
    if cv < 0.01 or std < 1e-6:
        print(f"  {feat:45s} std={std:.6f}, mean={df[feat].mean():.6f}, CV={cv:.4f}")

# === 5. Check which futr_exog are actually time-varying vs static ===
print(f"\n{'='*80}")
print("5. FUTR_EXOG TEMPORAL VARIANCE (static vs time-varying)")
print(f"{'='*80}")
for feat in config.futr_exog:
    if feat not in df.columns:
        print(f"  {feat:45s} [MISSING]")
        continue
    within_well_std = df.groupby("unique_id")[feat].std().mean()
    between_well_std = df.groupby("unique_id")[feat].mean().std()
    print(f"  {feat:45s} within-well-std={within_well_std:.4f}, between-well-std={between_well_std:.4f}" +
          (" ** STATIC **" if within_well_std < 0.01 else ""))

# === 6. Summary recommendation ===
print(f"\n{'='*80}")
print("6. FEATURE RANKING (combined: 0.5*norm_MI + 0.5*norm_|r|)")
print(f"{'='*80}")
corr_norm = corr / corr.max() if corr.max() > 0 else corr
mi_norm = mi_series / mi_series.max() if mi_series.max() > 0 else mi_series
combined = (0.5 * corr_norm + 0.5 * mi_norm).sort_values(ascending=False)
for i, (feat, val) in enumerate(combined.items()):
    marker = "✓" if val > 0.3 else "✗"
    print(f"  {marker} {i+1:2d}. {feat:45s} score = {val:.4f}")
