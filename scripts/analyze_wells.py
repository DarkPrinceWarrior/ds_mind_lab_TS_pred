"""Analyze problematic wells to understand why R² is negative."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig
from src.wlpr_pipeline import load_raw_data, load_coordinates, load_distance_matrix, prepare_model_frames

config = PipelineConfig(model_type="xlinear")
raw = load_raw_data(Path("MODEL_23.09.25.csv"), validate=False)
coords = load_coordinates(Path("Distance.xlsx"))
distances = load_distance_matrix(Path("Distance.xlsx"))
frames = prepare_model_frames(raw, coords, config, distances=distances)
train_df = frames["train_df"]

# Also load predictions
preds = pd.read_csv("artifacts_xlinear_attn_causal_stage_geo_tuned/wlpr_predictions.csv")

print("=" * 80)
print("PER-WELL STATISTICS")
print("=" * 80)

wells = sorted(train_df["unique_id"].unique())
for well in wells:
    wd = train_df[train_df["unique_id"] == well]
    wp = preds[preds["unique_id"].astype(str) == str(well)] if "unique_id" in preds.columns else pd.DataFrame()
    
    y = wd["y"]
    print(f"\nWell {well}:")
    print(f"  Train samples: {len(wd)}")
    print(f"  Target (y) stats: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")
    print(f"  CV (std/mean):    {y.std()/max(y.mean(), 0.01):.4f}")
    
    # Check last 12 months trend
    last12 = y.iloc[-12:] if len(y) >= 12 else y
    print(f"  Last 12mo mean:   {last12.mean():.2f}, std={last12.std():.2f}")
    
    # Test predictions
    if len(wp) > 0 and "y" in wp.columns and "y_hat" in wp.columns:
        test_y = wp["y"].values
        test_yhat = wp["y_hat"].values
        mae = np.abs(test_y - test_yhat).mean()
        test_range = test_y.max() - test_y.min()
        print(f"  Test actual:      mean={test_y.mean():.2f}, range={test_range:.2f}")
        print(f"  Test MAE:         {mae:.4f}")
        print(f"  MAE/range ratio:  {mae/max(test_range, 0.01):.4f}")
        print(f"  MAE/mean ratio:   {mae/max(abs(test_y.mean()), 0.01):.4f}")

# Highlight problematic
print("\n" + "=" * 80)
print("PROBLEMATIC WELLS SUMMARY")
print("=" * 80)
print("Wells with negative R²: 14, 23, 35, 38, 45, 50")
print("These wells have VERY small variance in test period.")
print("R² = 1 - SS_res/SS_tot. When SS_tot (variance) is tiny,")
print("even small absolute errors cause massive negative R².")
print()

# Check scale differences
print("=" * 80)
print("SCALE COMPARISON ACROSS WELLS")
print("=" * 80)
means = []
for well in wells:
    wd = train_df[train_df["unique_id"] == well]
    means.append({"well": well, "mean_y": wd["y"].mean(), "std_y": wd["y"].std()})
means_df = pd.DataFrame(means).sort_values("mean_y", ascending=False)
print(means_df.to_string(index=False))
print(f"\nMax/Min mean ratio: {means_df['mean_y'].max() / max(means_df['mean_y'].min(), 0.01):.1f}x")
