"""
Scenario: shut off injector well 34 from 2021-01 and compare forecast vs baseline.

Train on full data (identical for both scenarios).
For the shutoff scenario, zero out well 34 WWIR from 2021-01 onward in the raw
data BEFORE building injection features, so the forecast covariates reflect
the absence of injection support.
"""
from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig
from src.wlpr_pipeline import (
    load_raw_data,
    load_coordinates,
    load_distance_matrix,
    prepare_model_frames,
    train_and_forecast,
    evaluate_predictions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SHUTOFF_WELL = "34"
SHUTOFF_DATE = pd.Timestamp("2021-01-01")

DATA_PATH = Path("MODEL_23.09.25.csv")
DISTANCES_PATH = Path("Distance.xlsx")
OUTPUT_DIR = Path("artifacts/scenario_shutoff_34")


def run_scenario(raw_df: pd.DataFrame, coords, config, distances, label: str):
    """Run full pipeline and return predictions + frames."""
    logger.info("=" * 60)
    logger.info("Running scenario: %s", label)
    logger.info("=" * 60)
    frames = prepare_model_frames(raw_df, coords, config, distances=distances)
    preds = train_and_forecast(frames, config)
    metrics, merged = evaluate_predictions(
        preds, frames["test_df"], frames["train_df"],
    )
    return frames, preds, metrics, merged


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig()

    raw_df = load_raw_data(DATA_PATH)
    coords = load_coordinates(DISTANCES_PATH)
    distances = load_distance_matrix(DISTANCES_PATH)

    # --- Baseline: normal run ---
    frames_base, preds_base, metrics_base, merged_base = run_scenario(
        raw_df, coords, config, distances, "BASELINE (all injectors active)"
    )

    # --- Shutoff: zero WWIR for well 34 from 2021-01 ---
    raw_shutoff = raw_df.copy()
    mask = (
        (raw_shutoff["well"] == SHUTOFF_WELL)
        & (raw_shutoff["date"] >= SHUTOFF_DATE)
        & (raw_shutoff["type"] == "INJ")
    )
    n_zeroed = mask.sum()
    logger.info(
        "Zeroing WWIR for well %s from %s: %d rows affected",
        SHUTOFF_WELL, SHUTOFF_DATE.date(), n_zeroed,
    )
    raw_shutoff.loc[mask, "wwir"] = 0.0
    raw_shutoff.loc[mask, "wwit_diff"] = 0.0
    # Freeze cumulative injection at the value just before shutoff
    if n_zeroed > 0:
        pre_shutoff = raw_shutoff[
            (raw_shutoff["well"] == SHUTOFF_WELL)
            & (raw_shutoff["date"] < SHUTOFF_DATE)
            & (raw_shutoff["type"] == "INJ")
        ]
        if not pre_shutoff.empty:
            frozen_wwit = pre_shutoff.sort_values("date").iloc[-1]["wwit"]
            raw_shutoff.loc[mask, "wwit"] = frozen_wwit

    frames_shut, preds_shut, metrics_shut, merged_shut = run_scenario(
        raw_shutoff, coords, config, distances,
        f"SHUTOFF well {SHUTOFF_WELL} from {SHUTOFF_DATE.date()}"
    )

    # --- Compare ---
    test_start = frames_base["test_start"]
    logger.info("=" * 60)
    logger.info("COMPARISON: Forecast period %s — %s", test_start.date(),
                merged_base["ds"].max().date() if not merged_base.empty else "?")
    logger.info("=" * 60)

    # Identify which producers are connected to well 34
    pair_summary = frames_base.get("injection_summary", pd.DataFrame())
    inj_col = "inj_id" if "inj_id" in pair_summary.columns else "inj"
    prod_col = "prod_id" if "prod_id" in pair_summary.columns else "prod"
    if not pair_summary.empty and inj_col in pair_summary.columns:
        connected = pair_summary[pair_summary[inj_col].astype(str) == SHUTOFF_WELL]
        connected_prods = sorted(connected[prod_col].astype(str).unique())
    else:
        connected_prods = []
    logger.info("Producers connected to injector %s: %s", SHUTOFF_WELL, connected_prods)

    # Merge predictions
    comp = merged_base[["unique_id", "ds", "y", "y_hat"]].rename(
        columns={"y_hat": "y_hat_baseline"}
    ).merge(
        merged_shut[["unique_id", "ds", "y_hat"]].rename(
            columns={"y_hat": "y_hat_shutoff"}
        ),
        on=["unique_id", "ds"],
        how="inner",
    )
    comp["diff"] = comp["y_hat_shutoff"] - comp["y_hat_baseline"]
    comp["diff_pct"] = np.where(
        comp["y_hat_baseline"].abs() > 0.01,
        100 * comp["diff"] / comp["y_hat_baseline"],
        0.0,
    )

    # Per-well summary
    report_rows = []
    for well in sorted(comp["unique_id"].unique()):
        wc = comp[comp["unique_id"] == well]
        base_mean = wc["y_hat_baseline"].mean()
        shut_mean = wc["y_hat_shutoff"].mean()
        actual_mean = wc["y"].mean()
        diff_mean = wc["diff"].mean()
        diff_pct = 100 * diff_mean / base_mean if abs(base_mean) > 0.01 else 0
        is_connected = well in connected_prods
        report_rows.append({
            "well": well,
            "connected_to_34": is_connected,
            "actual_mean": round(actual_mean, 2),
            "baseline_mean": round(base_mean, 2),
            "shutoff_mean": round(shut_mean, 2),
            "diff_mean": round(diff_mean, 2),
            "diff_pct": round(diff_pct, 2),
        })

    report = pd.DataFrame(report_rows)
    report = report.sort_values("diff_pct")

    print("\n" + "=" * 90)
    print(f"SCENARIO: Injector well {SHUTOFF_WELL} shut off from {SHUTOFF_DATE.date()}")
    print(f"Forecast period: {test_start.date()} — {comp['ds'].max().date()}")
    print("=" * 90)
    print(f"\n{'Well':>6} | {'Connected':>9} | {'Actual':>8} | {'Baseline':>8} | {'Shutoff':>8} | {'Diff':>8} | {'Diff%':>7}")
    print("-" * 90)
    for _, r in report.iterrows():
        marker = " ***" if r["connected_to_34"] else ""
        print(f"{r['well']:>6} | {'YES' if r['connected_to_34'] else 'no':>9} | "
              f"{r['actual_mean']:>8.2f} | {r['baseline_mean']:>8.2f} | "
              f"{r['shutoff_mean']:>8.2f} | {r['diff_mean']:>+8.2f} | "
              f"{r['diff_pct']:>+7.2f}%{marker}")

    print("\n" + "-" * 90)
    overall_base = metrics_base["overall"]
    overall_shut = metrics_shut["overall"]
    print(f"Overall WMAPE baseline: {overall_base.get('wmape', 0):.4f}%")
    print(f"Overall WMAPE shutoff:  {overall_shut.get('wmape', 0):.4f}%")

    # Save
    report.to_csv(OUTPUT_DIR / "comparison_report.csv", index=False)
    comp.to_csv(OUTPUT_DIR / "detailed_comparison.csv", index=False)

    summary = {
        "scenario": f"Shutoff injector {SHUTOFF_WELL} from {SHUTOFF_DATE.date()}",
        "test_start": str(test_start.date()),
        "n_producers": len(report),
        "connected_producers": connected_prods,
        "baseline_wmape": overall_base.get("wmape"),
        "shutoff_wmape": overall_shut.get("wmape"),
        "most_affected_wells": report.head(3)[["well", "diff_pct"]].to_dict("records"),
    }
    with open(OUTPUT_DIR / "scenario_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
