from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from volve_config import VolveConfig
from volve_pipeline import (
    load_volve_data,
    prepare_volve_frames,
    run_walk_forward_validation,
    train_and_forecast,
    evaluate_predictions,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)


def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        elif value is not None:
            print(f"  {key:25s}: {value}")
    print(f"{'='*60}\n")


def generate_forecast_pdf(
    merged: pd.DataFrame,
    metrics: dict,
    output_dir: Path,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wells = sorted(merged["unique_id"].unique())
        n_wells = len(wells)
        fig, axes = plt.subplots(n_wells, 1, figsize=(14, 4 * n_wells), squeeze=False)
        for i, well in enumerate(wells):
            ax = axes[i, 0]
            wd = merged[merged["unique_id"] == well].sort_values("ds")
            ax.plot(wd["ds"], wd["y"], "o-", label="Actual", markersize=2, linewidth=0.8)
            ax.plot(wd["ds"], wd["y_hat"], "s--", label="Predicted", markersize=2, linewidth=0.8)
            if "cp_lo" in wd.columns and "cp_hi" in wd.columns:
                ax.fill_between(wd["ds"], wd["cp_lo"], wd["cp_hi"], alpha=0.2, label="90% CI")
            well_metrics = metrics.get("by_well", {}).get(well, {})
            mae_val = well_metrics.get("mae", "N/A")
            r2_val = well_metrics.get("r2", "N/A")
            title = f"{well}"
            if isinstance(mae_val, float):
                title += f" | MAE={mae_val:.1f}"
            if isinstance(r2_val, float):
                title += f" | R\u00b2={r2_val:.4f}"
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8)
            ax.set_ylabel("Oil Rate (Sm\u00b3/d)")
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("Date")
        fig.suptitle("Volve - XLinear Daily Oil Rate Forecast", fontsize=14, y=1.01)
        fig.tight_layout()
        pdf_path = output_dir / "volve_forecasts.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved forecast PDF: %s", pdf_path)
        return pdf_path
    except Exception as exc:
        logger.warning("Could not generate forecast PDF: %s", exc)
        return output_dir / "volve_forecasts.pdf"


def generate_full_history_pdf(
    frames: dict,
    merged: pd.DataFrame,
    output_dir: Path,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_df = frames["train_df"]
        wells = sorted(train_df["unique_id"].unique())
        n_wells = len(wells)
        fig, axes = plt.subplots(n_wells, 1, figsize=(16, 4 * n_wells), squeeze=False)
        for i, well in enumerate(wells):
            ax = axes[i, 0]
            wt = train_df[train_df["unique_id"] == well].sort_values("ds")
            wm = merged[merged["unique_id"] == well].sort_values("ds")
            ax.plot(wt["ds"], wt["y"], "-", label="Train", alpha=0.6, linewidth=0.5)
            ax.plot(wm["ds"], wm["y"], "o-", label="Test actual", markersize=2, linewidth=0.8)
            ax.plot(wm["ds"], wm["y_hat"], "s--", label="Test predicted", markersize=2, linewidth=0.8)
            if "cp_lo" in wm.columns and "cp_hi" in wm.columns:
                ax.fill_between(wm["ds"], wm["cp_lo"], wm["cp_hi"], alpha=0.2, label="90% CI")
            test_start = frames.get("test_start")
            if test_start is not None:
                ax.axvline(test_start, color="red", linestyle=":", alpha=0.5, label="Test start")
            ax.set_title(well)
            ax.legend(loc="best", fontsize=8)
            ax.set_ylabel("Oil Rate (Sm\u00b3/d)")
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("Date")
        fig.suptitle("Volve - Full Oil Production History + Forecast", fontsize=14, y=1.01)
        fig.tight_layout()
        pdf_path = output_dir / "volve_full_history.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved full history PDF: %s", pdf_path)
        return pdf_path
    except Exception as exc:
        logger.warning("Could not generate full history PDF: %s", exc)
        return output_dir / "volve_full_history.pdf"


def save_artifacts(
    pred_df: pd.DataFrame,
    metrics: dict,
    frames: dict,
    config: VolveConfig,
    output_dir: Path,
    cv_results: dict = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "volve_predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    metadata = {
        "config": asdict(config),
        "target_wells": frames["target_wells"],
        "test_start": frames["test_start"].strftime("%Y-%m-%d"),
        "train_cutoff": frames["train_cutoff"].strftime("%Y-%m-%d")
        if isinstance(frames.get("train_cutoff"), pd.Timestamp)
        else None,
        "train_rows": int(len(frames["train_df"])),
        "test_rows": int(len(frames["test_df"])),
    }
    if cv_results:
        cv_path = output_dir / "cv_metrics.json"
        with open(cv_path, "w") as f:
            json.dump(cv_results, f, indent=2, default=str)
        metadata["cv_aggregate"] = cv_results.get("aggregate")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Artifacts saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volve daily oil rate forecasting (XLinear)")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent / "Volve production data.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts_volve")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--disable-cv", action="store_true")
    parser.add_argument("--disable-conformal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    start_time = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Volve Daily Oil Rate Forecasting Pipeline (XLinear)")
    logger.info("Timestamp: %s", datetime.now().isoformat())
    logger.info("=" * 60)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    config = VolveConfig()
    if args.horizon is not None:
        config.horizon = args.horizon
        config.val_horizon = args.horizon
    if args.input_size is not None:
        config.input_size = args.input_size
    if args.max_steps is not None:
        config.xlinear_max_steps = args.max_steps
    if args.disable_cv:
        config.cv_enabled = False
    if args.disable_conformal:
        config.conformal_enabled = False

    raw_df = load_volve_data(args.data_path)
    frames = prepare_volve_frames(raw_df, config)

    cv_results = run_walk_forward_validation(frames, config)
    if cv_results and cv_results.get("aggregate"):
        print_metrics(cv_results["aggregate"], "Walk-Forward CV Aggregate Metrics")

    preds = train_and_forecast(frames, config)

    conformal_profile = cv_results.get("conformal_profile") if isinstance(cv_results, dict) else None
    if config.conformal_enabled and conformal_profile:
        from conformal import apply_conformal_intervals
        preds = apply_conformal_intervals(preds, conformal_profile, horizon=config.horizon)
        logger.info("Applied conformal intervals to forecast")

    metrics, merged = evaluate_predictions(preds, frames["test_df"], frames["train_df"])
    print_metrics(metrics["overall"], "Overall Test Metrics")
    for well, wm in metrics.get("by_well", {}).items():
        print_metrics(wm, f"Well: {well}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_forecast_pdf(merged, metrics, output_dir)
    generate_full_history_pdf(frames, merged, output_dir)
    save_artifacts(preds, metrics, frames, config, output_dir, cv_results=cv_results)

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("Artifacts: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
