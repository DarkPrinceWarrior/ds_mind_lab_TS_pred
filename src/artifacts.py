from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from .config import PipelineConfig
    from .data_validation import validate_and_report
    from .logging_config import setup_logging
    from .metrics_extended import calculate_metrics_by_horizon, print_metrics_summary
    from .mlflow_tracking import create_tracker
    from .caching import CacheManager
    from .wlpr_pipeline import (
        load_raw_data,
        load_coordinates,
        load_distance_matrix,
        prepare_model_frames,
        run_walk_forward_validation,
        train_and_forecast,
        evaluate_predictions,
    )
    from .visualization import (
        generate_forecast_pdf,
        generate_full_history_pdf,
        generate_residuals_pdf,
    )
    from .visualization_features import generate_feature_analysis_pdf
except ImportError:  # pragma: no cover
    from config import PipelineConfig
    from data_validation import validate_and_report
    from logging_config import setup_logging
    from metrics_extended import calculate_metrics_by_horizon, print_metrics_summary
    from mlflow_tracking import create_tracker
    from caching import CacheManager
    from wlpr_pipeline import (
        load_raw_data,
        load_coordinates,
        load_distance_matrix,
        prepare_model_frames,
        run_walk_forward_validation,
        train_and_forecast,
        evaluate_predictions,
    )
    from visualization import (
        generate_forecast_pdf,
        generate_full_history_pdf,
        generate_residuals_pdf,
    )
    from visualization_features import generate_feature_analysis_pdf

logger = logging.getLogger(__name__)


def save_artifacts(
    pred_df: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    frames: Dict[str, pd.DataFrame],
    config: PipelineConfig,
    output_dir: Path,
    pdf_paths: Optional[Dict[str, str]] = None,
    cv_results: Optional[Dict[str, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "wlpr_predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    metadata = {
        "config": asdict(config),
        "target_wells": frames["target_wells"],
        "test_start": frames["test_start"].strftime("%Y-%m-%d"),
        "train_cutoff": frames.get("train_cutoff").strftime("%Y-%m-%d") if isinstance(frames.get("train_cutoff"), pd.Timestamp) else None,
        "train_rows": int(len(frames["train_df"])),
        "test_rows": int(len(frames["test_df"])),
    }
    kernel_metadata = frames.get("kernel_metadata")
    if kernel_metadata:
        metadata["kernel_selection"] = kernel_metadata
    inj_summary = frames.get("injection_summary")
    if isinstance(inj_summary, pd.DataFrame) and not inj_summary.empty:
        summary_path = output_dir / "injection_lag_summary.csv"
        inj_summary.to_csv(summary_path, index=False)
        metadata["injection_summary_path"] = str(summary_path)
        metadata["injection_pairs"] = int(len(inj_summary))
    if pdf_paths:
        metadata["pdf_reports"] = {key: str(value) for key, value in pdf_paths.items()}
    if cv_results:
        cv_path = output_dir / "cv_metrics.json"
        with open(cv_path, "w", encoding="utf-8") as handle:
            json.dump(cv_results, handle, indent=2)
        metadata["walk_forward_cv"] = {
            "folds": len(cv_results.get("folds", [])),
            "aggregate": cv_results.get("aggregate"),
            "details_path": str(cv_path),
        }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Artifacts saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WLPR forecasting pipeline using Chronos-2")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("MODEL_23.09.25.csv"),
        help="Path to the monthly well dataset",
    )
    parser.add_argument(
        "--distances-path",
        type=Path,
        default=Path("Distance.xlsx"),
        help="Path to combined distances + coordinates file (.xlsx)",
    )
    parser.add_argument(
        "--coords-path",
        type=Path,
        default=None,
        help="(legacy) Separate coordinates file. If omitted, coordinates are read from --distances-path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store predictions and metrics",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching of intermediate results",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--chronos-model",
        type=str,
        default=None,
        help="Chronos-2 hub model name (e.g., amazon/chronos-2 or autogluon/chronos-2-small)",
    )
    parser.add_argument(
        "--chronos-revision",
        type=str,
        default=None,
        help="Chronos-2 hub model revision (branch, tag, commit)",
    )
    parser.add_argument(
        "--chronos-local-dir",
        type=Path,
        default=None,
        help="Local directory for Chronos-2 model cache",
    )
    parser.add_argument(
        "--chronos-input-len",
        type=int,
        default=None,
        help="Chronos-2 input chunk length (defaults to input_size)",
    )
    parser.add_argument(
        "--chronos-output-len",
        type=int,
        default=None,
        help="Chronos-2 output chunk length (defaults to horizon)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    setup_logging(
        log_dir=log_dir,
        level=args.log_level,
        console=True,
        file_logging=True,
        rotation="size",
        colored=True,
    )

    start_time = time.perf_counter()
    logger.info("=" * 80)
    logger.info("Starting WLPR Forecasting Pipeline (Chronos-2)")
    logger.info("Timestamp: %s", datetime.now().isoformat())
    logger.info("=" * 80)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")
    coords_source = args.coords_path if args.coords_path else args.distances_path
    if coords_source is None or not coords_source.exists():
        raise FileNotFoundError(f"Coordinate/distance file not found at {coords_source}")

    config = PipelineConfig()
    if args.chronos_model:
        config.chronos_hub_model_name = args.chronos_model
    if args.chronos_revision:
        config.chronos_hub_model_revision = args.chronos_revision
    if args.chronos_local_dir:
        config.chronos_local_dir = str(args.chronos_local_dir)
    if args.chronos_input_len:
        config.chronos_input_chunk_length = int(args.chronos_input_len)
    if args.chronos_output_len:
        config.chronos_output_chunk_length = int(args.chronos_output_len)

    if not args.disable_cache:
        try:
            from . import wlpr_pipeline as _pipeline_mod
        except ImportError:  # pragma: no cover
            import wlpr_pipeline as _pipeline_mod  # type: ignore[no-redef]
        cache = CacheManager(cache_dir=output_dir / ".cache", enabled=True)
        _pipeline_mod._cache = cache
        logger.info("Caching enabled at: %s", cache.cache_dir)
    else:
        logger.info("Caching disabled")

    tracker = None
    if args.enable_mlflow:
        tracker = create_tracker(
            config=config,
            run_name=f"wlpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=args.mlflow_uri,
        )
        if tracker:
            tracker.start_run()
            tracker.log_config(config)
            tracker.set_tags({
                "pipeline": "wlpr_forecasting",
                "model": "Chronos-2",
            })
            logger.info("MLflow tracking enabled")

    raw_df = load_raw_data(args.data_path, validate=not args.skip_validation)

    if not args.skip_validation:
        try:
            coords_temp = load_coordinates(coords_source)
            quality_report = validate_and_report(
                raw_df,
                coords=coords_temp,
                save_report=True,
                output_path=str(output_dir),
            )
            if tracker:
                tracker.log_dict(quality_report.to_dict(), "data_quality_report")
                tracker.log_metrics({
                    "data_total_rows": quality_report.total_rows,
                    "data_total_wells": quality_report.total_wells,
                    "data_duplicate_rows": quality_report.duplicate_rows,
                })
        except Exception as exc:
            logger.warning("Data validation failed: %s", exc)

    coords = load_coordinates(coords_source)
    distances = None
    if args.distances_path and args.distances_path.exists():
        distances = load_distance_matrix(args.distances_path)
    elif args.distances_path:
        logger.warning(
            "Distance file not found at %s. Falling back to coordinate-based distances.",
            args.distances_path,
        )

    frames = prepare_model_frames(raw_df, coords, config, distances=distances)

    cv_results = run_walk_forward_validation(
        frames, coords, config, distances=distances,
    )
    if tracker and cv_results:
        tracker.log_dict(cv_results, "cv_results")
        if "aggregate" in cv_results and cv_results["aggregate"]:
            tracker.log_metrics(
                {f"cv_{k}": v for k, v in cv_results["aggregate"].items() if v is not None},
                step=0,
            )

    preds = train_and_forecast(frames, config)

    metrics, merged = evaluate_predictions(
        preds, frames["test_df"], frames["train_df"], use_extended_metrics=True,
    )
    print_metrics_summary(metrics["overall"], "Overall Test Metrics")

    horizon_metrics = calculate_metrics_by_horizon(merged, config.horizon)
    logger.info("Horizon-specific metrics calculated for %d steps", len(horizon_metrics))

    if tracker:
        overall_flat = {f"test_{k}": v for k, v in metrics["overall"].items() if v is not None}
        tracker.log_metrics(overall_flat, step=1)
        for step, step_metrics in horizon_metrics.items():
            step_flat = {f"horizon_{step}_{k}": v for k, v in step_metrics.items() if v is not None}
            tracker.log_metrics(step_flat, step=step)
        tracker.log_dict(metrics, "test_metrics_detailed")

    forecast_pdf = generate_forecast_pdf(merged, metrics, output_dir)
    full_history_pdf = generate_full_history_pdf(frames, merged, metrics, config, output_dir)
    residuals_pdf = generate_residuals_pdf(merged, metrics, output_dir)
    feature_pdf = generate_feature_analysis_pdf(frames["train_df"], output_dir)
    pdf_paths = {
        "test_forecast": str(forecast_pdf),
        "full_history": str(full_history_pdf),
        "residuals": str(residuals_pdf),
        "feature_analysis": str(feature_pdf),
    }

    save_artifacts(
        preds, metrics, frames, config, output_dir,
        pdf_paths=pdf_paths, cv_results=cv_results,
    )

    if tracker:
        for name, path in pdf_paths.items():
            tracker.log_artifact(Path(path), "reports")
        tracker.log_artifact(output_dir / "metrics.json", "metrics")
        tracker.log_artifact(output_dir / "metadata.json", "metadata")
        tracker.log_artifact(output_dir / "wlpr_predictions.csv", "predictions")
        if (output_dir / "injection_lag_summary.csv").exists():
            tracker.log_artifact(output_dir / "injection_lag_summary.csv", "features")

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info("Total execution time: %.2f seconds (%.2f minutes)", elapsed, elapsed / 60)
    logger.info(
        "Overall metrics: %s",
        {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics["overall"].items() if v is not None},
    )
    if cv_results and cv_results.get("aggregate"):
        logger.info(
            "Walk-forward CV aggregate: %s",
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in cv_results["aggregate"].items() if v is not None},
        )
    logger.info("Artifacts saved to: %s", output_dir)
    logger.info("=" * 80)

    if tracker:
        tracker.end_run()
        logger.info("MLflow run completed: %s", tracker.run_id)


if __name__ == "__main__":
    main()
