from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from .config import PipelineConfig
    from .conformal import apply_conformal_intervals
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
    from conformal import apply_conformal_intervals
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


def _dist_runtime_from_env() -> tuple[int, int, bool]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    return rank, world_size, rank == 0


def _wait_for_file(path: Path, timeout_s: int = 7200, poll_s: float = 2.0) -> None:
    start = time.perf_counter()
    while not path.exists():
        if (time.perf_counter() - start) > timeout_s:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(poll_s)


def _ddp_frames_cache_path(output_dir: Path) -> Path:
    return output_dir / ".ddp_frames.pkl"


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
        attention_alpha = inj_summary.attrs.get("attention_alpha_timeseries")
        if isinstance(attention_alpha, pd.DataFrame) and not attention_alpha.empty:
            alpha_path = output_dir / "alpha_timeseries.parquet"
            try:
                attention_alpha.to_parquet(alpha_path, index=False)
            except Exception:
                alpha_path = output_dir / "alpha_timeseries.csv"
                attention_alpha.to_csv(alpha_path, index=False)
            metadata["attention_alpha_timeseries_path"] = str(alpha_path)
            metadata["attention_alpha_timeseries_rows"] = int(len(attention_alpha))
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
        conformal_profile = cv_results.get("conformal_profile")
        if isinstance(conformal_profile, dict) and conformal_profile.get("grouped_profiles"):
            conformal_path = output_dir / "grouped_conformal_profile.json"
            with open(conformal_path, "w", encoding="utf-8") as handle:
                json.dump(conformal_profile, handle, indent=2)
            metadata["grouped_conformal_profile_path"] = str(conformal_path)

    attr_artifacts = {
        "graph_fusion_weights": ("graph_fusion_weights.parquet", pred_df.attrs.get("graph_fusion_weights")),
        "edge_allocations": ("edge_allocations.parquet", pred_df.attrs.get("edge_allocations")),
        "scenario_edge_deltas": ("scenario_edge_deltas.parquet", pred_df.attrs.get("scenario_edge_deltas")),
        "well_event_metrics": ("well_event_metrics.csv", pred_df.attrs.get("well_event_metrics")),
        "physics_history": ("physics_history.csv", pred_df.attrs.get("physics_history")),
    }
    for key, (filename, payload) in attr_artifacts.items():
        if not isinstance(payload, pd.DataFrame) or payload.empty:
            continue
        path = output_dir / filename
        if path.suffix == ".parquet":
            try:
                payload.to_parquet(path, index=False)
            except Exception:
                path = path.with_suffix(".csv")
                payload.to_csv(path, index=False)
        else:
            payload.to_csv(path, index=False)
        metadata[f"{key}_path"] = str(path)

    training_summary = pred_df.attrs.get("training_summary")
    if isinstance(training_summary, dict) and training_summary:
        training_path = output_dir / "stgnn_training_summary.json"
        with open(training_path, "w", encoding="utf-8") as handle:
            json.dump(training_summary, handle, indent=2, default=str)
        metadata["stgnn_training_summary_path"] = str(training_path)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Artifacts saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WLPR forecasting pipeline using Chronos-2, XLinear, or STGNN PyG")
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
    parser.add_argument(
        "--model",
        type=str,
        default="chronos2",
        choices=["chronos2", "xlinear", "stgnn_pyg"],
        help="Forecasting model to use (default: chronos2)",
    )
    parser.add_argument(
        "--disable-conformal",
        action="store_true",
        help="Disable conformal prediction intervals",
    )
    parser.add_argument(
        "--disable-cv",
        action="store_true",
        help="Disable walk-forward cross-validation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override number of walk-forward CV folds",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=None,
        help="Conformal miscoverage level alpha (default from config: 0.1 = 90%% PI)",
    )
    parser.add_argument(
        "--conformal-method",
        type=str,
        default=None,
        choices=["icp", "wcp_exp", "wcp_linear"],
        help="Conformal calibration method",
    )
    parser.add_argument(
        "--conformal-exp-decay",
        type=float,
        default=None,
        help="Exponential decay for WCP-exp (recent residuals weighted higher)",
    )
    parser.add_argument(
        "--conformal-min-samples",
        type=int,
        default=None,
        help="Minimum residual samples per horizon before fallback to global epsilon",
    )
    parser.add_argument(
        "--conformal-global",
        action="store_true",
        help="Use one global conformal epsilon for all horizon steps",
    )
    parser.add_argument(
        "--disable-inj-attention",
        action="store_true",
        help="Disable attention-based injector->producer aggregation features",
    )
    parser.add_argument(
        "--inj-attention-target-mode",
        type=str,
        default=None,
        choices=["delta", "level"],
        help="Training target for attention weights: delta WLPR or level WLPR",
    )
    parser.add_argument(
        "--inj-attention-steps",
        type=int,
        default=None,
        help="max_iter for sklearn attention fitting",
    )
    parser.add_argument(
        "--inj-attention-prior-strength",
        type=float,
        default=None,
        help="Regularization strength toward kernel-based prior weights",
    )
    parser.add_argument(
        "--inj-attention-smooth-strength",
        type=float,
        default=None,
        help="Temporal smoothness regularization for causal_stage_geo alpha(t)",
    )
    parser.add_argument(
        "--inj-attention-future-anchor-strength",
        type=float,
        default=None,
        help="Anchor strength to train-last alpha for future horizon",
    )
    parser.add_argument(
        "--inj-attention-geo-condition-strength",
        type=float,
        default=None,
        help="Strength of geo-conditioned prior blending into attention weights",
    )
    parser.add_argument(
        "--disable-inj-attention-stage-adaptive",
        action="store_true",
        help="Disable stage-adaptive gating in causal_stage_geo attention",
    )
    parser.add_argument(
        "--stgnn-max-epochs",
        type=int,
        default=None,
        help="Override max epochs for STGNN PyG training",
    )
    parser.add_argument(
        "--stgnn-early-stop-patience",
        type=int,
        default=None,
        help="Override early stopping patience for STGNN PyG training",
    )
    parser.add_argument(
        "--stgnn-batch-size",
        type=int,
        default=None,
        help="Override mini-batch size for STGNN PyG training",
    )
    parser.add_argument(
        "--stgnn-num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers for STGNN PyG training",
    )
    parser.add_argument(
        "--stgnn-use-amp",
        action="store_true",
        help="Enable automatic mixed precision (bf16 autocast on CUDA) for STGNN PyG training",
    )
    parser.add_argument(
        "--stgnn-variant",
        type=str,
        default=None,
        choices=["single_relation_multitask", "single_relation_multitask_noalloc", "multitask_nograph", "legacy_multigraph"],
        help="STGNN PyG architecture variant",
    )
    parser.add_argument(
        "--physics-warmup-epochs",
        type=int,
        default=None,
        help="Override warmup epochs before staged physics regularization reaches full weight",
    )
    parser.add_argument(
        "--physics-weight-max",
        type=float,
        default=None,
        help="Override maximum staged physics loss weight",
    )
    parser.add_argument(
        "--physics-lambda-crm",
        type=float,
        default=None,
        help="Override CRM residual coefficient inside physics loss",
    )
    parser.add_argument(
        "--physics-lambda-simplex",
        type=float,
        default=None,
        help="Override simplex regularization coefficient for edge allocations",
    )
    parser.add_argument(
        "--physics-lambda-nonneg",
        type=float,
        default=None,
        help="Override non-negativity regularization coefficient for physics loss",
    )
    parser.add_argument(
        "--physics-lambda-smoothness",
        type=float,
        default=None,
        help="Override temporal smoothness regularization coefficient for physics loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_rank, dist_world_size, dist_is_main = _dist_runtime_from_env()

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
    logger.info("Starting WLPR Forecasting Pipeline (%s)", args.model.upper())
    logger.info("Timestamp: %s", datetime.now().isoformat())
    if dist_world_size > 1:
        logger.info("Distributed launch detected: rank=%d, world_size=%d", dist_rank, dist_world_size)
    logger.info("=" * 80)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")
    coords_source = args.coords_path if args.coords_path else args.distances_path
    if coords_source is None or not coords_source.exists():
        raise FileNotFoundError(f"Coordinate/distance file not found at {coords_source}")

    config = PipelineConfig()
    config.model_type = args.model
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
    if args.disable_conformal:
        config.conformal_enabled = False
    if args.disable_cv:
        config.cv_enabled = False
    if args.cv_folds is not None:
        config.cv_folds = int(args.cv_folds)
    if args.conformal_alpha is not None:
        config.conformal_alpha = float(args.conformal_alpha)
    if args.conformal_method:
        config.conformal_method = str(args.conformal_method)
    if args.conformal_exp_decay is not None:
        config.conformal_exp_decay = float(args.conformal_exp_decay)
    if args.conformal_min_samples is not None:
        config.conformal_min_samples = int(args.conformal_min_samples)
    if args.conformal_global:
        config.conformal_per_horizon = False
    if args.disable_inj_attention:
        config.inj_attention_enabled = False
    if args.inj_attention_target_mode:
        config.inj_attention_target_mode = str(args.inj_attention_target_mode)
    if args.inj_attention_steps is not None:
        config.inj_attention_steps = int(args.inj_attention_steps)
    if args.inj_attention_prior_strength is not None:
        config.inj_attention_prior_strength = float(args.inj_attention_prior_strength)
    if args.inj_attention_smooth_strength is not None:
        config.inj_attention_smooth_strength = float(args.inj_attention_smooth_strength)
    if args.inj_attention_future_anchor_strength is not None:
        config.inj_attention_future_anchor_strength = float(args.inj_attention_future_anchor_strength)
    if args.inj_attention_geo_condition_strength is not None:
        config.inj_attention_geo_condition_strength = float(args.inj_attention_geo_condition_strength)
    if args.disable_inj_attention_stage_adaptive:
        config.inj_attention_stage_adaptive = False
    if args.stgnn_max_epochs is not None:
        config.stgnn_max_epochs = int(args.stgnn_max_epochs)
    if args.stgnn_early_stop_patience is not None:
        config.stgnn_early_stop_patience = int(args.stgnn_early_stop_patience)
    if args.stgnn_batch_size is not None:
        config.stgnn_batch_size = int(args.stgnn_batch_size)
    if args.stgnn_num_workers is not None:
        config.stgnn_num_workers = int(args.stgnn_num_workers)
    if args.stgnn_use_amp:
        config.stgnn_use_amp = True
    if args.stgnn_variant:
        config.stgnn_variant = str(args.stgnn_variant)
    if args.physics_warmup_epochs is not None:
        config.physics_warmup_epochs = int(args.physics_warmup_epochs)
    if args.physics_weight_max is not None:
        config.physics_weight_max = float(args.physics_weight_max)
    if args.physics_lambda_crm is not None:
        config.physics_lambda_crm = float(args.physics_lambda_crm)
    if args.physics_lambda_simplex is not None:
        config.physics_lambda_simplex = float(args.physics_lambda_simplex)
    if args.physics_lambda_nonneg is not None:
        config.physics_lambda_nonneg = float(args.physics_lambda_nonneg)
    if args.physics_lambda_smoothness is not None:
        config.physics_lambda_smoothness = float(args.physics_lambda_smoothness)

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
    if args.enable_mlflow and not dist_is_main:
        logger.info("Skipping MLflow on non-zero DDP rank %d", dist_rank)
    if args.enable_mlflow and dist_is_main:
        tracker = create_tracker(
            config=config,
            run_name=f"wlpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=args.mlflow_uri,
        )
        if tracker:
            tracker.start_run()
            tracker.log_config(config)
            model_label = {"chronos2": "Chronos-2", "xlinear": "XLinear", "stgnn_pyg": "STGNN PyG"}.get(args.model, args.model)
            tracker.set_tags({
                "pipeline": "wlpr_forecasting",
                "model": model_label,
            })
            logger.info("MLflow tracking enabled")

    ddp_graph_mode = dist_world_size > 1 and args.model == "stgnn_pyg"
    frames_cache_path = _ddp_frames_cache_path(output_dir)
    coords = None
    distances = None
    cv_results = {}

    if ddp_graph_mode and not dist_is_main:
        logger.info("Rank %d waiting for rank 0 prepared frames: %s", dist_rank, frames_cache_path)
        _wait_for_file(frames_cache_path)
        with open(frames_cache_path, "rb") as handle:
            payload = pickle.load(handle)
        frames = payload["frames"]
        coords = payload.get("coords")
        distances = payload.get("distances")
        logger.info("Rank %d loaded cached frames from rank 0", dist_rank)
    else:
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
        if ddp_graph_mode and dist_is_main:
            with open(frames_cache_path, "wb") as handle:
                pickle.dump({"frames": frames, "coords": coords, "distances": distances}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Rank 0 cached prepared frames for DDP: %s", frames_cache_path)

    if not (ddp_graph_mode and not dist_is_main):
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
    if dist_world_size > 1 and not dist_is_main and args.model == "stgnn_pyg":
        logger.info("Rank %d finished distributed STGNN work; rank 0 will run final evaluation and artifact export.", dist_rank)
        if tracker:
            tracker.end_run()
        return
    conformal_profile = cv_results.get("conformal_profile") if isinstance(cv_results, dict) else None
    if config.conformal_enabled and conformal_profile:
        preds = apply_conformal_intervals(preds, conformal_profile, horizon=config.horizon)
        logger.info(
            "Applied conformal intervals to test forecast: method=%s, alpha=%.3f",
            conformal_profile.get("method"),
            float(conformal_profile.get("alpha", config.conformal_alpha)),
        )
    elif config.conformal_enabled:
        logger.warning("Conformal enabled, but no calibration profile found in CV results.")

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
        for optional_name in [
            "graph_fusion_weights.parquet",
            "graph_fusion_weights.csv",
            "edge_allocations.parquet",
            "edge_allocations.csv",
            "scenario_edge_deltas.parquet",
            "scenario_edge_deltas.csv",
            "grouped_conformal_profile.json",
            "well_event_metrics.csv",
            "physics_history.csv",
            "stgnn_training_summary.json",
        ]:
            optional_path = output_dir / optional_name
            if optional_path.exists():
                tracker.log_artifact(optional_path, "graph_artifacts")

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
