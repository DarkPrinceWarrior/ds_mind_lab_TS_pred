from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .config import PipelineConfig
except ImportError:  # pragma: no cover
    from config import PipelineConfig

logger = logging.getLogger(__name__)


def _format_metrics_text(metrics: Dict[str, Dict[str, Dict[str, float]]], unique_id: str) -> str:
    per_well = metrics.get("by_well", {}).get(str(unique_id))
    if per_well is None:
        return "MAE: n/a\nWMAPE: n/a\nMASE: n/a\nRMSE: n/a"

    def _fmt(value: Optional[float], percent: bool = False) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{value:.2f}{'%' if percent else ''}"

    return "\n".join(
        [
            f"MAE: {_fmt(per_well.get('mae'))}",
            f"WMAPE: {_fmt(per_well.get('wmape'), percent=True)}",
            f"MASE: {_fmt(per_well.get('mase'))}",
            f"RMSE: {_fmt(per_well.get('rmse'))}",
        ]
    )


def generate_forecast_pdf(
    merged: pd.DataFrame, metrics: Dict[str, Dict[str, float]], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / "wlpr_forecasts.pdf"
    with PdfPages(pdf_path) as pdf:
        for unique_id, group in merged.groupby("unique_id"):
            group = group.sort_values("ds")
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            ax.plot(
                group["ds"],
                group["y"],
                label="Actual (Test)",
                marker="o",
                linewidth=1.5,
            )
            ax.plot(
                group["ds"],
                group["y_hat"],
                label="Forecast",
                marker="x",
                linewidth=1.5,
            )
            ax.set_title(f"Well {unique_id} WLPR Forecast vs Actual")
            ax.set_ylabel("WLPR (m3/day)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            ax.legend()
            metrics_text = _format_metrics_text(metrics, str(unique_id))
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def generate_full_history_pdf(
    frames: Dict[str, pd.DataFrame],
    merged: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    config: PipelineConfig,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    train_df = frames["train_df"][["unique_id", "ds", "y"]].copy()
    test_df = frames["test_df"][["unique_id", "ds", "y"]].copy()
    full_df = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    test_start = pd.Timestamp(frames["test_start"])
    val_offset = pd.DateOffset(months=config.val_horizon)
    pdf_path = output_dir / "wlpr_full_history.pdf"
    with PdfPages(pdf_path) as pdf:
        for unique_id in frames["target_wells"]:
            series = full_df[full_df["unique_id"] == unique_id]
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            ax.plot(
                series["ds"],
                series["y"],
                label="Actual WLPR",
                color="black",
                linewidth=1.4,
            )
            forecast = merged[merged["unique_id"] == unique_id]
            if not forecast.empty:
                ax.plot(
                    forecast["ds"],
                    forecast["y_hat"],
                    label="Forecast (Test)",
                    marker="x",
                    linewidth=1.5,
                    color="tab:orange",
                )
            train_start = series["ds"].min()
            test_end = series["ds"].max()
            val_start = max(train_start, test_start - val_offset)
            if val_start > test_start:
                val_start = test_start
            if train_start < val_start:
                ax.axvspan(train_start, val_start, alpha=0.08, color="tab:blue", label="Train")
            if val_start < test_start:
                ax.axvspan(val_start, test_start, alpha=0.08, color="tab:green", label="Validation")
            if test_start < test_end:
                ax.axvspan(test_start, test_end, alpha=0.08, color="tab:red", label="Test")
            ax.set_title(f"Well {unique_id} WLPR History (Train/Val/Test)")
            ax.set_ylabel("WLPR (m3/day)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            ax.legend()
            metrics_text = _format_metrics_text(metrics, str(unique_id))
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def generate_residuals_pdf(
    merged: pd.DataFrame, metrics: Dict[str, Dict[str, float]], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    residuals = merged.copy()
    residuals["residual"] = residuals["y"] - residuals["y_hat"]
    pdf_path = output_dir / "wlpr_residuals.pdf"
    with PdfPages(pdf_path) as pdf:
        for unique_id, group in residuals.groupby("unique_id"):
            group = group.sort_values("ds")
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.plot(
                group["ds"],
                group["residual"],
                label="Residual (Actual - Forecast)",
                marker="o",
                linewidth=1.5,
                color="tab:purple",
            )
            ax.fill_between(
                group["ds"],
                0.0,
                group["residual"],
                color="tab:purple",
                alpha=0.25,
            )
            ax.set_title(f"Well {unique_id} WLPR Residuals (Test)")
            ax.set_ylabel("Residual (m3/day)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            ax.legend()
            metrics_text = _format_metrics_text(metrics, str(unique_id))
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path
