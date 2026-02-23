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

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SHUTOFF_WELL = "34"
SHUTOFF_DATE = pd.Timestamp("2021-01-01")

DATA_PATH = Path("MODEL_23.09.25.csv")
DISTANCES_PATH = Path("Distance.xlsx")
OUTPUT_DIR = Path("artifacts/scenario_shutoff_34")


def _generate_scenario_pdf(
    comp: pd.DataFrame,
    report: pd.DataFrame,
    frames_base: dict,
    merged_base: pd.DataFrame,
    merged_shut: pd.DataFrame,
    connected_prods: list,
    test_start: pd.Timestamp,
    out_dir: Path,
    model_name: str,
):
    pdf_path = out_dir / "scenario_shutoff_34.pdf"

    pair_summary = frames_base.get("injection_summary", pd.DataFrame())
    inj_col = "inj_id" if "inj_id" in pair_summary.columns else "inj"
    prod_col = "prod_id" if "prod_id" in pair_summary.columns else "prod"
    weight_map = {}
    if not pair_summary.empty and inj_col in pair_summary.columns:
        w34_pairs = pair_summary[pair_summary[inj_col].astype(str) == SHUTOFF_WELL]
        weight_map = dict(zip(w34_pairs[prod_col].astype(str), w34_pairs["weight"]))

    wells = sorted(comp["unique_id"].unique())
    wells_connected = sorted([w for w in wells if w in connected_prods],
                             key=lambda w: -weight_map.get(w, 0))
    wells_other = sorted([w for w in wells if w not in connected_prods])
    ordered_wells = wells_connected + wells_other

    model_label = {"chronos2": "Chronos-2", "timexer": "TimeXer"}.get(model_name, model_name)

    with PdfPages(pdf_path) as pdf:
        # --- Стр. 1: Сводная таблица ---
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        ax.set_title(
            f"Сценарий: остановка нагнетательной скв. {SHUTOFF_WELL} с {SHUTOFF_DATE.strftime('%d.%m.%Y')}\n"
            f"Модель: {model_label} | Период прогноза: "
            f"{test_start.strftime('%d.%m.%Y')} \u2014 {comp['ds'].max().strftime('%d.%m.%Y')}",
            fontsize=14, fontweight="bold", pad=20,
        )
        col_labels = [
            "\u2116\nскв.", "Связь\nсо скв. 34", "Вес\nCRM",
            "Факт\n(сред.)", "Базовый\nпрогноз", "Прогноз\nбез 34",
            "Разница\n(м\u00b3/сут)", "Разница\n(%)",
        ]
        table_data = []
        for _, r in report.sort_values("diff_pct").iterrows():
            w = str(r["well"])
            table_data.append([
                w,
                "ДА" if r["connected_to_34"] else "",
                f"{weight_map.get(w, 0):.3f}" if w in weight_map else "",
                f"{r['actual_mean']:.1f}",
                f"{r['baseline_mean']:.1f}",
                f"{r['shutoff_mean']:.1f}",
                f"{r['diff_mean']:+.2f}",
                f"{r['diff_pct']:+.2f}%",
            ])
        tbl = ax.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, row_data in enumerate(table_data, start=1):
            if row_data[1] == "\u0414\u0410":
                for j in range(len(col_labels)):
                    tbl[i, j].set_facecolor("#D6E4F0")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Стр. 2: Столбчатая диаграмма отклонений ---
        fig, ax = plt.subplots(figsize=(11, 6))
        rpt_sorted = report.sort_values("diff_pct")
        colors = ["#C00000" if d < -0.5 else "#4472C4" if d < 0 else "#70AD47"
                  for d in rpt_sorted["diff_pct"]]
        bars = ax.barh(rpt_sorted["well"].astype(str), rpt_sorted["diff_pct"], color=colors)
        ax.set_xlabel("Изменение прогноза (%)")
        ax.set_ylabel("Скважина")
        ax.set_title(
            f"Влияние остановки нагнетательной скв. {SHUTOFF_WELL} на прогноз добычи\n"
            f"Модель: {model_label}",
            fontsize=12, fontweight="bold",
        )
        ax.axvline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, rpt_sorted["diff_pct"]):
            offset = 0.05 * np.sign(bar.get_width()) if bar.get_width() != 0 else 0.05
            ax.text(bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}%", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Постраничные графики по скважинам (только прогнозный период) ---
        for well in ordered_wells:
            wc = comp[comp["unique_id"] == well].sort_values("ds")
            is_conn = well in connected_prods
            crm_w = weight_map.get(well, 0)

            fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                                     gridspec_kw={"height_ratios": [3, 1]})

            ax = axes[0]
            ax.plot(wc["ds"], wc["y"], "ko-", linewidth=2, markersize=7,
                    label="Факт")
            ax.plot(wc["ds"], wc["y_hat_baseline"], "b^--", linewidth=2,
                    markersize=7, label="Базовый прогноз")
            ax.plot(wc["ds"], wc["y_hat_shutoff"], "rs--", linewidth=2,
                    markersize=7, label=f"Прогноз без скв. {SHUTOFF_WELL}")

            conn_str = (f"СВЯЗАНА со скв. 34 (вес CRM = {crm_w:.3f})"
                        if is_conn else "не связана со скв. 34")
            ax.set_title(
                f"Скважина {well} | {conn_str}\n"
                f"Модель: {model_label}",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylabel("Дебит жидкости (м\u00b3/сут)")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.bar(wc["ds"], wc["diff"], width=20, color="tab:red", alpha=0.7)
            ax2.axhline(0, color="black", linewidth=0.5)
            ax2.set_ylabel("\u0394 (м\u00b3/сут)")
            ax2.set_xlabel("Дата")
            ax2.set_title("Отклонение: прогноз без скв. 34 \u2212 базовый прогноз",
                          fontsize=9)
            ax2.grid(True, alpha=0.3)

            mean_diff = wc["diff"].mean()
            mean_pct = wc["diff_pct"].mean()
            ax2.text(0.98, 0.95,
                     f"Ср. отклонение: {mean_diff:+.2f} м\u00b3/сут ({mean_pct:+.2f}%)",
                     transform=ax2.transAxes, fontsize=9, ha="right", va="top",
                     bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8})

            fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info("Scenario PDF saved to %s", pdf_path)


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chronos2", choices=["chronos2", "timexer"])
    args = parser.parse_args()

    base_artifacts = Path("artifacts_timexer") if args.model == "timexer" else Path("artifacts")
    out_dir = base_artifacts / "scenario_shutoff_34"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(model_type=args.model)

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
    print(f"СЦЕНАРИЙ: Остановка нагн. скв. {SHUTOFF_WELL} с {SHUTOFF_DATE.strftime('%d.%m.%Y')}")
    print(f"Период прогноза: {test_start.strftime('%d.%m.%Y')} — {comp['ds'].max().strftime('%d.%m.%Y')}")
    print(f"Модель: {args.model.upper()}")
    print("=" * 90)
    print(f"\n{'Скв.':>6} | {'Связь':>6} | {'Факт':>8} | {'Базовый':>8} | {'Без 34':>8} | {'Разн.':>8} | {'Разн.%':>7}")
    print("-" * 90)
    for _, r in report.iterrows():
        marker = " ***" if r["connected_to_34"] else ""
        print(f"{r['well']:>6} | {'ДА' if r['connected_to_34'] else '':>6} | "
              f"{r['actual_mean']:>8.2f} | {r['baseline_mean']:>8.2f} | "
              f"{r['shutoff_mean']:>8.2f} | {r['diff_mean']:>+8.2f} | "
              f"{r['diff_pct']:>+7.2f}%{marker}")

    print("\n" + "-" * 90)
    overall_base = metrics_base["overall"]
    overall_shut = metrics_shut["overall"]
    print(f"WMAPE базовый:  {overall_base.get('wmape', 0):.4f}%")
    print(f"WMAPE без 34:   {overall_shut.get('wmape', 0):.4f}%")

    # --- Generate PDF report ---
    _generate_scenario_pdf(
        comp, report, frames_base, merged_base, merged_shut,
        connected_prods, test_start, out_dir, args.model,
    )

    # Save
    report.to_csv(out_dir / "comparison_report.csv", index=False)
    comp.to_csv(out_dir / "detailed_comparison.csv", index=False)

    summary = {
        "model": args.model,
        "scenario": f"Shutoff injector {SHUTOFF_WELL} from {SHUTOFF_DATE.date()}",
        "test_start": str(test_start.date()),
        "n_producers": len(report),
        "connected_producers": connected_prods,
        "baseline_wmape": overall_base.get("wmape"),
        "shutoff_wmape": overall_shut.get("wmape"),
        "most_affected_wells": report.head(3)[["well", "diff_pct"]].to_dict("records"),
    }
    with open(out_dir / "scenario_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
