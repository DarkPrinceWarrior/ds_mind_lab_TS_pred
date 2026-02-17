"""Feature importance and influence visualization for WLPR pipeline.

Generates a multi-page PDF with:
  1. Correlation heatmap (all features vs WLPR)
  2. Mutual information bar chart (nonlinear importance)
  3. Per-well graph feature scatter plots
  4. Neighbor-aggregated features vs WLPR time series overlay
  5. Graph topology embedding (Node2Vec / Spectral) colored by mean WLPR
  6. CRM connectivity vs mean production scatter
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRAPH_STATIC_COLS = [
    "n2v_0", "n2v_1", "n2v_2", "n2v_3",
    "spectral_0", "spectral_1", "spectral_2", "spectral_3",
    "closeness_centrality", "pagerank", "eigenvector_centrality", "clustering_coeff",
    "crm_max_connectivity",
]

GRAPH_DYNAMIC_COLS = [
    "neighbor_avg_wlpr", "neighbor_avg_womr",
    "inj_wwir_crm_weighted",
]

PRODUCTION_COLS = [
    "wlpt", "womt", "womr", "wthp",
    "inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted",
]

FEATURE_GROUPS = {
    "Production / Injection": PRODUCTION_COLS,
    "Graph (dynamic)": GRAPH_DYNAMIC_COLS,
    "Graph (static topology)": GRAPH_STATIC_COLS,
    "Fourier / Embeddings": [
        "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
        "ts_embed_0", "ts_embed_1", "ts_embed_2",
    ],
}


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def generate_feature_analysis_pdf(
    train_df: pd.DataFrame,
    output_dir: Path,
    target_col: str = "y",
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.feature_selection import mutual_info_regression

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "feature_analysis.pdf"

    all_feature_cols = []
    for cols in FEATURE_GROUPS.values():
        all_feature_cols.extend(_safe_cols(train_df, cols))
    all_feature_cols = list(dict.fromkeys(all_feature_cols))

    df = train_df[["unique_id", "ds", target_col] + _safe_cols(train_df, all_feature_cols)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    present_features = [c for c in all_feature_cols if c in df.columns]

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Correlation bar chart (grouped + colored) ---
        correlations = df[present_features].corrwith(df[target_col]).sort_values()
        group_colors = {}
        for gname, gcols in FEATURE_GROUPS.items():
            for c in gcols:
                group_colors[c] = gname
        palette = {
            "Production / Injection": "#1f77b4",
            "Graph (dynamic)": "#d62728",
            "Graph (static topology)": "#2ca02c",
            "Fourier / Embeddings": "#9467bd",
        }
        bar_colors = [palette.get(group_colors.get(c, ""), "#7f7f7f") for c in correlations.index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(correlations) * 0.32)))
        bars = ax.barh(range(len(correlations)), correlations.values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(correlations.index, fontsize=8)
        ax.set_xlabel("Pearson Correlation with WLPR")
        ax.set_title("Feature Correlation with Target (WLPR)")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=palette[g], label=g) for g in palette if any(c in correlations.index for c in FEATURE_GROUPS[g])]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 2: Mutual Information (nonlinear importance) ---
        X = df[present_features].values.astype(np.float64)
        y = df[target_col].values.astype(np.float64)
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean, y_clean = X[valid_mask], y[valid_mask]

        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42, n_neighbors=5)
        mi_series = pd.Series(mi_scores, index=present_features).sort_values(ascending=True)
        bar_colors_mi = [palette.get(group_colors.get(c, ""), "#7f7f7f") for c in mi_series.index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(mi_series) * 0.32)))
        ax.barh(range(len(mi_series)), mi_series.values, color=bar_colors_mi, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(mi_series)))
        ax.set_yticklabels(mi_series.index, fontsize=8)
        ax.set_xlabel("Mutual Information (nats)")
        ax.set_title("Nonlinear Feature Importance (Mutual Information with WLPR)")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 3: Per-well correlation heatmap for graph features ---
        graph_all = _safe_cols(df, GRAPH_STATIC_COLS + GRAPH_DYNAMIC_COLS)
        nz_graph = [c for c in graph_all if df[c].std() > 1e-8]
        if nz_graph:
            wells = sorted(df["unique_id"].unique())
            corr_matrix = pd.DataFrame(index=wells, columns=nz_graph, dtype=float)
            for well in wells:
                wd = df[df["unique_id"] == well]
                for c in nz_graph:
                    if wd[c].std() > 1e-8:
                        corr_matrix.loc[well, c] = wd[c].corr(wd[target_col])
                    else:
                        corr_matrix.loc[well, c] = 0.0
            corr_matrix = corr_matrix.astype(float)

            fig, ax = plt.subplots(figsize=(max(8, len(nz_graph) * 0.7), max(5, len(wells) * 0.45)))
            im = ax.imshow(corr_matrix.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(nz_graph)))
            ax.set_xticklabels(nz_graph, rotation=55, ha="right", fontsize=7)
            ax.set_yticks(range(len(wells)))
            ax.set_yticklabels([f"Well {w}" for w in wells], fontsize=8)
            ax.set_title("Per-Well Correlation: Graph Features vs WLPR")
            for i in range(len(wells)):
                for j in range(len(nz_graph)):
                    val = corr_matrix.iloc[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if abs(val) > 0.5 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 4: neighbor_avg_wlpr vs WLPR time series per well ---
        neighbor_col = "neighbor_avg_wlpr"
        if neighbor_col in df.columns:
            wells = sorted(df["unique_id"].unique())
            n_wells = len(wells)
            ncols = 3
            nrows = (n_wells + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.0), squeeze=False)
            fig.suptitle("Neighbor-Aggregated WLPR vs Own WLPR (per well)", fontsize=13, y=1.01)
            for idx, well in enumerate(wells):
                ax = axes[idx // ncols][idx % ncols]
                wd = df[df["unique_id"] == well].sort_values("ds")
                ax.plot(wd["ds"], wd[target_col], label="WLPR", linewidth=1.2, color="#1f77b4")
                ax2 = ax.twinx()
                ax2.plot(wd["ds"], wd[neighbor_col], label="Neighbor avg", linewidth=1.0, color="#d62728", alpha=0.7)
                r = wd[target_col].corr(wd[neighbor_col])
                ax.set_title(f"Well {well}  (r={r:.2f})", fontsize=9)
                ax.tick_params(axis="x", labelsize=7, rotation=30)
                ax.tick_params(axis="y", labelsize=7)
                ax2.tick_params(axis="y", labelsize=7, labelcolor="#d62728")
                if idx == 0:
                    ax.set_ylabel("WLPR", fontsize=8)
                    ax2.set_ylabel("Neighbor avg WLPR", fontsize=8, color="#d62728")
            for idx in range(n_wells, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 5: Node2Vec & Spectral embeddings colored by mean WLPR ---
        well_stats = df.groupby("unique_id").agg(
            mean_wlpr=(target_col, "mean"),
            **{c: (c, "first") for c in _safe_cols(df, GRAPH_STATIC_COLS)},
        ).reset_index()

        for emb_prefix, emb_name in [("n2v_", "Node2Vec"), ("spectral_", "Spectral")]:
            emb_cols = [c for c in well_stats.columns if c.startswith(emb_prefix)]
            if len(emb_cols) >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
                fig.suptitle(f"{emb_name} Embedding Space (colored by mean WLPR)", fontsize=12)
                for ax_idx, (cx, cy) in enumerate([(0, 1), (2, 3)]):
                    if cx >= len(emb_cols) or cy >= len(emb_cols):
                        break
                    ax = axes[ax_idx]
                    sc = ax.scatter(
                        well_stats[emb_cols[cx]], well_stats[emb_cols[cy]],
                        c=well_stats["mean_wlpr"], cmap="viridis", s=120, edgecolors="black", linewidth=0.8,
                    )
                    for _, row in well_stats.iterrows():
                        ax.annotate(f"W{row['unique_id']}", (row[emb_cols[cx]], row[emb_cols[cy]]),
                                    fontsize=7, ha="center", va="bottom", textcoords="offset points", xytext=(0, 5))
                    ax.set_xlabel(emb_cols[cx])
                    ax.set_ylabel(emb_cols[cy])
                    ax.grid(True, alpha=0.3)
                    fig.colorbar(sc, ax=ax, label="Mean WLPR (m3/day)")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # --- Page 6: CRM connectivity vs mean production scatter ---
        crm_cols = _safe_cols(well_stats, ["crm_max_connectivity", "crm_total_connectivity"])
        cen_cols = _safe_cols(well_stats, ["pagerank", "eigenvector_centrality", "closeness_centrality", "clustering_coeff"])
        scatter_cols = [c for c in crm_cols + cen_cols if well_stats[c].std() > 1e-8]
        if scatter_cols:
            ncols_s = min(3, len(scatter_cols))
            nrows_s = (len(scatter_cols) + ncols_s - 1) // ncols_s
            fig, axes = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s * 4.5, nrows_s * 4.0), squeeze=False)
            fig.suptitle("Graph Metrics vs Mean WLPR (per well)", fontsize=12, y=1.01)
            for i, col in enumerate(scatter_cols):
                ax = axes[i // ncols_s][i % ncols_s]
                ax.scatter(well_stats[col], well_stats["mean_wlpr"], s=80, edgecolors="black", linewidth=0.6, color="#2ca02c")
                for _, row in well_stats.iterrows():
                    ax.annotate(f"W{row['unique_id']}", (row[col], row["mean_wlpr"]),
                                fontsize=7, ha="center", va="bottom", textcoords="offset points", xytext=(0, 5))
                r = well_stats[col].corr(well_stats["mean_wlpr"])
                ax.set_title(f"{col}\n(r={r:.3f})", fontsize=9)
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel("Mean WLPR", fontsize=8)
                ax.grid(True, alpha=0.3)
            for i in range(len(scatter_cols), nrows_s * ncols_s):
                axes[i // ncols_s][i % ncols_s].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 7: Injection features cross-correlation with WLPR per well ---
        inj_cols = _safe_cols(df, ["inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted"])
        if inj_cols:
            wells = sorted(df["unique_id"].unique())
            n_wells = len(wells)
            ncols_i = 3
            nrows_i = (n_wells + ncols_i - 1) // ncols_i
            fig, axes = plt.subplots(nrows_i, ncols_i, figsize=(14, nrows_i * 3.0), squeeze=False)
            fig.suptitle("Injection Features vs WLPR (per well)", fontsize=13, y=1.01)
            for idx, well in enumerate(wells):
                ax = axes[idx // ncols_i][idx % ncols_i]
                wd = df[df["unique_id"] == well].sort_values("ds")
                ax.plot(wd["ds"], wd[target_col], label="WLPR", linewidth=1.2, color="black")
                ax2 = ax.twinx()
                colors_inj = ["#d62728", "#ff7f0e", "#2ca02c"]
                for j, ic in enumerate(inj_cols):
                    ax2.plot(wd["ds"], wd[ic], label=ic.replace("inj_", ""), linewidth=0.8, color=colors_inj[j], alpha=0.7)
                ax.set_title(f"Well {well}", fontsize=9)
                ax.tick_params(axis="x", labelsize=7, rotation=30)
                ax.tick_params(axis="y", labelsize=7)
                ax2.tick_params(axis="y", labelsize=7)
                if idx == 0:
                    ax.set_ylabel("WLPR", fontsize=8)
                    ax2.set_ylabel("Injection features", fontsize=8)
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")
            for idx in range(n_wells, nrows_i * ncols_i):
                axes[idx // ncols_i][idx % ncols_i].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info("Feature analysis PDF saved to %s (%d pages)", pdf_path, 7)
    return pdf_path
