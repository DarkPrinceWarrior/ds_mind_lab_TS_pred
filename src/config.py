from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _default_kernel_candidates() -> List[Dict[str, Any]]:
    return [
        {"type": "idw", "calibrate": True, "params": {"p": 1.5}, "param_grid": {"p": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]}},
        {"type": "exponential", "calibrate": True, "params": {"scale": 400.0}, "param_grid": {"scale": [200.0, 400.0, 600.0, 800.0, 1000.0]}},
        {"type": "gaussian", "calibrate": True, "params": {"scale": 400.0}, "param_grid": {"scale": [200.0, 400.0, 600.0, 800.0, 1000.0]}},
        {"type": "matern", "calibrate": True, "params": {"scale": 400.0, "nu": 1.5}, "param_grid": {"scale": [200.0, 400.0, 600.0, 800.0], "nu": [0.5, 1.5, 2.5]}},
        {"type": "rational_quadratic", "calibrate": True, "params": {"scale": 400.0, "alpha": 1.0}, "param_grid": {"scale": [200.0, 400.0, 600.0, 800.0], "alpha": [0.5, 1.0, 2.0]}},
    ]


@dataclass
class PipelineConfig:
    horizon: int = 6
    val_horizon: int = 6
    freq: str = "MS"
    min_history: int = 60
    input_size: int = 48
    random_seed: int = 42
    cv_folds: int = 6
    cv_step: int = 6
    cv_enabled: bool = True
    # Physics parameters
    physics_weight: float = 0.1
    physics_injection_coeff: float = 0.05
    physics_damping: float = 0.01
    physics_smoothing_weight: float = 0.0
    physics_features: List[str] = field(default_factory=lambda: ["inj_wwir_lag_weighted"])
    physics_base_loss: str = "huber"
    # Injection kernel parameters
    inj_top_k: int = 5
    inj_kernel_type: str = "idw"
    inj_kernel_p: float = 2.0
    inj_kernel_params: Dict[str, float] = field(default_factory=dict)
    inj_kernel_calibrate: bool = True
    inj_kernel_param_grid: Dict[str, List[float]] = field(default_factory=dict)
    inj_kernel_candidates: List[Dict[str, Any]] = field(default_factory=_default_kernel_candidates)
    inj_distance_anisotropy: Optional[Dict[str, Any]] = None
    inj_directional_bias: Optional[Dict[str, Any]] = None
    use_crm_filter: bool = True
    tau_bound_multiplier: float = 2.0
    lag_min_overlap: int = 6
    physics_estimates: Optional[Dict[str, float]] = None
    # Preprocessing parameters
    enable_physics_preprocessing: bool = True
    preprocessing_structural_break_threshold: float = 0.7
    preprocessing_outlier_contamination: float = 0.05
    preprocessing_enable_smoothing: bool = False
    preprocessing_smooth_window_length: int = 7
    preprocessing_smooth_polyorder: int = 2
    preprocessing_bilateral_sigma_space: float = 3.0
    # Graph feature parameters
    graph_n2v_dimensions: int = 4
    graph_spectral_components: int = 4
    graph_neighbor_k: int = 5
    graph_neighbor_agg_cols: List[str] = field(
        default_factory=lambda: ["wlpr", "womr"],
    )
    # DTW-based dynamic similarity graph parameters
    graph_dtw_agg_cols: List[str] = field(
        default_factory=lambda: ["wlpr", "womr"],
    )
    graph_dtw_k: int = 5
    # Production clustering + graph sparsification (SGP-GCN SPC)
    graph_sparsify: bool = True
    graph_sparsify_max_k: int = 4
    graph_sparsify_inter_quantile: float = 0.5
    # Feature lists
    # MDI feature selection (RF importance) removed:
    #   - fourier_* (MDI<0.002 globally, Chronos-2 has own positional encoding)
    #   - month_sin/month_cos (MDI~0, redundant with Chronos-2 time encoding)
    #   - pagerank, eigenvector_centrality, clustering_coeff (MDI=0 per-well)
    # Kept: wlpt (per-well MDI=0.03-0.08), ts_embed_*, graph embeddings
    hist_exog: List[str] = field(
        default_factory=lambda: [
            "wlpt", "womt", "womr", "wthp",
            "inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted",
            "inj_top1_contribution", "inj_top2_contribution", "inj_top3_contribution",
            "ts_embed_0", "ts_embed_1", "ts_embed_2",
            "neighbor_avg_wlpr", "neighbor_avg_womr",
            "dtw_neighbor_avg_wlpr", "dtw_neighbor_avg_womr",
            "productivity_index", "dp_drawdown",
        ]
    )
    futr_exog: List[str] = field(
        default_factory=lambda: [
            "inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted",
            "inj_top1_contribution", "inj_top2_contribution", "inj_top3_contribution",
            "n2v_0", "n2v_1", "n2v_2", "n2v_3",
            "spectral_0", "spectral_1", "spectral_2", "spectral_3",
            "closeness_centrality",
            "crm_max_connectivity",
        ]
    )
    static_exog: List[str] = field(
        default_factory=lambda: [
            "coord_x", "coord_y", "coord_z",
            "well_depth", "dist_from_center",
            "quadrant_0", "quadrant_1", "quadrant_2", "quadrant_3",
        ]
    )
    # Chronos-2 configuration
    chronos_hub_model_name: str = "amazon/chronos-2"
    chronos_hub_model_revision: Optional[str] = None
    chronos_local_dir: Optional[str] = None
    chronos_input_chunk_length: Optional[int] = None
    chronos_output_chunk_length: Optional[int] = None
    chronos_probabilistic: bool = True
    chronos_quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.9],
    )
    chronos_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Model selection
    model_type: str = "chronos2"  # "chronos2" or "timexer"
    # TimeXer configuration
    # Architecture: ~135K params, ratio ~81:1 with 1664 training windows.
    # Regularized via dropout=0.25, robust scaler, early stopping, lr decay.
    timexer_patch_len: int = 6       # 6 months per patch = half-year semantics
    timexer_hidden_size: int = 64    # enough capacity for 29 covariates
    timexer_n_heads: int = 4         # head_dim=16, stable attention
    timexer_e_layers: int = 2        # 2 layers for self-attn + cross-attn depth
    timexer_d_ff: int = 64           # compact FFN (1:1 with hidden)
    timexer_dropout: float = 0.25    # moderate regularization
    timexer_max_steps: int = 1000
    timexer_learning_rate: float = 5e-4
    timexer_batch_size: int = 32
    timexer_windows_batch_size: int = 64
    timexer_early_stop_patience: int = 10
    timexer_val_check_steps: int = 50
    timexer_scaler_type: str = "robust"  # outlier-resistant (wells 14, 23)
    timexer_loss: str = "mae"        # robust to outliers, default in paper
    timexer_num_lr_decays: int = 3   # 3 LR decays across max_steps
