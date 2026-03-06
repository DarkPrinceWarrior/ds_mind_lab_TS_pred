from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    input_size: int = 36
    random_seed: int = 42
    cv_folds: int = 6
    cv_step: int = 6
    cv_enabled: bool = True

    # Conformal prediction (post-processing UQ)
    conformal_enabled: bool = True
    conformal_alpha: float = 0.1
    conformal_method: str = "wcp_exp"  # "icp", "wcp_exp", "wcp_linear"
    conformal_per_horizon: bool = True
    conformal_exp_decay: float = 0.97
    conformal_min_samples: int = 30
    conformal_group_key: str = "well_cluster"  # "well", "cluster", "regime", "well_cluster"

    # Legacy physics parameters (baseline path)
    physics_weight: float = 0.1
    physics_injection_coeff: float = 0.05
    physics_damping: float = 0.01
    physics_smoothing_weight: float = 0.0
    physics_features: List[str] = field(default_factory=lambda: ["inj_wwir_lag_attn", "inj_wwir_lag_weighted"])
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
    inj_attention_enabled: bool = True
    inj_attention_method: str = "causal_stage_geo"
    inj_attention_target_mode: str = "delta"
    inj_attention_steps: int = 300
    inj_attention_prior_strength: float = 0.2
    inj_attention_smooth_strength: float = 0.05
    inj_attention_future_anchor_strength: float = 0.25
    inj_attention_geo_condition_strength: float = 0.35
    inj_attention_stage_adaptive: bool = True
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
    graph_backend: str = "pyg"
    graph_types: List[str] = field(default_factory=lambda: ["topo", "bin", "cond", "dyn"])
    node_types: List[str] = field(default_factory=lambda: ["producer", "injector"])
    use_hetero_graph: bool = True
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
    hist_exog: List[str] = field(
        default_factory=lambda: [
            "wlpt", "womt", "womr", "wthp",
            "inj_wwir_lag_weighted",
            "inj_wwit_diff_lag_attn", "inj_wwir_crm_attn",
            "inj_top1_contribution", "inj_top2_contribution",
            "dtw_neighbor_avg_wlpr",
            "productivity_index", "dp_drawdown",
        ]
    )
    futr_exog: List[str] = field(
        default_factory=lambda: [
            "inj_wwir_lag_weighted",
            "inj_wwit_diff_lag_attn", "inj_wwir_crm_attn",
            "inj_top1_contribution", "inj_top2_contribution",
        ]
    )
    static_exog: List[str] = field(
        default_factory=lambda: [
            "coord_x", "coord_y", "coord_z", "dist_from_center",
            "n2v_1", "n2v_2", "n2v_3",
            "spectral_0", "spectral_1", "spectral_2", "spectral_3",
            "crm_max_connectivity",
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
    model_type: str = "chronos2"  # "chronos2", "xlinear", "stgnn_pyg"

    # XLinear configuration (NeuralForecast)
    xlinear_hidden_size: int = 128
    xlinear_temporal_ff: int = 256
    xlinear_channel_ff: int = 24
    xlinear_temporal_dropout: float = 0.15
    xlinear_channel_dropout: float = 0.10
    xlinear_embed_dropout: float = 0.10
    xlinear_head_dropout: float = 0.10
    xlinear_max_steps: int = 2000
    xlinear_learning_rate: float = 1e-3
    xlinear_batch_size: int = 64
    xlinear_windows_batch_size: int = 64
    xlinear_early_stop_patience: int = 15
    xlinear_val_check_steps: int = 50
    xlinear_scaler_type: str = "robust"
    xlinear_loss: str = "huber"
    xlinear_num_lr_decays: int = 5

    # STGNN / PyG configuration
    stgnn_hidden_dim: int = 64
    stgnn_temporal_hidden_dim: int = 64
    stgnn_edge_dim: int = 8
    stgnn_message_passing_steps: int = 2
    stgnn_layers: int = 2
    stgnn_relation_conv: Dict[str, str] = field(
        default_factory=lambda: {
            "topo": "SAGE",
            "bin": "GAT",
            "cond": "NNConv",
            "dyn": "TransformerConv",
            "causal": "TransformerConv",
        }
    )
    stgnn_temporal_backbone: str = "gru"  # "gru", "tcn", "transformer"
    stgnn_graph_fusion: str = "attention"
    stgnn_use_reverse_edges: bool = True
    stgnn_edge_dropout: float = 0.0
    stgnn_feature_dropout: float = 0.1
    stgnn_batch_size: int = 4
    stgnn_max_epochs: int = 120
    stgnn_learning_rate: float = 1e-3
    stgnn_weight_decay: float = 1e-4
    stgnn_scheduler_patience: int = 8
    stgnn_early_stop_patience: int = 16
    stgnn_num_workers: int = 0
    stgnn_use_amp: bool = False
    stgnn_prod_feature_cols: List[str] = field(
        default_factory=lambda: [
            "wlpr", "womr", "wlpt", "wthp", "fw",
            "pseudo_productivity_index", "dp_drawdown",
            "wlpr_diff1", "wlpr_cumsum1", "womr_diff1", "womr_cumsum1",
            "dtw_neighbor_avg_wlpr",
        ]
    )
    stgnn_inj_feature_cols: List[str] = field(
        default_factory=lambda: [
            "wwir", "wwit", "wwit_diff", "wwir_diff1", "wwir_cumsum1",
        ]
    )

    # Graph-native physics regularization
    physics_loss_enabled: bool = True
    physics_loss_mode: str = "crm_residual"
    physics_warmup_epochs: int = 10
    physics_weight_init: float = 0.0
    physics_weight_max: float = 0.15
    physics_lambda_crm: float = 0.05
    physics_lambda_nonneg: float = 0.01
    physics_lambda_cumulative: float = 0.01
    physics_lambda_shutin: float = 0.02
    physics_lambda_simplex: float = 0.01
    physics_lambda_smoothness: float = 0.005

    # Scenario / intervention settings
    scenario_mode: str = "graph_edit_plus_controls"  # "covariate_only", "graph_edit", "graph_edit_plus_controls"
    scenario_edit_graph: bool = True
    scenario_graph_types: List[str] = field(default_factory=lambda: ["bin", "cond", "dyn", "causal"])

    def is_graph_model(self) -> bool:
        return str(self.model_type).strip().lower() == "stgnn_pyg"

    def resolved_graph_types(self) -> List[str]:
        values = [str(item).strip().lower() for item in self.graph_types if str(item).strip()]
        return values or ["topo", "bin", "cond", "dyn"]

    def resolved_node_types(self) -> List[str]:
        values = [str(item).strip().lower() for item in self.node_types if str(item).strip()]
        return values or ["producer", "injector"]

    def resolved_stgnn_feature_columns(self) -> Dict[str, List[str]]:
        producer_cols = list(dict.fromkeys(self.stgnn_prod_feature_cols + ["productivity_index"]))
        injector_cols = list(dict.fromkeys(self.stgnn_inj_feature_cols))
        return {"producer": producer_cols, "injector": injector_cols}

    def resolved_conformal_group_levels(self) -> List[Tuple[str, List[str]]]:
        key = (self.conformal_group_key or "").strip().lower()
        if key == "well":
            return [("well", ["unique_id"])]
        if key == "cluster":
            return [("cluster", ["prod_cluster"])]
        if key == "regime":
            return [("regime", ["regime_id"])]
        return [
            ("well_cluster", ["unique_id", "prod_cluster"]),
            ("cluster", ["prod_cluster"]),
        ]
