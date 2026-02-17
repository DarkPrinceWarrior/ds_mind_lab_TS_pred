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
    preprocessing_smooth_window_length: int = 7
    preprocessing_smooth_polyorder: int = 2
    # Feature lists
    hist_exog: List[str] = field(
        default_factory=lambda: [
            "wlpt", "womt", "womr", "wwit", "wthp",
            "inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted",
            "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2", "fourier_sin_3", "fourier_cos_3",
            "ts_embed_0", "ts_embed_1", "ts_embed_2",
        ]
    )
    futr_exog: List[str] = field(
        default_factory=lambda: [
            "month_sin", "month_cos", "time_idx", "type_prod", "type_inj",
            "inj_wwir_lag_weighted", "inj_wwit_diff_lag_weighted", "inj_wwir_crm_weighted",
            "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2", "fourier_sin_3", "fourier_cos_3",
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
    chronos_kwargs: Dict[str, Any] = field(default_factory=dict)
