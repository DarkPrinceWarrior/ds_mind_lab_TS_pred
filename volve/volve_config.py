from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VolveConfig:
    horizon: int = 30
    val_horizon: int = 30
    freq: str = "D"
    min_history: int = 365
    input_size: int = 60
    random_seed: int = 42
    cv_folds: int = 3
    cv_step: int = 30
    cv_enabled: bool = True
    conformal_enabled: bool = True
    conformal_alpha: float = 0.1
    conformal_method: str = "wcp_exp"
    conformal_per_horizon: bool = True
    conformal_exp_decay: float = 0.97
    conformal_min_samples: int = 30

    hist_exog: List[str] = field(
        default_factory=lambda: [
            "water_rate",
            "gas_rate",
            "watercut",
            "avg_whp",
            "avg_dp_tubing",
            "avg_choke_size",
            "on_stream_hrs",
            "avg_downhole_pressure",
            "avg_downhole_temperature",
            "field_inj_rate",
        ]
    )
    futr_exog: List[str] = field(
        default_factory=lambda: [
            "field_inj_rate",
        ]
    )
    static_exog: List[str] = field(default_factory=list)

    # XLinear configuration
    xlinear_hidden_size: int = 64
    xlinear_temporal_ff: int = 128
    xlinear_channel_ff: int = 16
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
