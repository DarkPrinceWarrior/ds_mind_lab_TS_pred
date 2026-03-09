from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch_geometric.data import Batch
except ImportError:  # pragma: no cover
    Batch = None

try:
    from .graph_dataset import build_temporal_graph_windows
    from .models_stgnn import STGNNPyG
    from .physics_regularizers import build_physics_loss_breakdown
except ImportError:  # pragma: no cover
    from graph_dataset import build_temporal_graph_windows
    from models_stgnn import STGNNPyG
    from physics_regularizers import build_physics_loss_breakdown

logger = logging.getLogger(__name__)


@dataclass
class STGNNTrainingArtifacts:
    predictions: pd.DataFrame
    graph_fusion_weights: pd.DataFrame
    edge_allocations: pd.DataFrame
    physics_history: pd.DataFrame
    well_event_metrics: pd.DataFrame
    training_summary: Dict[str, Any]


@dataclass
class _DistributedRuntime:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _require_pyg_batch() -> None:
    if Batch is None:
        raise ImportError(
            "stgnn_pyg batching requires torch-geometric. Install torch-geometric for the active torch/CUDA build."
        )


class _WindowDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def _collate_temporal_windows(batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    _require_pyg_batch()
    history_len = len(batch_samples[0]["history"])
    history = [
        Batch.from_data_list([sample["history"][step] for sample in batch_samples])
        for step in range(history_len)
    ]
    target = torch.stack([sample["target"] for sample in batch_samples], dim=0)
    return {
        "history": history,
        "target": target,
        "samples": batch_samples,
        "history_dates": [sample["history_dates"] for sample in batch_samples],
        "forecast_dates": [sample["forecast_dates"] for sample in batch_samples],
        "cutoff_dates": [sample["cutoff_date"] for sample in batch_samples],
    }


def _distributed_runtime() -> _DistributedRuntime:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if enabled and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return _DistributedRuntime(enabled=enabled, rank=rank, local_rank=local_rank, world_size=world_size)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _all_reduce_mean(value: float, runtime: _DistributedRuntime, device: torch.device) -> float:
    if not runtime.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / float(runtime.world_size)
    return float(tensor.item())


def _broadcast_float(value: float, runtime: _DistributedRuntime, device: torch.device) -> float:
    if not runtime.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def _broadcast_artifacts(
    artifacts: Optional[STGNNTrainingArtifacts],
    runtime: _DistributedRuntime,
) -> STGNNTrainingArtifacts:
    if not runtime.enabled:
        if artifacts is None:
            raise ValueError("Expected STGNN artifacts on single-process execution.")
        return artifacts
    payload = [artifacts if runtime.is_main else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise ValueError("Rank 0 did not produce STGNN artifacts.")
    return payload[0]


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _select_loss(name: str) -> nn.Module:
    normalized = str(name or "huber").strip().lower()
    if normalized == "mae":
        return nn.L1Loss()
    if normalized == "mse":
        return nn.MSELoss()
    return nn.HuberLoss()


def _find_window_splits(
    graph_bundle: Dict[str, Any],
    train_cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
    config: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    windows = build_temporal_graph_windows(graph_bundle, input_size=int(config.input_size), horizon=int(config.horizon))
    if not windows:
        return [], [], None
    offset = pd.tseries.frequencies.to_offset(config.freq)
    val_start = pd.Timestamp(test_start) - offset * max(int(config.val_horizon), 1)
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_sample: Optional[Dict[str, Any]] = None

    for sample in windows:
        forecast_start = pd.Timestamp(sample["forecast_dates"][0])
        forecast_end = pd.Timestamp(sample["forecast_dates"][-1])
        if forecast_end <= pd.Timestamp(train_cutoff):
            if forecast_start >= val_start:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
        elif forecast_start >= pd.Timestamp(test_start) and sample["cutoff_date"] <= pd.Timestamp(train_cutoff):
            test_sample = sample

    if not val_samples and train_samples:
        val_samples = [train_samples[-1]]
        train_samples = train_samples[:-1] or train_samples
    if test_sample is None:
        candidates = [sample for sample in windows if pd.Timestamp(sample["forecast_dates"][0]) >= pd.Timestamp(test_start)]
        if candidates:
            test_sample = candidates[0]
    return train_samples, val_samples, test_sample


def _history_to_device(history: List[Any], device: torch.device) -> List[Any]:
    return [snapshot.to(device) for snapshot in history]


def _node_feature_matrix(
    multigraph_spec: Dict[str, Any],
    dates: List[pd.Timestamp],
    *,
    node_type: str,
    feature_name: str,
) -> Optional[torch.Tensor]:
    feature_names = list(multigraph_spec.get("graph_metadata", {}).get(f"{node_type}_feature_names", []))
    if feature_name not in feature_names:
        return None
    feature_idx = feature_names.index(feature_name)
    values = []
    for ds in dates:
        matrix = multigraph_spec.get("node_dynamic_features_by_time", {}).get(pd.Timestamp(ds), {}).get(node_type)
        if matrix is None:
            continue
        values.append(np.asarray(matrix[:, feature_idx], dtype=float))
    if not values:
        return None
    arr = np.stack(values, axis=0)
    return torch.as_tensor(arr, dtype=torch.float32)


def _node_feature_matrix_any(
    multigraph_spec: Dict[str, Any],
    dates: List[pd.Timestamp],
    *,
    node_type: str,
    candidates: List[str],
) -> Optional[torch.Tensor]:
    for feature_name in candidates:
        values = _node_feature_matrix(multigraph_spec, dates, node_type=node_type, feature_name=feature_name)
        if values is not None:
            return values
    return None


def _select_physics_graph_type(diagnostics: Dict[str, Any]) -> Optional[str]:
    preferred = ["causal", "dyn", "cond", "bin"]
    available = set(diagnostics.get("edge_allocations", {}).keys()) & set(diagnostics.get("edge_context", {}).keys())
    for graph_type in preferred:
        if graph_type in available:
            return graph_type
    return next(iter(available), None) if available else None


def _slice_physics_payload(
    diagnostics: Dict[str, Any],
    graph_type: str,
    sample_idx: int,
) -> Optional[Dict[str, Any]]:
    edge_context = diagnostics.get("edge_context", {}).get(graph_type)
    latest_alloc = diagnostics.get("edge_allocations", {}).get(graph_type)
    alloc_history = diagnostics.get("allocation_history", {}).get(graph_type)
    if edge_context is None or latest_alloc is None:
        return None

    edge_batch = edge_context.get("edge_batch")
    if edge_batch is None:
        return {
            "edge_index": edge_context["edge_index"],
            "edge_attr": edge_context.get("edge_attr"),
            "alloc": latest_alloc.squeeze(-1),
            "alloc_history": alloc_history,
        }

    mask = edge_batch == int(sample_idx)
    if not torch.any(mask):
        return None
    edge_attr = edge_context.get("edge_attr")
    return {
        "edge_index": torch.stack(
            [edge_context["src_local_index"][mask], edge_context["dst_local_index"][mask]],
            dim=0,
        ),
        "edge_attr": edge_attr[mask] if edge_attr is not None else None,
        "alloc": latest_alloc.squeeze(-1)[mask],
        "alloc_history": alloc_history[:, mask] if alloc_history is not None else None,
    }


def _build_physics_inputs(
    sample: Dict[str, Any],
    multigraph_spec: Dict[str, Any],
    diagnostics: Dict[str, Any],
    device: torch.device,
    sample_idx: int = 0,
) -> Dict[str, Optional[torch.Tensor]]:
    graph_type = _select_physics_graph_type(diagnostics)
    if graph_type is None:
        return {}
    payload = _slice_physics_payload(diagnostics, graph_type, sample_idx)
    if payload is None:
        return {}

    history_dates = [pd.Timestamp(ds) for ds in sample.get("history_dates", [])]
    forecast_dates = [pd.Timestamp(ds) for ds in sample.get("forecast_dates", [])]
    if not history_dates or not forecast_dates:
        return {}
    injector_dates = history_dates + forecast_dates
    inj_rates = _node_feature_matrix_any(
        multigraph_spec,
        injector_dates,
        node_type="injector",
        candidates=["wwir"],
    )
    if inj_rates is None:
        return {}

    future_J = _node_feature_matrix_any(
        multigraph_spec,
        forecast_dates,
        node_type="producer",
        candidates=["pseudo_productivity_index", "productivity_index"],
    )
    future_dp = _node_feature_matrix_any(
        multigraph_spec,
        forecast_dates,
        node_type="producer",
        candidates=["dp_drawdown"],
    )
    future_bhp = _node_feature_matrix_any(
        multigraph_spec,
        forecast_dates,
        node_type="producer",
        candidates=["wbhp", "wthp"],
    )

    edge_index = payload["edge_index"].to(device)
    edge_attr = payload.get("edge_attr")
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    latest_alloc = payload["alloc"].to(device)
    alloc_history = payload.get("alloc_history")
    alloc_history = alloc_history.to(device).unsqueeze(0) if alloc_history is not None else None
    edge_feature_names = list(multigraph_spec.get("graph_metadata", {}).get("edge_feature_names", {}).get(graph_type, []))
    lag_idx = edge_feature_names.index("lag") if "lag" in edge_feature_names else None
    tau_idx = edge_feature_names.index("tau") if "tau" in edge_feature_names else None

    inj_rates = inj_rates.to(device)
    horizon = len(forecast_dates)
    history_len = len(history_dates)
    num_producers = len(multigraph_spec.get("graph_metadata", {}).get("producer_ids", []))
    future_alloc = latest_alloc.unsqueeze(0).repeat(horizon, 1)
    producer_forcing = torch.zeros((horizon, num_producers), device=device, dtype=torch.float32)
    producer_tau_num = torch.zeros((num_producers,), device=device, dtype=torch.float32)
    producer_tau_den = torch.zeros((num_producers,), device=device, dtype=torch.float32)

    for edge_pos in range(edge_index.size(1)):
        inj_idx = int(edge_index[0, edge_pos].item())
        prod_idx = int(edge_index[1, edge_pos].item())
        lag_steps = 0
        tau_value = torch.tensor(1.0, device=device)
        if edge_attr is not None and edge_attr.numel() > 0:
            if lag_idx is not None:
                lag_steps = max(int(round(float(edge_attr[edge_pos, lag_idx].detach().cpu().item()))), 0)
            if tau_idx is not None:
                tau_value = torch.clamp(edge_attr[edge_pos, tau_idx].abs(), min=1.0)
        decay = torch.exp(-torch.tensor(float(lag_steps), device=device) / torch.clamp(tau_value, min=1.0))
        producer_tau_num[prod_idx] = producer_tau_num[prod_idx] + latest_alloc[edge_pos] * tau_value
        producer_tau_den[prod_idx] = producer_tau_den[prod_idx] + latest_alloc[edge_pos].abs()

        for step in range(horizon):
            source_t = min(max(history_len + step - lag_steps, 0), inj_rates.size(0) - 1)
            producer_forcing[step, prod_idx] = (
                producer_forcing[step, prod_idx]
                + future_alloc[step, edge_pos] * decay * inj_rates[source_t, inj_idx]
            )

    producer_tau = producer_tau_num / torch.clamp(producer_tau_den, min=1e-6)
    tau_future = producer_tau.unsqueeze(0).repeat(horizon, 1)

    return {
        "graph_type": graph_type,
        "producer_forcing": producer_forcing.unsqueeze(0),
        "alloc_weights": future_alloc.unsqueeze(0),
        "alloc_history": alloc_history,
        "allocation_group_index": edge_index[1].detach().clone(),
        "allocation_num_groups": num_producers,
        "tau": tau_future.unsqueeze(0),
        "J": future_J.to(device).unsqueeze(0) if future_J is not None else None,
        "Vp": future_dp.to(device).unsqueeze(0) if future_dp is not None else None,
        "bhp": future_bhp.to(device).unsqueeze(0) if future_bhp is not None else None,
    }


def _scenario_shutin_mask(
    frames: Dict[str, Any],
    sample: Dict[str, Any],
    multigraph_spec: Dict[str, Any],
    device: torch.device,
) -> Optional[torch.Tensor]:
    return None


def _compute_batch_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    loss_fn: nn.Module,
    multigraph_spec: Dict[str, Any],
    frames: Dict[str, Any],
    config: Any,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    history = _history_to_device(batch["history"], device)
    target = batch["target"].to(device, non_blocking=True)
    use_amp = bool(getattr(config, "stgnn_use_amp", False)) and device.type == "cuda"

    with _autocast_context(device, use_amp):
        pred, diagnostics = model(history)

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    pred = pred.float()
    target = target.float()
    data_loss = loss_fn(pred, target)
    total_loss = data_loss
    log_parts = {
        "data_loss": float(data_loss.detach().cpu()),
        "batch_size": float(pred.size(0)),
    }

    if bool(config.physics_loss_enabled):
        schedule = 0.0
        warmup_epochs = max(int(config.physics_warmup_epochs), 1)
        if epoch >= warmup_epochs:
            progress = min(
                1.0,
                float(epoch - warmup_epochs + 1) / float(max(int(config.stgnn_max_epochs) - warmup_epochs + 1, 1)),
            )
            schedule = float(config.physics_weight_init) + progress * (
                float(config.physics_weight_max) - float(config.physics_weight_init)
            )
        physics_parts: Dict[str, List[float]] = {}
        physics_totals: List[torch.Tensor] = []
        for sample_idx, sample in enumerate(batch["samples"]):
            physics_inputs = _build_physics_inputs(sample, multigraph_spec, diagnostics, device, sample_idx=sample_idx)
            if physics_inputs.get("producer_forcing") is None:
                continue
            physics = build_physics_loss_breakdown(
                pred[sample_idx].transpose(0, 1).unsqueeze(0),
                physics_inputs["producer_forcing"],
                alloc_weights=physics_inputs.get("alloc_weights"),
                alloc_history=physics_inputs.get("alloc_history"),
                allocation_group_index=physics_inputs.get("allocation_group_index"),
                allocation_num_groups=physics_inputs.get("allocation_num_groups"),
                shutin_mask=_scenario_shutin_mask(frames, sample, multigraph_spec, device),
                bhp=physics_inputs.get("bhp"),
                tau=physics_inputs.get("tau"),
                J=physics_inputs.get("J"),
                Vp=physics_inputs.get("Vp"),
                lambdas={
                    "crm_residual": float(config.physics_lambda_crm),
                    "nonnegative": float(config.physics_lambda_nonneg),
                    "cumulative_monotonic": float(config.physics_lambda_cumulative),
                    "allocation_simplex": float(config.physics_lambda_simplex),
                    "shutin_consistency": float(config.physics_lambda_shutin),
                    "temporal_smoothness": float(config.physics_lambda_smoothness),
                },
            )
            physics_totals.append(physics["total"])
            for name, value in physics.items():
                if name == "total":
                    continue
                physics_parts.setdefault(name, []).append(float(value.detach().cpu()))
        if physics_totals:
            mean_physics = torch.stack(physics_totals).mean()
            physics_loss = schedule * mean_physics
            total_loss = total_loss + physics_loss
            log_parts["physics_weight"] = schedule
            log_parts["physics_loss"] = float(physics_loss.detach().cpu())
            for name, values in physics_parts.items():
                log_parts[f"physics_{name}"] = float(np.mean(values)) if values else 0.0
        else:
            log_parts["physics_weight"] = schedule
            log_parts["physics_loss"] = 0.0
    return total_loss, log_parts, pred.detach()


def _evaluate_loader(
    model: nn.Module,
    loader: Optional[DataLoader],
    loss_fn: nn.Module,
    multigraph_spec: Dict[str, Any],
    frames: Dict[str, Any],
    config: Any,
    device: torch.device,
    epoch: int,
) -> float:
    if loader is None:
        return float("inf")
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            loss, _, _ = _compute_batch_loss(model, batch, loss_fn, multigraph_spec, frames, config, device, epoch)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _prediction_frame(
    pred: torch.Tensor,
    forecast_dates: List[pd.Timestamp],
    producer_ids: List[str],
) -> pd.DataFrame:
    pred_np = pred.detach().cpu().numpy()
    records = []
    for prod_idx, well in enumerate(producer_ids):
        for step, ds in enumerate(forecast_dates):
            records.append({"unique_id": well, "ds": pd.Timestamp(ds), "y_hat": float(pred_np[prod_idx, step])})
    return pd.DataFrame.from_records(records)


def _fusion_weights_frame(model: STGNNPyG) -> pd.DataFrame:
    if model.latest_fusion_weights is None:
        return pd.DataFrame(columns=["graph_type", "weight"])
    weights = model.latest_fusion_weights.detach().cpu().numpy()
    return pd.DataFrame({"graph_type": model.graph_types, "weight": weights})


def _edge_allocations_frame(model: STGNNPyG, multigraph_spec: Dict[str, Any]) -> pd.DataFrame:
    pair_table = multigraph_spec.get("pair_table", pd.DataFrame())
    rows = []
    for graph_type, values in (model.latest_edge_allocations or {}).items():
        alloc = values.detach().cpu().numpy().reshape(-1)
        if pair_table is not None and not pair_table.empty and len(pair_table) >= len(alloc):
            subset = pair_table.iloc[: len(alloc)]
            for idx, value in enumerate(alloc):
                rows.append(
                    {
                        "graph_type": graph_type,
                        "inj_id": str(subset.iloc[idx]["inj_id"]),
                        "prod_id": str(subset.iloc[idx]["prod_id"]),
                        "allocation": float(value),
                    }
                )
        else:
            for idx, value in enumerate(alloc):
                rows.append({"graph_type": graph_type, "inj_id": None, "prod_id": None, "allocation": float(value), "edge_idx": idx})
    return pd.DataFrame.from_records(rows)


def _well_event_metrics(pred_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = test_df[["unique_id", "ds", "y"]].merge(pred_df, on=["unique_id", "ds"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["unique_id", "mae", "rmse"])
    records = []
    for unique_id, group in merged.groupby("unique_id"):
        err = group["y"].to_numpy(dtype=float) - group["y_hat"].to_numpy(dtype=float)
        records.append(
            {
                "unique_id": str(unique_id),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("mae", ascending=False).reset_index(drop=True)


def fit_and_forecast_stgnn(
    frames: Dict[str, Any],
    config: Any,
) -> STGNNTrainingArtifacts:
    runtime = _distributed_runtime()
    graph_bundle = frames.get("graph_dataset")
    multigraph_spec = frames.get("multigraph_spec", {})
    if not graph_bundle or not graph_bundle.get("snapshots"):
        raise ValueError("stgnn_pyg requires graph_dataset snapshots in frames.")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{runtime.local_rank}" if runtime.enabled else "cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")

    train_samples, val_samples, test_sample = _find_window_splits(
        graph_bundle,
        train_cutoff=pd.Timestamp(frames["train_cutoff"]),
        test_start=pd.Timestamp(frames["test_start"]),
        config=config,
    )
    if not train_samples or test_sample is None:
        raise ValueError("Not enough graph windows to train/evaluate stgnn_pyg.")

    batch_size = max(int(getattr(config, "stgnn_batch_size", 1)), 1)
    num_workers = max(int(getattr(config, "stgnn_num_workers", 0)), 0)
    pin_memory = device.type == "cuda"

    train_dataset = _WindowDataset(train_samples)
    val_dataset = _WindowDataset(val_samples or train_samples[-1:])
    test_dataset = _WindowDataset([test_sample])
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True, drop_last=False)
        if runtime.enabled else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_temporal_windows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_temporal_windows,
    ) if runtime.is_main else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=_collate_temporal_windows,
    )

    model_core = STGNNPyG(graph_bundle["metadata"], config).to(device)
    init_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=_collate_temporal_windows,
    )
    init_batch = next(iter(init_loader))
    if runtime.is_main:
        logger.info("STGNN init: running lazy-parameter warmup on 1 temporal window")
    with torch.no_grad():
        _ = model_core(_history_to_device(init_batch["history"], device))
    if runtime.is_main:
        logger.info("STGNN init: lazy-parameter warmup complete")

    if runtime.enabled:
        if runtime.is_main:
            logger.info("STGNN init: wrapping model with DistributedDataParallel")
        model: nn.Module = DDP(
            model_core,
            device_ids=[runtime.local_rank] if device.type == "cuda" else None,
            output_device=runtime.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    else:
        model = model_core
    if runtime.is_main:
        logger.info("STGNN init: model ready for training")

    loss_fn = _select_loss(getattr(config, "xlinear_loss", "huber")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.stgnn_learning_rate),
        weight_decay=float(config.stgnn_weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(int(config.stgnn_scheduler_patience), 1),
    )
    use_amp = bool(getattr(config, "stgnn_use_amp", False)) and device.type == "cuda"

    physics_logs: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0
    max_epochs = max(int(config.stgnn_max_epochs), 1)

    if runtime.is_main:
        logger.info(
            "STGNN training start: train_windows=%d, val_windows=%d, horizon=%d, input_size=%d, batch_size=%d, max_epochs=%d, device=%s, world_size=%d, amp=%s",
            len(train_samples),
            len(val_samples),
            int(config.horizon),
            int(config.input_size),
            batch_size,
            max_epochs,
            device,
            runtime.world_size,
            use_amp,
        )

    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, log_parts, _ = _compute_batch_loss(model, batch, loss_fn, multigraph_spec, frames, config, device, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if runtime.is_main:
                log_parts["epoch"] = epoch
                physics_logs.append(log_parts)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_train_loss = _all_reduce_mean(mean_train_loss, runtime, device)
        if runtime.is_main:
            val_loss = _evaluate_loader(model, val_loader, loss_fn, multigraph_spec, frames, config, device, epoch)
        else:
            val_loss = 0.0
        val_loss = _broadcast_float(val_loss, runtime, device)
        scheduler.step(val_loss)

        if runtime.is_main and (epoch == 1 or epoch == max_epochs or epoch % 5 == 0):
            logger.info(
                "STGNN epoch %d/%d: train_loss=%.5f, val_loss=%.5f, best_val=%.5f",
                epoch,
                max_epochs,
                mean_train_loss,
                float(val_loss),
                float(min(best_val, val_loss)),
            )
        if val_loss < best_val:
            best_val = val_loss
            if runtime.is_main:
                best_state = {key: value.detach().cpu().clone() for key, value in _unwrap_model(model).state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max(int(config.stgnn_early_stop_patience), 1):
                if runtime.is_main:
                    logger.info(
                        "STGNN early stopping at epoch %d: best_val=%.5f, patience=%d",
                        epoch,
                        float(best_val),
                        patience,
                    )
                break

    artifacts: Optional[STGNNTrainingArtifacts] = None
    if runtime.is_main:
        if best_state is not None:
            _unwrap_model(model).load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(test_loader))
            _, _, pred = _compute_batch_loss(model, test_batch, loss_fn, multigraph_spec, frames, config, device, epoch=0)
        pred = pred.squeeze(0)
        plain_model = _unwrap_model(model)
        pred_df = _prediction_frame(pred, test_sample["forecast_dates"], graph_bundle["metadata"]["producer_ids"])
        fusion_df = _fusion_weights_frame(plain_model)
        edge_alloc_df = _edge_allocations_frame(plain_model, multigraph_spec)
        physics_df = pd.DataFrame.from_records(physics_logs)
        well_metrics_df = _well_event_metrics(pred_df, frames["test_df"])
        training_summary = {
            "train_windows": len(train_samples),
            "val_windows": len(val_samples),
            "best_val_loss": best_val,
            "epochs_trained": int(physics_df["epoch"].max()) if not physics_df.empty else 0,
            "device": str(device),
            "world_size": runtime.world_size,
            "ddp_enabled": runtime.enabled,
            "batch_size": batch_size,
            "amp_enabled": use_amp,
        }
        logger.info(
            "STGNN training complete: epochs=%d, best_val=%.5f, forecast_rows=%d",
            training_summary["epochs_trained"],
            float(best_val),
            len(pred_df),
        )
        artifacts = STGNNTrainingArtifacts(
            predictions=pred_df,
            graph_fusion_weights=fusion_df,
            edge_allocations=edge_alloc_df,
            physics_history=physics_df,
            well_event_metrics=well_metrics_df,
            training_summary=training_summary,
        )
    return _broadcast_artifacts(artifacts, runtime)
