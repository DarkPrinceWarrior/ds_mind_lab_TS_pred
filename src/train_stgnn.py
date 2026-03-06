from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from .graph_dataset import build_temporal_graph_windows
    from .models_stgnn import STGNNPyG
    from .physics_regularizers import build_physics_loss_breakdown
except ImportError:  # pragma: no cover
    from graph_dataset import build_temporal_graph_windows
    from models_stgnn import STGNNPyG
    from physics_regularizers import build_physics_loss_breakdown


@dataclass
class STGNNTrainingArtifacts:
    predictions: pd.DataFrame
    graph_fusion_weights: pd.DataFrame
    edge_allocations: pd.DataFrame
    physics_history: pd.DataFrame
    well_event_metrics: pd.DataFrame
    training_summary: Dict[str, Any]


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


def _future_injector_tensor(multigraph_spec: Dict[str, Any], future_dates: List[pd.Timestamp]) -> Optional[torch.Tensor]:
    injector_feature_names = list(multigraph_spec.get("graph_metadata", {}).get("injector_feature_names", []))
    if "wwir" not in injector_feature_names:
        return None
    wwir_idx = injector_feature_names.index("wwir")
    values = []
    for ds in future_dates:
        matrix = multigraph_spec.get("node_dynamic_features_by_time", {}).get(pd.Timestamp(ds), {}).get("injector")
        if matrix is None:
            continue
        values.append(np.asarray(matrix[:, [wwir_idx]], dtype=float).sum(axis=0, keepdims=False))
    if not values:
        return None
    arr = np.stack(values, axis=0).reshape(len(values), 1)  # [time, 1]
    return torch.as_tensor(arr, dtype=torch.float32)


def _edge_allocation_history(sample: Dict[str, Any], graph_type: str = "dyn") -> Optional[torch.Tensor]:
    allocations = []
    for snapshot in sample["history"]:
        for edge_type in snapshot.edge_types:
            if edge_type[1] != graph_type or edge_type[0] != "injector":
                continue
            edge_attr = snapshot[edge_type].edge_attr
            if edge_attr.numel() == 0:
                continue
            allocations.append(edge_attr[:, :1].transpose(0, 1))
            break
    if not allocations:
        return None
    return torch.stack(allocations, dim=1)  # [1, time, edges]


def _compute_sample_loss(
    model: STGNNPyG,
    sample: Dict[str, Any],
    loss_fn: nn.Module,
    multigraph_spec: Dict[str, Any],
    config: Any,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    history = _history_to_device(sample["history"], device)
    target = sample["target"].to(device)
    pred, diagnostics = model(history)
    data_loss = loss_fn(pred, target)
    log_parts = {"data_loss": float(data_loss.detach().cpu())}

    total_loss = data_loss
    if bool(config.physics_loss_enabled):
        schedule = 0.0
        warmup_epochs = max(int(config.physics_warmup_epochs), 1)
        if epoch >= warmup_epochs:
            progress = min(1.0, float(epoch - warmup_epochs + 1) / float(max(int(config.stgnn_max_epochs) - warmup_epochs + 1, 1)))
            schedule = float(config.physics_weight_init) + progress * (float(config.physics_weight_max) - float(config.physics_weight_init))
        inj_future = _future_injector_tensor(multigraph_spec, sample["forecast_dates"])
        alloc_history = _edge_allocation_history(sample)
        if inj_future is not None:
            physics = build_physics_loss_breakdown(
                pred.transpose(0, 1).unsqueeze(0),
                inj_future.unsqueeze(0).to(device),
                alloc_weights=alloc_history.to(device) if alloc_history is not None else None,
                lambdas={
                    "crm_residual": float(config.physics_lambda_crm),
                    "nonnegative": float(config.physics_lambda_nonneg),
                    "cumulative_monotonic": float(config.physics_lambda_cumulative),
                    "allocation_simplex": float(config.physics_lambda_simplex),
                    "shutin_consistency": float(config.physics_lambda_shutin),
                    "temporal_smoothness": float(config.physics_lambda_smoothness),
                },
            )
            physics_loss = schedule * physics["total"]
            total_loss = total_loss + physics_loss
            log_parts["physics_weight"] = schedule
            log_parts["physics_loss"] = float(physics_loss.detach().cpu())
            for name, value in physics.items():
                if name == "total":
                    continue
                log_parts[f"physics_{name}"] = float(value.detach().cpu())
    return total_loss, log_parts, pred


def _evaluate_samples(
    model: STGNNPyG,
    samples: List[Dict[str, Any]],
    loss_fn: nn.Module,
    multigraph_spec: Dict[str, Any],
    config: Any,
    device: torch.device,
    epoch: int,
) -> float:
    if not samples:
        return float("inf")
    losses = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            loss, _, _ = _compute_sample_loss(model, sample, loss_fn, multigraph_spec, config, device, epoch)
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
    graph_bundle = frames.get("graph_dataset")
    multigraph_spec = frames.get("multigraph_spec", {})
    if not graph_bundle or not graph_bundle.get("snapshots"):
        raise ValueError("stgnn_pyg requires graph_dataset snapshots in frames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, val_samples, test_sample = _find_window_splits(
        graph_bundle,
        train_cutoff=pd.Timestamp(frames["train_cutoff"]),
        test_start=pd.Timestamp(frames["test_start"]),
        config=config,
    )
    if not train_samples or test_sample is None:
        raise ValueError("Not enough graph windows to train/evaluate stgnn_pyg.")

    model = STGNNPyG(graph_bundle["metadata"], config).to(device)
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

    physics_logs: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    for epoch in range(1, max(int(config.stgnn_max_epochs), 1) + 1):
        model.train()
        train_losses = []
        for sample in train_samples:
            optimizer.zero_grad(set_to_none=True)
            loss, log_parts, _ = _compute_sample_loss(model, sample, loss_fn, multigraph_spec, config, device, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            log_parts["epoch"] = epoch
            physics_logs.append(log_parts)
        val_loss = _evaluate_samples(model, val_samples or train_samples[-1:], loss_fn, multigraph_spec, config, device, epoch)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max(int(config.stgnn_early_stop_patience), 1):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, pred = _compute_sample_loss(model, test_sample, loss_fn, multigraph_spec, config, device, epoch=0)

    pred_df = _prediction_frame(pred, test_sample["forecast_dates"], graph_bundle["metadata"]["producer_ids"])
    fusion_df = _fusion_weights_frame(model)
    edge_alloc_df = _edge_allocations_frame(model, multigraph_spec)
    physics_df = pd.DataFrame.from_records(physics_logs)
    well_metrics_df = _well_event_metrics(pred_df, frames["test_df"])
    training_summary = {
        "train_windows": len(train_samples),
        "val_windows": len(val_samples),
        "best_val_loss": best_val,
        "epochs_trained": int(physics_df["epoch"].max()) if not physics_df.empty else 0,
        "device": str(device),
    }
    return STGNNTrainingArtifacts(
        predictions=pred_df,
        graph_fusion_weights=fusion_df,
        edge_allocations=edge_alloc_df,
        physics_history=physics_df,
        well_event_metrics=well_metrics_df,
        training_summary=training_summary,
    )
