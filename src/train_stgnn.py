from __future__ import annotations

from dataclasses import dataclass
import logging
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

logger = logging.getLogger(__name__)


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


def _build_physics_inputs(
    sample: Dict[str, Any],
    multigraph_spec: Dict[str, Any],
    diagnostics: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    graph_type = _select_physics_graph_type(diagnostics)
    if graph_type is None:
        return {}
    edge_context = diagnostics.get("edge_context", {}).get(graph_type)
    latest_alloc = diagnostics.get("edge_allocations", {}).get(graph_type)
    alloc_history = diagnostics.get("allocation_history", {}).get(graph_type)
    if edge_context is None or latest_alloc is None:
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

    edge_index = edge_context["edge_index"]
    edge_attr = edge_context.get("edge_attr")
    edge_feature_names = list(multigraph_spec.get("graph_metadata", {}).get("edge_feature_names", {}).get(graph_type, []))
    lag_idx = edge_feature_names.index("lag") if "lag" in edge_feature_names else None
    tau_idx = edge_feature_names.index("tau") if "tau" in edge_feature_names else None

    inj_rates = inj_rates.to(device)
    latest_alloc = latest_alloc.squeeze(-1).to(device)
    alloc_history = alloc_history.to(device).unsqueeze(0) if alloc_history is not None else None
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None

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


def _compute_sample_loss(
    model: STGNNPyG,
    sample: Dict[str, Any],
    loss_fn: nn.Module,
    multigraph_spec: Dict[str, Any],
    frames: Dict[str, Any],
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
        physics_inputs = _build_physics_inputs(sample, multigraph_spec, diagnostics, device)
        if physics_inputs.get("producer_forcing") is not None:
            physics = build_physics_loss_breakdown(
                pred.transpose(0, 1).unsqueeze(0),
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
    frames: Dict[str, Any],
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
            loss, _, _ = _compute_sample_loss(model, sample, loss_fn, multigraph_spec, frames, config, device, epoch)
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
    max_epochs = max(int(config.stgnn_max_epochs), 1)

    logger.info(
        "STGNN training start: train_windows=%d, val_windows=%d, horizon=%d, input_size=%d, max_epochs=%d, device=%s",
        len(train_samples),
        len(val_samples),
        int(config.horizon),
        int(config.input_size),
        max_epochs,
        device,
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for sample in train_samples:
            optimizer.zero_grad(set_to_none=True)
            loss, log_parts, _ = _compute_sample_loss(model, sample, loss_fn, multigraph_spec, frames, config, device, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            log_parts["epoch"] = epoch
            physics_logs.append(log_parts)
        val_loss = _evaluate_samples(model, val_samples or train_samples[-1:], loss_fn, multigraph_spec, frames, config, device, epoch)
        scheduler.step(val_loss)
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if epoch == 1 or epoch == max_epochs or epoch % 5 == 0:
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max(int(config.stgnn_early_stop_patience), 1):
                logger.info(
                    "STGNN early stopping at epoch %d: best_val=%.5f, patience=%d",
                    epoch,
                    float(best_val),
                    patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, pred = _compute_sample_loss(model, test_sample, loss_fn, multigraph_spec, frames, config, device, epoch=0)

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
    logger.info(
        "STGNN training complete: epochs=%d, best_val=%.5f, forecast_rows=%d",
        training_summary["epochs_trained"],
        float(best_val),
        len(pred_df),
    )
    return STGNNTrainingArtifacts(
        predictions=pred_df,
        graph_fusion_weights=fusion_df,
        edge_allocations=edge_alloc_df,
        physics_history=physics_df,
        well_event_metrics=well_metrics_df,
        training_summary=training_summary,
    )
