from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from .features_graph import apply_scenario_to_graphs
except ImportError:  # pragma: no cover
    from features_graph import apply_scenario_to_graphs


def _require_pyg():
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "stgnn_pyg requires torch-geometric. Install a torch-geometric wheel matching your torch/CUDA build."
        ) from exc
    return HeteroData


def _safe_tensor(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def to_heterodata_snapshots(
    multigraph_spec: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    HeteroData = _require_pyg()
    dates = [pd.Timestamp(ds) for ds in multigraph_spec.get("dates", [])]
    producer_static = np.asarray(multigraph_spec.get("node_static_features", {}).get("producer", np.zeros((0, 0))), dtype=float)
    injector_static = np.asarray(multigraph_spec.get("node_static_features", {}).get("injector", np.zeros((0, 0))), dtype=float)
    snapshots: List[Any] = []

    for ds in dates:
        payload = multigraph_spec["node_dynamic_features_by_time"][pd.Timestamp(ds)]
        producer_dynamic = np.asarray(payload.get("producer", np.zeros((producer_static.shape[0], 0))), dtype=float)
        injector_dynamic = np.asarray(payload.get("injector", np.zeros((injector_static.shape[0], 0))), dtype=float)
        data = HeteroData()
        data["producer"].x = _safe_tensor(
            np.concatenate([producer_static, producer_dynamic], axis=1) if producer_static.size or producer_dynamic.size else np.zeros((producer_dynamic.shape[0], 0), dtype=float)
        )
        data["injector"].x = _safe_tensor(
            np.concatenate([injector_static, injector_dynamic], axis=1) if injector_static.size or injector_dynamic.size else np.zeros((injector_dynamic.shape[0], 0), dtype=float)
        )
        data["producer"].num_nodes = int(data["producer"].x.shape[0])
        data["injector"].num_nodes = int(data["injector"].x.shape[0])
        target = multigraph_spec.get("producer_targets_by_time", {}).get(pd.Timestamp(ds))
        if target is not None:
            data["producer"].y = _safe_tensor(np.asarray(target, dtype=float))
        data.ds = pd.Timestamp(ds)

        for graph_type, edge_dict in multigraph_spec.get("edge_index_dict_by_graph", {}).items():
            attrs_by_type = multigraph_spec.get("edge_attr_dict_by_graph_and_time", {}).get(graph_type, {}).get(pd.Timestamp(ds), {})
            for edge_type, edge_index in edge_dict.items():
                edge_attr = attrs_by_type.get(edge_type)
                if edge_attr is None:
                    static_attrs = multigraph_spec.get("edge_static_attrs", {}).get(graph_type, {}).get(edge_type)
                    edge_attr = static_attrs if static_attrs is not None else np.zeros((edge_index.shape[1], 0), dtype=float)
                data[edge_type].edge_index = _safe_tensor(edge_index, dtype=torch.long)
                data[edge_type].edge_attr = _safe_tensor(np.asarray(edge_attr, dtype=float))
        snapshots.append(data)

    metadata = {
        "producer_ids": list(multigraph_spec.get("graph_metadata", {}).get("producer_ids", [])),
        "injector_ids": list(multigraph_spec.get("graph_metadata", {}).get("injector_ids", [])),
        "producer_feature_names": list(multigraph_spec.get("graph_metadata", {}).get("producer_feature_names", [])),
        "injector_feature_names": list(multigraph_spec.get("graph_metadata", {}).get("injector_feature_names", [])),
        "producer_static_feature_names": list(multigraph_spec.get("graph_metadata", {}).get("producer_static_feature_names", [])),
        "injector_static_feature_names": list(multigraph_spec.get("graph_metadata", {}).get("injector_static_feature_names", [])),
        "edge_feature_names": dict(multigraph_spec.get("graph_metadata", {}).get("edge_feature_names", {})),
        "relation_groups": dict(multigraph_spec.get("graph_metadata", {}).get("relation_groups", {})),
        "graph_types": list(multigraph_spec.get("graph_types", [])),
    }
    return {
        "dates": dates,
        "snapshots": snapshots,
        "metadata": metadata,
        "multigraph_spec": multigraph_spec,
    }


def build_temporal_graph_windows(
    graph_bundle: Dict[str, Any],
    input_size: int,
    horizon: int,
) -> List[Dict[str, Any]]:
    snapshots = list(graph_bundle.get("snapshots", []))
    dates = [pd.Timestamp(ds) for ds in graph_bundle.get("dates", [])]
    producer_ids = list(graph_bundle.get("metadata", {}).get("producer_ids", []))
    windows: List[Dict[str, Any]] = []
    if not snapshots or len(snapshots) < input_size + horizon:
        return windows

    for end_idx in range(input_size - 1, len(snapshots) - horizon):
        history = snapshots[end_idx - input_size + 1: end_idx + 1]
        future_dates = dates[end_idx + 1: end_idx + 1 + horizon]
        target_steps: List[np.ndarray] = []
        for future_idx in range(end_idx + 1, end_idx + 1 + horizon):
            target_steps.append(snapshots[future_idx]["producer"].y.cpu().numpy())
        target = np.concatenate(target_steps, axis=1) if target_steps else np.zeros((len(producer_ids), horizon), dtype=float)
        windows.append(
            {
                "history": history,
                "history_dates": dates[end_idx - input_size + 1: end_idx + 1],
                "forecast_dates": future_dates,
                "target": torch.as_tensor(target, dtype=torch.float32),
                "cutoff_date": dates[end_idx],
            }
        )
    return windows


def apply_scenario_to_graph_bundle(
    graph_bundle: Dict[str, Any],
    scenario: Optional[Dict[str, Any]],
    config: Any,
) -> Dict[str, Any]:
    if not scenario:
        return deepcopy(graph_bundle)
    edited_spec = apply_scenario_to_graphs(graph_bundle.get("multigraph_spec", {}), scenario)
    edited_bundle = to_heterodata_snapshots(edited_spec, config)
    edited_bundle["scenario"] = dict(scenario)
    edited_bundle["scenario_edge_deltas"] = edited_spec.get("scenario_edge_deltas", pd.DataFrame())
    return edited_bundle
