from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv, NNConv, SAGEConv, TransformerConv
except ImportError:  # pragma: no cover
    GATv2Conv = None
    HeteroConv = None
    NNConv = None
    SAGEConv = None
    TransformerConv = None


def _require_pyg() -> None:
    if HeteroConv is None or SAGEConv is None:
        raise ImportError(
            "stgnn_pyg requires torch-geometric. Install torch-geometric for the active torch/CUDA build."
        )


def _relation_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def _build_relation_conv(kind: str, hidden_dim: int, edge_dim: int):
    _require_pyg()
    normalized = str(kind or "sage").strip().lower()
    if normalized in {"sage", "graphsage"}:
        return SAGEConv((-1, -1), hidden_dim)
    if normalized in {"gat", "gatv2"} and GATv2Conv is not None:
        return GATv2Conv((-1, -1), hidden_dim, edge_dim=edge_dim, add_self_loops=False)
    if normalized in {"transformer", "transformerconv"} and TransformerConv is not None:
        return TransformerConv((-1, -1), hidden_dim, edge_dim=edge_dim, beta=True)
    if normalized == "nnconv" and NNConv is not None:
        edge_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim),
        )
        return NNConv(hidden_dim, hidden_dim, edge_network, aggr="mean")
    return SAGEConv((-1, -1), hidden_dim)


class TemporalBackbone(nn.Module):
    def __init__(self, mode: str, hidden_dim: int, temporal_hidden_dim: int):
        super().__init__()
        self.mode = str(mode or "gru").strip().lower()
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        if self.mode == "gru":
            self.net = nn.GRU(hidden_dim, temporal_hidden_dim, batch_first=True)
            self.out_proj = nn.Linear(temporal_hidden_dim, hidden_dim)
        elif self.mode == "tcn":
            self.net = nn.Sequential(
                nn.Conv1d(hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.out_proj = nn.Linear(temporal_hidden_dim, hidden_dim)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 16),
                batch_first=True,
            )
            self.net = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Temporal backbone expects [num_nodes, time, hidden], got {tuple(x.shape)}")
        if self.mode == "gru":
            out, _ = self.net(x)
            return self.out_proj(out[:, -1, :])
        if self.mode == "tcn":
            out = self.net(x.transpose(1, 2)).transpose(1, 2)
            return self.out_proj(out[:, -1, :])
        out = self.net(x)
        return self.out_proj(out[:, -1, :])


class GraphFusionAttention(nn.Module):
    def __init__(self, hidden_dim: int, graph_types: List[str]):
        super().__init__()
        self.graph_types = list(graph_types)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, branch_embeddings: Dict[str, Dict[str, torch.Tensor]]) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        graph_summaries: List[torch.Tensor] = []
        for graph_type in self.graph_types:
            payload = branch_embeddings[graph_type]
            pieces = [value.mean(dim=0) for value in payload.values() if value.numel() > 0]
            summary = torch.stack(pieces, dim=0).mean(dim=0) if pieces else torch.zeros_like(next(iter(payload.values())).mean(dim=0))
            graph_summaries.append(summary)
        stacked = torch.stack(graph_summaries, dim=0)
        weights = torch.softmax(self.scorer(stacked).squeeze(-1), dim=0)
        fused: Dict[str, torch.Tensor] = {}
        node_types = list(next(iter(branch_embeddings.values())).keys())
        for node_type in node_types:
            fused[node_type] = sum(
                weights[idx] * branch_embeddings[graph_type][node_type]
                for idx, graph_type in enumerate(self.graph_types)
            )
        return fused, weights


class MultiGraphHeteroEncoder(nn.Module):
    def __init__(self, metadata: Dict[str, Any], config: Any):
        super().__init__()
        _require_pyg()
        self.graph_types = list(metadata.get("graph_types", []))
        self.node_types = ["producer", "injector"]
        self.hidden_dim = int(config.stgnn_hidden_dim)
        self.edge_feature_names = dict(metadata.get("edge_feature_names", {}))
        self.relation_groups = dict(metadata.get("relation_groups", {}))
        self.input_proj = nn.ModuleDict({node_type: nn.LazyLinear(self.hidden_dim) for node_type in self.node_types})
        self.dropout = nn.Dropout(float(config.stgnn_feature_dropout))
        self.branch_convs = nn.ModuleDict()
        self.branch_uses_edge_attr: Dict[str, bool] = {}

        for graph_type in self.graph_types:
            relation_kind = config.stgnn_relation_conv.get(graph_type, "SAGE")
            use_edge_attr = str(relation_kind).strip().lower() not in {"sage", "graphsage"}
            self.branch_uses_edge_attr[graph_type] = use_edge_attr
            layers = nn.ModuleList()
            for _ in range(max(int(config.stgnn_layers), 1)):
                convs = {}
                edge_dim = max(len(self.edge_feature_names.get(graph_type, [])), 1)
                for edge_type in self.relation_groups.get(graph_type, []):
                    convs[edge_type] = _build_relation_conv(relation_kind, self.hidden_dim, edge_dim=edge_dim)
                layers.append(HeteroConv(convs, aggr="sum"))
            self.branch_convs[graph_type] = layers

    def project_inputs(self, snapshot: Any) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            x = snapshot[node_type].x
            projected[node_type] = self.dropout(F.relu(self.input_proj[node_type](x)))
        return projected

    def forward_branch(self, snapshot: Any, graph_type: str) -> Dict[str, torch.Tensor]:
        x_dict = self.project_inputs(snapshot)
        for conv in self.branch_convs[graph_type]:
            edge_index_dict = {
                edge_type: snapshot[edge_type].edge_index
                for edge_type in self.relation_groups.get(graph_type, [])
                if hasattr(snapshot[edge_type], "edge_index")
            }
            if not edge_index_dict:
                return x_dict
            if self.branch_uses_edge_attr.get(graph_type, False):
                edge_attr_dict = {edge_type: snapshot[edge_type].edge_attr for edge_type in edge_index_dict}
                out = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                out = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: self.dropout(F.relu(out.get(node_type, x_dict[node_type])))
                for node_type in self.node_types
            }
        return x_dict


class ProducerForecastHead(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STGNNPyG(nn.Module):
    def __init__(self, metadata: Dict[str, Any], config: Any):
        super().__init__()
        self.metadata = metadata
        self.config = config
        self.graph_types = list(metadata.get("graph_types", []))
        self.node_types = ["producer", "injector"]
        hidden_dim = int(config.stgnn_hidden_dim)
        temporal_hidden_dim = int(config.stgnn_temporal_hidden_dim)
        self.encoder = MultiGraphHeteroEncoder(metadata, config)
        self.temporal = nn.ModuleDict(
            {
                graph_type: nn.ModuleDict(
                    {node_type: TemporalBackbone(config.stgnn_temporal_backbone, hidden_dim, temporal_hidden_dim) for node_type in self.node_types}
                )
                for graph_type in self.graph_types
            }
        )
        self.fusion = GraphFusionAttention(hidden_dim, self.graph_types)
        self.head = ProducerForecastHead(hidden_dim, int(config.horizon))
        self.latest_fusion_weights: torch.Tensor | None = None
        self.latest_edge_allocations: Dict[str, torch.Tensor] = {}

    def forward(self, history: List[Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        branch_sequences: Dict[str, Dict[str, List[torch.Tensor]]] = {
            graph_type: {node_type: [] for node_type in self.node_types}
            for graph_type in self.graph_types
        }
        for snapshot in history:
            for graph_type in self.graph_types:
                encoded = self.encoder.forward_branch(snapshot, graph_type)
                for node_type in self.node_types:
                    branch_sequences[graph_type][node_type].append(encoded[node_type])

        branch_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        for graph_type in self.graph_types:
            branch_embeddings[graph_type] = {}
            for node_type in self.node_types:
                seq = torch.stack(branch_sequences[graph_type][node_type], dim=1)
                branch_embeddings[graph_type][node_type] = self.temporal[graph_type][node_type](seq)

        fused, fusion_weights = self.fusion(branch_embeddings)
        self.latest_fusion_weights = fusion_weights.detach()
        pred = self.head(fused["producer"])

        edge_allocations: Dict[str, torch.Tensor] = {}
        last_snapshot = history[-1]
        for graph_type in self.graph_types:
            allocations: List[torch.Tensor] = []
            for edge_type in self.metadata.get("relation_groups", {}).get(graph_type, []):
                if edge_type[0] != "injector" or not hasattr(last_snapshot[edge_type], "edge_attr"):
                    continue
                edge_attr = last_snapshot[edge_type].edge_attr
                if edge_attr.numel() == 0:
                    continue
                allocations.append(edge_attr[:, :1])
            if allocations:
                edge_allocations[graph_type] = torch.cat(allocations, dim=0)
        self.latest_edge_allocations = edge_allocations
        diagnostics = {
            "fusion_weights": fusion_weights,
            "edge_allocations": edge_allocations,
            "branch_embeddings": branch_embeddings,
        }
        return pred, diagnostics
