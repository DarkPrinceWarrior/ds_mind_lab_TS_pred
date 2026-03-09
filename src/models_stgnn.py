from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    weights = torch.zeros_like(scores)
    for segment in range(max(int(num_segments), 0)):
        mask = index == segment
        if torch.any(mask):
            segment_weights = torch.softmax(scores[mask].float(), dim=0).to(dtype=scores.dtype)
            weights[mask] = segment_weights
    return weights


def _split_by_batch(values: torch.Tensor, batch_index: Optional[torch.Tensor]) -> torch.Tensor:
    if batch_index is None:
        return values
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
    counts = torch.bincount(batch_index, minlength=num_graphs).tolist()
    chunks = torch.split(values, counts, dim=0)
    return torch.stack(chunks, dim=0)


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


class EdgeAllocationHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        group_index: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_attr is None:
            edge_attr = src_emb.new_zeros((src_emb.size(0), 0))
        features = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        logits = self.net(features).squeeze(-1)
        weights = _segment_softmax(logits, group_index, num_groups).unsqueeze(-1)
        return weights, logits


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
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
        self.variant = str(getattr(config, "stgnn_variant", metadata.get("stgnn_variant", "legacy_multigraph"))).strip().lower()
        self.target_names = list(metadata.get("target_names", ["wlpr"]))
        self.target_dim = max(len(self.target_names), 1)
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
        self.fusion = None if (self.variant == "single_relation_multitask" or len(self.graph_types) == 1) else GraphFusionAttention(hidden_dim, self.graph_types)
        self.head = ProducerForecastHead(hidden_dim, int(config.horizon) * self.target_dim)
        self.forward_edge_types: Dict[str, tuple[str, str, str]] = {}
        self.edge_heads = nn.ModuleDict()
        for graph_type in self.graph_types:
            for edge_type in self.metadata.get("relation_groups", {}).get(graph_type, []):
                if edge_type[0] == "injector" and edge_type[2] == "producer":
                    self.forward_edge_types[graph_type] = edge_type
                    self.edge_heads[graph_type] = EdgeAllocationHead(hidden_dim)
                    break
        self.latest_fusion_weights: torch.Tensor | None = None
        self.latest_edge_allocations: Dict[str, torch.Tensor] = {}
        self.latest_allocation_history: Dict[str, torch.Tensor] = {}

    def _compute_edge_allocations(
        self,
        graph_type: str,
        snapshot: Any,
        encoded: Dict[str, torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        edge_type = self.forward_edge_types.get(graph_type)
        if edge_type is None or not hasattr(snapshot[edge_type], "edge_index"):
            return None, None
        edge_index = snapshot[edge_type].edge_index
        if edge_index.numel() == 0:
            return None, None
        edge_attr = getattr(snapshot[edge_type], "edge_attr", None)
        src_emb = encoded["injector"][edge_index[0]]
        dst_emb = encoded["producer"][edge_index[1]]
        weights, logits = self.edge_heads[graph_type](
            src_emb,
            dst_emb,
            edge_attr,
            edge_index[1],
            encoded["producer"].size(0),
        )
        context = {
            "edge_type": edge_type,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "src_index": edge_index[0],
            "dst_index": edge_index[1],
            "logits": logits,
        }
        producer_batch = getattr(snapshot["producer"], "batch", None)
        injector_batch = getattr(snapshot["injector"], "batch", None)
        if producer_batch is not None and injector_batch is not None:
            num_graphs = int(producer_batch.max().item()) + 1 if producer_batch.numel() else 1
            prod_counts = torch.bincount(producer_batch, minlength=num_graphs)
            inj_counts = torch.bincount(injector_batch, minlength=num_graphs)
            prod_ptr = torch.cat([prod_counts.new_zeros(1), torch.cumsum(prod_counts, dim=0)], dim=0)
            inj_ptr = torch.cat([inj_counts.new_zeros(1), torch.cumsum(inj_counts, dim=0)], dim=0)
            edge_batch = producer_batch[edge_index[1]]
            context["edge_batch"] = edge_batch
            context["src_batch"] = injector_batch[edge_index[0]]
            context["dst_batch"] = edge_batch
            context["src_local_index"] = edge_index[0] - inj_ptr[injector_batch[edge_index[0]]]
            context["dst_local_index"] = edge_index[1] - prod_ptr[edge_batch]
        return weights, context

    def forward(self, history: List[Any]) -> tuple[torch.Tensor, Dict[str, Any]]:
        branch_sequences: Dict[str, Dict[str, List[torch.Tensor]]] = {
            graph_type: {node_type: [] for node_type in self.node_types}
            for graph_type in self.graph_types
        }
        allocation_sequences: Dict[str, List[torch.Tensor]] = {
            graph_type: []
            for graph_type in self.graph_types
            if graph_type in self.edge_heads
        }
        edge_context: Dict[str, Dict[str, Any]] = {}
        for snapshot in history:
            for graph_type in self.graph_types:
                encoded = self.encoder.forward_branch(snapshot, graph_type)
                for node_type in self.node_types:
                    branch_sequences[graph_type][node_type].append(encoded[node_type])
                allocation_weights, context = self._compute_edge_allocations(graph_type, snapshot, encoded)
                if allocation_weights is not None:
                    allocation_sequences.setdefault(graph_type, []).append(allocation_weights)
                if context is not None:
                    edge_context[graph_type] = context

        branch_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        for graph_type in self.graph_types:
            branch_embeddings[graph_type] = {}
            for node_type in self.node_types:
                seq = torch.stack(branch_sequences[graph_type][node_type], dim=1)
                branch_embeddings[graph_type][node_type] = self.temporal[graph_type][node_type](seq)

        if self.fusion is None:
            graph_type = self.graph_types[0]
            fused = branch_embeddings[graph_type]
            fusion_weights = torch.ones((1,), device=fused["producer"].device, dtype=fused["producer"].dtype)
        else:
            fused, fusion_weights = self.fusion(branch_embeddings)
        self.latest_fusion_weights = fusion_weights.detach()
        pred = self.head(fused["producer"])
        if self.target_dim > 1:
            pred = pred.view(pred.size(0), int(self.config.horizon), self.target_dim)
        producer_batch = getattr(history[-1]["producer"], "batch", None)
        pred = _split_by_batch(pred, producer_batch)

        edge_allocations: Dict[str, torch.Tensor] = {}
        allocation_history: Dict[str, torch.Tensor] = {}
        for graph_type, sequence in allocation_sequences.items():
            if not sequence:
                continue
            history_tensor = torch.stack(sequence, dim=0).squeeze(-1)
            allocation_history[graph_type] = history_tensor
            edge_allocations[graph_type] = sequence[-1]
        self.latest_edge_allocations = edge_allocations
        self.latest_allocation_history = allocation_history
        diagnostics = {
            "fusion_weights": fusion_weights,
            "edge_allocations": edge_allocations,
            "allocation_history": allocation_history,
            "branch_embeddings": branch_embeddings,
            "edge_context": edge_context,
            "producer_batch": producer_batch,
        }
        return pred, diagnostics
