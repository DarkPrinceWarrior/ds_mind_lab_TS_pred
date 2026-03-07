"""Graph-based feature engineering for well production forecasting.

Builds graph representations of the well network and computes:
- Topology embeddings (Node2Vec / Spectral) encoding each well's position
- Centrality metrics (degree, betweenness, closeness, PageRank)
- Neighbor-aggregated production signals (1-hop GCN-style message passing)
- CRM connectivity summary features
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_well_graph(
    coords: pd.DataFrame,
    distances: Optional[pd.DataFrame] = None,
    well_types: Optional[Dict[str, str]] = None,
    crm_weights: Optional[pd.DataFrame] = None,
    distance_threshold: Optional[float] = None,
) -> nx.Graph:
    """Build a weighted undirected graph of wells.

    Edge weight = 1 / distance (proximity). If *crm_weights* are provided
    they are stored as a separate ``crm`` edge attribute so that downstream
    code can build multiple graph views.

    Args:
        coords: DataFrame with columns [well, x, y, z].
        distances: Pre-computed pairwise distance matrix (wells x wells).
        well_types: Mapping well_id -> "PROD" / "INJ".
        crm_weights: Optional DataFrame with CRM f_ij coefficients.
        distance_threshold: Drop edges longer than this (meters). If None,
            keep all edges.

    Returns:
        A ``networkx.Graph`` with node attributes (type, x, y, z) and edge
        attributes (distance, proximity, crm).
    """
    G = nx.Graph()
    wells = coords["well"].astype(str).tolist()
    coord_map = coords.set_index("well")[["x", "y", "z"]].to_dict("index")

    for w in wells:
        c = coord_map.get(w, {"x": 0.0, "y": 0.0, "z": 0.0})
        wtype = well_types.get(w, "UNK") if well_types else "UNK"
        G.add_node(w, type=wtype, x=c["x"], y=c["y"], z=c["z"])

    for i, w1 in enumerate(wells):
        for w2 in wells[i + 1:]:
            if distances is not None and w1 in distances.index and w2 in distances.columns:
                d = float(distances.loc[w1, w2])
            else:
                c1, c2 = coord_map.get(w1, {}), coord_map.get(w2, {})
                d = np.sqrt(
                    (c1.get("x", 0) - c2.get("x", 0)) ** 2
                    + (c1.get("y", 0) - c2.get("y", 0)) ** 2
                    + (c1.get("z", 0) - c2.get("z", 0)) ** 2
                )
            if np.isnan(d) or d <= 0:
                continue
            if distance_threshold is not None and d > distance_threshold:
                continue
            proximity = 1.0 / d
            crm_val = 0.0
            if crm_weights is not None and w1 in crm_weights.index and w2 in crm_weights.columns:
                crm_val = float(crm_weights.loc[w1, w2]) if not np.isnan(crm_weights.loc[w1, w2]) else 0.0
            G.add_edge(w1, w2, distance=d, proximity=proximity, crm=crm_val)

    logger.info("Built well graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Production clustering + graph sparsification (SGP-GCN)
# ---------------------------------------------------------------------------

def cluster_wells_by_production(
    df: pd.DataFrame,
    target_wells: List[str],
    value_col: str = "wlpr",
    max_k: int = 4,
    train_cutoff: Optional[pd.Timestamp] = None,
) -> Dict[str, int]:
    """Cluster wells by normalized production profile using K-Means.

    Selects optimal cluster count (2..max_k) via silhouette score.
    Returns mapping well_id -> cluster_label.
    """
    profiles = []
    well_order = []
    for well in target_wells:
        mask = df["well"].astype(str) == str(well)
        if train_cutoff is not None:
            mask = mask & (df["ds"] <= train_cutoff)
        series = df.loc[mask].sort_values("ds")[value_col].values.astype(float)
        if len(series) == 0:
            series = np.array([0.0])
        mu, std = series.mean(), series.std()
        normed = (series - mu) / std if std > 1e-9 else series - mu
        profiles.append(normed)
        well_order.append(well)

    # Pad/truncate to common length
    max_len = max(len(p) for p in profiles)
    X = np.zeros((len(profiles), max_len))
    for i, p in enumerate(profiles):
        X[i, :len(p)] = p

    if len(well_order) < 3:
        return {w: 0 for w in well_order}

    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, min(max_k + 1, len(well_order))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    cluster_map = {well_order[i]: int(best_labels[i]) for i in range(len(well_order))}
    cluster_sizes = {}
    for lbl in best_labels:
        cluster_sizes[int(lbl)] = cluster_sizes.get(int(lbl), 0) + 1

    logger.info(
        "Production clustering: k=%d, silhouette=%.3f, sizes=%s",
        best_k, best_score, cluster_sizes,
    )
    return cluster_map


def sparsify_graph_by_clusters(
    G: nx.Graph,
    cluster_map: Dict[str, int],
    inter_cluster_quantile: float = 0.5,
) -> nx.Graph:
    """Remove weak inter-cluster edges from the graph (SGP-GCN SPC algorithm).

    Edges between wells in the same cluster are kept.
    Edges between wells in different clusters are removed if their
    proximity weight is below the *inter_cluster_quantile* of all
    inter-cluster edge weights.

    Returns a new (pruned) graph; the original is not modified.
    """
    G_sparse = G.copy()
    inter_weights = []
    inter_edges = []
    for u, v, data in G.edges(data=True):
        cu = cluster_map.get(str(u), -1)
        cv = cluster_map.get(str(v), -1)
        if cu != cv and cu >= 0 and cv >= 0:
            inter_weights.append(data.get("proximity", 0.0))
            inter_edges.append((u, v))

    if not inter_weights:
        return G_sparse

    threshold = float(np.quantile(inter_weights, inter_cluster_quantile))
    removed = 0
    for (u, v), w in zip(inter_edges, inter_weights):
        if w < threshold:
            G_sparse.remove_edge(u, v)
            removed += 1

    logger.info(
        "Graph sparsification: removed %d/%d inter-cluster edges (threshold=%.6f, quantile=%.0f%%)",
        removed, len(inter_edges), threshold, inter_cluster_quantile * 100,
    )
    return G_sparse


# ---------------------------------------------------------------------------
# Centrality metrics
# ---------------------------------------------------------------------------

def compute_centrality_features(G: nx.Graph) -> pd.DataFrame:
    """Compute graph centrality metrics for every node.

    Returns a DataFrame indexed by well with columns:
        degree_centrality, betweenness_centrality, closeness_centrality,
        pagerank, eigenvector_centrality, clustering_coeff
    """
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="distance")
    closeness = nx.closeness_centrality(G, distance="distance")
    pr = nx.pagerank(G, weight="proximity")
    try:
        eigenvec = nx.eigenvector_centrality_numpy(G, weight="proximity")
    except Exception:
        eigenvec = {n: 0.0 for n in G.nodes()}
    clustering = nx.clustering(G, weight="proximity")

    records = []
    for node in G.nodes():
        records.append({
            "well": node,
            "degree_centrality": degree.get(node, 0.0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "closeness_centrality": closeness.get(node, 0.0),
            "pagerank": pr.get(node, 0.0),
            "eigenvector_centrality": eigenvec.get(node, 0.0),
            "clustering_coeff": clustering.get(node, 0.0),
        })
    df = pd.DataFrame(records)
    logger.info("Computed 6 centrality features for %d wells", len(df))
    return df


# ---------------------------------------------------------------------------
# Graph embeddings (Node2Vec + Spectral)
# ---------------------------------------------------------------------------

def compute_node2vec_embeddings(
    G: nx.Graph,
    dimensions: int = 4,
    walk_length: int = 20,
    num_walks: int = 50,
    p: float = 1.0,
    q: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Node2Vec embeddings that encode structural position in the well graph.

    Uses return parameter *p* and in-out parameter *q* to balance between
    BFS-like (local neighbourhood) and DFS-like (global structure) walks.
    Setting q > 1 biases toward BFS, which captures local clustering --
    appropriate for spatial well graphs.
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        logger.warning("node2vec not installed, skipping Node2Vec embeddings")
        return pd.DataFrame(columns=["well"] + [f"n2v_{i}" for i in range(dimensions)])

    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=walk_length,
        num_walks=num_walks, p=p, q=q, seed=seed, quiet=True,
        weight_key="proximity",
    )
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    records = []
    for node in G.nodes():
        vec = model.wv[str(node)] if str(node) in model.wv else np.zeros(dimensions)
        row = {"well": node}
        for i in range(dimensions):
            row[f"n2v_{i}"] = float(vec[i])
        records.append(row)

    df = pd.DataFrame(records)
    logger.info("Computed %d-dim Node2Vec embeddings for %d wells", dimensions, len(df))
    return df


def compute_spectral_embeddings(
    G: nx.Graph,
    n_components: int = 4,
) -> pd.DataFrame:
    """Compute spectral embeddings from the graph Laplacian.

    Uses the smallest non-trivial eigenvectors of the normalized Laplacian,
    which capture global connectivity patterns (Fiedler vector = community
    structure, higher eigenvectors = finer partitions).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n < n_components + 1:
        logger.warning("Too few nodes (%d) for %d spectral components", n, n_components)
        return pd.DataFrame(columns=["well"] + [f"spectral_{i}" for i in range(n_components)])

    node_idx = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = data.get("proximity", 1.0)
        adj[i, j] = w
        adj[j, i] = w

    L = laplacian(adj, normed=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    embedding = eigenvectors[:, 1: n_components + 1]

    records = []
    for i, node in enumerate(nodes):
        row = {"well": node}
        for c in range(embedding.shape[1]):
            row[f"spectral_{c}"] = float(embedding[i, c])
        records.append(row)

    df = pd.DataFrame(records)
    logger.info("Computed %d-dim spectral embeddings for %d wells", embedding.shape[1], len(df))
    return df


# ---------------------------------------------------------------------------
# DTW similarity between well production curves
# ---------------------------------------------------------------------------

def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute DTW distance between two 1-D time series using full cost matrix."""
    n, m = len(s1), len(s2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (s1[i - 1] - s2[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, m]))


def compute_dtw_similarity_matrix(
    df: pd.DataFrame,
    target_wells: List[str],
    value_col: str = "wlpr",
    sigma: Optional[float] = None,
    train_cutoff: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute pairwise DTW similarity matrix between well production curves.

    Uses the rate-of-change (diff) of *value_col* following STA-MGCN paper:
    DTW on derivative curves captures shape similarity regardless of scale.

    Similarity = exp(-dtw_dist^2 / (2 * sigma^2))  (Gaussian kernel).
    If *sigma* is None, it is set to the median DTW distance (self-tuning).

    Only uses data up to *train_cutoff* to prevent leakage.
    """
    well_curves: Dict[str, np.ndarray] = {}
    for well in target_wells:
        mask = df["well"].astype(str) == str(well)
        if train_cutoff is not None:
            mask = mask & (df["ds"] <= train_cutoff)
        series = df.loc[mask].sort_values("ds")[value_col].values
        diff = np.diff(series) if len(series) > 1 else np.array([0.0])
        mu, std = diff.mean(), diff.std()
        well_curves[well] = (diff - mu) / std if std > 1e-9 else diff - mu

    n = len(target_wells)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance(well_curves[target_wells[i]], well_curves[target_wells[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    if sigma is None:
        upper = dist_matrix[np.triu_indices(n, k=1)]
        sigma = float(np.median(upper)) if len(upper) > 0 else 1.0
    sigma = max(sigma, 1e-9)

    sim_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(sim_matrix, 0.0)

    sim_df = pd.DataFrame(sim_matrix, index=target_wells, columns=target_wells)
    logger.info(
        "Computed DTW similarity matrix for %d wells (sigma=%.2f, median_dist=%.2f)",
        n, sigma, float(np.median(dist_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
    )
    return sim_df


def compute_dtw_neighbor_aggregated_features(
    df: pd.DataFrame,
    dtw_sim: pd.DataFrame,
    feature_cols: List[str],
    k: int = 5,
) -> pd.DataFrame:
    """Weighted average of neighbors' features using DTW similarity weights.

    Same logic as geographic neighbor aggregation but uses DTW-based
    similarity instead of distance-based proximity.

    New columns: ``dtw_neighbor_avg_{col}`` for each col in *feature_cols*.
    """
    df = df.copy()
    wells = sorted(df["well"].astype(str).unique())
    new_cols = [f"dtw_neighbor_avg_{col}" for col in feature_cols]
    for col in new_cols:
        df[col] = 0.0

    neighbor_weights: Dict[str, List[Tuple[str, float]]] = {}
    for well in wells:
        if well not in dtw_sim.index:
            neighbor_weights[well] = []
            continue
        sims = dtw_sim.loc[well].to_dict()
        neighbors = [(nbr, w) for nbr, w in sims.items() if nbr != well and w > 0]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[:k]
        total_w = sum(w for _, w in neighbors)
        if total_w > 0:
            neighbors = [(n, w / total_w) for n, w in neighbors]
        neighbor_weights[well] = neighbors

    dates = sorted(df["ds"].unique())
    for date in dates:
        date_mask = df["ds"] == date
        date_data = df.loc[date_mask].copy()
        date_data["well"] = date_data["well"].astype(str)
        date_data = date_data.set_index("well")
        for well in wells:
            if well not in date_data.index:
                continue
            well_mask = date_mask & (df["well"].astype(str) == well)
            nbrs = neighbor_weights.get(well, [])
            if not nbrs:
                continue
            agg = np.zeros(len(feature_cols))
            for nbr, w in nbrs:
                if nbr in date_data.index:
                    vals = date_data.loc[nbr, feature_cols]
                    if isinstance(vals, pd.DataFrame):
                        vals = vals.iloc[0]
                    agg += w * vals.astype(float).fillna(0.0).values
            for i, col in enumerate(feature_cols):
                df.loc[well_mask, f"dtw_neighbor_avg_{col}"] = agg[i]

    logger.info(
        "Computed DTW neighbor-aggregated features for %d columns x %d wells",
        len(feature_cols), len(wells),
    )
    return df


# ---------------------------------------------------------------------------
# Neighbor-aggregated production (1-hop GCN-style message passing)
# ---------------------------------------------------------------------------

def compute_neighbor_aggregated_features(
    df: pd.DataFrame,
    G: nx.Graph,
    feature_cols: List[str],
    k: int = 5,
    weight_key: str = "proximity",
) -> pd.DataFrame:
    """For each well at each timestep, compute weighted average of neighbors' features.

    This is equivalent to a single GCN message-passing layer:
        h_i = sum_j (w_ij / sum_k w_ik) * x_j

    Args:
        df: DataFrame with columns [well, ds] + feature_cols.
        G: Well graph.
        feature_cols: Columns to aggregate from neighbors.
        k: Max neighbors to consider.
        weight_key: Edge attribute to use as weight.

    Returns:
        DataFrame with new columns ``neighbor_avg_{col}`` for each col.
    """
    df = df.copy()
    wells = sorted(df["well"].unique())
    new_cols = [f"neighbor_avg_{col}" for col in feature_cols]
    for col in new_cols:
        df[col] = 0.0

    neighbor_weights: Dict[str, List[Tuple[str, float]]] = {}
    for well in wells:
        if well not in G:
            neighbor_weights[well] = []
            continue
        neighbors = []
        for nbr, edata in G[well].items():
            w = edata.get(weight_key, 1.0)
            neighbors.append((nbr, w))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[:k]
        total_w = sum(w for _, w in neighbors)
        if total_w > 0:
            neighbors = [(n, w / total_w) for n, w in neighbors]
        neighbor_weights[well] = neighbors

    dates = sorted(df["ds"].unique())
    for date in dates:
        date_mask = df["ds"] == date
        date_data = df.loc[date_mask].set_index("well")
        for well in wells:
            if well not in date_data.index:
                continue
            well_mask = date_mask & (df["well"] == well)
            nbrs = neighbor_weights.get(well, [])
            if not nbrs:
                continue
            agg = np.zeros(len(feature_cols))
            for nbr, w in nbrs:
                if nbr in date_data.index:
                    vals = date_data.loc[nbr, feature_cols]
                    if isinstance(vals, pd.DataFrame):
                        vals = vals.iloc[0]
                    agg += w * vals.astype(float).fillna(0.0).values
            for i, col in enumerate(feature_cols):
                df.loc[well_mask, f"neighbor_avg_{col}"] = agg[i]

    logger.info("Computed neighbor-aggregated features for %d columns x %d wells", len(feature_cols), len(wells))
    return df


# ---------------------------------------------------------------------------
# CRM connectivity summary
# ---------------------------------------------------------------------------

def compute_crm_connectivity_features(
    pair_summary: pd.DataFrame,
    target_wells: List[str],
) -> pd.DataFrame:
    """Summarize CRM connectivity per producer well.

    From the injection pair summary, compute:
        crm_total_connectivity: sum of CRM weights from all injectors
        crm_max_connectivity: strongest single injector connection
        crm_num_injectors: number of connected injectors
        crm_avg_lag: average optimal lag (months)
        crm_avg_tau: average time constant
    """
    if pair_summary.empty:
        cols = ["well", "crm_total_connectivity", "crm_max_connectivity",
                "crm_num_injectors", "crm_avg_lag", "crm_avg_tau"]
        return pd.DataFrame(columns=cols)

    prod_col = "prod_id" if "prod_id" in pair_summary.columns else "prod_well"
    records = []
    for well in target_wells:
        well_pairs = pair_summary[pair_summary[prod_col].astype(str) == str(well)]
        if well_pairs.empty:
            records.append({
                "well": well, "crm_total_connectivity": 0.0,
                "crm_max_connectivity": 0.0, "crm_num_injectors": 0,
                "crm_avg_lag": 0.0, "crm_avg_tau": 0.0,
            })
            continue
        weights = well_pairs["weight"].fillna(0.0)
        records.append({
            "well": well,
            "crm_total_connectivity": float(weights.sum()),
            "crm_max_connectivity": float(weights.max()),
            "crm_num_injectors": int(len(well_pairs)),
            "crm_avg_lag": float(well_pairs["lag"].mean()) if "lag" in well_pairs.columns else 0.0,
            "crm_avg_tau": float(well_pairs["tau"].mean()) if "tau" in well_pairs.columns else 0.0,
        })
    df = pd.DataFrame(records)
    logger.info("Computed CRM connectivity features for %d producers", len(df))
    return df


# ---------------------------------------------------------------------------
# Public API: build all graph features at once
# ---------------------------------------------------------------------------

def build_graph_features(
    prod_df: pd.DataFrame,
    coords: pd.DataFrame,
    target_wells: List[str],
    pair_summary: pd.DataFrame,
    distances: Optional[pd.DataFrame] = None,
    well_types: Optional[Dict[str, str]] = None,
    n2v_dimensions: int = 4,
    spectral_components: int = 4,
    neighbor_agg_cols: Optional[List[str]] = None,
    neighbor_k: int = 5,
    seed: int = 42,
    dtw_agg_cols: Optional[List[str]] = None,
    dtw_k: int = 5,
    dtw_value_col: str = "wlpr",
    train_cutoff: Optional[pd.Timestamp] = None,
    sparsify_graph: bool = True,
    sparsify_max_k: int = 4,
    sparsify_inter_quantile: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build all graph-based features and merge into prod_df.

    Returns:
        (enriched_prod_df, static_graph_df) where static_graph_df contains
        per-well graph features for use as tiled covariates.
    """
    prod_df = prod_df.copy()
    well_col = "well" if "well" in prod_df.columns else "unique_id"
    for source_col in ["wlpr", "womr"]:
        if source_col in prod_df.columns:
            diff_col = f"{source_col}_diff1"
            cumsum_col = f"{source_col}_cumsum1"
            if diff_col not in prod_df.columns:
                prod_df[diff_col] = prod_df.groupby(well_col)[source_col].diff().fillna(0.0)
            if cumsum_col not in prod_df.columns:
                prod_df[cumsum_col] = prod_df.groupby(well_col)[source_col].cumsum().fillna(0.0)
    if "productivity_index" in prod_df.columns and "pseudo_productivity_index" not in prod_df.columns:
        prod_df["pseudo_productivity_index"] = prod_df["productivity_index"]

    G = build_well_graph(coords, distances=distances, well_types=well_types)

    # Centrality & embeddings use the full (dense) graph
    centrality = compute_centrality_features(G)
    n2v = compute_node2vec_embeddings(G, dimensions=n2v_dimensions, seed=seed)
    spectral = compute_spectral_embeddings(G, n_components=spectral_components)
    crm_conn = compute_crm_connectivity_features(pair_summary, target_wells)

    static_df = centrality.merge(n2v, on="well", how="outer")
    static_df = static_df.merge(spectral, on="well", how="outer")
    static_df = static_df.merge(crm_conn, on="well", how="outer")
    static_df = static_df.fillna(0.0)

    # Scale embeddings to [-1, 1] range for stable covariate input
    embed_cols = [c for c in static_df.columns if c.startswith(("n2v_", "spectral_"))]
    if embed_cols:
        scaler = StandardScaler()
        static_df[embed_cols] = scaler.fit_transform(static_df[embed_cols])

    static_cols = [c for c in static_df.columns if c != "well"]
    for col in static_cols:
        mapping = static_df.set_index("well")[col].to_dict()
        prod_df[col] = prod_df[well_col].astype(str).map(mapping).fillna(0.0)

    # Production clustering + graph sparsification for neighbor aggregation
    G_agg = G
    if sparsify_graph and len(target_wells) >= 3:
        cluster_map = cluster_wells_by_production(
            prod_df, target_wells,
            value_col=dtw_value_col if dtw_value_col in prod_df.columns else "wlpr",
            max_k=sparsify_max_k,
            train_cutoff=train_cutoff,
        )
        G_agg = sparsify_graph_by_clusters(G, cluster_map, inter_cluster_quantile=sparsify_inter_quantile)
        # Store cluster labels as a static feature
        prod_df["prod_cluster"] = prod_df[well_col].astype(str).map(cluster_map).fillna(-1).astype(int)

    if neighbor_agg_cols:
        available = [c for c in neighbor_agg_cols if c in prod_df.columns]
        if available:
            prod_df = compute_neighbor_aggregated_features(
                prod_df, G_agg, feature_cols=available, k=neighbor_k,
            )

    # DTW-based neighbor aggregation (dynamic similarity graph)
    if dtw_agg_cols:
        available_dtw = [c for c in dtw_agg_cols if c in prod_df.columns]
        if available_dtw and dtw_value_col in prod_df.columns:
            dtw_sim = compute_dtw_similarity_matrix(
                prod_df, target_wells,
                value_col=dtw_value_col,
                train_cutoff=train_cutoff,
            )
            prod_df = compute_dtw_neighbor_aggregated_features(
                prod_df, dtw_sim, feature_cols=available_dtw, k=dtw_k,
            )

    logger.info(
        "Graph features complete: %d static cols + geo-neighbor for %d cols + dtw-neighbor for %d cols",
        len(static_cols), len(neighbor_agg_cols or []), len(dtw_agg_cols or []),
    )
    return prod_df, static_df


def _safe_numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _series_to_matrix(
    df: pd.DataFrame,
    entity_col: str,
    entities: List[str],
    feature_cols: List[str],
    dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, np.ndarray]:
    if not feature_cols:
        return {pd.Timestamp(ds): np.zeros((len(entities), 0), dtype=float) for ds in dates}
    work = df.copy()
    work[entity_col] = work[entity_col].astype(str)
    work["ds"] = pd.to_datetime(work["ds"])
    work = _safe_numeric_frame(work, feature_cols)
    matrices: Dict[pd.Timestamp, np.ndarray] = {}
    for ds in dates:
        snap = work[work["ds"] == pd.Timestamp(ds)].set_index(entity_col)
        rows: List[np.ndarray] = []
        for entity in entities:
            if entity in snap.index:
                values = snap.loc[entity, feature_cols]
                if isinstance(values, pd.DataFrame):
                    values = values.iloc[0]
                rows.append(values.to_numpy(dtype=float))
            else:
                rows.append(np.zeros(len(feature_cols), dtype=float))
        matrices[pd.Timestamp(ds)] = np.vstack(rows) if rows else np.zeros((0, len(feature_cols)), dtype=float)
    return matrices


def _coords_static_features(coords: pd.DataFrame, wells: List[str]) -> pd.DataFrame:
    coords_frame = coords.copy()
    coords_frame["well"] = coords_frame["well"].astype(str)
    coords_frame = coords_frame.rename(columns={"x": "coord_x", "y": "coord_y", "z": "coord_z"})
    for col in ["coord_x", "coord_y", "coord_z"]:
        if col not in coords_frame.columns:
            coords_frame[col] = 0.0
    center = coords_frame[["coord_x", "coord_y", "coord_z"]].mean()
    coords_frame["dist_from_center"] = np.sqrt(
        (coords_frame["coord_x"] - center["coord_x"]) ** 2
        + (coords_frame["coord_y"] - center["coord_y"]) ** 2
        + (coords_frame["coord_z"] - center["coord_z"]) ** 2
    )
    return coords_frame[coords_frame["well"].isin(wells)].drop_duplicates("well")


def _build_topology_edges(
    coords: pd.DataFrame,
    producer_ids: List[str],
    *,
    distances: Optional[pd.DataFrame] = None,
    top_k: int = 4,
) -> Tuple[
    Dict[Tuple[str, str, str], np.ndarray],
    Dict[pd.Timestamp, Dict[Tuple[str, str, str], np.ndarray]],
    List[str],
    Dict[Tuple[str, str, str], np.ndarray],
]:
    if not producer_ids:
        return {}, {}, ["distance_m", "proximity"], {}
    topo_coords = _coords_static_features(coords, producer_ids).set_index("well")
    dist_frame: Optional[pd.DataFrame] = None
    if distances is not None and not distances.empty:
        dist_frame = distances.copy()
        dist_frame.index = dist_frame.index.astype(str)
        dist_frame.columns = dist_frame.columns.astype(str)
    src_nodes: List[int] = []
    dst_nodes: List[int] = []
    attrs: List[List[float]] = []
    for i, src in enumerate(producer_ids):
        candidates: List[Tuple[str, float]] = []
        for dst in producer_ids:
            if src == dst:
                continue
            if dist_frame is not None and src in dist_frame.index and dst in dist_frame.columns:
                distance_val = float(dist_frame.loc[src, dst])
            else:
                if src not in topo_coords.index or dst not in topo_coords.index:
                    distance_val = np.nan
                else:
                    delta = topo_coords.loc[src, ["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=float) - topo_coords.loc[dst, ["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=float)
                    distance_val = float(np.linalg.norm(delta))
            if not np.isfinite(distance_val) or distance_val <= 0:
                continue
            candidates.append((dst, distance_val))
        candidates.sort(key=lambda item: item[1])
        for dst, distance_val in candidates[:max(int(top_k), 1)]:
            src_nodes.append(i)
            dst_nodes.append(producer_ids.index(dst))
            attrs.append([distance_val, 1.0 / max(distance_val, 1e-6)])
    edge_index = np.asarray([src_nodes, dst_nodes], dtype=np.int64) if src_nodes else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=float) if attrs else np.zeros((0, 2), dtype=float)
    edge_type = ("producer", "topo", "producer")
    return {edge_type: edge_index}, {}, ["distance_m", "proximity"], {edge_type: edge_attr}


def build_multigraph_spec(
    prod_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    pair_table: pd.DataFrame,
    coords: pd.DataFrame,
    distances: Optional[pd.DataFrame],
    config: Any,
) -> Dict[str, Any]:
    prod_frame = prod_df.copy()
    inj_frame = inj_df.copy()
    prod_frame["well"] = prod_frame["well"].astype(str)
    prod_frame["ds"] = pd.to_datetime(prod_frame["ds"])
    inj_frame["well"] = inj_frame["well"].astype(str)
    inj_frame["ds"] = pd.to_datetime(inj_frame["ds"])

    for source_col in ["wlpr", "womr"]:
        if source_col in prod_frame.columns:
            diff_col = f"{source_col}_diff1"
            cumsum_col = f"{source_col}_cumsum1"
            if diff_col not in prod_frame.columns:
                prod_frame[diff_col] = prod_frame.groupby("well")[source_col].diff().fillna(0.0)
            if cumsum_col not in prod_frame.columns:
                prod_frame[cumsum_col] = prod_frame.groupby("well")[source_col].cumsum().fillna(0.0)
    if "productivity_index" in prod_frame.columns and "pseudo_productivity_index" not in prod_frame.columns:
        prod_frame["pseudo_productivity_index"] = prod_frame["productivity_index"]
    if "wwir" in inj_frame.columns:
        if "wwir_diff1" not in inj_frame.columns:
            inj_frame["wwir_diff1"] = inj_frame.groupby("well")["wwir"].diff().fillna(0.0)
        if "wwir_cumsum1" not in inj_frame.columns:
            inj_frame["wwir_cumsum1"] = inj_frame.groupby("well")["wwir"].cumsum().fillna(0.0)

    producer_ids = sorted(prod_frame["well"].astype(str).unique())
    injector_ids = sorted(inj_frame["well"].astype(str).unique())
    dates = sorted(pd.to_datetime(prod_frame["ds"]).unique())
    graph_types = [graph_type for graph_type in config.resolved_graph_types() if graph_type in {"topo", "bin", "cond", "dyn", "causal"}]
    pair_attrs = dict(getattr(pair_table, "attrs", {})) if pair_table is not None else {}
    pair_table = pair_table.copy() if pair_table is not None else pd.DataFrame()
    if pair_attrs:
        pair_table.attrs.update(pair_attrs)
    if not pair_table.empty:
        pair_table["prod_id"] = pair_table["prod_id"].astype(str)
        pair_table["inj_id"] = pair_table["inj_id"].astype(str)

    producer_static_cols = [col for col in config.static_exog if col in prod_frame.columns]
    producer_static = (
        prod_frame.groupby("well")[producer_static_cols].first().reindex(producer_ids).fillna(0.0)
        if producer_static_cols
        else pd.DataFrame(index=producer_ids)
    )
    injector_static_base = _coords_static_features(coords, injector_ids).set_index("well").reindex(injector_ids)
    injector_static = injector_static_base.fillna(0.0)

    producer_feature_cols = [col for col in config.resolved_stgnn_feature_columns()["producer"] if col in prod_frame.columns]
    injector_feature_cols = [col for col in config.resolved_stgnn_feature_columns()["injector"] if col in inj_frame.columns]
    producer_dynamic = _series_to_matrix(prod_frame, "well", producer_ids, producer_feature_cols, dates)
    injector_dynamic = _series_to_matrix(inj_frame, "well", injector_ids, injector_feature_cols, dates)

    cluster_map: Dict[str, int] = {}
    if "prod_cluster" in prod_frame.columns:
        cluster_map = (
            prod_frame[["well", "prod_cluster"]]
            .drop_duplicates("well")
            .set_index("well")["prod_cluster"]
            .fillna(-1)
            .astype(int)
            .to_dict()
        )

    edge_index_dict_by_graph: Dict[str, Dict[Tuple[str, str, str], np.ndarray]] = {}
    edge_attr_dict_by_graph_and_time: Dict[str, Dict[pd.Timestamp, Dict[Tuple[str, str, str], np.ndarray]]] = {}
    edge_feature_names: Dict[str, List[str]] = {}
    edge_static_attrs: Dict[str, Dict[Tuple[str, str, str], np.ndarray]] = {}
    relation_groups: Dict[str, List[Tuple[str, str, str]]] = {}

    topo_edges, _, topo_names, topo_static_attrs = _build_topology_edges(
        coords,
        producer_ids,
        distances=distances,
        top_k=config.graph_neighbor_k,
    )
    if "topo" in graph_types:
        edge_index_dict_by_graph["topo"] = topo_edges
        edge_feature_names["topo"] = topo_names
        relation_groups["topo"] = list(topo_edges.keys())
        edge_static_attrs["topo"] = topo_static_attrs
        edge_attr_dict_by_graph_and_time["topo"] = {
            pd.Timestamp(ds): {edge_type: topo_static_attrs[edge_type].copy() for edge_type in topo_edges}
            for ds in dates
        }

    edge_static = pair_table.attrs.get("graph_edge_static", pd.DataFrame()) if hasattr(pair_table, "attrs") else pd.DataFrame()
    edge_temporal = pair_table.attrs.get("graph_edge_temporal", pd.DataFrame()) if hasattr(pair_table, "attrs") else pd.DataFrame()
    if isinstance(edge_static, pd.DataFrame) and not edge_static.empty and injector_ids and producer_ids:
        inj_index = {well: idx for idx, well in enumerate(injector_ids)}
        prod_index = {well: idx for idx, well in enumerate(producer_ids)}
        edge_static = edge_static.copy()
        edge_static["prod_id"] = edge_static["prod_id"].astype(str)
        edge_static["inj_id"] = edge_static["inj_id"].astype(str)
        valid_pairs = edge_static["prod_id"].isin(prod_index) & edge_static["inj_id"].isin(inj_index)
        edge_static = edge_static[valid_pairs].sort_values(["inj_id", "prod_id"]).reset_index(drop=True)

        base_src = edge_static["inj_id"].map(inj_index).to_numpy(dtype=np.int64)
        base_dst = edge_static["prod_id"].map(prod_index).to_numpy(dtype=np.int64)
        base_edge_index = np.vstack([base_src, base_dst]) if len(edge_static) else np.zeros((2, 0), dtype=np.int64)

        temporal_lookup: Dict[pd.Timestamp, pd.DataFrame] = {}
        if isinstance(edge_temporal, pd.DataFrame) and not edge_temporal.empty:
            dyn = edge_temporal.copy()
            dyn["ds"] = pd.to_datetime(dyn["ds"])
            dyn["prod_id"] = dyn["prod_id"].astype(str)
            dyn["inj_id"] = dyn["inj_id"].astype(str)
            temporal_lookup = {pd.Timestamp(ds): frame.set_index(["inj_id", "prod_id"]) for ds, frame in dyn.groupby("ds")}

        graph_specs = {
            "bin": {
                "features": ["edge_exists", "kernel_weight", "distance_m", "metric_distance_m"],
                "builder": lambda row, dyn_row: [
                    1.0,
                    float(row.get("kernel_weight", row.get("weight", 0.0))),
                    float(row.get("distance_m", 0.0)),
                    float(row.get("metric_distance_m", row.get("distance_m", 0.0))),
                ],
            },
            "cond": {
                "features": ["kernel_weight", "crm_weight", "corr", "lag", "tau", "causal_score", "attn_alpha_train_mean"],
                "builder": lambda row, dyn_row: [
                    float(row.get("kernel_weight", row.get("weight", 0.0))),
                    float(row.get("crm_weight", row.get("kernel_weight", row.get("weight", 0.0)))),
                    float(row.get("corr", 0.0)),
                    float(row.get("lag", 0.0)),
                    float(row.get("tau", 0.0)),
                    float(row.get("causal_score", 0.0)),
                    float(row.get("attn_alpha_train_mean", row.get("crm_weight", 0.0))),
                ],
            },
            "dyn": {
                "features": ["alpha_t", "lagged_rate_t", "crm_rate_t", "lagged_wwit_diff_t", "contribution_t", "stage_id", "regime_id"],
                "builder": lambda row, dyn_row: [
                    float((dyn_row or {}).get("alpha_t", row.get("attn_alpha_train_mean", row.get("kernel_weight", 0.0)))),
                    float((dyn_row or {}).get("lagged_rate_t", 0.0)),
                    float((dyn_row or {}).get("crm_rate_t", 0.0)),
                    float((dyn_row or {}).get("lagged_wwit_diff_t", 0.0)),
                    float((dyn_row or {}).get("contribution_t", 0.0)),
                    float((dyn_row or {}).get("stage_id", 0.0) or 0.0),
                    float((dyn_row or {}).get("regime_id", 0.0) or 0.0),
                ],
            },
            "causal": {
                "features": ["causal_score", "attn_alpha_train_mean", "attn_alpha_train_last", "attn_alpha_full_last", "lag", "tau"],
                "builder": lambda row, dyn_row: [
                    float(row.get("causal_score", 0.0)),
                    float(row.get("attn_alpha_train_mean", row.get("crm_weight", 0.0))),
                    float(row.get("attn_alpha_train_last", row.get("crm_weight", 0.0))),
                    float(row.get("attn_alpha_full_last", row.get("crm_weight", 0.0))),
                    float(row.get("lag", 0.0)),
                    float(row.get("tau", 0.0)),
                ],
            },
        }

        for graph_type in graph_types:
            if graph_type == "topo" or graph_type not in graph_specs:
                continue
            edge_type = ("injector", graph_type, "producer")
            edge_index_dict_by_graph[graph_type] = {edge_type: base_edge_index.copy()}
            edge_feature_names[graph_type] = graph_specs[graph_type]["features"]
            relation_groups[graph_type] = [edge_type]
            edge_static_attrs[graph_type] = {}
            time_map: Dict[pd.Timestamp, Dict[Tuple[str, str, str], np.ndarray]] = {}
            static_arr = np.vstack([graph_specs[graph_type]["builder"](row, None) for _, row in edge_static.iterrows()]) if len(edge_static) else np.zeros((0, len(graph_specs[graph_type]["features"])), dtype=float)
            edge_static_attrs[graph_type][edge_type] = static_arr
            for ds in dates:
                dyn_frame = temporal_lookup.get(pd.Timestamp(ds))
                rows: List[List[float]] = []
                for _, row in edge_static.iterrows():
                    dyn_row = None
                    if dyn_frame is not None and (row["inj_id"], row["prod_id"]) in dyn_frame.index:
                        dyn_record = dyn_frame.loc[(row["inj_id"], row["prod_id"])]
                        if isinstance(dyn_record, pd.DataFrame):
                            dyn_record = dyn_record.iloc[0]
                        dyn_row = dyn_record.to_dict()
                    rows.append(graph_specs[graph_type]["builder"](row, dyn_row))
                time_map[pd.Timestamp(ds)] = {edge_type: np.asarray(rows, dtype=float) if rows else np.zeros((0, len(graph_specs[graph_type]["features"])), dtype=float)}
            edge_attr_dict_by_graph_and_time[graph_type] = time_map

    if config.stgnn_use_reverse_edges:
        for graph_type, edge_map in list(edge_index_dict_by_graph.items()):
            reversed_edges: Dict[Tuple[str, str, str], np.ndarray] = {}
            static_attrs_map = edge_static_attrs.setdefault(graph_type, {})
            for edge_type, edge_index in list(edge_map.items()):
                src_type, relation, dst_type = edge_type
                reverse_type = (dst_type, f"rev_{relation}", src_type)
                reversed_edges[reverse_type] = edge_index[[1, 0], :].copy() if edge_index.size else edge_index.copy()
                relation_groups.setdefault(graph_type, []).append(reverse_type)
                if edge_type in static_attrs_map:
                    static_attrs_map[reverse_type] = static_attrs_map[edge_type].copy()
            edge_map.update(reversed_edges)
            if graph_type in edge_attr_dict_by_graph_and_time:
                for ds, attrs_by_type in edge_attr_dict_by_graph_and_time[graph_type].items():
                    for edge_type, edge_attr in list(attrs_by_type.items()):
                        src_type, relation, dst_type = edge_type
                        reverse_type = (dst_type, f"rev_{relation}", src_type)
                        attrs_by_type[reverse_type] = edge_attr.copy()

    actual_graph_types = [graph_type for graph_type in graph_types if graph_type in edge_index_dict_by_graph]

    graph_metadata = {
        "producer_ids": producer_ids,
        "injector_ids": injector_ids,
        "producer_feature_names": producer_feature_cols,
        "injector_feature_names": injector_feature_cols,
        "producer_static_feature_names": list(producer_static.columns),
        "injector_static_feature_names": list(injector_static.columns),
        "edge_feature_names": edge_feature_names,
        "relation_groups": relation_groups,
        "graph_types": actual_graph_types,
    }

    return {
        "graph_types": actual_graph_types,
        "dates": [pd.Timestamp(ds) for ds in dates],
        "node_static_features": {
            "producer": producer_static.to_numpy(dtype=float) if not producer_static.empty else np.zeros((len(producer_ids), 0), dtype=float),
            "injector": injector_static.to_numpy(dtype=float) if not injector_static.empty else np.zeros((len(injector_ids), 0), dtype=float),
        },
        "node_dynamic_features_by_time": {
            pd.Timestamp(ds): {
                "producer": producer_dynamic[pd.Timestamp(ds)],
                "injector": injector_dynamic[pd.Timestamp(ds)],
            }
            for ds in dates
        },
        "producer_targets_by_time": {
            pd.Timestamp(ds): producer_dynamic[pd.Timestamp(ds)][:, [producer_feature_cols.index("wlpr")]] if "wlpr" in producer_feature_cols else np.zeros((len(producer_ids), 1), dtype=float)
            for ds in dates
        },
        "edge_index_dict_by_graph": edge_index_dict_by_graph,
        "edge_attr_dict_by_graph_and_time": edge_attr_dict_by_graph_and_time,
        "edge_static_attrs": edge_static_attrs,
        "cluster_map": cluster_map,
        "graph_metadata": graph_metadata,
        "pair_table": edge_static if isinstance(edge_static, pd.DataFrame) else pd.DataFrame(),
    }


def apply_scenario_to_graphs(multigraph_spec: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    if not multigraph_spec:
        return multigraph_spec
    scenario_type = str((scenario or {}).get("type", "")).strip().lower()
    if scenario_type != "injector_shutoff":
        return deepcopy(multigraph_spec)

    edited = deepcopy(multigraph_spec)
    injector_id = str(scenario.get("well", "")).strip()
    cutoff = pd.Timestamp(scenario.get("date"))
    inj_ids = edited.get("graph_metadata", {}).get("injector_ids", [])
    if injector_id not in inj_ids:
        return edited
    inj_idx = inj_ids.index(injector_id)

    scenario_deltas: List[Dict[str, Any]] = []
    for ds, node_payload in edited.get("node_dynamic_features_by_time", {}).items():
        if pd.Timestamp(ds) < cutoff:
            continue
        injector_features = node_payload.get("injector")
        if injector_features is not None and injector_features.shape[0] > inj_idx:
            original = injector_features[inj_idx].copy()
            injector_features[inj_idx, :] = 0.0
            scenario_deltas.append(
                {
                    "ds": pd.Timestamp(ds),
                    "graph_type": "controls",
                    "inj_id": injector_id,
                    "prod_id": None,
                    "delta_l1": float(np.abs(original).sum()),
                }
            )
        for graph_type, attrs_by_time in edited.get("edge_attr_dict_by_graph_and_time", {}).items():
            if graph_type not in {"bin", "cond", "dyn", "causal"}:
                continue
            edge_map = edited.get("edge_index_dict_by_graph", {}).get(graph_type, {})
            for edge_type, edge_index in edge_map.items():
                if not edge_type[0] == "injector":
                    continue
                if pd.Timestamp(ds) not in attrs_by_time or edge_type not in attrs_by_time[pd.Timestamp(ds)]:
                    continue
                mask = edge_index[0] == inj_idx
                if not np.any(mask):
                    continue
                edge_attr = attrs_by_time[pd.Timestamp(ds)][edge_type]
                before = edge_attr[mask].copy()
                if graph_type == "bin":
                    edge_attr[mask] = 0.0
                elif graph_type == "cond":
                    edge_attr[mask] = edge_attr[mask] * 0.1
                else:
                    edge_attr[mask] = 0.0
                prod_ids = edited.get("graph_metadata", {}).get("producer_ids", [])
                prod_indices = edge_index[1][mask]
                for row_idx, prod_pos in enumerate(prod_indices):
                    scenario_deltas.append(
                        {
                            "ds": pd.Timestamp(ds),
                            "graph_type": graph_type,
                            "inj_id": injector_id,
                            "prod_id": prod_ids[int(prod_pos)] if int(prod_pos) < len(prod_ids) else None,
                            "delta_l1": float(np.abs(before[row_idx] - edge_attr[mask][row_idx]).sum()),
                        }
                    )
    edited["scenario_edge_deltas"] = pd.DataFrame.from_records(scenario_deltas)
    edited["scenario"] = dict(scenario)
    return edited
