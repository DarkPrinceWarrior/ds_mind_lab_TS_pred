"""Graph-based feature engineering for well production forecasting.

Builds graph representations of the well network and computes:
- Topology embeddings (Node2Vec / Spectral) encoding each well's position
- Centrality metrics (degree, betweenness, closeness, PageRank)
- Neighbor-aggregated production signals (1-hop GCN-style message passing)
- CRM connectivity summary features
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build all graph-based features and merge into prod_df.

    Returns:
        (enriched_prod_df, static_graph_df) where static_graph_df contains
        per-well graph features for use as tiled covariates.
    """
    G = build_well_graph(coords, distances=distances, well_types=well_types)

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

    prod_df = prod_df.copy()
    well_col = "well" if "well" in prod_df.columns else "unique_id"
    static_cols = [c for c in static_df.columns if c != "well"]
    for col in static_cols:
        mapping = static_df.set_index("well")[col].to_dict()
        prod_df[col] = prod_df[well_col].astype(str).map(mapping).fillna(0.0)

    if neighbor_agg_cols:
        available = [c for c in neighbor_agg_cols if c in prod_df.columns]
        if available:
            prod_df = compute_neighbor_aggregated_features(
                prod_df, G, feature_cols=available, k=neighbor_k,
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
