"""System detection and separatrix construction.

Two systems (cognitive vs physical) are detected by fitting a Gaussian
mixture model in 4D edge space — each edge is represented as
(x_src, y_src, x_tgt, y_tgt). Edges are weight-replicated so the GMM
respects flow probabilities.

After fitting, each SOC code is assigned to the system that holds the
plurality of its outgoing weight. From these assignments we derive:

* Per-system attractor poles (weighted centroids of system members).
* The dynamic separatrix: the zero-isocline of the kernel-regressed field
  of crossing transitions, projected onto the line between the two poles.
* The GMM partition boundary: the locus where two systems carry equal
  kernel-weighted vote.

In practice the two separatrices nearly coincide; the dynamic version is
preferred for the paper figures because it is defined directly from the
flow data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mobility.fields import (
    EdgeArrays, Grid, FieldConfig,
    kernel_regression,
)


# ---------------------------------------------------------------------------
# GMM-based system detection
# ---------------------------------------------------------------------------

@dataclass
class SystemAssignment:
    """Outcome of a GMM run with all derived quantities."""

    seed: int
    edge_labels: np.ndarray              # one label per edge
    soc_system: dict[str, int]           # SOC -> system label (plurality)
    flow_within: float                   # share of weight within systems (%)
    flow_between: float                  # share of weight crossing (%)
    flow_matrix: np.ndarray              # (k, k) inter-system weight matrix in %
    silhouette: float
    min_sys_frac: float                  # smallest system as share of total weight
    log_likelihood: float
    n_components: int
    weight_total: float


def _flow_stats(
    df_edges: pd.DataFrame,
    edge_labels: np.ndarray,
    n_components: int,
) -> tuple[float, float, np.ndarray, dict[str, int], float]:
    """Aggregate edge labels to SOC labels and compute flow statistics."""
    src = df_edges["src_soc2018"].to_numpy()
    tgt = df_edges["tgt_soc2018"].to_numpy()
    w = df_edges["wP"].to_numpy(dtype=float)

    # SOC -> plurality system, weighted by outgoing edges
    soc_system: dict[str, int] = {}
    for soc in np.unique(src):
        mask = (src == soc)
        votes = np.zeros(n_components, dtype=float)
        for k in range(n_components):
            votes[k] = float(w[mask & (edge_labels == k)].sum())
        soc_system[soc] = int(np.argmax(votes))

    # Inter-system weight matrix
    fm = np.zeros((n_components, n_components), dtype=float)
    for i in range(len(df_edges)):
        ms = soc_system.get(src[i], -1)
        mt = soc_system.get(tgt[i], -1)
        if ms >= 0 and mt >= 0:
            fm[ms, mt] += w[i]

    total = fm.sum()
    within = float(np.trace(fm) / total) if total > 0 else 0.0
    fm_pct = (fm / total) * 100.0 if total > 0 else fm

    # Smallest-system share
    sys_w = np.zeros(n_components, dtype=float)
    for soc, k in soc_system.items():
        sys_w[k] += float(df_edges.loc[df_edges["src_soc2018"] == soc, "wP"].sum())
    min_frac = float(sys_w.min() / sys_w.sum()) if sys_w.sum() > 0 else 0.0

    return within, 1.0 - within, fm_pct, soc_system, min_frac


def detect_systems(
    df_edges: pd.DataFrame,
    *,
    n_components: int = 2,
    n_runs: int = 1,
    n_init: int = 5,
    min_system_frac: float = 0.15,
    select_by: str = "flow_between",
    weight_replication_max: int = 100,
    silhouette_sample: int = 2000,
    base_seed: int = 0,
) -> tuple[SystemAssignment, list[SystemAssignment]]:
    """Fit GMM(s) on the 4D edge space, choose the best run.

    Parameters
    ----------
    df_edges : DataFrame
        Edge table with ``x_src, y_src, x_tgt, y_tgt, wP``.
    n_components : int
        Number of mixture components (systems).
    n_runs : int
        Number of seeds to try.
    n_init : int
        ``n_init`` for each GMM fit.
    min_system_frac : float
        Reject runs where any system holds less than this share of weight.
    select_by : {"flow_between", "silhouette"}
        Selection criterion among valid runs. ``flow_between`` minimises
        cross-system weight; ``silhouette`` maximises cluster separation.
    weight_replication_max : int
        Cap on the integer multiplicity used for weight-replicated fitting.
    silhouette_sample : int
        Subsample size for silhouette computation.
    base_seed : int
        Seed offset; the seeds used are ``base_seed + i`` for i in range(n_runs).

    Returns
    -------
    best : SystemAssignment
        Selected run.
    all_runs : list[SystemAssignment]
        All runs (in original seed order).
    """
    if select_by not in ("flow_between", "silhouette"):
        raise ValueError("select_by must be 'flow_between' or 'silhouette'")

    X = df_edges[["x_src", "y_src", "x_tgt", "y_tgt"]].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    w_int = np.clip(np.round(weight_replication_max * w / w.max()), 1,
                    weight_replication_max).astype(int)
    X_rep = np.repeat(X_sc, w_int, axis=0)

    runs: list[SystemAssignment] = []
    for i in range(n_runs):
        seed = base_seed + i
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
            n_init=n_init,
        )
        model.fit(X_rep)
        labels = model.predict(X_sc)

        rng = np.random.default_rng(seed)
        sub = rng.choice(len(X_sc), size=min(silhouette_sample, len(X_sc)), replace=False)
        try:
            sil = float(silhouette_score(X_sc[sub], labels[sub], sample_weight=w[sub]))
        except Exception:
            sil = float("nan")

        within, between, fm_pct, soc_sys, min_frac = _flow_stats(
            df_edges, labels, n_components,
        )
        runs.append(SystemAssignment(
            seed=seed,
            edge_labels=labels,
            soc_system=soc_sys,
            flow_within=within * 100,
            flow_between=between * 100,
            flow_matrix=fm_pct,
            silhouette=sil,
            min_sys_frac=min_frac * 100,
            log_likelihood=float(model.score(X_sc)),
            n_components=n_components,
            weight_total=float(w.sum()),
        ))

    valid = [r for r in runs if (r.min_sys_frac / 100) >= min_system_frac]
    if not valid:
        # fall back to lowest flow_between
        best = min(runs, key=lambda r: r.flow_between)
    else:
        if select_by == "silhouette":
            best = max(valid, key=lambda r: r.silhouette)
        else:
            best = min(valid, key=lambda r: r.flow_between)

    return best, runs


# ---------------------------------------------------------------------------
# Per-system aggregations
# ---------------------------------------------------------------------------

def assign_edge_systems(
    df_edges: pd.DataFrame,
    soc_system: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each edge to (src_system, tgt_system, is_crossing)."""
    src_sys = df_edges["src_soc2018"].map(soc_system).fillna(-1).astype(int).to_numpy()
    tgt_sys = df_edges["tgt_soc2018"].map(soc_system).fillna(-1).astype(int).to_numpy()
    crossing = (src_sys >= 0) & (tgt_sys >= 0) & (src_sys != tgt_sys)
    return src_sys, tgt_sys, crossing


def attractor_poles(
    df_edges: pd.DataFrame,
    soc_system: dict[str, int],
    n_components: int,
) -> np.ndarray:
    """Compute one (x, y) pole per system as a weight-weighted centroid.

    Weights are total outgoing wP per SOC.
    """
    socs = list(soc_system.keys())
    src_w = (
        df_edges.groupby("src_soc2018")["wP"].sum()
        .reindex(socs).fillna(0.0)
    )
    centres = (
        df_edges[["src_soc2018", "x_src", "y_src"]]
        .drop_duplicates(subset="src_soc2018")
        .set_index("src_soc2018")
        .reindex(socs)
    )

    poles = np.zeros((n_components, 2), dtype=float)
    for k in range(n_components):
        mask = np.array([soc_system[s] == k for s in socs])
        w_k = src_w.to_numpy(dtype=float)[mask]
        x_k = centres["x_src"].to_numpy(dtype=float)[mask]
        y_k = centres["y_src"].to_numpy(dtype=float)[mask]
        finite = np.isfinite(x_k) & np.isfinite(y_k) & np.isfinite(w_k) & (w_k > 0)
        if not finite.any():
            poles[k] = (np.nan, np.nan)
            continue
        ws = w_k[finite].sum()
        poles[k, 0] = (w_k[finite] * x_k[finite]).sum() / ws
        poles[k, 1] = (w_k[finite] * y_k[finite]).sum() / ws
    return poles


# ---------------------------------------------------------------------------
# Separatrix
# ---------------------------------------------------------------------------

@dataclass
class Separatrix:
    """Separatrix scalar field on the disk grid; contour at 0 is the boundary."""

    field: np.ndarray         # (n_grid, n_grid), nan outside disk
    method: str               # "dynamic" or "gmm_partition"


def dynamic_separatrix(
    edges: EdgeArrays,
    grid: Grid,
    cfg: FieldConfig,
    crossing_mask: np.ndarray,
    poles: np.ndarray,
) -> Separatrix:
    """Compute the dynamic separatrix from crossing-edge directions.

    The separatrix scalar is the projection of the kernel-averaged crossing
    direction onto the unit vector pointing from pole 1 to pole 0. The
    boundary is the zero-isocline.
    """
    U, V, _ = kernel_regression(edges, grid, cfg, edge_mask=crossing_mask)

    diff = poles[0] - poles[1]
    norm = float(np.linalg.norm(diff))
    if norm == 0:
        raise ValueError("poles coincide; cannot define dynamic separatrix")
    d_unit = diff / norm

    proj = (np.where(np.isfinite(U), U, 0.0) * d_unit[0]
            + np.where(np.isfinite(V), V, 0.0) * d_unit[1])
    proj[grid.mask_disk] = np.nan
    return Separatrix(field=proj, method="dynamic")


def gmm_partition_separatrix(
    edges: EdgeArrays,
    grid: Grid,
    cfg: FieldConfig,
    src_systems: np.ndarray,
    n_components: int = 2,
) -> Separatrix:
    """Kernel-weighted vote difference between systems on the disk grid.

    Boundary is the zero-isocline. Only meaningful for n_components == 2.
    """
    if n_components != 2:
        raise NotImplementedError("gmm_partition_separatrix supports n_components=2 only")

    import numpy as np  # local for clarity
    try:
        import cupy as xp_cupy
        use_gpu = cfg.use_gpu
    except Exception:
        use_gpu = False
    xp = xp_cupy if use_gpu else np

    h2 = float(cfg.bandwidth ** 2)
    n_pts = grid.n_pts
    vote_diff_flat = np.zeros(n_pts, dtype=float)

    for k in range(n_components):
        mask_k = (src_systems == k)
        if not mask_k.any():
            continue
        x_k = xp.asarray(edges.x_src[mask_k])
        y_k = xp.asarray(edges.y_src[mask_k])
        w_k = xp.asarray(edges.wP[mask_k])

        votes_k = np.zeros(n_pts, dtype=float)
        for bs in range(0, len(grid.disk_idx), cfg.batch):
            bidx = grid.disk_idx[bs: bs + cfg.batch]
            gx_b = xp.asarray(grid.gx_flat[bidx])
            gy_b = xp.asarray(grid.gy_flat[bidx])
            d2 = (gx_b[:, None] - x_k[None, :]) ** 2 \
               + (gy_b[:, None] - y_k[None, :]) ** 2
            kern = xp.exp(-0.5 * d2 / h2) * w_k[None, :]
            wsum = kern.sum(axis=1)
            if use_gpu:
                votes_k[bidx] = xp_cupy.asnumpy(wsum)
            else:
                votes_k[bidx] = wsum
        sign = 1.0 if k == 0 else -1.0
        vote_diff_flat += sign * votes_k

    field = vote_diff_flat.reshape(cfg.n_grid, cfg.n_grid)
    field[grid.mask_disk] = np.nan
    return Separatrix(field=field, method="gmm_partition")


# ---------------------------------------------------------------------------
# Edge-set partitioning by system
# ---------------------------------------------------------------------------

def partition_edges_by_system(
    df_edges: pd.DataFrame,
    soc_system: dict,
    n_components: int = 2,
) -> dict:
    """Split the edge table into within-system and crossing subsets.

    Parameters
    ----------
    df_edges : DataFrame
        Edge table with src/tgt SOC columns.
    soc_system : dict
        SOC -> system label mapping from a SystemAssignment.
    n_components : int
        Number of systems.

    Returns
    -------
    dict[str, DataFrame]
        Keys are ``"within_0"``, ``"within_1"``, ..., and ``"crossing"``.
        Each value is the corresponding subset of ``df_edges``.
    """
    src_sys = df_edges["src_soc2018"].map(soc_system)
    tgt_sys = df_edges["tgt_soc2018"].map(soc_system)

    out = {}
    for k in range(n_components):
        mask = (src_sys == k) & (tgt_sys == k)
        out[f"within_{k}"] = df_edges[mask].copy()

    crossing_mask = (
        src_sys.notna() & tgt_sys.notna() & (src_sys != tgt_sys)
    )
    out["crossing"] = df_edges[crossing_mask].copy()

    return out

    