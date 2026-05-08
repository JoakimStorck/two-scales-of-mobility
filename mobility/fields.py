"""Mobility field computation via Gaussian kernel regression.

For each grid point, we compute a weight-locally-averaged direction vector
(U, V) and the kernel weight sum W. The averaging kernel is centred at each
edge's source occupation; the value averaged is the unit direction of the
edge.

Field types:

* **Absolute.** Edges contribute their normalised direction vector.
* **Residual.** The global drift is subtracted; only the deviation from the
  global flow remains.
* **Subset.** Same as absolute but restricted to a subset of edges (within
  a system, crossing a boundary, etc).

The expensive part is the grid-by-edge distance computation. We accelerate
with CuPy when available and batch over grid points to control memory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Optional GPU backend. Falls back to NumPy if CuPy is missing or has no GPU.
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


@dataclass(frozen=True)
class FieldConfig:
    """Parameters for kernel regression on the polar disk."""

    bandwidth: float = 0.08
    n_grid: int = 120
    grid_min: float = -1.0
    grid_max: float = 1.0
    disk_radius_sq: float = 0.98   # mask points outside this from the disk
    batch: int = 1024
    min_wsum: float = 1e-10
    use_gpu: bool = True            # honoured only if CuPy is importable

    def is_gpu_active(self) -> bool:
        return self.use_gpu and _HAS_CUPY


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    grid_1d: np.ndarray
    GX: np.ndarray
    GY: np.ndarray
    mask_disk: np.ndarray   # True for grid points OUTSIDE the disk
    gx_flat: np.ndarray
    gy_flat: np.ndarray
    disk_idx: np.ndarray    # indices of in-disk grid points
    n_pts: int


def make_grid(cfg: FieldConfig) -> Grid:
    """Build the regular grid used for field evaluation."""
    grid_1d = np.linspace(cfg.grid_min, cfg.grid_max, cfg.n_grid)
    GX, GY = np.meshgrid(grid_1d, grid_1d)
    in_disk_2d = (GX ** 2 + GY ** 2) <= cfg.disk_radius_sq
    mask_disk = ~in_disk_2d
    gx_flat = GX.flatten()
    gy_flat = GY.flatten()
    in_disk_flat = (gx_flat ** 2 + gy_flat ** 2) <= cfg.disk_radius_sq
    disk_idx = np.where(in_disk_flat)[0]
    return Grid(
        grid_1d=grid_1d, GX=GX, GY=GY,
        mask_disk=mask_disk,
        gx_flat=gx_flat, gy_flat=gy_flat,
        disk_idx=disk_idx, n_pts=len(gx_flat),
    )


# ---------------------------------------------------------------------------
# Edge preparation
# ---------------------------------------------------------------------------

@dataclass
class EdgeArrays:
    """Edge data in the form needed by kernel regression."""

    x_src: np.ndarray
    y_src: np.ndarray
    dx_norm: np.ndarray
    dy_norm: np.ndarray
    wP: np.ndarray
    n_edges: int


def prepare_edges(df_edges: pd.DataFrame) -> EdgeArrays:
    """Extract the arrays needed for kernel regression from an edge table.

    Edges with zero displacement get unit-vector components of zero, which
    contributes a zero direction to the field at their source location.
    """
    x_src = df_edges["x_src"].to_numpy(dtype=float)
    y_src = df_edges["y_src"].to_numpy(dtype=float)
    dx = df_edges["x_tgt"].to_numpy(dtype=float) - x_src
    dy = df_edges["y_tgt"].to_numpy(dtype=float) - y_src
    wP = df_edges["wP"].to_numpy(dtype=float)

    d_len = np.hypot(dx, dy)
    valid = d_len > 1e-10
    dx_norm = np.where(valid, dx / d_len, 0.0)
    dy_norm = np.where(valid, dy / d_len, 0.0)

    return EdgeArrays(
        x_src=x_src, y_src=y_src,
        dx_norm=dx_norm, dy_norm=dy_norm,
        wP=wP, n_edges=len(x_src),
    )


# ---------------------------------------------------------------------------
# Core kernel regression
# ---------------------------------------------------------------------------

def kernel_regression(
    edges: EdgeArrays,
    grid: Grid,
    cfg: FieldConfig,
    *,
    edge_mask: np.ndarray | None = None,
    direction_x: np.ndarray | None = None,
    direction_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian kernel regression on the disk grid.

    For each in-disk grid point :math:`g`:

    .. math::
        U(g) = \\frac{\\sum_e w_e \\, k(g, p_e) \\, \\hat{d}_{e,x}}
                    {\\sum_e w_e \\, k(g, p_e)},

    with :math:`k(g, p) = \\exp(-\\|g-p\\|^2 / (2 h^2))`. Same for V.
    W is the unnormalised denominator; it acts as a kernel-density estimate
    of edge activity.

    Parameters
    ----------
    edges : EdgeArrays
        Output of :func:`prepare_edges`.
    grid : Grid
        Output of :func:`make_grid`.
    cfg : FieldConfig
        Bandwidth, batch size, etc.
    edge_mask : ndarray of bool, optional
        Restrict the regression to a subset of edges (e.g. within-system).
    direction_x, direction_y : ndarray, optional
        Override the direction vectors to average. Defaults to the
        normalised displacement carried by ``edges``. Used to compute
        residual fields by passing the residual direction.

    Returns
    -------
    U, V, W : ndarray, shape (n_grid, n_grid)
        Mean direction-x, mean direction-y, kernel weight sum. NaN outside
        the disk and at points with negligible weight.
    """
    if edge_mask is None:
        x_src = edges.x_src
        y_src = edges.y_src
        wP = edges.wP
        dx = edges.dx_norm if direction_x is None else np.asarray(direction_x, float)
        dy = edges.dy_norm if direction_y is None else np.asarray(direction_y, float)
    else:
        x_src = edges.x_src[edge_mask]
        y_src = edges.y_src[edge_mask]
        wP = edges.wP[edge_mask]
        if direction_x is None:
            dx = edges.dx_norm[edge_mask]
        else:
            dx = np.asarray(direction_x, float)[edge_mask]
        if direction_y is None:
            dy = edges.dy_norm[edge_mask]
        else:
            dy = np.asarray(direction_y, float)[edge_mask]

    if len(x_src) == 0:
        n = cfg.n_grid
        return (np.full((n, n), np.nan),
                np.full((n, n), np.nan),
                np.full((n, n), np.nan))

    xp = _cp if cfg.is_gpu_active() else np
    h2 = float(cfg.bandwidth ** 2)

    x_src_g = xp.asarray(x_src)
    y_src_g = xp.asarray(y_src)
    dx_g = xp.asarray(dx)
    dy_g = xp.asarray(dy)
    w_g = xp.asarray(wP)

    n_pts = grid.n_pts
    U_f = np.full(n_pts, np.nan)
    V_f = np.full(n_pts, np.nan)
    W_f = np.full(n_pts, np.nan)

    for bs in range(0, len(grid.disk_idx), cfg.batch):
        bidx = grid.disk_idx[bs: bs + cfg.batch]
        gx_b = xp.asarray(grid.gx_flat[bidx])
        gy_b = xp.asarray(grid.gy_flat[bidx])

        d2 = (gx_b[:, None] - x_src_g[None, :]) ** 2 \
           + (gy_b[:, None] - y_src_g[None, :]) ** 2
        kern = xp.exp(-0.5 * d2 / h2) * w_g[None, :]
        wsum = kern.sum(axis=1)

        valid = wsum > cfg.min_wsum
        safe = xp.where(valid, wsum, xp.ones_like(wsum))
        u_b = xp.where(valid, (kern * dx_g[None, :]).sum(axis=1) / safe, xp.nan)
        v_b = xp.where(valid, (kern * dy_g[None, :]).sum(axis=1) / safe, xp.nan)

        if cfg.is_gpu_active():
            U_f[bidx] = _cp.asnumpy(u_b)
            V_f[bidx] = _cp.asnumpy(v_b)
            W_f[bidx] = _cp.asnumpy(wsum)
        else:
            U_f[bidx] = u_b
            V_f[bidx] = v_b
            W_f[bidx] = wsum

    n = cfg.n_grid
    U = U_f.reshape(n, n); U[grid.mask_disk] = np.nan
    V = V_f.reshape(n, n); V[grid.mask_disk] = np.nan
    W = W_f.reshape(n, n); W[grid.mask_disk] = np.nan
    return U, V, W


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_field(
    U: np.ndarray, V: np.ndarray, W: np.ndarray,
    grid: Grid,
    *,
    R_pctile: float = 95.0,
    W_pctile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute resultant strength R = ||(U, V)|| and normalised W, both in [0, 1].

    R is divided by its 95th percentile and clipped; same for log1p(W).
    """
    R = np.hypot(U, V)
    R95 = np.nanpercentile(R, R_pctile)
    R = np.clip(R / (R95 + 1e-12), 0.0, 1.0)
    R[grid.mask_disk] = np.nan

    Wn = np.log1p(W)
    W95 = np.nanpercentile(Wn, W_pctile)
    Wn = np.clip(Wn / (W95 + 1e-12), 0.0, 1.0)
    Wn[grid.mask_disk] = np.nan
    return R, Wn


# ---------------------------------------------------------------------------
# Global drift and residual decomposition
# ---------------------------------------------------------------------------

def global_drift(edges: EdgeArrays) -> tuple[float, float]:
    """Weighted mean of the unit edge directions across all edges."""
    w_sum = float(edges.wP.sum())
    if w_sum == 0:
        return 0.0, 0.0
    u = float(np.sum(edges.wP * edges.dx_norm) / w_sum)
    v = float(np.sum(edges.wP * edges.dy_norm) / w_sum)
    return u, v


def residual_directions(
    edges: EdgeArrays,
    drift: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Subtract the global drift and re-normalise to unit vectors.

    Returns the residual unit-direction arrays plus the drift used.
    """
    if drift is None:
        drift = global_drift(edges)
    du, dv = drift

    dx_res = edges.dx_norm - du
    dy_res = edges.dy_norm - dv
    d_res = np.hypot(dx_res, dy_res)
    valid = d_res > 1e-10
    dx_res_n = np.where(valid, dx_res / d_res, 0.0)
    dy_res_n = np.where(valid, dy_res / d_res, 0.0)
    return dx_res_n, dy_res_n, drift


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Computed field with metadata."""

    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    R: np.ndarray
    W_norm: np.ndarray
    n_edges: int
    drift: tuple[float, float] | None = None


def _cache_key(
    name: str,
    cfg: FieldConfig,
    edge_signature: str,
    extra: dict | None = None,
) -> str:
    """Build a deterministic filename suffix for caching."""
    payload = {
        "name": name,
        "cfg": asdict(cfg),
        "edges": edge_signature,
    }
    if extra:
        payload["extra"] = extra
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:12]


def edge_signature(edges: EdgeArrays) -> str:
    """Short hash summarising the edge set, used for cache invalidation.

    If you change the edge filtering (P_THRESHOLD, TOP_K, etc) the signature
    changes and the cache is automatically rebuilt.
    """
    blob = (
        edges.x_src.tobytes()
        + edges.y_src.tobytes()
        + edges.dx_norm.tobytes()
        + edges.dy_norm.tobytes()
        + edges.wP.tobytes()
    )
    return hashlib.md5(blob).hexdigest()[:12]


def compute_field(
    edges: EdgeArrays,
    grid: Grid,
    cfg: FieldConfig,
    *,
    edge_mask: np.ndarray | None = None,
    direction_x: np.ndarray | None = None,
    direction_y: np.ndarray | None = None,
    drift: tuple[float, float] | None = None,
) -> FieldResult:
    """Compute a single field and return it bundled with normalised arrays."""
    U, V, W = kernel_regression(
        edges, grid, cfg,
        edge_mask=edge_mask,
        direction_x=direction_x,
        direction_y=direction_y,
    )
    R, W_norm = normalise_field(U, V, W, grid)
    n_e = int(edges.n_edges if edge_mask is None else int(np.sum(edge_mask)))
    return FieldResult(
        U=U, V=V, W=W, R=R, W_norm=W_norm,
        n_edges=n_e, drift=drift,
    )


def compute_or_load(
    name: str,
    edges: EdgeArrays,
    grid: Grid,
    cfg: FieldConfig,
    cache_dir: Path | str,
    *,
    edge_mask: np.ndarray | None = None,
    direction_x: np.ndarray | None = None,
    direction_y: np.ndarray | None = None,
    drift: tuple[float, float] | None = None,
    force_recompute: bool = False,
    extra_key: dict | None = None,
) -> FieldResult:
    """Compute a field, or load from disk cache if available.

    The cache key is derived from ``name``, ``cfg``, the edge signature, and
    ``extra_key``. Changing any of these invalidates the cache.

    Parameters
    ----------
    cache_dir : path-like
        Directory to write cache files into. Created if missing.
    force_recompute : bool
        If True, always recompute even if a cached file exists.
    extra_key : dict, optional
        Additional content that should be part of the cache key (e.g. an
        edge-mask hash if you compute many fields off the same edges).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sig = edge_signature(edges)
    if edge_mask is not None:
        mask_hash = hashlib.md5(edge_mask.tobytes()).hexdigest()[:12]
        extra_key = {**(extra_key or {}), "mask": mask_hash}
    if direction_x is not None or direction_y is not None:
        dx_hash = hashlib.md5(np.asarray(direction_x, float).tobytes()).hexdigest()[:8] \
                  if direction_x is not None else "x"
        dy_hash = hashlib.md5(np.asarray(direction_y, float).tobytes()).hexdigest()[:8] \
                  if direction_y is not None else "y"
        extra_key = {**(extra_key or {}), "dx": dx_hash, "dy": dy_hash}

    key = _cache_key(name, cfg, sig, extra_key)
    fp = cache_dir / f"field_{name}_{key}.npz"

    if fp.exists() and not force_recompute:
        data = np.load(fp, allow_pickle=False)
        return FieldResult(
            U=data["U"], V=data["V"], W=data["W"],
            R=data["R"], W_norm=data["W_norm"],
            n_edges=int(data["n_edges"]),
            drift=tuple(data["drift"]) if "drift" in data.files else None,
        )

    result = compute_field(
        edges, grid, cfg,
        edge_mask=edge_mask,
        direction_x=direction_x, direction_y=direction_y,
        drift=drift,
    )
    payload = dict(
        U=result.U, V=result.V, W=result.W,
        R=result.R, W_norm=result.W_norm,
        n_edges=np.array(result.n_edges),
    )
    if result.drift is not None:
        payload["drift"] = np.array(result.drift)
    np.savez(fp, **payload)
    return result