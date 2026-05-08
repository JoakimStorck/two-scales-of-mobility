"""Directional decomposition of transitions along the angular dimension.

For each edge, we project the displacement vector onto the tangential
direction at the source, defined as t = (-sin(xi), cos(xi)). A positive
projection means counterclockwise motion (increasing xi); a negative one
means clockwise motion.

We then bin sources by xi and compute weighted-mean tangential projection
within each bin, separately for each system and for crossing edges.
The result, smoothed with a small circular window, traces how mobility
direction varies with starting angle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tangential_projection(
    df_edges: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    *,
    normalise: bool = True,
) -> pd.DataFrame:
    """Add tangential and radial projections to an edge table.

    Parameters
    ----------
    df_edges : DataFrame
        Edge table with ``x_src, y_src, x_tgt, y_tgt, src_soc2018``.
    df_soc_centres : DataFrame
        SOC centres with ``soc2018, xi_soc``. Used to look up the source
        angle in radians.
    normalise : bool
        If True, project the unit-length displacement so the projection
        gives direction only. If False, project the raw displacement
        (in xy units).

    Returns
    -------
    DataFrame
        Copy of ``df_edges`` with three new columns:
        ``xi_src_rad``, ``xi_src_deg``, ``proj_xi``.
    """
    df = df_edges.copy()
    xi_map = df_soc_centres.set_index("soc2018")["xi_soc"]
    df["xi_src_rad"] = df["src_soc2018"].map(xi_map).to_numpy(dtype=float)
    df["xi_src_deg"] = (np.degrees(df["xi_src_rad"]) % 360.0)

    dx = df["x_tgt"].to_numpy(dtype=float) - df["x_src"].to_numpy(dtype=float)
    dy = df["y_tgt"].to_numpy(dtype=float) - df["y_src"].to_numpy(dtype=float)

    xi = df["xi_src_rad"].to_numpy(dtype=float)
    tan_x = -np.sin(xi)
    tan_y = np.cos(xi)
    raw_proj = dx * tan_x + dy * tan_y

    if normalise:
        d_len = np.hypot(dx, dy) + 1e-12
        df["proj_xi"] = raw_proj / d_len
    else:
        df["proj_xi"] = raw_proj

    return df.dropna(subset=["xi_src_rad", "xi_src_deg"])


def _smooth_circular(arr: np.ndarray, win: int) -> np.ndarray:
    """Circular moving-average smoothing that respects NaN."""
    n = len(arr)
    out = np.full_like(arr, np.nan)
    half = win // 2
    for i in range(n):
        idx = [(i + j - half) % n for j in range(win)]
        vals = arr[idx]
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            out[i] = finite.mean()
    out[~np.isfinite(arr)] = np.nan
    return out


def angular_curve(
    df_edges_proj: pd.DataFrame,
    *,
    n_bins: int = 36,
    smooth_window: int = 3,
    min_weight: float = 1e-6,
    activity_threshold_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin edges by source xi and compute weighted-mean tangential projection.

    Parameters
    ----------
    df_edges_proj : DataFrame
        Output of :func:`compute_tangential_projection`. Must include
        ``xi_src_deg``, ``proj_xi``, ``wP``.
    n_bins : int
        Number of equal-width angular bins covering [0, 360).
    smooth_window : int
        Width of the circular moving-average smoothing.
    min_weight : float
        Bins with total weight below this are set to NaN.
    activity_threshold_frac : float
        After binning, bins with weight below ``frac * max_weight`` are
        further masked.

    Returns
    -------
    bin_centres_deg : ndarray
        Geographic angle of each bin centre, in [0, 360).
    curve : ndarray
        Smoothed weighted-mean tangential projection per bin (NaN where
        activity is too low).
    se : ndarray
        Standard error of the weighted mean per bin, smoothed circularly
        with the same window. NaN where the curve is NaN.
    weights : ndarray
        Total weight per bin (unsmoothed, for diagnostics).
    """
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    df = df_edges_proj.copy()
    df["bin"] = pd.cut(df["xi_src_deg"], bins=bin_edges,
                       labels=False, include_lowest=True)

    proj = np.full(n_bins, np.nan)
    se = np.full(n_bins, np.nan)
    wt = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        sub = df[df["bin"] == b]
        x = sub["proj_xi"].to_numpy(dtype=float)
        w = sub["wP"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        wt[b] = float(w[m].sum())
        if m.sum() < 2 or wt[b] < min_weight:
            continue

        x_m, w_m = x[m], w[m]
        w_sum = w_m.sum()
        mean = float(np.sum(w_m * x_m) / w_sum)
        proj[b] = mean

        # Weighted variance of the bin
        var = float(np.sum(w_m * (x_m - mean) ** 2) / w_sum)
        # Effective sample size (Kish)
        n_eff = float(w_sum ** 2 / np.sum(w_m ** 2))
        if n_eff > 1:
            se[b] = float(np.sqrt(var / n_eff))

    smoothed_proj = _smooth_circular(proj, smooth_window)
    smoothed_se = _smooth_circular(se, smooth_window)

    if wt.max() > 0:
        low_act = wt < activity_threshold_frac * wt.max()
        smoothed_proj[low_act] = np.nan
        smoothed_se[low_act] = np.nan

    return bin_centres, smoothed_proj, smoothed_se, wt