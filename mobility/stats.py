"""Weighted statistical primitives.

Used throughout the analysis for ECDFs, weighted quantiles and means,
weighted correlations, and rank correlations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return only entries where x and w are finite and w > 0."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[mask], w[mask]


def wmean(x, w) -> float:
    """Weighted mean. Returns NaN on empty input."""
    x, w = _clean(x, w)
    return float(np.sum(w * x) / np.sum(w)) if x.size else np.nan


def wquantile(x, w, q: float) -> float:
    """Weighted quantile via the empirical CDF.

    Linear interpolation between adjacent CDF values.
    """
    x, w = _clean(x, w)
    if x.size == 0:
        return np.nan
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) / ws.sum()
    return float(np.interp(q, cw, xs))


def weighted_ecdf(x, w) -> tuple[np.ndarray, np.ndarray]:
    """Weighted empirical CDF.

    Returns
    -------
    xu : ndarray
        Unique sorted x values.
    cdf : ndarray
        Cumulative weight share, in [0, 1].
    """
    x, w = _clean(x, w)
    if x.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    x, w = x[order], w[order]
    xu, inv = np.unique(x, return_inverse=True)
    wsum = np.zeros_like(xu, dtype=float)
    np.add.at(wsum, inv, w)
    cw = np.cumsum(wsum)
    return xu, cw / cw[-1]


def weighted_cdf_at(xs: np.ndarray, cdf: np.ndarray, t: float) -> float:
    """Evaluate a step ECDF at threshold ``t`` (right-continuous)."""
    if xs.size == 0:
        return np.nan
    idx = np.searchsorted(xs, t, side="right") - 1
    if idx < 0:
        return 0.0
    return float(cdf[idx])


def weighted_corr(x, y, w) -> float:
    """Weighted Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if x.size < 3:
        return float("nan")
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy = np.sqrt(np.sum(w * (y - my) ** 2))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def spearman_rank_corr(x, y) -> float:
    """Unweighted Spearman rank correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")