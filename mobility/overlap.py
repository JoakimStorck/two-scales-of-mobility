"""Signed overlap between source and target task circles.

For two occupations a, b with task radii R_a, R_b and centre-distance d_ab,
the signed overlap is:

    s_ab = R_a + R_b - d_ab

which is positive when the circles overlap, zero at tangency, negative when
separated. The normalised signed overlap divides by R_src so it has the same
units as the normalised hop length u_R = d_ab / R_src.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.stats import weighted_corr


def add_signed_overlap(df_edges: pd.DataFrame) -> pd.DataFrame:
    """Add signed overlap columns to an edge table.

    Requires that the edge table already contains source and target task
    radii. We need both, so this function expects the caller to have merged
    target radii in (the standard :func:`build_edges` only adds the source
    radius). This function does that merge step internally if a target
    radius column is missing.

    Parameters
    ----------
    df_edges : DataFrame
        Edge table from :func:`mobility.transitions.build_edges`. Must
        contain ``R_src_task_rms``; if ``R_tgt_task_rms`` is absent it must
        be supplied via :func:`merge_target_radii` first.

    Returns
    -------
    DataFrame with three new columns:
    ``[overlap_signed, overlap_norm_src, R_tgt_task_rms]``.

    Notes
    -----
    ``overlap_norm_src = (R_src + R_tgt - d_xy) / R_src``.
    """
    if "R_tgt_task_rms" not in df_edges.columns:
        raise KeyError(
            "df_edges must contain 'R_tgt_task_rms'. "
            "Call merge_target_radii() first."
        )

    df = df_edges.copy()
    R_src = df["R_src_task_rms"].to_numpy(dtype=float)
    R_tgt = df["R_tgt_task_rms"].to_numpy(dtype=float)
    d_xy = df["d_xy"].to_numpy(dtype=float)

    df["overlap_signed"] = R_src + R_tgt - d_xy
    df["overlap_norm_src"] = np.where(
        np.isfinite(R_src) & (R_src > 0),
        df["overlap_signed"].to_numpy(dtype=float) / R_src,
        np.nan,
    )
    return df


def merge_target_radii(
    df_edges: pd.DataFrame,
    df_task_radii: pd.DataFrame,
) -> pd.DataFrame:
    """Merge target-side task radii into the edge table.

    The edge table from :func:`build_edges` only carries the source-side
    radius (since u and u_R are normalised by the source). To compute the
    signed overlap we also need the target radius.
    """
    radii = (
        df_task_radii.set_index("soc2018")["R_task_rms"]
        .rename("R_tgt_task_rms")
    )
    return df_edges.merge(
        radii, left_on="tgt_soc2018", right_index=True, how="left",
    )


def overlap_summary(df_edges: pd.DataFrame) -> dict[str, float]:
    """Compute summary statistics for the overlap-vs-hop relationship.

    Returns weighted and unweighted Pearson correlations between
    ``overlap_norm_src`` and ``u_R``, plus the share of edges with
    positive overlap (weighted by wP).
    """
    required = {"overlap_norm_src", "u_R", "wP"}
    missing = sorted(required - set(df_edges.columns))
    if missing:
        raise KeyError(f"missing columns: {missing}")

    o = df_edges["overlap_norm_src"].to_numpy(dtype=float)
    u = df_edges["u_R"].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    mask = np.isfinite(o) & np.isfinite(u) & np.isfinite(w) & (w > 0)

    o, u, w = o[mask], u[mask], w[mask]
    if o.size < 3:
        return {
            "n_valid": int(o.size),
            "pearson_unweighted": float("nan"),
            "pearson_weighted": float("nan"),
            "share_positive_w": float("nan"),
        }

    pearson_unw = float(np.corrcoef(o, u)[0, 1])
    pearson_w = weighted_corr(o, u, w)
    share_pos = float(np.sum(w[o > 0]) / np.sum(w))
    return {
        "n_valid": int(o.size),
        "pearson_unweighted": pearson_unw,
        "pearson_weighted": pearson_w,
        "share_positive_w": share_pos,
    }