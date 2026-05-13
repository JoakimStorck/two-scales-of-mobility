"""Mobility field plot for arbitrary K.

Generalises mobility_systems_combined to K systems. Two panels:

* Left: within-system flows. Each system rendered as a coloured vector
  field with arrows showing local mean direction. Occupation points
  overlaid in matching colours.
* Right: crossing-system flows. Single vector field showing where
  cross-subsystem mobility points and how strong it is.

Poles are marked as stars in both panels. No separatrix is drawn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.fields import (
    EdgeArrays, Grid, FieldConfig,
    prepare_edges, make_grid, compute_or_load,
)
from mobility.systems import assign_edge_systems, attractor_poles


# Default palette (same as k_sweep)
SYSTEM_PALETTE = [
    "#4477AA",   # blue
    "#EE7733",   # orange
    "#117733",   # green
    "#AA3377",   # magenta
    "#DDCC77",   # sand
    "#66CCEE",   # cyan
    "#CC3311",   # red
    "#882255",   # wine
]


def compute_fields_for_K(
    df_edges: pd.DataFrame,
    assignment,
    *,
    bandwidth: float = 0.08,
    n_grid: int = 120,
    batch: int = 1024,
    cache_dir=None,
) -> dict:
    """Compute per-system and crossing kernel fields for one K-partition.

    Returns
    -------
    dict
        Contains:
        - 'edges'         : EdgeArrays
        - 'grid'          : Grid
        - 'cfg'           : FieldConfig
        - 'system_fields' : {k: FieldResult}
        - 'crossing_field': FieldResult
        - 'poles'         : ndarray (K, 2)
    """
    edges = prepare_edges(df_edges)
    cfg = FieldConfig(bandwidth=bandwidth, n_grid=n_grid, batch=batch)
    grid = make_grid(cfg)

    src_sys, tgt_sys, crossing = assign_edge_systems(
        df_edges, assignment.soc_system,
    )

    system_fields = {}
    for k in range(assignment.n_components):
        mask_k = (src_sys == k) & (tgt_sys == k)
        if cache_dir is not None:
            f_k = compute_or_load(
                name=f"K{assignment.n_components}_sys{k}",
                edges=edges, grid=grid, cfg=cfg,
                cache_dir=cache_dir,
                edge_mask=mask_k,
            )
        else:
            # No-cache fallback
            from mobility.fields import compute_field
            f_k = compute_field(edges, grid, cfg, edge_mask=mask_k)
        system_fields[k] = f_k

    if cache_dir is not None:
        f_cross = compute_or_load(
            name=f"K{assignment.n_components}_crossing",
            edges=edges, grid=grid, cfg=cfg,
            cache_dir=cache_dir,
            edge_mask=crossing,
        )
    else:
        from mobility.fields import compute_field
        f_cross = compute_field(edges, grid, cfg, edge_mask=crossing)

    poles = attractor_poles(
        df_edges, assignment.soc_system, assignment.n_components,
    )

    return {
        "edges": edges,
        "grid": grid,
        "cfg": cfg,
        "system_fields": system_fields,
        "crossing_field": f_cross,
        "poles": poles,
    }


def plot_mobility_fields_K(
    fields_data: dict,
    df_soc_centres: pd.DataFrame,
    assignment,
    df_edges: pd.DataFrame,
    *,
    out_path=None,
    palette=None,
    arrow_stride: int = 8,
    arrow_scale: float = 22.0,
    point_size_max: float = 70.0,
    crossing_color: str = "0.25",
    title_left: str = None,
    title_right: str = None,
):
    """Two-panel mobility field plot.

    Parameters
    ----------
    fields_data : dict
        Output of compute_fields_for_K.
    df_soc_centres : DataFrame
        SOC centres with x_soc, y_soc.
    assignment : SystemAssignment
        For soc_system mapping (used to colour occupation points).
    df_edges : DataFrame
        For outflow weights (used to size occupation points).
    out_path : path, optional
        Save destination.
    palette : list of str, optional
        Colour palette for systems. Defaults to SYSTEM_PALETTE.
    arrow_stride : int
        Subsample factor for vector field. Higher = fewer arrows.
        E.g. 8 keeps every 8th grid point in each direction.
    arrow_scale : float
        matplotlib quiver scale; larger = shorter arrows.
    point_size_max : float
        Maximum scatter point size (smallest is 8).
    crossing_color : str
        Colour for crossing-field arrows.
    title_left, title_right : str, optional
        Override panel titles.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if palette is None:
        palette = SYSTEM_PALETTE

    K = assignment.n_components
    grid = fields_data["grid"]
    system_fields = fields_data["system_fields"]
    crossing_field = fields_data["crossing_field"]
    poles = fields_data["poles"]

    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    centres = df_soc_centres.set_index("soc2018")
    max_w = float(src_w.max())

    # Subsample grid points for arrows
    GX = grid.GX
    GY = grid.GY
    s = arrow_stride
    gx_sub = GX[::s, ::s]
    gy_sub = GY[::s, ::s]

    fig, (ax_L, ax_R) = plt.subplots(
        1, 2, figsize=(12.0, 6.0), constrained_layout=True,
    )

    # ---------------- LEFT: within-system flows ----------------
    th = np.linspace(0, 2 * np.pi, 200)
    ax_L.plot(np.cos(th), np.sin(th), color="0.85", linewidth=0.8)

    for k in range(K):
        f_k = system_fields[k]
        U = f_k.U[::s, ::s]
        V = f_k.V[::s, ::s]
        R = np.hypot(U, V)
        # Only draw arrows where the field has appreciable magnitude
        threshold = np.nanpercentile(R, 30) if np.isfinite(R).any() else 0.0
        keep = np.isfinite(U) & np.isfinite(V) & (R >= threshold)
        ax_L.quiver(
            gx_sub[keep], gy_sub[keep], U[keep], V[keep],
            color=palette[k % len(palette)],
            scale=arrow_scale, width=0.0035, alpha=0.85,
            headwidth=4, headlength=5,
        )

    # Occupation points coloured by system, in front of arrows
    for soc in centres.index:
        if soc not in assignment.soc_system:
            continue
        lab = assignment.soc_system[soc]
        x = float(centres.loc[soc, "x_soc"])
        y = float(centres.loc[soc, "y_soc"])
        w = float(src_w.get(soc, 0.0))
        size = 6 + point_size_max * (w / max_w if max_w > 0 else 0)
        ax_L.scatter(
            x, y, s=size,
            c=palette[lab % len(palette)],
            alpha=0.55, edgecolor="none", zorder=5,
        )

    # Poles as stars
    for k, p in enumerate(poles):
        if np.all(np.isfinite(p)):
            ax_L.scatter(
                p[0], p[1], s=260,
                c=palette[k % len(palette)],
                marker="*", edgecolor="black", linewidth=1.4,
                zorder=10,
            )

    ax_L.set_xlim(-1.05, 1.05)
    ax_L.set_ylim(-1.05, 1.05)
    ax_L.set_aspect("equal")
    ax_L.set_xticks([])
    ax_L.set_yticks([])
    ax_L.set_title(
        title_left or f"Within-system flows (K = {K})",
        fontsize=11,
    )
    for spine in ax_L.spines.values():
        spine.set_visible(False)

    # ---------------- RIGHT: crossing flows ----------------
    ax_R.plot(np.cos(th), np.sin(th), color="0.85", linewidth=0.8)

    U_c = crossing_field.U[::s, ::s]
    V_c = crossing_field.V[::s, ::s]
    R_c = np.hypot(U_c, V_c)
    threshold = np.nanpercentile(R_c, 30) if np.isfinite(R_c).any() else 0.0
    keep = np.isfinite(U_c) & np.isfinite(V_c) & (R_c >= threshold)
    ax_R.quiver(
        gx_sub[keep], gy_sub[keep], U_c[keep], V_c[keep],
        color=crossing_color,
        scale=arrow_scale, width=0.0035, alpha=0.85,
        headwidth=4, headlength=5,
    )

    # Occupation points (faded, for context)
    for soc in centres.index:
        if soc not in assignment.soc_system:
            continue
        lab = assignment.soc_system[soc]
        x = float(centres.loc[soc, "x_soc"])
        y = float(centres.loc[soc, "y_soc"])
        w = float(src_w.get(soc, 0.0))
        size = 6 + point_size_max * (w / max_w if max_w > 0 else 0)
        ax_R.scatter(
            x, y, s=size,
            c=palette[lab % len(palette)],
            alpha=0.35, edgecolor="none", zorder=5,
        )

    # Poles
    for k, p in enumerate(poles):
        if np.all(np.isfinite(p)):
            ax_R.scatter(
                p[0], p[1], s=260,
                c=palette[k % len(palette)],
                marker="*", edgecolor="black", linewidth=1.4,
                zorder=10,
            )

    ax_R.set_xlim(-1.05, 1.05)
    ax_R.set_ylim(-1.05, 1.05)
    ax_R.set_aspect("equal")
    ax_R.set_xticks([])
    ax_R.set_yticks([])
    ax_R.set_title(
        title_right or f"Crossing flows (K = {K})",
        fontsize=11,
    )
    for spine in ax_R.spines.values():
        spine.set_visible(False)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
