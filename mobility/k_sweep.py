"""K sweep: fit GMMs with K = 2, ..., K_max and visualise side by side.

This is the natural extension of k3_comparison. We want to see how
successively finer partitions carve the task-space geometry.

Each fit re-uses the same detect_systems machinery. The visualisation
shows each partition as a separate subplot on a small-multiples grid,
with SOCs coloured by their assigned system and sized by outflow weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.systems import detect_systems, attractor_poles


# Distinct colour palette for up to 8 systems. Designed to be readable
# both on screen and in print.
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


def run_k_sweep(
    df_edges: pd.DataFrame,
    *,
    K_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    n_runs: int = 5,
    n_init: int = 5,
    min_system_frac: float = 0.03,  # very permissive; higher K is allowed to be unbalanced
    base_seed: int = 0,
) -> dict:
    """Fit GMMs for each K in K_values.

    Returns
    -------
    dict
        Maps K -> SystemAssignment (the best run for that K).
    """
    out = {}
    for K in K_values:
        best, _ = detect_systems(
            df_edges,
            n_components=K,
            n_runs=n_runs,
            n_init=n_init,
            min_system_frac=min_system_frac,
            select_by="flow_between",
            base_seed=base_seed,
        )
        out[K] = best
        print(f"K={K}: flow_within={best.flow_within:.1f}%, "
              f"flow_between={best.flow_between:.1f}%, "
              f"min_sys_frac={best.min_sys_frac:.1f}%")
    return out


def characterise_systems(
    soc_system: dict[str, int],
    df_soc_centres: pd.DataFrame,
    df_edges: pd.DataFrame,
    title_map: pd.Series | None = None,
    n_top: int = 5,
) -> pd.DataFrame:
    """One row per system: size, mean angle (circular), mean radius, top SOCs.

    Same logic as k3_comparison.characterise_k3_systems but generalised.
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    centres = df_soc_centres.set_index("soc2018")

    rows = []
    total_w = float(src_w.sum())
    for k in sorted(set(soc_system.values())):
        socs_k = [s for s, lab in soc_system.items() if lab == k]
        if not socs_k:
            continue
        sub = centres.loc[centres.index.intersection(socs_k)]
        w_k = src_w.reindex(sub.index).fillna(0.0).to_numpy(dtype=float)
        w_sum = float(w_k.sum())

        xi_arr = sub["xi_soc"].to_numpy(dtype=float)
        if w_sum > 0:
            mean_cos = np.sum(w_k * np.cos(xi_arr)) / w_sum
            mean_sin = np.sum(w_k * np.sin(xi_arr)) / w_sum
            mean_xi = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)
            mean_chi = float(np.sum(w_k * sub["chi_soc"].to_numpy(dtype=float)) / w_sum)
        else:
            mean_xi = float("nan")
            mean_chi = float("nan")

        # Top occupations
        top = (
            pd.Series(w_k, index=sub.index)
            .sort_values(ascending=False)
            .head(n_top)
        )
        if title_map is not None:
            top_labels = [str(title_map.get(s, s)) for s in top.index]
        else:
            top_labels = list(top.index)

        rows.append({
            "system": k,
            "n_soc": int(len(socs_k)),
            "share_of_total": round(100 * w_sum / total_w, 1) if total_w > 0 else 0.0,
            "mean_xi_deg": round(mean_xi, 1) if np.isfinite(mean_xi) else None,
            "mean_chi": round(mean_chi, 3) if np.isfinite(mean_chi) else None,
            "top_occupations": "; ".join(top_labels),
        })
    return pd.DataFrame(rows)


def plot_k_sweep_grid(
    assignments: dict,
    df_soc_centres: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    out_path: str | None = None,
    n_cols: int = 3,
    palette: list = None,
):
    """Small-multiples grid: one panel per K, showing the partition.

    Each SOC is plotted at its (x_soc, y_soc), coloured by system, sized
    by outflow weight. Pole markers (attractors) overlaid as stars.

    Parameters
    ----------
    assignments : dict
        Maps K -> SystemAssignment (output of run_k_sweep).
    df_soc_centres, df_edges : DataFrames
        Standard inputs.
    out_path : path-like, optional
        Save destination.
    n_cols : int
        Number of columns in the grid.
    palette : list of str, optional
        Colour list. Defaults to SYSTEM_PALETTE.
    """
    import matplotlib.pyplot as plt

    if palette is None:
        palette = SYSTEM_PALETTE

    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    centres = df_soc_centres.set_index("soc2018")
    max_w = float(src_w.max())

    Ks = sorted(assignments.keys())
    n = len(Ks)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 4.4 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, K in zip(axes, Ks):
        assignment = assignments[K]
        soc_system = assignment.soc_system

        # Unit circle
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.85", linewidth=0.8)
        ax.axhline(0, color="0.94", linewidth=0.5, zorder=0)
        ax.axvline(0, color="0.94", linewidth=0.5, zorder=0)

        # SOCs
        for soc in centres.index:
            if soc not in soc_system:
                continue
            lab = soc_system[soc]
            x = float(centres.loc[soc, "x_soc"])
            y = float(centres.loc[soc, "y_soc"])
            w = float(src_w.get(soc, 0.0))
            size = 8 + 60 * (w / max_w if max_w > 0 else 0)
            color = palette[lab % len(palette)]
            ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolor="none")

        # Poles
        try:
            poles = attractor_poles(df_edges, soc_system, K)
            for k, p in enumerate(poles):
                if np.all(np.isfinite(p)):
                    ax.scatter(p[0], p[1], s=220, c=palette[k % len(palette)],
                               marker="*", edgecolor="black", linewidth=1.2,
                               zorder=10)
        except Exception:
            pass

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"K = {K}\n(crossing flow: {assignment.flow_between:.1f}%)",
            fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
