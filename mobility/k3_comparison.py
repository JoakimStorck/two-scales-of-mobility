"""K=3 inspection: how does a third component split the K=2 partition?

This is a diagnostic for the model-selection argument. The main paper
uses K=2 subsystems. We want to know whether K=3 splits one of the K=2
systems in a meaningful way (e.g. production vs service within the
physical system) or whether it carves the geometry along an unrelated
axis (e.g. high vs low task radius within the same sector).

Workflow:
    1. Run GMM with K=3 on the same edge data.
    2. Build a SOC-by-SOC crosstab: each SOC's K=2 label vs its K=3 label.
    3. Identify which K=2 system gets split, and characterise the split
       by looking at the angular position (xi) and centre distances of
       the SOCs in each new subsystem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.systems import detect_systems, attractor_poles


def run_k3(
    df_edges: pd.DataFrame,
    *,
    n_runs: int = 5,
    n_init: int = 5,
    min_system_frac: float = 0.05,  # lower than K=2 default; K=3 may be unbalanced
    base_seed: int = 0,
):
    """Fit a K=3 GMM using the same detect_systems machinery."""
    best_k3, runs_k3 = detect_systems(
        df_edges,
        n_components=3,
        n_runs=n_runs,
        n_init=n_init,
        min_system_frac=min_system_frac,
        select_by="flow_between",
        base_seed=base_seed,
    )
    return best_k3, runs_k3


def crosstab_k2_vs_k3(
    soc_system_k2: dict[str, int],
    soc_system_k3: dict[str, int],
    df_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Crosstab: count of SOCs in each (K2, K3) cell, weighted by outflow.

    Returns a small DataFrame with K2 labels as rows and K3 labels as
    columns. Each cell holds the total outflow weight of SOCs in that
    (K2, K3) combination, plus an unweighted SOC count.
    """
    # Outgoing weight per SOC
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()

    common_socs = sorted(set(soc_system_k2.keys()) & set(soc_system_k3.keys()))
    rows = []
    for soc in common_socs:
        rows.append({
            "soc2018": soc,
            "k2_label": soc_system_k2[soc],
            "k3_label": soc_system_k3[soc],
            "outflow": float(src_w.get(soc, 0.0)),
        })
    df = pd.DataFrame(rows)

    n_table = pd.crosstab(df["k2_label"], df["k3_label"])
    w_table = df.groupby(["k2_label", "k3_label"])["outflow"].sum().unstack(fill_value=0.0)

    return n_table, w_table


def characterise_k3_systems(
    soc_system_k3: dict[str, int],
    df_soc_centres: pd.DataFrame,
    df_edges: pd.DataFrame,
    title_map: pd.Series | None = None,
    n_top: int = 8,
) -> pd.DataFrame:
    """One row per K=3 system: size, mean angle, mean radius, top occupations.

    Top occupations are ranked by outflow weight within the system.
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()

    centres = df_soc_centres.set_index("soc2018")
    rows = []
    for k in sorted(set(soc_system_k3.values())):
        socs_k = [s for s, lab in soc_system_k3.items() if lab == k]
        if not socs_k:
            continue
        sub = centres.loc[centres.index.intersection(socs_k)]
        w_k = src_w.reindex(sub.index).fillna(0.0).to_numpy(dtype=float)
        total_w = float(w_k.sum())

        # Weighted mean angle (circular)
        xi_arr = sub["xi_soc"].to_numpy(dtype=float)
        if total_w > 0:
            mean_cos = np.sum(w_k * np.cos(xi_arr)) / total_w
            mean_sin = np.sum(w_k * np.sin(xi_arr)) / total_w
            mean_xi = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)
            mean_chi = float(np.sum(w_k * sub["chi_soc"].to_numpy(dtype=float)) / total_w)
        else:
            mean_xi = float("nan")
            mean_chi = float("nan")

        # Top occupations by outflow
        top_socs = (
            pd.Series(w_k, index=sub.index)
            .sort_values(ascending=False)
            .head(n_top)
        )
        if title_map is not None:
            top_labels = [
                f"{title_map.get(s, s)} ({s})"
                for s in top_socs.index
            ]
        else:
            top_labels = list(top_socs.index)

        rows.append({
            "k3_system": k,
            "n_soc": int(len(socs_k)),
            "total_outflow": round(total_w, 2),
            "share_of_total": round(100 * total_w / float(src_w.sum()), 1),
            "mean_xi_deg": round(mean_xi, 1) if np.isfinite(mean_xi) else None,
            "mean_chi": round(mean_chi, 3) if np.isfinite(mean_chi) else None,
            "top_occupations": "; ".join(top_labels[:n_top]),
        })
    return pd.DataFrame(rows)


def plot_k3_geography(
    soc_system_k3: dict[str, int],
    soc_system_k2: dict[str, int],
    df_soc_centres: pd.DataFrame,
    df_edges: pd.DataFrame,
    poles_k3: np.ndarray,
    poles_k2: np.ndarray | None = None,
    out_path: str | None = None,
):
    """Two-panel plot: K=2 partition on the left, K=3 partition on the right.

    Each SOC is shown at its (x_soc, y_soc) position, coloured by system.
    Size scales with outflow weight. Poles are marked as stars.
    """
    import matplotlib.pyplot as plt

    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    centres = df_soc_centres.set_index("soc2018")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    colors_k2 = {0: "#4477AA", 1: "#EE7733", -1: "0.7"}
    colors_k3 = {0: "#4477AA", 1: "#EE7733", 2: "#117733", -1: "0.7"}

    for ax, labels, colors, poles, title in [
        (axes[0], soc_system_k2, colors_k2, poles_k2, "K = 2"),
        (axes[1], soc_system_k3, colors_k3, poles_k3, "K = 3"),
    ]:
        # Unit circle
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="0.85", linewidth=0.8)
        ax.axhline(0, color="0.92", linewidth=0.5, zorder=0)
        ax.axvline(0, color="0.92", linewidth=0.5, zorder=0)

        for soc in centres.index:
            if soc not in labels:
                continue
            lab = labels[soc]
            x = float(centres.loc[soc, "x_soc"])
            y = float(centres.loc[soc, "y_soc"])
            w = float(src_w.get(soc, 0.0))
            size = 8 + 60 * (w / src_w.max() if src_w.max() > 0 else 0)
            ax.scatter(x, y, s=size, c=colors.get(lab, "0.7"),
                       alpha=0.7, edgecolor="none")

        if poles is not None:
            for p in poles:
                if np.all(np.isfinite(p)):
                    ax.scatter(p[0], p[1], s=200, c="black", marker="*",
                               edgecolor="white", linewidth=1.5, zorder=10)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
