"""Mobility cost as a function of partition resolution.

For each K we compute:

* Median normalised hop length u_R within each system (weighted by flow).
* Pooled within-system median u_R (single number per K).
* Crossing-edge median u_R.
* Share of total flow that crosses system boundaries (= flow_between).

The hypothesis: as K increases, within-system mobility becomes shorter
(narrower systems contain only the closest occupations) while crossing
flow increases (more boundaries to cross). The trade-off characterises
the hierarchical structure of segmentation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.stats import wquantile
from mobility.systems import assign_edge_systems


def cost_by_K(
    assignments: dict,
    df_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Compute within/crossing mobility cost statistics for each K.

    Parameters
    ----------
    assignments : dict
        Output of k_sweep.run_k_sweep. Maps K -> SystemAssignment.
    df_edges : DataFrame
        Edge table with x_src, y_src, x_tgt, y_tgt, wP, u_R columns.

    Returns
    -------
    DataFrame
        One row per K, with columns:
        - K
        - n_systems
        - flow_within (%)
        - flow_between (%)
        - within_median_uR        : pooled within-system median u_R
        - within_p75_uR
        - within_p25_uR
        - within_w_sum             : total flow weight within systems
        - crossing_median_uR
        - crossing_p75_uR
        - crossing_p25_uR
        - crossing_w_sum
        - cost_ratio               : crossing_median_uR / within_median_uR
    """
    rows = []
    for K, assignment in sorted(assignments.items()):
        src_sys, tgt_sys, crossing = assign_edge_systems(
            df_edges, assignment.soc_system,
        )
        within = (src_sys >= 0) & (tgt_sys >= 0) & (src_sys == tgt_sys)

        u_R = df_edges["u_R"].to_numpy(dtype=float)
        w = df_edges["wP"].to_numpy(dtype=float)

        u_in, w_in = u_R[within], w[within]
        u_cr, w_cr = u_R[crossing], w[crossing]

        within_median = wquantile(u_in, w_in, 0.50) if w_in.sum() > 0 else float("nan")
        within_p25 = wquantile(u_in, w_in, 0.25) if w_in.sum() > 0 else float("nan")
        within_p75 = wquantile(u_in, w_in, 0.75) if w_in.sum() > 0 else float("nan")

        crossing_median = wquantile(u_cr, w_cr, 0.50) if w_cr.sum() > 0 else float("nan")
        crossing_p25 = wquantile(u_cr, w_cr, 0.25) if w_cr.sum() > 0 else float("nan")
        crossing_p75 = wquantile(u_cr, w_cr, 0.75) if w_cr.sum() > 0 else float("nan")

        cost_ratio = (crossing_median / within_median
                      if within_median and within_median > 0
                      else float("nan"))

        rows.append({
            "K": int(K),
            "n_systems": int(assignment.n_components),
            "flow_within": float(assignment.flow_within),
            "flow_between": float(assignment.flow_between),
            "within_median_uR": within_median,
            "within_p25_uR": within_p25,
            "within_p75_uR": within_p75,
            "within_w_sum": float(w_in.sum()),
            "crossing_median_uR": crossing_median,
            "crossing_p25_uR": crossing_p25,
            "crossing_p75_uR": crossing_p75,
            "crossing_w_sum": float(w_cr.sum()),
            "cost_ratio": cost_ratio,
        })
    return pd.DataFrame(rows)


def cost_per_system(
    assignments: dict,
    df_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Per-system within-mobility statistics for each K.

    More detailed than cost_by_K: shows the within-system median u_R for
    each individual system, not just the pooled value. Helps see whether
    systems differ in their internal mobility cost.

    Returns
    -------
    DataFrame
        One row per (K, system_id).
    """
    rows = []
    for K, assignment in sorted(assignments.items()):
        src_sys, tgt_sys, _ = assign_edge_systems(
            df_edges, assignment.soc_system,
        )
        u_R = df_edges["u_R"].to_numpy(dtype=float)
        w = df_edges["wP"].to_numpy(dtype=float)

        for s in range(assignment.n_components):
            mask = (src_sys == s) & (tgt_sys == s)
            u_s, w_s = u_R[mask], w[mask]
            if w_s.sum() <= 0:
                continue
            rows.append({
                "K": int(K),
                "system": int(s),
                "n_edges": int(mask.sum()),
                "w_sum": float(w_s.sum()),
                "share_of_within": float(w_s.sum()) / float(
                    w[(src_sys >= 0) & (tgt_sys >= 0) & (src_sys == tgt_sys)].sum()
                ),
                "median_uR": wquantile(u_s, w_s, 0.50),
                "p25_uR": wquantile(u_s, w_s, 0.25),
                "p75_uR": wquantile(u_s, w_s, 0.75),
            })
    return pd.DataFrame(rows)


def plot_cost_by_K(
    df_cost: pd.DataFrame,
    out_path: str | None = None,
):
    """Two-panel plot: within/crossing median u_R, and flow shares.

    Panel A: median u_R for within-system and crossing edges, with IQR.
    Panel B: share of total flow within vs crossing systems.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax_uR, ax_flow = axes

    K = df_cost["K"].to_numpy()

    # --- Panel A: cost ---
    w_med = df_cost["within_median_uR"].to_numpy()
    w_lo = df_cost["within_p25_uR"].to_numpy()
    w_hi = df_cost["within_p75_uR"].to_numpy()
    c_med = df_cost["crossing_median_uR"].to_numpy()
    c_lo = df_cost["crossing_p25_uR"].to_numpy()
    c_hi = df_cost["crossing_p75_uR"].to_numpy()

    ax_uR.plot(K, w_med, color="tab:blue", marker="o", linewidth=1.8,
               label="Within-system median")
    ax_uR.fill_between(K, w_lo, w_hi, color="tab:blue", alpha=0.18,
                       label="Within-system IQR")
    ax_uR.plot(K, c_med, color="tab:orange", marker="s", linewidth=1.8,
               label="Crossing median")
    ax_uR.fill_between(K, c_lo, c_hi, color="tab:orange", alpha=0.18,
                       label="Crossing IQR")

    ax_uR.axhline(1.0, color="0.5", linewidth=0.7, linestyle=":")
    ax_uR.text(K[-1], 1.04, "one task radius",
               color="0.5", fontsize=8, ha="right", va="bottom")

    ax_uR.set_xlabel("Number of systems K")
    ax_uR.set_ylabel("Normalised transition length $u_R$")
    ax_uR.set_xticks(K)
    ax_uR.grid(alpha=0.3)
    ax_uR.legend(loc="best", frameon=True, fontsize=9)

    # --- Panel B: flow shares ---
    ax_flow.plot(K, df_cost["flow_within"].to_numpy(), color="tab:blue",
                 marker="o", linewidth=1.8, label="Within-system flow")
    ax_flow.plot(K, df_cost["flow_between"].to_numpy(), color="tab:orange",
                 marker="s", linewidth=1.8, label="Crossing flow")
    ax_flow.set_xlabel("Number of systems K")
    ax_flow.set_ylabel("Share of total flow (%)")
    ax_flow.set_xticks(K)
    ax_flow.set_ylim(0, 100)
    ax_flow.grid(alpha=0.3)
    ax_flow.legend(loc="best", frameon=True, fontsize=9)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
