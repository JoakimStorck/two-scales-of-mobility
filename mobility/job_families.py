"""Mapping GMM-derived mobility systems to O*NET Job Families.

The GMM partitions the labor market based on flow patterns. O*NET's Job
Families partition it based on functional classification (the BLS view).
Comparing the two tells us how data-driven mobility structure relates to
the official taxonomy.

Two perspectives:

* **System composition.** For each GMM system, which Job Families dominate?
  This characterises each system in familiar terms (e.g. system 0 is mostly
  Office Support + Educational Instruction).
* **Job Family fragmentation.** For each Job Family, is it concentrated in
  one GMM system or split across several? This tells us which functional
  categories correspond to coherent mobility communities and which span
  multiple subsystems.

Plus a sanity check: comparing GMM partition to the Job Family partition
as if both were clusterings. Adjusted Rand Index (ARI) or normalised
mutual information (NMI) gives a single agreement measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_job_families(
    path,
    df_soc_centres: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load the O*NET Job Family table and aggregate to SOC2018 level.

    The CSV has columns Code, Occupation, Job Family at the O*NET-detailed
    level (e.g. 13-2011.00). We strip the decimal suffix to get SOC2018
    and keep one Job Family per SOC. If multiple O*NET-detailed codes
    under the same SOC are mapped to different families (rare), we keep
    the modal one.

    Parameters
    ----------
    path : str
        Path to All_Job_Families.csv.
    df_soc_centres : DataFrame, optional
        If provided, restrict to SOCs present in this table.

    Returns
    -------
    DataFrame with columns ``soc2018, job_family``.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Code": "onet_code", "Job Family": "job_family"})
    df["soc2018"] = df["onet_code"].astype(str).str.split(".").str[0]

    # If multiple ONET codes under one SOC map to different families,
    # use the mode (most frequent).
    soc_fam = (
        df.groupby("soc2018")["job_family"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )

    if df_soc_centres is not None:
        soc_fam = soc_fam[soc_fam["soc2018"].isin(df_soc_centres["soc2018"])]

    return soc_fam.reset_index(drop=True)


# ---------------------------------------------------------------------------
# System composition: which Job Families dominate each GMM system?
# ---------------------------------------------------------------------------

def system_composition(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
    top_n: int = None,
) -> pd.DataFrame:
    """For each GMM system, the share of each Job Family.

    Parameters
    ----------
    soc_system : dict
        SOC -> system label from a SystemAssignment.
    df_jobfam : DataFrame
        Output of load_job_families.
    df_edges : DataFrame
        Edge table (used for outflow weights).
    weight_by_outflow : bool
        If True, weight each SOC by its total outgoing flow; if False,
        unweighted SOC counts.
    top_n : int, optional
        Show only the top N Job Families per system (rest grouped as
        "Other"). If None, show all.

    Returns
    -------
    DataFrame
        Long form: system, job_family, weight, share_within_system (%).
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc, sys in soc_system.items():
        fam = fam_map.get(soc, "Unknown")
        if weight_by_outflow:
            w = float(src_w.get(soc, 0.0))
        else:
            w = 1.0
        rows.append({"system": sys, "job_family": fam, "weight": w})

    df = pd.DataFrame(rows)
    agg = df.groupby(["system", "job_family"], as_index=False)["weight"].sum()

    # Share within each system
    sys_totals = agg.groupby("system")["weight"].sum()
    agg["share_within_system"] = (
        100 * agg["weight"] / agg["system"].map(sys_totals)
    ).round(1)

    agg = agg.sort_values(
        ["system", "weight"], ascending=[True, False]
    ).reset_index(drop=True)

    if top_n is not None:
        keep_rows = []
        for sys, sub in agg.groupby("system"):
            top = sub.head(top_n)
            rest = sub.iloc[top_n:]
            if len(rest) > 0:
                other = pd.DataFrame([{
                    "system": sys,
                    "job_family": f"Other ({len(rest)} families)",
                    "weight": rest["weight"].sum(),
                    "share_within_system": round(
                        rest["share_within_system"].sum(), 1
                    ),
                }])
                keep_rows.append(pd.concat([top, other], ignore_index=True))
            else:
                keep_rows.append(top)
        agg = pd.concat(keep_rows, ignore_index=True)

    return agg


# ---------------------------------------------------------------------------
# Job Family fragmentation: how is each family split across GMM systems?
# ---------------------------------------------------------------------------

def family_fragmentation(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
) -> pd.DataFrame:
    """For each Job Family, the share landing in each GMM system.

    Returns one row per (family, system) pair. The dominant_share column
    is the maximum share, giving a per-family "concentration" measure:
    high values mean the family lives mostly in one system.
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc, sys in soc_system.items():
        fam = fam_map.get(soc, "Unknown")
        if weight_by_outflow:
            w = float(src_w.get(soc, 0.0))
        else:
            w = 1.0
        rows.append({"job_family": fam, "system": sys, "weight": w})

    df = pd.DataFrame(rows)
    agg = df.groupby(["job_family", "system"], as_index=False)["weight"].sum()
    fam_totals = agg.groupby("job_family")["weight"].sum()
    agg["share_of_family"] = (
        100 * agg["weight"] / agg["job_family"].map(fam_totals)
    ).round(1)

    # Dominant-share per family (concentration measure)
    dom = agg.groupby("job_family")["share_of_family"].max()
    agg["dominant_share"] = agg["job_family"].map(dom)

    agg = agg.sort_values(
        ["dominant_share", "job_family", "system"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    return agg


# ---------------------------------------------------------------------------
# Overall agreement: ARI and NMI between GMM and Job Family partitions
# ---------------------------------------------------------------------------

def partition_agreement(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
) -> dict:
    """Adjusted Rand Index and Normalised Mutual Information.

    Treats both partitions (GMM systems and Job Families) as clusterings
    over the common set of SOCs and reports standard agreement metrics.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        adjusted_mutual_info_score,
    )

    fam_map = df_jobfam.set_index("soc2018")["job_family"]
    common = sorted(set(soc_system.keys()) & set(fam_map.index))

    labels_gmm = np.array([soc_system[s] for s in common])
    labels_fam = np.array([fam_map[s] for s in common])

    return {
        "n_soc": len(common),
        "n_gmm_systems": int(len(set(labels_gmm))),
        "n_job_families": int(len(set(labels_fam))),
        "ARI": float(adjusted_rand_score(labels_fam, labels_gmm)),
        "NMI": float(normalized_mutual_info_score(labels_fam, labels_gmm)),
        "AMI": float(adjusted_mutual_info_score(labels_fam, labels_gmm)),
    }


# ---------------------------------------------------------------------------
# Sweep: agreement as function of K
# ---------------------------------------------------------------------------

def agreement_by_K(
    assignments: dict,
    df_jobfam: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ARI/NMI for each K in a K-sweep.

    Hypothesis: agreement rises with K up to some point (more GMM systems
    = finer match to functional categories) then declines as GMM begins
    fragmenting beyond the natural family granularity.
    """
    rows = []
    for K, assignment in sorted(assignments.items()):
        agree = partition_agreement(assignment.soc_system, df_jobfam)
        rows.append({"K": int(K), **agree})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composition matrix as heatmap
# ---------------------------------------------------------------------------

def composition_matrix(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
    normalise: str = "system",   # 'system', 'family', or 'none'
) -> pd.DataFrame:
    """Build a (Job Family) x (System) matrix of weights or shares.

    Parameters
    ----------
    normalise : {'system', 'family', 'none'}
        If 'system', columns sum to 100 (share of each system by family).
        If 'family', rows sum to 100 (share of each family by system).
        If 'none', raw weight totals.
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc, sys in soc_system.items():
        fam = fam_map.get(soc, "Unknown")
        w = float(src_w.get(soc, 0.0)) if weight_by_outflow else 1.0
        rows.append({"job_family": fam, "system": sys, "weight": w})

    df = pd.DataFrame(rows)
    mat = (
        df.groupby(["job_family", "system"])["weight"].sum()
        .unstack(fill_value=0.0)
    )

    if normalise == "system":
        col_sum = mat.sum(axis=0).replace(0, np.nan)
        mat = (100 * mat / col_sum).round(1)
    elif normalise == "family":
        row_sum = mat.sum(axis=1).replace(0, np.nan)
        mat = (100 * mat.div(row_sum, axis=0)).round(1)

    return mat


def plot_composition_heatmap(
    matrix: pd.DataFrame,
    *,
    out_path: str | None = None,
    title: str = "Composition matrix",
    cmap: str = "viridis",
):
    """Heatmap of the composition matrix from composition_matrix."""
    import matplotlib.pyplot as plt

    # Sort rows by their dominant system for visual coherence
    if matrix.shape[1] > 1:
        dominant_sys = matrix.idxmax(axis=1)
        row_order = matrix.index[
            np.lexsort((matrix.index, dominant_sys.values))
        ]
        matrix = matrix.loc[row_order]

    fig, ax = plt.subplots(
        figsize=(0.6 * matrix.shape[1] + 3.5,
                 0.3 * matrix.shape[0] + 1.5),
        constrained_layout=True,
    )

    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"Sys {c}" for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title(title)

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if np.isfinite(v) and v >= 5:
                color = "white" if v > matrix.values.max() * 0.5 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="%")

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
