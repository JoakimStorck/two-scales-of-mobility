"""Global-share tables: Job Family vs GMM system, one table per K.

Each cell is the share of total mobility weight in the intersection
(family, system), computed as

    cell = weight(family AND system) / total_weight * 100

so the entire table sums to ~100% (modulo SOCs without a Job Family
assignment). Row sums give each family's total share of mobility;
column sums give each system's total share.

The tables are directly comparable across K because the unit is the
same in every cell: percent of total mobility.

In addition to the per-K tables, this module provides a summary table
that condenses the K-sweep into a single view: for each Job Family,
the share of the family's weight that lies in its dominant GMM system,
shown as a column per K value. Families with high dominant shares
across all K are mobility-coherent; families with shares that drop
sharply as K rises are split by GMM into multiple subsystems.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def global_share_table(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
    round_to: int = 2,
    add_totals: bool = True,
    sort_families_by: str = "row_total",   # 'row_total' or 'name'
) -> pd.DataFrame:
    """One table: rows = Job Families, columns = GMM systems, cells = % of total."""
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc, sys in soc_system.items():
        fam = fam_map.get(soc, None)
        if fam is None:
            continue
        if weight_by_outflow:
            w = float(src_w.get(soc, 0.0))
        else:
            w = 1.0
        rows.append({"job_family": fam, "system": sys, "weight": w})

    df = pd.DataFrame(rows)
    total_w = df["weight"].sum()
    if total_w <= 0:
        return pd.DataFrame()

    mat = (
        df.groupby(["job_family", "system"])["weight"].sum()
        .unstack(fill_value=0.0)
    )
    mat = (100.0 * mat / total_w).round(round_to)

    mat = mat[sorted(mat.columns)]
    mat.columns = [f"Sys {c}" for c in mat.columns]

    if sort_families_by == "row_total":
        mat = mat.assign(_total=mat.sum(axis=1)).sort_values(
            "_total", ascending=False
        ).drop(columns="_total")
    elif sort_families_by == "name":
        mat = mat.sort_index()

    if add_totals:
        col_totals = mat.sum(axis=0).round(round_to)
        row_totals = mat.sum(axis=1).round(round_to)
        mat["TOTAL"] = row_totals
        mat.loc["TOTAL"] = list(col_totals) + [round(col_totals.sum(), round_to)]

    mat.index.name = "Job Family"
    return mat


def all_global_share_tables(
    assignments: dict,
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
    round_to: int = 2,
    sort_families_by: str = "row_total",
) -> dict:
    """Produce one global-share table per K in assignments."""
    out = {}
    for K, assignment in sorted(assignments.items()):
        out[K] = global_share_table(
            assignment.soc_system,
            df_jobfam,
            df_edges,
            weight_by_outflow=weight_by_outflow,
            round_to=round_to,
            add_totals=True,
            sort_families_by=sort_families_by,
        )
    return out


def save_tables_to_csv(tables: dict, dir_path, prefix: str = "jobfam_global_K"):
    """Save each table to a CSV with name '<prefix><K>.csv'."""
    from pathlib import Path
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for K, df in tables.items():
        fp = dir_path / f"{prefix}{K}.csv"
        df.to_csv(fp)
        print(f"  Saved: {fp}")


# ---------------------------------------------------------------------------
# Summary table: dominant share per family across K-values
# ---------------------------------------------------------------------------

def _family_dominant_share(
    soc_system: dict[str, int],
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
) -> pd.Series:
    """Per-family dominant share: max(% of family weight in any one system).

    A value of 100 means the family is entirely contained in a single GMM
    system. Lower values mean the family is split.
    """
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc, sys in soc_system.items():
        fam = fam_map.get(soc, None)
        if fam is None:
            continue
        w = float(src_w.get(soc, 0.0)) if weight_by_outflow else 1.0
        rows.append({"job_family": fam, "system": sys, "weight": w})

    df = pd.DataFrame(rows)
    agg = df.groupby(["job_family", "system"])["weight"].sum().reset_index()
    fam_total = agg.groupby("job_family")["weight"].sum()
    agg["share_of_family"] = 100 * agg["weight"] / agg["job_family"].map(fam_total)

    dominant = agg.groupby("job_family")["share_of_family"].max()
    return dominant


def _family_total_share(
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    soc_universe: set,
    *,
    weight_by_outflow: bool = True,
) -> pd.Series:
    """Total share of mobility weight in each Job Family (in %)."""
    src_w = df_edges.groupby("src_soc2018")["wP"].sum()
    fam_map = df_jobfam.set_index("soc2018")["job_family"]

    rows = []
    for soc in soc_universe:
        fam = fam_map.get(soc, None)
        if fam is None:
            continue
        w = float(src_w.get(soc, 0.0)) if weight_by_outflow else 1.0
        rows.append({"job_family": fam, "weight": w})

    df = pd.DataFrame(rows)
    fam_w = df.groupby("job_family")["weight"].sum()
    return (100 * fam_w / fam_w.sum()).round(2)


def coherence_summary(
    assignments: dict,
    df_jobfam: pd.DataFrame,
    df_edges: pd.DataFrame,
    *,
    weight_by_outflow: bool = True,
    round_to: int = 1,
    sort_by: str = "total_share",     # 'total_share', 'name', or 'K2'
) -> pd.DataFrame:
    """One-table summary of how coherently GMM treats each Job Family.

    Rows are Job Families. The first column is total_share — the family's
    share of total mobility — for context. The remaining columns are
    dom_K2, dom_K3, ..., dom_KN: the dominant share at each K, i.e. the
    fraction of the family's weight that lies in its single largest GMM
    system at that K.

    A family with dominant shares near 100 across all K is mobility-
    coherent: GMM never splits it. A family whose dominant share drops
    sharply with K is split by GMM into multiple subsystems.

    Parameters
    ----------
    assignments : dict
        Maps K -> SystemAssignment.
    df_jobfam : DataFrame
        Columns: soc2018, job_family.
    df_edges : DataFrame
        Edge table.
    weight_by_outflow : bool
        If True, weight each SOC by its outgoing flow.
    round_to : int
        Decimal places for the cell values.
    sort_by : {'total_share', 'name', 'K2'}
        Row order. 'total_share' (default) orders families by importance.
        'K2' orders by descending dominant share at K=2.

    Returns
    -------
    DataFrame
        Indexed by Job Family.
    """
    Ks = sorted(assignments.keys())

    soc_universe = set()
    for K in Ks:
        soc_universe |= set(assignments[K].soc_system.keys())

    total_share = _family_total_share(
        df_jobfam, df_edges, soc_universe,
        weight_by_outflow=weight_by_outflow,
    )

    cols = {}
    for K in Ks:
        cols[f"dom_K{K}"] = _family_dominant_share(
            assignments[K].soc_system,
            df_jobfam, df_edges,
            weight_by_outflow=weight_by_outflow,
        )

    df = pd.DataFrame(cols)
    df.insert(0, "total_share", total_share)
    df = df.round(round_to)
    df.index.name = "Job Family"

    if sort_by == "total_share":
        df = df.sort_values("total_share", ascending=False)
    elif sort_by == "name":
        df = df.sort_index()
    elif sort_by == "K2":
        first_dom = f"dom_K{Ks[0]}"
        df = df.sort_values(first_dom, ascending=False)

    return df