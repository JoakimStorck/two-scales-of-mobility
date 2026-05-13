"""Narrative summary table of the K-sweep.

Produces the compact 'what does each K reveal' table for the AR paper.
Three of the four columns are computed automatically:

* K (number of subsystems)
* Crossing flow (% of total)
* Smallest system (% of total)

The fourth column — narrative description of new functional structure
— is left for the author to fill in based on characterise_systems
output and Job Family composition.

The function exports both a CSV with the numeric columns and a stub
column for the narrative text, and a LaTeX snippet ready for
inclusion in the manuscript.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hierarchy_summary(
    assignments: dict,
    df_jobfam: pd.DataFrame | None = None,
    df_edges: pd.DataFrame | None = None,
    *,
    round_to: int = 1,
) -> pd.DataFrame:
    """Build the narrative summary DataFrame.

    Parameters
    ----------
    assignments : dict
        Maps K -> SystemAssignment (from k_sweep.run_k_sweep).
    df_jobfam : DataFrame, optional
        If provided, also reports the dominant Job Family per smallest
        system at each K, as a hint for filling in the narrative column.
    df_edges : DataFrame, optional
        Edge table, needed if df_jobfam is provided.
    round_to : int
        Decimal places for numeric columns.

    Returns
    -------
    DataFrame with columns:
        K, n_systems, crossing_flow_pct, smallest_system_pct,
        hint_dominant_family_smallest (if jobfam given),
        narrative (empty, to be filled in by author).
    """
    rows = []
    for K, asn in sorted(assignments.items()):
        # System weights (share of total) — needed to find the smallest
        if df_edges is not None:
            src_w = df_edges.groupby("src_soc2018")["wP"].sum()
            sys_weights = {}
            for soc, lab in asn.soc_system.items():
                sys_weights[lab] = sys_weights.get(lab, 0.0) + float(
                    src_w.get(soc, 0.0)
                )
            total_w = sum(sys_weights.values())
            sys_shares = {
                k: 100 * v / total_w for k, v in sys_weights.items()
            }
            smallest_id = min(sys_shares, key=sys_shares.get)
            smallest_share = sys_shares[smallest_id]
        else:
            smallest_share = float(asn.min_sys_frac)
            smallest_id = None

        row = {
            "K": int(K),
            "n_systems": int(asn.n_components),
            "crossing_flow_pct": round(float(asn.flow_between), round_to),
            "smallest_system_pct": round(smallest_share, round_to),
        }

        # Optional: hint about the dominant Job Family in the smallest system
        if df_jobfam is not None and df_edges is not None and smallest_id is not None:
            fam_map = df_jobfam.set_index("soc2018")["job_family"]
            fam_weights = {}
            for soc, lab in asn.soc_system.items():
                if lab != smallest_id:
                    continue
                fam = fam_map.get(soc, None)
                if fam is None:
                    continue
                fam_weights[fam] = fam_weights.get(fam, 0.0) + float(
                    src_w.get(soc, 0.0)
                )
            if fam_weights:
                total_w_sys = sum(fam_weights.values())
                top_fam = max(fam_weights, key=fam_weights.get)
                top_share = 100 * fam_weights[top_fam] / total_w_sys
                row["hint_smallest_top_family"] = (
                    f"{top_fam} ({top_share:.0f}%)"
                )
            else:
                row["hint_smallest_top_family"] = ""

        row["narrative"] = ""  # placeholder for author text
        rows.append(row)

    return pd.DataFrame(rows)


def to_latex_table(
    df: pd.DataFrame,
    *,
    caption: str = "Hierarchical structure of mobility segmentation. "
                   "Each row shows the partition produced by GMM with K "
                   "subsystems, the share of total mobility weight that "
                   "crosses subsystem boundaries, and the size of the "
                   "smallest subsystem. The final column describes the "
                   "new functional structure that emerges at that level.",
    label: str = "tab:hierarchy",
    include_hint: bool = False,
) -> str:
    """Render the summary as a LaTeX booktabs table.

    Designed for direct paste into the manuscript. The narrative column
    is rendered as a paragraph cell with a fixed width so longer text
    wraps cleanly.
    """
    cols_show = ["K", "crossing_flow_pct", "smallest_system_pct"]
    headers = ["K", "Crossing flow (\\%)", "Smallest system (\\%)"]
    if include_hint and "hint_smallest_top_family" in df.columns:
        cols_show.append("hint_smallest_top_family")
        headers.append("Smallest system's top family")
    cols_show.append("narrative")
    headers.append("Emerging functional structure")

    # Build column specifier: small numeric columns + a wider paragraph
    col_spec = "c" * (len(cols_show) - 1) + "p{6.5cm}"

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for c in cols_show:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
