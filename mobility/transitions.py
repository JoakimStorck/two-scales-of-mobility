"""IPUMS-CPS transition matrix construction.

The pipeline:

1. Load the aggregated source-destination-weight CSV.
2. Build a square row-normalised matrix P over the SOC2018 universe.
3. Build an edge table where each row is a source-destination pair with
   probability above threshold, plus geometric distance and normalised
   hop length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mobility.geometry import norm_soc2018, polar_to_xy


# ---------------------------------------------------------------------------
# Loading and matrix construction
# ---------------------------------------------------------------------------

def load_ipums_transitions(path) -> pd.DataFrame:
    """Load the aggregated IPUMS-CPS transitions file.

    Required columns: ``source``, ``destination``, ``WTFINL``.

    Returns a DataFrame with normalised columns
    ``[source_soc, dest_soc, w]`` and only positive finite weights.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    required = {"source", "destination", "WTFINL"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"IPUMS file missing columns: {missing}")

    df["source_soc"] = df["source"].map(norm_soc2018)
    df["dest_soc"] = df["destination"].map(norm_soc2018)
    df["w"] = pd.to_numeric(df["WTFINL"], errors="coerce")
    df = df.dropna(subset=["source_soc", "dest_soc", "w"])
    df = df[np.isfinite(df["w"]) & (df["w"] > 0)].copy()
    return df[["source_soc", "dest_soc", "w"]].reset_index(drop=True)


def build_transition_matrix(
    df_ipums: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the row-normalised SOC2018 transition matrix.

    Parameters
    ----------
    df_ipums : DataFrame
        Output of :func:`load_ipums_transitions`.

    Returns
    -------
    F : ndarray, shape (N, N)
        Raw flow matrix (sum of weights). Diagonal includes self-flows
        if any were present in the input.
    P : ndarray, shape (N, N)
        Row-normalised transition matrix. Rows that summed to zero in F
        remain zero in P.
    soc_index : DataFrame
        Two columns ``[id, soc2018]`` mapping matrix indices to SOC codes.
    """
    soc_list = sorted(set(df_ipums["source_soc"]) | set(df_ipums["dest_soc"]))
    soc_to_id = {s: i for i, s in enumerate(soc_list)}
    n = len(soc_list)

    grouped = (
        df_ipums.groupby(["source_soc", "dest_soc"], as_index=False)["w"].sum()
    )
    src_idx = grouped["source_soc"].map(soc_to_id).to_numpy(dtype=int)
    dst_idx = grouped["dest_soc"].map(soc_to_id).to_numpy(dtype=int)
    weights = grouped["w"].to_numpy(dtype=float)

    F = np.zeros((n, n), dtype=np.float64)
    np.add.at(F, (src_idx, dst_idx), weights)

    row_sum = F.sum(axis=1, keepdims=True)
    P = np.divide(
        F, row_sum,
        out=np.zeros_like(F, dtype=np.float64),
        where=(row_sum > 0),
    )

    soc_index = pd.DataFrame({"id": range(n), "soc2018": soc_list})
    return F, P, soc_index


# ---------------------------------------------------------------------------
# Edge table construction
# ---------------------------------------------------------------------------

def build_edges(
    P: np.ndarray,
    soc_index: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    df_task_radii: pd.DataFrame,
    p_threshold: float = 0.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Build the SOC2018 → SOC2018 edge table with geometric distances.

    Self-loops are dropped. For each source, edges below ``p_threshold``
    are excluded; if ``top_k`` is given, only the strongest ``top_k``
    destinations are kept.

    Parameters
    ----------
    P : ndarray
        Row-normalised transition matrix from :func:`build_transition_matrix`.
    soc_index : DataFrame
        SOC index from :func:`build_transition_matrix`.
    df_soc_centres : DataFrame
        Output of :func:`mobility.geometry.build_soc_centres`. Must contain
        ``soc2018``, ``x_soc``, ``y_soc``.
    df_task_radii : DataFrame
        Output of :func:`mobility.geometry.build_task_radii`. Must contain
        ``soc2018``, ``R_task_rms``.
    p_threshold : float
        Minimum probability to retain an edge.
    top_k : int | None
        If set, keep at most this many destinations per source.

    Returns
    -------
    DataFrame with columns
    ``[src_soc2018, tgt_soc2018, wP, d_xy, R_src_task_rms, u_R,
    x_src, y_src, x_tgt, y_tgt]``.

    The normalised hop length is ``u_R = d_xy / R_src_task_rms``, where
    ``R_src_task_rms`` is the RMS task radius of the source occupation.
    """
    soc_pool = sorted(
        set(soc_index["soc2018"])
        & set(df_soc_centres["soc2018"])
        & set(df_task_radii["soc2018"])
    )

    soc_to_id = dict(zip(soc_index["soc2018"], soc_index["id"]))
    id_to_soc = soc_index.set_index("id")["soc2018"]

    centres = df_soc_centres.set_index("soc2018")[["x_soc", "y_soc"]]
    radii = (
        df_task_radii.set_index("soc2018")[["R_task_rms"]]
        .rename(columns={"R_task_rms": "R_src_task_rms"})
    )

    rows = []
    for src in soc_pool:
        i = soc_to_id.get(src)
        if i is None:
            continue
        probs = P[i, :].copy()
        probs[i] = 0.0  # drop self-loop

        idx = np.where(probs > float(p_threshold))[0]
        if idx.size == 0:
            continue
        if top_k is not None and int(top_k) > 0 and idx.size > int(top_k):
            idx = idx[np.argsort(probs[idx])[-int(top_k):]]

        x_src = float(centres.loc[src, "x_soc"])
        y_src = float(centres.loc[src, "y_soc"])
        R_src = float(radii.loc[src, "R_src_task_rms"])

        for j in idx:
            tgt = id_to_soc.iloc[int(j)]
            if tgt not in centres.index or tgt not in radii.index:
                continue
            x_tgt = float(centres.loc[tgt, "x_soc"])
            y_tgt = float(centres.loc[tgt, "y_soc"])
            d_xy = float(np.hypot(x_src - x_tgt, y_src - y_tgt))
            u_R = d_xy / R_src if R_src > 0 else np.nan
            rows.append((
                src, tgt, float(probs[j]), d_xy,
                R_src, u_R,
                x_src, y_src, x_tgt, y_tgt,
            ))
    
    return pd.DataFrame(rows, columns=[
        "src_soc2018", "tgt_soc2018", "wP", "d_xy",
        "R_src_task_rms", "u_R",
        "x_src", "y_src", "x_tgt", "y_tgt",
    ])