"""Matplotlib plotting functions for the mobility paper.

Each function takes the relevant DataFrame plus an output path and produces
a single figure. Figures are written as PDF for the paper and PNG for quick
inspection.

Style choices follow the paper: minimal grid, clear labels, no colour by
default in distribution plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility.stats import weighted_ecdf, weighted_cdf_at, wquantile

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_paper_style() -> None:
    """Apply paper-friendly matplotlib defaults."""
    mpl.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })


# ---------------------------------------------------------------------------
# Figure 4(a): ECDF of absolute hop length
# ---------------------------------------------------------------------------

def plot_ecdf_absolute(
    df_edges: pd.DataFrame,
    out_path: Path | str,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """ECDF of absolute hop length d_xy, weighted by wP."""
    set_paper_style()

    d = df_edges["d_xy"].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    xs, cdf = weighted_ecdf(d, w)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(xs, cdf, where="post", lw=2)
    ax.set_xlabel(r"Absolute hop length $d$")
    ax.set_ylabel("Cumulative share")
    ax.set_ylim(0, 1.0)
    ax.set_title("ECDF of absolute hop lengths")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4(b): ECDF of normalised hop length u_R
# ---------------------------------------------------------------------------

def plot_ecdf_normalised(
    df_edges: pd.DataFrame,
    out_path: Path | str,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """ECDF of normalised hop length u_R = d_xy / R_src, weighted by wP.

    Annotates the share of weight within one RMS source radius.
    """
    set_paper_style()

    u = df_edges["u_R"].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    xs, cdf = weighted_ecdf(u, w)

    y_at_one = weighted_cdf_at(xs, cdf, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(xs, cdf, where="post", lw=2,
            label=r"$u_R = d / R_{\mathrm{rms,src}}$")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)

    # Reference at u_R = 1
    ax.axvline(1.0, lw=1.8, linestyle="--", alpha=0.7, label=r"$u_R = 1$")
    ax.hlines(y_at_one, xmin=0.0, xmax=1.0, lw=1.8, linestyle="--", alpha=0.7)
    ax.plot(1.0, y_at_one, "o", ms=6)
    ax.text(
        0.02, y_at_one + 0.02,
        f"{100 * y_at_one:.0f}% within one RMS radius",
        ha="left", va="bottom", fontsize=16,
    )

    ax.set_xlabel(r"Normalised hop length $u_R$")
    ax.set_ylabel("Cumulative share")
    ax.set_title(r"ECDF of normalised hop lengths ($u_R$)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4(c): scatter of overlap vs hop length
# ---------------------------------------------------------------------------

def plot_overlap_vs_hop(
    df_edges: pd.DataFrame,
    out_path: Path | str,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Scatter of normalised signed overlap vs normalised hop length u_R.

    Point size is proportional to wP. An OLS trend line is overlaid.
    """
    set_paper_style()

    o = df_edges["overlap_norm_src"].to_numpy(dtype=float)
    u = df_edges["u_R"].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    mask = np.isfinite(o) & np.isfinite(u) & np.isfinite(w) & (w > 0)
    o, u, w = o[mask], u[mask], w[mask]

    fig, ax = plt.subplots(figsize=figsize)
    w_max = float(w.max()) if w.size else 0.0
    s = (10.0 + 90.0 * (w / w_max)) if w_max > 0 else (20.0 * np.ones_like(w))
    ax.scatter(o, u, s=2 * s, alpha=0.25, edgecolors="none",
               label=f"Edges (n={len(o)})")

    if o.size >= 3 and np.nanmax(o) > np.nanmin(o):
        coef = np.polyfit(o, u, deg=1)
        xp = np.linspace(np.nanmin(o), np.nanmax(o), 200)
        ax.plot(xp, np.polyval(coef, xp), lw=2,
                label=f"OLS: y = {coef[0]:.3f} x + {coef[1]:.3f}")

    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Normalised signed overlap")
    ax.set_ylabel(r"Normalised hop length $u_R$")
    ax.set_title("Task-radius overlap vs hop length")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4(d): CDF of signed overlap
# ---------------------------------------------------------------------------

def plot_overlap_cdf(
    df_edges: pd.DataFrame,
    out_path: Path | str,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Weighted CDF of normalised signed overlap."""
    set_paper_style()

    o = df_edges["overlap_norm_src"].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)
    xs, cdf = weighted_ecdf(o, w)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(xs, cdf, where="post", lw=2)
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Normalised signed overlap")
    ax.set_ylabel("Cumulative share")
    ax.set_title("CDF of normalised task-radius overlap")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Mobility-field plots (Section 3.1.2)
# ===========================================================================

from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

from mobility.themes import set_theme, get_theme, Theme
from mobility.fields import FieldResult, Grid
from mobility.systems import Separatrix


def _smooth_field(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that respects NaN."""
    if sigma <= 0:
        return arr
    a = np.where(np.isfinite(arr), arr, 0.0)
    return np.where(np.isfinite(arr), gaussian_filter(a, sigma=sigma), np.nan)


def _draw_disk_frame(
    ax,
    theme: Theme,
    *,
    label_step_deg: int = 30,
    show_axes: bool = True,
):
    """Draw the unit circle, optional axes, and degree labels."""
    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(th), np.sin(th), "-",
            color=theme.foreground, lw=1.5, alpha=0.7, zorder=6)
    if show_axes:
        ax.axhline(0, color=theme.muted, lw=0.6, alpha=0.4)
        ax.axvline(0, color=theme.muted, lw=0.6, alpha=0.4)
    for deg in range(0, 360, label_step_deg):
        rad = np.radians(deg)
        ax.text(1.07 * np.cos(rad), 1.07 * np.sin(rad), f"{deg}°",
                ha="center", va="center",
                fontsize=11, color=theme.muted, alpha=0.85)
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def _draw_separatrix(ax, sep: Separatrix, grid: Grid, theme: Theme,
                     lw: float = 3.0):
    """Draw the separatrix as a single contour at 0."""
    ax.contour(
        grid.GX, grid.GY, sep.field, levels=[0.0],
        colors=[theme.accent_separatrix], linewidths=lw, zorder=7,
    )


def _streamplot_field(
    ax, grid: Grid, field: FieldResult, *,
    color_value,
    cmap,
    smooth_sigma: float = 1.0,
    activity_threshold: float = 0.05,
    lw_scale: float = 5.0,
    lw_min: float = 0.8,
    lw_max: float = 6.0,
    density: float = 1.9,
    arrowsize: float = 1.7,
    zorder: int = 4,
    norm=None,
):
    """Common streamplot wrapper used by all field plots."""
    U_s = _smooth_field(field.U, smooth_sigma)
    V_s = _smooth_field(field.V, smooth_sigma)
    active = field.W_norm > activity_threshold * np.nanmax(field.W_norm)
    U_m = np.where(active, U_s, np.nan)
    V_m = np.where(active, V_s, np.nan)
    lw = np.clip(field.W_norm * lw_scale, lw_min, lw_max)

    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

    ax.streamplot(
        grid.grid_1d, grid.grid_1d, U_m, V_m,
        color=color_value,
        cmap=cmap,
        norm=norm,
        linewidth=lw,
        density=density,
        arrowsize=arrowsize,
        arrowstyle="->",
        zorder=zorder,
    )


def plot_mobility_systems_combined(
    field_systems: dict,
    field_residual: FieldResult,
    field_global: FieldResult,
    grid: Grid,
    separatrix: Separatrix,
    poles: np.ndarray,
    occupations_xy: tuple,
    out_path,
    *,
    theme="paper",
    sys_colors=None,
    figsize: tuple = (27, 9),
    title_systems: str = "Closed systems",
    title_residual: str = "Residual system",
    title_global: str = "Global mobility field",
    dpi: int = 300,
):
    """Three-panel figure: closed systems, residual, global. Paper Figure 5.

    Parameters
    ----------
    field_systems : dict[int, FieldResult]
        Per-system fields. Keys are system labels (0, 1, ...).
    field_residual : FieldResult
        Field computed from crossing edges only.
    field_global : FieldResult
        Field computed from all edges.
    grid : Grid
    separatrix : Separatrix
    poles : ndarray, shape (n_systems, 2)
    occupations_xy : tuple of (xo, yo) arrays
        Background scatter of occupations.
    out_path : path-like
    theme : {"paper", "presentation"} or Theme
    sys_colors : list of colours, one per system
    """
    th = set_theme(theme)
    if sys_colors is None:
        sys_colors = [plt.cm.tab10(i) for i in range(len(field_systems))]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"wspace": 0.015},
    )
    fig.patch.set_facecolor(th.background)

    xo, yo = occupations_xy

    def _occ_scatter(ax, alpha):
        ax.scatter(xo, yo, s=5, color=th.muted, alpha=alpha,
                   linewidths=0, zorder=2)

    def _poles(ax, dim=False):
        for i, p in enumerate(poles):
            ax.plot(*p, "*", ms=15,
                    color=sys_colors[i],
                    alpha=0.25 if dim else 1.0, zorder=8)

    # --- Panel 1: closed systems ---
    _draw_disk_frame(ax1, th)
    _occ_scatter(ax1, 0.55 if th.name == "paper" else 0.18)
    _draw_separatrix(ax1, separatrix, grid, th)
    for k, fs in field_systems.items():
        _streamplot_field(
            ax1, grid, fs,
            color_value=np.full_like(fs.R, k),
            cmap=mpl.colors.ListedColormap([sys_colors[k]]),
            norm=mpl.colors.Normalize(vmin=0, vmax=max(field_systems) or 1),
            zorder=4 + k,
        )
    _poles(ax1, dim=False)
    legend_elements = [
        Line2D([0], [0], color=sys_colors[k], lw=3,
               label=f"System {k}")
        for k in field_systems
    ] + [Line2D([0], [0], color=th.accent_separatrix, lw=3, label="System border")]
    ax1.legend(handles=legend_elements, frameon=False, fontsize=13,
               loc="lower right",
               handlelength=1.8, borderpad=0.2, labelspacing=0.25,
               labelcolor=th.foreground)
    ax1.set_title(title_systems, pad=4, color=th.foreground)

    # --- Panel 2: residual ---
    _draw_disk_frame(ax2, th)
    _occ_scatter(ax2, 0.55 if th.name == "paper" else 0.18)
    _draw_separatrix(ax2, separatrix, grid, th)
    _streamplot_field(
        ax2, grid, field_residual,
        color_value=np.full_like(field_residual.R, 0),
        cmap=mpl.colors.ListedColormap([th.muted]),
        zorder=3,
    )
    _poles(ax2, dim=True)
    ax2.set_title(title_residual, pad=4, color=th.foreground)

    # --- Panel 3: global ---
    _draw_disk_frame(ax3, th)
    _occ_scatter(ax3, 0.55 if th.name == "paper" else 0.18)
    _draw_separatrix(ax3, separatrix, grid, th)
    _streamplot_field(
        ax3, grid, field_global,
        color_value=np.full_like(field_global.R, 0),
        cmap=mpl.colors.ListedColormap([th.foreground]),
        zorder=4,
    )
    _poles(ax3, dim=False)
    ax3.set_title(title_global, pad=4, color=th.foreground)

    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.94,
                        wspace=0.015)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.02, facecolor=th.background)
    plt.close(fig)


# ===========================================================================
# Figure 6: CDF of normalised transition lengths per system
# ===========================================================================

def plot_system_cdfs(
    edge_subsets: dict,
    out_path,
    *,
    theme="paper",
    sys_colors: list = None,
    crossing_color: str = "crimson",
    sys_labels: dict = None,
    figsize: tuple = (10, 6),
    xlim: tuple = (0, 5),
):
    """Three-curve CDF of normalised transition length per system.

    Paper Figure 6.

    Parameters
    ----------
    edge_subsets : dict
        Output of :func:`mobility.systems.partition_edges_by_system`.
        Each value must contain ``u_R`` and ``wP``.
    out_path : path-like
    theme : {"paper", "presentation"} or Theme
    sys_colors : list
        One colour per within-system curve, indexed by system label.
    crossing_color : str
        Colour for the crossing-system curve.
    sys_labels : dict, optional
        Override default system labels. Keys are subset keys
        (e.g. ``"within_0"``, ``"crossing"``).
    figsize : tuple
    xlim : tuple
    """
    th = set_theme(theme)

    n_components = sum(1 for k in edge_subsets if k.startswith("within_"))
    if sys_colors is None:
        sys_colors = [plt.cm.tab10(i) for i in range(n_components)]

    if sys_labels is None:
        sys_labels = {}
        for k in range(n_components):
            sys_labels[f"within_{k}"] = f"System {k}"
        sys_labels["crossing"] = "Residual system"

    fig, ax = plt.subplots(figsize=figsize)

    for k in range(n_components):
        key = f"within_{k}"
        sub = edge_subsets[key]
        u = sub["u_R"].to_numpy(dtype=float)
        w = sub["wP"].to_numpy(dtype=float)
        valid = np.isfinite(u) & np.isfinite(w) & (w > 0)
        if valid.sum() < 5:
            continue
        xs, cdf = weighted_ecdf(u[valid], w[valid])
        ax.step(xs, cdf, where="post", lw=2.5,
                color=sys_colors[k], label=sys_labels[key])

    sub = edge_subsets["crossing"]
    u = sub["u_R"].to_numpy(dtype=float)
    w = sub["wP"].to_numpy(dtype=float)
    valid = np.isfinite(u) & np.isfinite(w) & (w > 0)
    if valid.sum() >= 5:
        xs, cdf = weighted_ecdf(u[valid], w[valid])
        ax.step(xs, cdf, where="post", lw=2.5,
                color=crossing_color, label=sys_labels["crossing"])

    ax.axvline(1.0, color=th.foreground, lw=1.5, ls="--", alpha=0.7)
    ax.text(1.02, 0.05, r"$u_R = 1$",
            color=th.foreground, fontsize=14, va="bottom")
    ax.set_xlabel(r"Normalised transition length $u_R = d\,/\,R_{\mathrm{src}}$")
    ax.set_ylabel("Cumulative share")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc="lower right",
              labelcolor=th.foreground)
    ax.set_title("CDF of normalised transition lengths by system", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)


# ===========================================================================
# Figure 7: angular direction analysis
# ===========================================================================

def plot_angular_directions(
    curves: dict,
    out_path,
    *,
    poles: np.ndarray = None,
    pole_labels: list = None,
    theme="paper",
    sys_colors: list = None,
    crossing_color: str = "#22AA66",
    sys_labels: dict = None,
    figsize: tuple = (13, 6),
    origin_deg: float = 135.0,
    normalised: bool = True,
    show_band: bool = False,
    band_n_se: float = 1.0,
    band_alpha: float = 0.18,
):
    """Tangential projection of mobility along source angle xi.

    Paper Figure 7. The horizontal axis is rotated so that ``origin_deg``
    sits at x=0; this places the "natural" boundary between cognitive and
    physical clusters in the centre of the plot for visual clarity.

    Parameters
    ----------
    curves : dict
        Keys are subset names (``"within_0"``, ``"within_1"``, ``"crossing"``).
        Values are tuples ``(bin_centres_deg, smoothed_curve, smoothed_se, weights)``
        from :func:`mobility.directions.angular_curve`.
    out_path : path-like
    poles : ndarray, optional
        If given, vertical guides are drawn at each pole's angle.
    pole_labels : list of str, optional
        Labels for poles (e.g. ``["Pole 0", "Pole 1"]``).
    theme : {"paper", "presentation"} or Theme
    sys_colors : list
        Colours for within-system curves, indexed by system label.
    crossing_color : str
    sys_labels : dict, optional
        Override default subset labels.
    figsize : tuple
    origin_deg : float
        Geographic angle placed at x=0 in the plot.
    normalised : bool
        If True, the y-axis is fixed to [-1, 1].
    show_band : bool
        If True, draw a translucent +/- band of width ``band_n_se``
        standard errors around each curve.
    band_n_se : float
        Width of the band in standard errors (default 1.0).
    band_alpha : float
        Alpha for the bands.
    """
    th = set_theme(theme)

    n_components = sum(1 for k in curves if k.startswith("within_"))
    if sys_colors is None:
        sys_colors = [plt.cm.tab10(i) for i in range(n_components)]

    if sys_labels is None:
        sys_labels = {}
        for k in range(n_components):
            sys_labels[f"within_{k}"] = f"System {k}"
        sys_labels["crossing"] = "Residual system"

    def rot(deg):
        return ((deg - origin_deg) % 360) - 180

    first_key = next(iter(curves))
    bin_centres_deg = curves[first_key][0]
    x_rot = rot(bin_centres_deg)
    sort_idx = np.argsort(x_rot)
    x_plot = x_rot[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color=th.muted, lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color=th.foreground, lw=1.5, ls=":", alpha=0.5, zorder=5)
    if normalised:
        ax.axhspan(0, 1, alpha=0.05, color="green")
        ax.axhspan(-1, 0, alpha=0.05, color="red")
        ax.text(-175, 0.92, r"Counterclockwise (increasing $\xi$)",
                fontsize=13, color="green", alpha=0.8)
        ax.text(-175, -0.97, r"Clockwise (decreasing $\xi$)",
                fontsize=13, color="red", alpha=0.8)

    def _plot_curve(key, color, label, lw=2.5, ls="-"):
        if key not in curves:
            return
        # Support both 3-tuple (legacy) and 4-tuple (with SE)
        item = curves[key]
        if len(item) == 4:
            _, curve, se, _ = item
        else:
            _, curve, _ = item
            se = None

        c = curve[sort_idx]
        ax.plot(x_plot, c, color=color, lw=lw, ls=ls, label=label)

        if show_band and se is not None:
            s = se[sort_idx]
            lo = c - band_n_se * s
            hi = c + band_n_se * s
            if normalised:
                lo = np.clip(lo, -1, 1)
                hi = np.clip(hi, -1, 1)
            ax.fill_between(x_plot, lo, hi,
                            color=color, alpha=band_alpha, linewidth=0)

    for k in range(n_components):
        _plot_curve(f"within_{k}", sys_colors[k], sys_labels[f"within_{k}"])

    _plot_curve("crossing", crossing_color, sys_labels["crossing"],
                lw=2.0, ls="--")

    if poles is not None:
        if pole_labels is None:
            pole_labels = [f"Pole {i}" for i in range(len(poles))]
        for i, p in enumerate(poles):
            xi_pole_deg = float(np.degrees(np.arctan2(p[1], p[0])) % 360)
            xi_rot = rot(xi_pole_deg)
            color = sys_colors[i] if i < len(sys_colors) else th.foreground
            ax.axvline(xi_rot, color=color, lw=2.0, ls=":", alpha=0.8, zorder=5)
            ax.text(xi_rot, -0.88 if normalised else ax.get_ylim()[0] * 0.9,
                    f"{pole_labels[i]}\n{xi_pole_deg:.0f}°",
                    ha="center", fontsize=12, color=color, alpha=0.9)

    tick_positions = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    tick_geo = [(t + 180 + origin_deg) % 360 for t in tick_positions]
    direction_lookup = {
        0: "E", 45: "NE", 90: "N", 135: "NW",
        180: "W", 225: "SW", 270: "S", 315: "SE",
    }
    tick_labels = [
        f"{direction_lookup.get(round(geo) % 360, '')}\n{geo:.0f}°"
        for geo in tick_geo
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-180, 180)

    if normalised:
        ax.set_ylabel("Normalised tangential projection")
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylabel("Tangential displacement")

    ax.set_xlabel(r"Source direction $\xi$")
    ax.set_title(r"Mobility patterns -- movement along $\xi$ per system")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=14,
              labelcolor=th.foreground)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)

# ===========================================================================
# GMM diagnostic figure
# ===========================================================================

def plot_gmm_diagnostic(
    df_edges: pd.DataFrame,
    soc_system: dict,
    poles: np.ndarray,
    grid: Grid,
    separatrix: Separatrix,
    occupations_xy: tuple,
    out_path,
    *,
    flow_within_pct: float = None,
    flow_between_pct: float = None,
    theme="presentation",
    sys_colors: list = None,
    crossing_color: str = "crimson",
    figsize: tuple = (20, 10),
    dpi: int = 200,
):
    """Two-panel diagnostic of the GMM system assignment.

    Left:  SOC codes coloured by system membership.
    Right: All edges drawn as line segments, within-system in system colour,
           crossing-system in red.
    """
    th = set_theme(theme)
    n_components = len(set(soc_system.values()))
    if sys_colors is None:
        sys_colors = [plt.cm.tab10(i) for i in range(n_components)]

    xo, yo = occupations_xy

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(th.background)

    # Build SOC -> (x, y) lookup from the edge table
    soc_xy = (
        df_edges[["src_soc2018", "x_src", "y_src"]]
        .drop_duplicates(subset="src_soc2018")
        .set_index("src_soc2018")
    )

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor(th.background)
        ax.set_aspect("equal")
        ax.scatter(xo, yo, s=4,
                   color=th.foreground if th.name == "presentation" else "0.7",
                   alpha=0.10 if th.name == "presentation" else 0.35,
                   linewidths=0, zorder=2)
        _draw_separatrix(ax, separatrix, grid, th, lw=2.5)

        if ax_idx == 0:
            # Left: SOC nodes coloured by system
            for k in range(n_components):
                socs_k = [s for s, v in soc_system.items()
                          if v == k and s in soc_xy.index]
                xs = [float(soc_xy.loc[s, "x_src"]) for s in socs_k]
                ys = [float(soc_xy.loc[s, "y_src"]) for s in socs_k]
                ax.scatter(xs, ys, s=35, color=sys_colors[k],
                           alpha=0.90, linewidths=0, zorder=4,
                           label=f"System {k} ({len(socs_k)} SOC)")
            for i, p in enumerate(poles):
                ax.plot(*p, "*", ms=16, color=sys_colors[i], zorder=8)
            ax.set_title("SOC codes by system",
                         color=th.foreground, fontsize=16)

        else:
            # Right: edges as line segments
            from matplotlib.collections import LineCollection
            x_src_a = df_edges["x_src"].to_numpy(dtype=float)
            y_src_a = df_edges["y_src"].to_numpy(dtype=float)
            x_tgt_a = df_edges["x_tgt"].to_numpy(dtype=float)
            y_tgt_a = df_edges["y_tgt"].to_numpy(dtype=float)
            wP = df_edges["wP"].to_numpy(dtype=float)
            src_sys = df_edges["src_soc2018"].map(soc_system)
            tgt_sys = df_edges["tgt_soc2018"].map(soc_system)

            segs_within = {k: [] for k in range(n_components)}
            ws_within = {k: [] for k in range(n_components)}
            segs_between = []
            ws_between = []
            for i in range(len(df_edges)):
                ms = src_sys.iloc[i]
                mt = tgt_sys.iloc[i]
                seg = [(x_src_a[i], y_src_a[i]),
                       (x_tgt_a[i], y_tgt_a[i])]
                w = wP[i]
                if pd.notna(ms) and pd.notna(mt) and ms == mt:
                    segs_within[int(ms)].append(seg)
                    ws_within[int(ms)].append(w)
                elif pd.notna(ms) and pd.notna(mt):
                    segs_between.append(seg)
                    ws_between.append(w)

            w_max = float(wP.max())
            for k in range(n_components):
                if segs_within[k]:
                    ax.add_collection(LineCollection(
                        segs_within[k],
                        linewidths=[0.3 + 1.2 * w / w_max for w in ws_within[k]],
                        colors=sys_colors[k],
                        alpha=0.30, zorder=3,
                        label=f"Within system {k}",
                    ))
            if segs_between:
                ax.add_collection(LineCollection(
                    segs_between,
                    linewidths=[0.3 + 1.5 * w / w_max for w in ws_between],
                    colors=crossing_color,
                    alpha=0.35, zorder=3,
                    label="Between systems",
                ))
            for k in range(n_components):
                socs_k = [s for s, v in soc_system.items()
                          if v == k and s in soc_xy.index]
                xs = [float(soc_xy.loc[s, "x_src"]) for s in socs_k]
                ys = [float(soc_xy.loc[s, "y_src"]) for s in socs_k]
                ax.scatter(xs, ys, s=20, color=sys_colors[k],
                           alpha=0.80, linewidths=0, zorder=5)
            for i, p in enumerate(poles):
                ax.plot(*p, "*", ms=16, color=sys_colors[i], zorder=8)

            title = "Transitions"
            if flow_within_pct is not None and flow_between_pct is not None:
                title += (f"  |  within: {flow_within_pct:.0f}%  "
                          f"between: {flow_between_pct:.0f}%")
            ax.set_title(title, color=th.foreground, fontsize=16)

        # Frame
        th_circle = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(th_circle), np.sin(th_circle),
                "-", color=th.foreground, lw=1.2, alpha=0.5, zorder=6)
        ax.axhline(0, color=th.foreground, lw=0.4, alpha=0.2)
        ax.axvline(0, color=th.foreground, lw=0.4, alpha=0.2)
        for deg, lbl in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
            rad = np.radians(deg)
            ax.text(1.07 * np.cos(rad), 1.07 * np.sin(rad), lbl,
                    ha="center", va="center",
                    fontsize=12, color=th.foreground, alpha=0.7)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=12, loc="lower right",
                  labelcolor=th.foreground)

    fig.suptitle("GMM 4D system assignment",
                 color=th.foreground, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)


# ===========================================================================
# Presentation: combined mobility field
# ===========================================================================

def plot_mobility_field_combined(
    field_systems: dict,
    grid: Grid,
    separatrix: Separatrix,
    poles: np.ndarray,
    occupations_xy: tuple,
    out_path,
    *,
    theme="presentation",
    sys_colors: list = None,
    sys_labels: dict = None,
    figsize: tuple = (12, 12),
    dpi: int = 200,
    activity_threshold: float = 0.001,
):
    """Single-panel: both systems' fields overlaid in their own colours."""
    th = set_theme(theme)
    n_components = len(field_systems)
    if sys_colors is None:
        sys_colors = [plt.cm.tab10(i) for i in range(n_components)]
    if sys_labels is None:
        sys_labels = {k: f"System {k}" for k in field_systems}

    xo, yo = occupations_xy

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th.background)
    ax.set_facecolor(th.background)
    _draw_disk_frame(ax, th)
    ax.scatter(xo, yo, s=5,
               color=th.foreground if th.name == "presentation" else "0.7",
               alpha=0.12 if th.name == "presentation" else 0.45,
               linewidths=0, zorder=2)

    for k, fs in field_systems.items():
        _streamplot_field(
            ax, grid, fs,
            color_value=np.full_like(fs.R, k),
            cmap=mpl.colors.ListedColormap([sys_colors[k]]),
            norm=mpl.colors.Normalize(vmin=0, vmax=max(field_systems) or 1),
            activity_threshold=activity_threshold,
            zorder=4 + k,
        )

    _draw_separatrix(ax, separatrix, grid, th)
    for i, p in enumerate(poles):
        ax.plot(*p, "*", ms=16, color=sys_colors[i], zorder=8)

    legend_elements = [
        Line2D([0], [0], color=sys_colors[k], lw=2,
               label=sys_labels[k])
        for k in field_systems
    ] + [Line2D([0], [0], color=th.accent_separatrix, lw=2,
                label="Separatrix")]
    ax.legend(handles=legend_elements, frameon=False, fontsize=14,
              loc="lower right", labelcolor=th.foreground)
    ax.set_title("Combined mobility field (both systems)",
                 pad=14, color=th.foreground)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)


# ===========================================================================
# Presentation: residual mobility field
# ===========================================================================

def plot_mobility_field_residual(
    field_residual: FieldResult,
    grid: Grid,
    separatrix: Separatrix,
    poles: np.ndarray,
    occupations_xy: tuple,
    out_path,
    *,
    theme="presentation",
    residual_color: str = "#22AA66",
    pole_colors: list = None,
    figsize: tuple = (12, 12),
    dpi: int = 200,
    activity_threshold: float = 0.001,
):
    """Single-panel: residual (crossing) flow field."""
    th = set_theme(theme)
    if pole_colors is None:
        pole_colors = [plt.cm.tab10(i) for i in range(len(poles))]

    xo, yo = occupations_xy

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th.background)
    ax.set_facecolor(th.background)
    _draw_disk_frame(ax, th)
    ax.scatter(xo, yo, s=5,
               color=th.foreground if th.name == "presentation" else "0.7",
               alpha=0.12 if th.name == "presentation" else 0.45,
               linewidths=0, zorder=2)

    _streamplot_field(
        ax, grid, field_residual,
        color_value=np.full_like(field_residual.R, 0),
        cmap=mpl.colors.ListedColormap([residual_color]),
        activity_threshold=activity_threshold,
        zorder=4,
    )

    _draw_separatrix(ax, separatrix, grid, th)
    for i, p in enumerate(poles):
        ax.plot(*p, "*", ms=16, color=pole_colors[i], zorder=8)

    legend_elements = [
        Line2D([0], [0], color=residual_color, lw=2,
               label="Residual system (crossing)"),
        Line2D([0], [0], color=th.accent_separatrix, lw=2,
               label="Separatrix"),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=14,
              loc="lower right", labelcolor=th.foreground)
    ax.set_title("Residual system mobility field (crossing transitions)",
                 pad=14, color=th.foreground)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)

# ===========================================================================
# Residual subgroup figures (diagnostic, optional)
# ===========================================================================

from matplotlib.collections import LineCollection

from mobility.diagnostics import (
    SUBGROUP_ORDER, SUBGROUP_STYLES,
)


# Default edge styling for the detailed subgroups figure.
# Manager-related edges are drawn more prominently than non-mgr.
DEFAULT_EDGE_STYLES = {
    "Upward (-> mgr)":       dict(lw=0.7, alpha=0.35),
    "Downward (mgr ->)":     dict(lw=0.7, alpha=0.35),
    "Lateral (mgr <-> mgr)": dict(lw=1.5, alpha=0.70),
    "Non-mgr W->S (CCW)":    dict(lw=0.4, alpha=0.20),
    "Non-mgr S->NE (CW)":    dict(lw=0.4, alpha=0.20),
    "Non-mgr NE->NW (CCW)":  dict(lw=0.4, alpha=0.20),
    "Non-mgr NW->W (CCW)":   dict(lw=0.4, alpha=0.20),
    "Non-mgr unknown":       dict(lw=0.3, alpha=0.15),
}


def plot_residual_subgroups(
    df_edges_classified: pd.DataFrame,
    src_positions: pd.DataFrame,
    grid: Grid,
    separatrix: Separatrix,
    occupations_xy: tuple,
    out_path,
    *,
    theme="paper",
    subgroup_styles: dict = None,
    edge_styles: dict = None,
    figsize: tuple = (13, 13),
    dpi: int = 150,
    show_top_titles: int = 3,
    title: str = ("Residual system subgroups in task space\n"
                  "(size = outflow weight, source occupations)"),
):
    """Polar plot of residual-system subgroups with all edges and source nodes.

    Each subgroup gets a distinct colour and marker; edges are drawn faintly
    so that the source-position scatter dominates visually.

    Parameters
    ----------
    df_edges_classified : DataFrame
        Output of :func:`mobility.diagnostics.classify_subgroups`.
        Must include ``x_src, y_src, x_tgt, y_tgt, wP, subgroup``.
    src_positions : DataFrame
        Output of :func:`mobility.diagnostics.source_positions_by_subgroup`.
    grid, separatrix, occupations_xy
        Same as in other field plots.
    subgroup_styles : dict, optional
        Override default styling per subgroup.
    edge_styles : dict, optional
        Override edge styling per subgroup.
    show_top_titles : int
        Number of top source occupations to label per subgroup.
    """
    th = set_theme(theme)
    sg_styles = subgroup_styles or SUBGROUP_STYLES
    e_styles = edge_styles or DEFAULT_EDGE_STYLES

    xo, yo = occupations_xy

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th.background)
    _draw_disk_frame(ax, th, label_step_deg=30)
    ax.scatter(xo, yo, s=3, color="0.88" if th.name == "paper" else "0.35",
               alpha=0.5, linewidths=0, zorder=2)
    _draw_separatrix(ax, separatrix, grid, th, lw=2.5)

    # Edges per subgroup
    w_max = float(df_edges_classified["wP"].max())
    z_base = 3
    for sg in SUBGROUP_ORDER:
        if sg not in e_styles:
            continue
        sub = df_edges_classified[df_edges_classified["subgroup"] == sg]
        sub = sub.dropna(subset=["x_src", "y_src", "x_tgt", "y_tgt"])
        if len(sub) == 0:
            continue
        segs = [
            [(r.x_src, r.y_src), (r.x_tgt, r.y_tgt)]
            for r in sub.itertuples()
        ]
        lws = [
            e_styles[sg]["lw"] + 1.5 * r.wP / w_max
            for r in sub.itertuples()
        ]
        lc = LineCollection(
            segs,
            colors=sg_styles[sg]["color"],
            linewidths=lws,
            alpha=e_styles[sg]["alpha"],
            zorder=z_base,
        )
        ax.add_collection(lc)
        z_base += 1

    # Source-occupation scatter
    w_max_pos = float(src_positions["w_out"].max())
    for sg in SUBGROUP_ORDER:
        sub = src_positions[src_positions["subgroup"] == sg]
        if len(sub) == 0 or sg not in sg_styles:
            continue
        sizes = 20 + 100 * sub["w_out"] / w_max_pos
        ax.scatter(
            sub["x"], sub["y"],
            s=sizes,
            color=sg_styles[sg]["color"],
            marker=sg_styles[sg]["marker"],
            alpha=0.85,
            linewidths=0.4,
            edgecolors=th.background,
            zorder=10,
        )

    # Top-N occupation titles per subgroup
    if show_top_titles > 0 and "title" in src_positions.columns:
        for sg in SUBGROUP_ORDER:
            sub = src_positions[src_positions["subgroup"] == sg]
            if len(sub) == 0:
                continue
            top = sub.nlargest(show_top_titles, "w_out")
            for _, row in top.iterrows():
                if pd.isna(row.get("title")):
                    continue
                ax.text(
                    row["x"] + 0.02, row["y"] + 0.02,
                    str(row["title"]),
                    fontsize=7, color=th.foreground,
                    alpha=0.8, zorder=11,
                )

    # Legend
    legend_elements = []
    for sg in SUBGROUP_ORDER:
        if sg not in sg_styles:
            continue
        if len(src_positions[src_positions["subgroup"] == sg]) == 0:
            continue
        legend_elements.append(Line2D(
            [0], [0],
            marker=sg_styles[sg]["marker"],
            color="w",
            markerfacecolor=sg_styles[sg]["color"],
            markersize=10, label=sg,
        ))
    legend_elements.append(Line2D(
        [0], [0], color=th.accent_separatrix, lw=2,
        label="System border",
    ))
    ax.legend(
        handles=legend_elements, frameon=False,
        fontsize=11, loc="lower right",
        labelcolor=th.foreground,
    )

    ax.set_title(title, pad=10, fontsize=15, color=th.foreground)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)


def plot_residual_synthesis(
    flow_synthesis_df: pd.DataFrame,
    df_edges_classified: pd.DataFrame,
    grid: Grid,
    separatrix: Separatrix,
    occupations_xy: tuple,
    out_path,
    *,
    theme="paper",
    subgroup_styles: dict = None,
    figsize: tuple = (12, 12),
    dpi: int = 150,
    show_source_scatter: bool = True,
    title: str = ("Residual system - mean flow direction per subgroup\n"
                  "(circle = source centroid, star = destination centroid)"),
):
    """One representative arrow per subgroup, with a faint source scatter.

    This is the most readable summary of cross-system mobility for paper or
    presentation use.

    Parameters
    ----------
    flow_synthesis_df : DataFrame
        Output of :func:`mobility.diagnostics.flow_synthesis`.
    df_edges_classified : DataFrame
        Used only for the optional source-occupation scatter background.
    grid, separatrix, occupations_xy
        Same as in other field plots.
    subgroup_styles : dict, optional
        Override default styling.
    show_source_scatter : bool
        If True, draw a small faint scatter of all source occupations per
        subgroup (in the subgroup colour). If False, only arrows.
    """
    th = set_theme(theme)
    sg_styles = subgroup_styles or SUBGROUP_STYLES
    xo, yo = occupations_xy

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(th.background)
    _draw_disk_frame(ax, th, label_step_deg=30)
    ax.scatter(xo, yo, s=3, color="0.88" if th.name == "paper" else "0.35",
               alpha=0.5, linewidths=0, zorder=2)
    _draw_separatrix(ax, separatrix, grid, th, lw=2.5)

    for _, row in flow_synthesis_df.iterrows():
        sg = row["subgroup"]
        if sg == "Non-mgr unknown" or sg not in sg_styles:
            continue
        color = sg_styles[sg]["color"]

        if show_source_scatter:
            sub = df_edges_classified[df_edges_classified["subgroup"] == sg]
            ax.scatter(
                sub["x_src"], sub["y_src"],
                s=4, color=color, alpha=0.15, linewidths=0, zorder=3,
            )

        # Arrow from source centroid to destination centroid
        ax.annotate(
            "",
            xy=(row["x1"], row["y1"]),
            xytext=(row["x0"], row["y0"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color, lw=2.5, mutation_scale=20,
            ),
            zorder=7,
        )
        # Source label
        ax.text(
            row["x0"] - 0.03, row["y0"] - 0.03, sg,
            fontsize=8, color=color, ha="right", va="top",
            fontweight="bold", zorder=8,
        )
        # Source and destination markers
        ax.scatter(row["x0"], row["y0"], s=60, color=color, marker="o",
                   edgecolors=th.background, linewidths=0.8, zorder=8)
        ax.scatter(row["x1"], row["y1"], s=60, color=color, marker="*",
                   edgecolors=th.background, linewidths=0.8, zorder=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=sg_styles[sg]["color"], markersize=8,
               label=sg)
        for sg in SUBGROUP_ORDER
        if sg in sg_styles and sg != "Non-mgr unknown"
        and sg in set(flow_synthesis_df["subgroup"])
    ] + [
        Line2D([0], [0], color=th.accent_separatrix, lw=2,
               label="System border"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markersize=8, label="Source centroid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.5",
               markersize=10, label="Destination centroid"),
    ]
    ax.legend(handles=legend_elements, frameon=False,
              fontsize=9, loc="lower right",
              labelcolor=th.foreground)

    ax.set_title(title, pad=10, fontsize=14, color=th.foreground)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=th.background)
    plt.close(fig)
