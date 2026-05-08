"""Plotting themes for paper, presentation, and poster contexts.

Two main themes are provided:

* ``paper`` — white background, standard sizing, designed for inclusion in
  the LaTeX paper.
* ``presentation`` — dark background, larger text, designed for slides
  and posters.

Apply via :func:`set_theme`. Each theme returns a dict of background and
foreground colours that plot functions use for elements that matplotlib
rcParams do not control directly (e.g. axis spine colour overrides, scatter
edgecolors, separatrix colour).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class Theme:
    """Visual theme for a figure."""

    name: str
    background: str
    foreground: str          # primary text/axis colour
    muted: str               # secondary lines, light scatter
    accent_separatrix: str
    cmap_field: str          # colormap for field magnitudes
    cmap_diverging: str      # colormap for residual fields


PAPER = Theme(
    name="paper",
    background="white",
    foreground="black",
    muted="0.55",
    accent_separatrix="limegreen",
    cmap_field="YlOrRd",
    cmap_diverging="RdBu_r",
)

PRESENTATION = Theme(
    name="presentation",
    background="0.10",
    foreground="white",
    muted="0.65",
    accent_separatrix="limegreen",
    cmap_field="YlOrRd",
    cmap_diverging="RdBu_r",
)

THEMES = {"paper": PAPER, "presentation": PRESENTATION}


def get_theme(name: str | Theme) -> Theme:
    """Resolve a theme name or Theme object to a Theme."""
    if isinstance(name, Theme):
        return name
    if name not in THEMES:
        raise KeyError(f"unknown theme '{name}', use one of {list(THEMES)}")
    return THEMES[name]


def set_theme(name: str | Theme = "paper") -> Theme:
    """Apply a theme globally via matplotlib rcParams.

    Returns the resolved Theme so callers can read its colour fields when
    drawing elements not controlled by rcParams.
    """
    theme = get_theme(name)

    is_dark = theme.name == "presentation"
    rc = {
        "font.size":        20 if not is_dark else 18,
        "axes.titlesize":   22 if not is_dark else 20,
        "axes.labelsize":   20 if not is_dark else 18,
        "xtick.labelsize":  18 if not is_dark else 16,
        "ytick.labelsize":  18 if not is_dark else 16,
        "figure.facecolor": theme.background,
        "axes.facecolor":   theme.background,
        "axes.edgecolor":   theme.foreground,
        "axes.labelcolor":  theme.foreground,
        "xtick.color":      theme.foreground,
        "ytick.color":      theme.foreground,
        "text.color":       theme.foreground,
    }
    mpl.rcParams.update(rc)
    return theme