# bh_tracks.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# TV denoise (1D) - Chambolle-style
# -----------------------------
def tv_denoise_1d(y: np.ndarray, weight: float = 0.01, n_iter: int = 200) -> np.ndarray:
    """
    Simple 1D total-variation denoising (ROF) using a Chambolle-like projection method.
    Good for smoothing noisy curves while preserving jumps.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3 or weight <= 0:
        return y.copy()

    p = np.zeros(y.size - 1, dtype=float)  # dual on edges
    tau = 0.125  # stable step for 1D

    for _ in range(int(n_iter)):
        div_p = np.zeros_like(y)
        div_p[0] = -p[0]
        div_p[1:-1] = p[:-1] - p[1:]
        div_p[-1] = p[-1]

        u = y - weight * div_p
        grad_u = np.diff(u)

        p_new = p + (tau / weight) * grad_u
        p_new = p_new / np.maximum(1.0, np.abs(p_new))
        p = p_new

    div_p = np.zeros_like(y)
    div_p[0] = -p[0]
    div_p[1:-1] = p[:-1] - p[1:]
    div_p[-1] = p[-1]
    u = y - weight * div_p
    return u


# -----------------------------
# Track indexing helper
# -----------------------------
def _get_track(
    bh: int,
    bh_ids: np.ndarray,
    first_index: np.ndarray,
    num_entries: np.ndarray,
) -> Tuple[int, int]:
    """Return (start, count) slice info for BH id; (-1,0) if not found."""
    bh = int(bh)
    bh_ids = np.asarray(bh_ids, dtype=np.int64)
    hit = np.where(bh_ids == bh)[0]
    if hit.size == 0:
        return -1, 0
    j = int(hit[0])
    return int(first_index[j]), int(num_entries[j])


# -----------------------------
# Optional: merger vertical lines (NO X markers)
# -----------------------------
def _draw_merger_vlines(
    ax,
    events: Optional[List[dict]] = None,
    minor_ls: str = "-",
    major_ls: str = "--",
    alpha: float = 0.35,
    lw: float = 1.0,
):
    """
    Draw vertical lines at merger times if events are provided.
    events format: [{"t": float, ...}, ...]
    No 'x' markers anywhere.
    """
    if not events:
        return
    for e in events:
        if "t" not in e:
            continue
        t = float(e["t"])
        # keep it simple: all same style unless you want to differentiate later
        ax.axvline(t, linestyle=major_ls, alpha=alpha, linewidth=lw)


# -----------------------------
# Core plot (NO dominant mainline)
# -----------------------------
def plot_field_tracks(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_field,
    field_label: str,
    xlog: bool = False,
    ylog: bool = False,
    ylim=None,
    figsize=(9, 6),

    # smoothing
    denoise: bool = False,
    tv_weight: float = 0.01,
    tv_iter: int = 200,

    # merger vlines (optional)
    events: Optional[List[dict]] = None,
    minor_ls: str = "-",
    major_ls: str = "--",
    alpha: float = 0.35,
    lw: float = 1.0,

    # compatibility args (ignored safely)
    dominant_mode: str = "none",
    target_id=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    merge_counts=None,
    min_merges_main: int = 3,
    minor_cut: int = 2,
):
    """
    Plot each BH track in ids_to_plot for a given field.
    - Does NOT compute a dominant BH.
    - Does NOT draw X markers.
    - Accepts extra kwargs (target_id, dominant_mode, etc.) to stay compatible
      with your notebook calls, but ignores them.
    """
    ids_to_plot = np.asarray(ids_to_plot, dtype=np.int64)
    all_time = np.asarray(all_time, dtype=float)
    all_field = np.asarray(all_field, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # optional merger time vertical lines
    _draw_merger_vlines(
        ax,
        events=events,
        minor_ls=minor_ls,
        major_ls=major_ls,
        alpha=alpha,
        lw=lw,
    )

    for bh in ids_to_plot.tolist():
        start, count = _get_track(bh, bh_ids, first_index, num_entries)
        if start < 0 or count <= 1:
            continue

        sl = slice(start, start + count)
        t = all_time[sl]
        y = all_field[sl]

        if denoise:
            y = tv_denoise_1d(y, weight=float(tv_weight), n_iter=int(tv_iter))

        ax.plot(t, y, linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel(field_label)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


# -----------------------------
# Wrappers: mass, mdot, cs, rho
# -----------------------------
def plot_mass(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_mass,
    xlog=False,
    ylog=False,
    ylim=None,
    figsize=(9, 6),

    denoise=False,
    tv_weight=0.01,
    tv_iter=200,

    # optional merger vlines
    events=None,
    minor_ls="-",
    major_ls="--",
    alpha=0.35,
    lw=1.0,

    # compatibility args (so your old notebook calls won't crash)
    dominant_mode="none",
    target_id=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    merge_counts=None,
    min_merges_main=3,
    minor_cut=2,
):
    return plot_field_tracks(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_mass,
        field_label="BH Mass",
        xlog=xlog,
        ylog=ylog,
        ylim=ylim,
        figsize=figsize,
        denoise=denoise,
        tv_weight=tv_weight,
        tv_iter=tv_iter,
        events=events,
        minor_ls=minor_ls,
        major_ls=major_ls,
        alpha=alpha,
        lw=lw,
        dominant_mode=dominant_mode,
        target_id=target_id,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        merge_counts=merge_counts,
        min_merges_main=min_merges_main,
        minor_cut=minor_cut,
    )


def plot_mdot(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_mdot,
    xlog=False,
    ylog=False,
    ylim=None,
    figsize=(9, 6),

    denoise=False,
    tv_weight=0.01,
    tv_iter=200,

    events=None,
    minor_ls="-",
    major_ls="--",
    alpha=0.35,
    lw=1.0,

    dominant_mode="none",
    target_id=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    merge_counts=None,
    min_merges_main=3,
    minor_cut=2,
):
    return plot_field_tracks(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_mdot,
        field_label="BH Mdot",
        xlog=xlog,
        ylog=ylog,
        ylim=ylim,
        figsize=figsize,
        denoise=denoise,
        tv_weight=tv_weight,
        tv_iter=tv_iter,
        events=events,
        minor_ls=minor_ls,
        major_ls=major_ls,
        alpha=alpha,
        lw=lw,
        dominant_mode=dominant_mode,
        target_id=target_id,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        merge_counts=merge_counts,
        min_merges_main=min_merges_main,
        minor_cut=minor_cut,
    )


def plot_cs(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_cs,
    xlog=False,
    ylog=False,
    ylim=None,
    figsize=(9, 6),

    denoise=False,
    tv_weight=0.01,
    tv_iter=200,

    events=None,
    minor_ls="-",
    major_ls="--",
    alpha=0.35,
    lw=1.0,

    dominant_mode="none",
    target_id=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    merge_counts=None,
    min_merges_main=3,
    minor_cut=2,
):
    return plot_field_tracks(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_cs,
        field_label="Sound Speed (cs)",
        xlog=xlog,
        ylog=ylog,
        ylim=ylim,
        figsize=figsize,
        denoise=denoise,
        tv_weight=tv_weight,
        tv_iter=tv_iter,
        events=events,
        minor_ls=minor_ls,
        major_ls=major_ls,
        alpha=alpha,
        lw=lw,
        dominant_mode=dominant_mode,
        target_id=target_id,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        merge_counts=merge_counts,
        min_merges_main=min_merges_main,
        minor_cut=minor_cut,
    )


def plot_rho(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_rho,
    xlog=False,
    ylog=False,
    ylim=None,
    figsize=(9, 6),

    denoise=False,
    tv_weight=0.01,
    tv_iter=200,

    events=None,
    minor_ls="-",
    major_ls="--",
    alpha=0.35,
    lw=1.0,

    dominant_mode="none",
    target_id=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    merge_counts=None,
    min_merges_main=3,
    minor_cut=2,
):
    return plot_field_tracks(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_rho,
        field_label="Density (rho)",
        xlog=xlog,
        ylog=ylog,
        ylim=ylim,
        figsize=figsize,
        denoise=denoise,
        tv_weight=tv_weight,
        tv_iter=tv_iter,
        events=events,
        minor_ls=minor_ls,
        major_ls=major_ls,
        alpha=alpha,
        lw=lw,
        dominant_mode=dominant_mode,
        target_id=target_id,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        merge_counts=merge_counts,
        min_merges_main=min_merges_main,
        minor_cut=minor_cut,
    )
