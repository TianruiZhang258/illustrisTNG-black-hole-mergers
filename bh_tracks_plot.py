"""
bh_tracks_plot.py

Utilities to plot black-hole (BH) tracks (mass, mdot, cs, rho) with optional TV denoising
and optional vertical lines marking merger times.

This module expects you to provide:
- groupA, groupB (e.g., from your own two_groups(...) function)
- id_in, id_out, merger_time arrays (same length; merger_time corresponds to each merger event)

Typical usage
-------------
groupA, groupB = two_groups(target_id=..., id_in=id_in, id_out=id_out)

fig, ax = plot_mass(
    groupA=groupA, groupB=groupB,
    bh_ids=bh_ids, first_index=first_index, num_entries=num_entries,
    all_time=all_time, all_mass=all_mass,
    id_in=id_in, id_out=id_out, merger_time=merger_time,
    show_mergers=True,
    denoise=True, tv_weight=0.01, tv_n_iter=200,
)
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# TV denoise (你的版本)
# ----------------------------
def tv_denoise_1d(y, weight=0.01, n_iter=200):
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 3 or weight <= 0:
        return y.copy()

    p = np.zeros(y.size - 1, dtype=float)
    tau = 0.125

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


# ----------------------------
# helpers
# ----------------------------
def _auto_ids_to_plot(ids_to_plot, groupA, groupB):
    if ids_to_plot is not None:
        return np.asarray(ids_to_plot, dtype=np.int64).ravel()

    parts = []
    if groupA is not None:
        parts.append(np.asarray(groupA, dtype=np.int64).ravel())
    if groupB is not None:
        parts.append(np.asarray(groupB, dtype=np.int64).ravel())

    if not parts:
        return np.asarray([], dtype=np.int64)

    return np.unique(np.concatenate(parts)).astype(np.int64)


def _final_mass_of(bh, bh_ids, first_index, num_entries, all_mass):
    bh = int(bh)
    bh_ids = np.asarray(bh_ids, dtype=np.int64).ravel()
    first_index = np.asarray(first_index, dtype=np.int64).ravel()
    num_entries = np.asarray(num_entries, dtype=np.int64).ravel()
    all_mass = np.asarray(all_mass, dtype=float).ravel()

    hit = np.where(bh_ids == bh)[0]
    if hit.size == 0:
        return np.nan
    i = int(hit[0])
    s = int(first_index[i])
    n = int(num_entries[i])
    if n <= 0:
        return np.nan
    return float(all_mass[s + n - 1])


def _pick_primary_from_groupA(groupA, bh_ids, first_index, num_entries, all_mass):
    groupA = np.asarray(groupA, dtype=np.int64).ravel()
    if groupA.size == 0:
        return None
    masses = np.array(
        [_final_mass_of(b, bh_ids, first_index, num_entries, all_mass) for b in groupA],
        dtype=float,
    )
    if not np.isfinite(masses).any():
        return None
    return int(groupA[np.nanargmax(masses)])


def _track_time_range(bh, bh_ids, first_index, num_entries, all_time):
    """Return (t_min, t_max) for bh track, or (None, None) if missing."""
    bh = int(bh)
    bh_ids = np.asarray(bh_ids, dtype=np.int64).ravel()
    first_index = np.asarray(first_index, dtype=np.int64).ravel()
    num_entries = np.asarray(num_entries, dtype=np.int64).ravel()
    all_time = np.asarray(all_time, dtype=float).ravel()

    hit = np.where(bh_ids == bh)[0]
    if hit.size == 0:
        return None, None
    i = int(hit[0])
    s = int(first_index[i])
    n = int(num_entries[i])
    if n <= 0:
        return None, None
    t = all_time[s : s + n]
    t = t[np.isfinite(t)]
    if t.size == 0:
        return None, None
    return float(np.min(t)), float(np.max(t))


def _estimate_merge_time(bh1, bh2, bh_ids, first_index, num_entries, all_time):
    """
    Fallback estimate (not used if you pass merger_time):
    max(tmin1, tmin2)  (earliest time both exist).
    """
    t1_min, _ = _track_time_range(bh1, bh_ids, first_index, num_entries, all_time)
    t2_min, _ = _track_time_range(bh2, bh_ids, first_index, num_entries, all_time)
    if t1_min is None or t2_min is None:
        return None
    return float(max(t1_min, t2_min))


def _plot_field(
    *,
    ids_to_plot,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_field,
    ylabel="",
    ax=None,
    ylog=False,
    ylim=None,
    xlim=None,
    title=None,
    label_ids=True,
    legend=True,
    legend_kwargs=None,
    primary_id=None,
    denoise=False,
    tv_weight=0.01,
    tv_n_iter=200,
    # --- merger vertical lines ---
    show_mergers=False,
    groupA=None,
    groupB=None,
    id_in=None,
    id_out=None,
    merger_time=None,  # 真实合并时间数组（与你的 merger_file 里的 f["time"] 对应）
    merger_alpha=0.35,
    **kwargs,
):
    # ---------- 标准化输入 ----------
    ids_to_plot = np.asarray(ids_to_plot, dtype=np.int64).ravel()
    bh_ids = np.asarray(bh_ids, dtype=np.int64).ravel()
    first_index = np.asarray(first_index, dtype=np.int64).ravel()
    num_entries = np.asarray(num_entries, dtype=np.int64).ravel()
    all_time = np.asarray(all_time, dtype=float).ravel()
    all_field = np.asarray(all_field, dtype=float).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # BH_ID -> position map
    pos = {int(bh): i for i, bh in enumerate(bh_ids.tolist())}

    # ---------- plot tracks ----------
    plotted = 0
    for bh in ids_to_plot:
        bh = int(bh)
        if bh not in pos:
            continue

        i = pos[bh]
        s = int(first_index[i])
        n = int(num_entries[i])
        if n <= 1:
            continue

        t = all_time[s : s + n]
        y = all_field[s : s + n]

        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        if t.size <= 1:
            continue

        if denoise:
            y = tv_denoise_1d(y, weight=float(tv_weight), n_iter=int(tv_n_iter))

        is_primary = (primary_id is not None) and (bh == int(primary_id))
        lw = 3.0 if is_primary else 1.5

        if label_ids:
            lbl = f"PRIMARY {bh}" if is_primary else str(bh)
        else:
            lbl = None

        ax.plot(t, y, label=lbl, linewidth=lw, **kwargs)
        plotted += 1

    # ---------- merger vertical lines ----------
    if (
        show_mergers
        and (groupA is not None)
        and (groupB is not None)
        and (id_in is not None)
        and (id_out is not None)
        and (merger_time is not None)
    ):
        A = set(map(int, np.asarray(groupA, dtype=np.int64).ravel().tolist()))
        B = set(map(int, np.asarray(groupB, dtype=np.int64).ravel().tolist()))
        id_in = np.asarray(id_in, dtype=np.int64).ravel()
        id_out = np.asarray(id_out, dtype=np.int64).ravel()
        merger_time = np.asarray(merger_time, dtype=float).ravel()

        ids_set = set(map(int, ids_to_plot.tolist()))
        seen = set()

        for a, b, t in zip(id_in, id_out, merger_time):
            a = int(a)
            b = int(b)
            if not np.isfinite(t):
                continue

            # at least one endpoint currently shown
            if (a not in ids_set) and (b not in ids_set):
                continue

            # A-A: solid; A-B: dashed
            if (a in A) and (b in A):
                style = "-"
            elif ((a in A) and (b in B)) or ((a in B) and (b in A)):
                style = "--"
            else:
                continue

            key = (min(a, b), max(a, b), float(t), style)
            if key in seen:
                continue
            seen.add(key)

            ax.axvline(float(t), linestyle=style, linewidth=1.0, alpha=float(merger_alpha))

    # ---------- axes ----------
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(True, alpha=0.3)

    if legend and label_ids and plotted > 0:
        lk = dict(loc="best", fontsize=8, frameon=True)
        if legend_kwargs:
            lk.update(legend_kwargs)
        ax.legend(**lk)

    fig.tight_layout()
    return fig, ax


# ----------------------------
# wrappers
# ----------------------------
def plot_mass(
    *,
    ids_to_plot=None,
    groupA=None,
    groupB=None,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_mass,
    id_in=None,
    id_out=None,
    merger_time=None,
    show_mergers=False,
    **kwargs,
):
    ids_to_plot = _auto_ids_to_plot(ids_to_plot, groupA, groupB)

    primary_id = kwargs.pop("primary_id", None)
    if primary_id is None and groupA is not None:
        primary_id = _pick_primary_from_groupA(groupA, bh_ids, first_index, num_entries, all_mass)

    return _plot_field(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_mass,
        ylabel="BH Mass",
        primary_id=primary_id,
        label_ids=True,
        legend=True,
        show_mergers=show_mergers,
        groupA=groupA,
        groupB=groupB,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        **kwargs,
    )


def plot_mdot(
    *,
    ids_to_plot=None,
    groupA=None,
    groupB=None,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_mdot,
    all_mass=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    show_mergers=False,
    **kwargs,
):
    ids_to_plot = _auto_ids_to_plot(ids_to_plot, groupA, groupB)

    primary_id = kwargs.pop("primary_id", None)
    if primary_id is None and groupA is not None and all_mass is not None:
        primary_id = _pick_primary_from_groupA(groupA, bh_ids, first_index, num_entries, all_mass)

    return _plot_field(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_mdot,
        ylabel="BH Mdot",
        primary_id=primary_id,
        label_ids=True,
        legend=True,
        show_mergers=show_mergers,
        groupA=groupA,
        groupB=groupB,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        **kwargs,
    )


def plot_cs(
    *,
    ids_to_plot=None,
    groupA=None,
    groupB=None,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_cs,
    all_mass=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    show_mergers=False,
    **kwargs,
):
    ids_to_plot = _auto_ids_to_plot(ids_to_plot, groupA, groupB)

    primary_id = kwargs.pop("primary_id", None)
    if primary_id is None and groupA is not None and all_mass is not None:
        primary_id = _pick_primary_from_groupA(groupA, bh_ids, first_index, num_entries, all_mass)

    return _plot_field(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_cs,
        ylabel="Sound Speed (cs)",
        primary_id=primary_id,
        label_ids=True,
        legend=True,
        show_mergers=show_mergers,
        groupA=groupA,
        groupB=groupB,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        **kwargs,
    )


def plot_rho(
    *,
    ids_to_plot=None,
    groupA=None,
    groupB=None,
    bh_ids,
    first_index,
    num_entries,
    all_time,
    all_rho,
    all_mass=None,
    id_in=None,
    id_out=None,
    merger_time=None,
    show_mergers=False,
    **kwargs,
):
    ids_to_plot = _auto_ids_to_plot(ids_to_plot, groupA, groupB)

    primary_id = kwargs.pop("primary_id", None)
    if primary_id is None and groupA is not None and all_mass is not None:
        primary_id = _pick_primary_from_groupA(groupA, bh_ids, first_index, num_entries, all_mass)

    return _plot_field(
        ids_to_plot=ids_to_plot,
        bh_ids=bh_ids,
        first_index=first_index,
        num_entries=num_entries,
        all_time=all_time,
        all_field=all_rho,
        ylabel="Density (rho)",
        primary_id=primary_id,
        label_ids=True,
        legend=True,
        show_mergers=show_mergers,
        groupA=groupA,
        groupB=groupB,
        id_in=id_in,
        id_out=id_out,
        merger_time=merger_time,
        **kwargs,
    )


__all__ = [
    "tv_denoise_1d",
    "plot_mass",
    "plot_mdot",
    "plot_cs",
    "plot_rho",
]
