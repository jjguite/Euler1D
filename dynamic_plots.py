# for real-time initialization and updating of matplotlib plots
#   --> used in main_with_plots from main.py

import matplotlib.pyplot as plt

# ─── PLOTTING SETUP ─────────────────────────────────────────────────────────
def plot_init(x, u, p, on_key, *, buffer_frac=0.10, eps=1e-8):
    """
    buffer_frac : extra space (fraction of data range) above/below the curve
    eps         : fallback buffer when data range is ~0 (avoids singular ylim)
    """

    ib, ie = p.ib, p.ie
    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)

    # connect the key‐press callback
    fig.canvas.mpl_connect('key_press_event', on_key)

    line, = ax.plot(
        x[ib:ie + 1],
        u[ib:ie + 1],
        'ro-',
        markersize=3,
        mfc='none',
        linewidth=0.5,
    )
    ax.grid()

    # initial limits with a small buffer
    ymin, ymax = u[ib:ie + 1].min(), u[ib:ie + 1].max()
    yrange     = ymax - ymin
    buf        = buffer_frac * yrange if yrange > 0 else eps
    lo, hi     = ymin - buf, ymax + buf
    ax.set_ylim(lo, hi)

    # remember current limits for smooth expansion later
    ax._cur_lo, ax._cur_hi = lo, hi

    return fig, ax, line, buffer_frac, eps


# ─── redraw & label ─────────────────────────────────────────────────────────
def plot_update(u, ax, line, buffer_frac, eps, p):
    """
    Update line data and smoothly expand or shrink y-limits:
      - If new range exceeds current, we expand (minimize jitter on oscillations).
      - If new range is fully inside current, we shrink to fit.
    """

    ib, ie = p.ib, p.ie
    line.set_ydata(u[ib:ie + 1])

    # Compute new data extrema and buffer
    ymin, ymax = u[ib:ie + 1].min(), u[ib:ie + 1].max()
    yrange     = ymax - ymin
    buf        = buffer_frac * yrange if yrange > 0 else eps
    desired_lo = ymin - buf
    desired_hi = ymax + buf

    # Decide whether to expand or shrink
    if desired_lo < ax._cur_lo or desired_hi > ax._cur_hi:
        # grow only in the needed direction
        new_lo = min(desired_lo, ax._cur_lo)
        new_hi = max(desired_hi, ax._cur_hi)
    else:
        # both desired_lo/high are inside current --> fit exactly
        new_lo, new_hi = desired_lo, desired_hi

    # Store and apply
    ax._cur_lo, ax._cur_hi = new_lo, new_hi
    ax.set_ylim(new_lo, new_hi)
