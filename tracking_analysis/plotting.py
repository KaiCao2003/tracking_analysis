import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _draw_dashed_connection(ax, start_pt, end_pt, start_t, end_t, norm, is3d=False, steps=20):
    """Draw a dashed connection with rainbow gradient."""
    pts = np.linspace(start_pt, end_pt, steps)
    times = np.linspace(start_t, end_t, steps)
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_times = times[:-1]

    if is3d:
        lc = Line3DCollection(segments, cmap="rainbow", norm=norm, linestyles="--")
        lc.set_array(seg_times)
        ax.add_collection3d(lc)
    else:
        lc = LineCollection(segments, cmap="rainbow", norm=norm, linestyles="--")
        lc.set_array(seg_times)
        ax.add_collection(lc)


def _annotate_ranges(ax, ranges, times):
    if not ranges:
        return
    txt = "; ".join(
        f"{times[a]:.2f}-{times[b-1]:.2f}" if b - a > 1 else f"{times[a]:.2f}"
        for a, b in ranges
    )
    ax.annotate(
        f"Filtered times: {txt}",
        xy=(0.5, -0.15),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
    )


def plot_trajectory_2d(pos, times, time_markers, out_path, anomalies=None,
                       full_size=False):
    fig, ax = plt.subplots(figsize=(16, 10) if full_size else None)
    points = pos[:, :2]
    mask = ~np.isnan(points).any(axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_mask = mask[:-1] & mask[1:]
    segments = segments[seg_mask]
    seg_times = times[:-1][seg_mask]
    norm = Normalize(times.min(), times.max())
    lc = LineCollection(segments, cmap="rainbow", norm=norm)
    lc.set_array(seg_times)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", "box")
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time (s)")

    if anomalies:
        for start, end in anomalies:
            if start > 0 and end < len(points):
                _draw_dashed_connection(
                    ax,
                    points[start - 1],
                    points[end],
                    times[start - 1],
                    times[end],
                    norm,
                    is3d=False,
                )
    # Add markers on the trajectory
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(
                pos[tm, 0],
                pos[tm, 1],
                marker="v",
                s=60,
                edgecolor="black",
                color="red",
                zorder=10,
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectory')
    _annotate_ranges(ax, anomalies, times)
    fig.savefig(out_path)
    plt.close(fig)

def plot_trajectory_3d(pos, times, time_markers, out_path, anomalies=None,
                       full_size=False):
    fig = plt.figure(figsize=(16, 10) if full_size else None)
    ax = fig.add_subplot(111, projection="3d")
    mask = ~np.isnan(pos).any(axis=1)
    segments = np.stack([pos[:-1], pos[1:]], axis=1)
    seg_mask = mask[:-1] & mask[1:]
    segments = segments[seg_mask]
    seg_times = times[:-1][seg_mask]
    norm = Normalize(times.min(), times.max())
    lc = Line3DCollection(segments, cmap="rainbow", norm=norm)
    lc.set_array(seg_times)
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(pos[:, 0], pos[:, 1], pos[:, 2])
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 1))
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time (s)")

    if anomalies:
        for start, end in anomalies:
            if start > 0 and end < len(pos):
                _draw_dashed_connection(
                    ax,
                    pos[start - 1],
                    pos[end],
                    times[start - 1],
                    times[end],
                    norm,
                    is3d=True,
                )

    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(
                pos[tm, 0],
                pos[tm, 1],
                pos[tm, 2],
                marker="v",
                s=60,
                edgecolor="black",
                color="red",
                zorder=10,
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    _annotate_ranges(ax, anomalies, times)
    fig.savefig(out_path)
    plt.close(fig)

def plot_time_series(values, times, ylabel, time_markers, out_path, anomalies=None,
                     full_size=False, x_limit=None, y_limit=None):
    """Plot a time series with optional anomaly gaps."""
    fig, ax = plt.subplots(figsize=(16, 10) if full_size else None)
    line, = ax.plot(times, values)

    # dashed connections across filtered ranges
    if anomalies:
        for start, end in anomalies:
            if start > 0 and end < len(values):
                ax.plot(
                    [times[start - 1], times[end]],
                    [values[start - 1], values[end]],
                    linestyle="--",
                    color=line.get_color(),
                    linewidth=1,
                    zorder=line.get_zorder() - 1,
                )

    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(times[tm], values[tm], marker='v', color='red', zorder=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Time")
    if x_limit is not None:
        ax.set_xlim(0, x_limit)
    if y_limit is not None:
        ax.set_ylim(0, y_limit)

    if anomalies:
        _annotate_ranges(ax, anomalies, times)

    fig.savefig(out_path)
    plt.close(fig)
