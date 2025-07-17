import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
    fig, ax = plt.subplots(figsize=(12, 8) if full_size else None)
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
                ax.plot(
                    [points[start - 1, 0], points[end, 0]],
                    [points[start - 1, 1], points[end, 1]],
                    linestyle="--",
                    color="tab:orange",
                )
    # Add markers on the trajectory
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(pos[tm,0], pos[tm,1], marker='v', s=60, edgecolor='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectory')
    _annotate_ranges(ax, anomalies, times)
    fig.savefig(out_path)
    plt.close(fig)

def plot_trajectory_3d(pos, times, time_markers, out_path, anomalies=None,
                       full_size=False):
    fig = plt.figure(figsize=(12, 8) if full_size else None)
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
                ax.plot(
                    [pos[start - 1, 0], pos[end, 0]],
                    [pos[start - 1, 1], pos[end, 1]],
                    [pos[start - 1, 2], pos[end, 2]],
                    linestyle="--",
                    color="tab:orange",
                )

    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(
                pos[tm, 0],
                pos[tm, 1],
                pos[tm, 2],
                marker='v',
                s=60,
                edgecolor='black'
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
    fig, ax = plt.subplots(figsize=(12, 8) if full_size else None)
    ax.plot(times, values)

    # dashed connections across filtered ranges
    if anomalies:
        for start, end in anomalies:
            if start > 0 and end < len(values):
                ax.plot(
                    [times[start - 1], times[end]],
                    [values[start - 1], values[end]],
                    linestyle="--",
                    color="tab:orange",
                    linewidth=1
                )

    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.axvline(x=times[tm], linestyle="--", linewidth=1)

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
