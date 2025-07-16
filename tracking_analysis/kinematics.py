import numpy as np
from scipy.signal import savgol_filter

def _smooth(data, window, polyorder):
    if not window or window < 3:
        return data
    # window must be odd
    wl = window if window % 2 == 1 else window + 1
    return savgol_filter(data, wl, polyorder, axis=0)

def _window_speed(pos, times, window):
    """Compute speed using the custom window-based averaging method."""
    if window % 2 != 0:
        raise ValueError("window must be even for windowed smoothing")
    half = window // 2
    n = len(times)
    speeds = []
    t_mid = []
    for start in range(0, n - window + 1):
        mid = start + half
        end = start + window
        dt1 = times[start + half - 1] - times[start]
        dt2 = times[end - 1] - times[mid]
        if dt1 == 0 or dt2 == 0:
            speeds.append(np.nan)
        else:
            d1 = np.linalg.norm(pos[start + half - 1] - pos[start]) / dt1
            d2 = np.linalg.norm(pos[end - 1] - pos[mid]) / dt2
            speeds.append((d1 + d2) / 2)
        t_mid.append((times[start + half - 1] + times[mid]) / 2)
    return np.array(speeds), np.array(t_mid)


def compute_linear_velocity(
    pos,
    times,
    smoothing=False,
    window=5,
    polyorder=2,
    method="savgol",
):
    """Compute linear speed from positions.

    Parameters
    ----------
    pos : ndarray
        Array of positions ``(N,3)``.
    times : ndarray
        Time values corresponding to each position.
    smoothing : bool, optional
        Whether to smooth the position data before differentiating.
    window : int, optional
        Smoothing window size.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filtering.
    method : {'savgol', 'window'}, optional
        Smoothing method. ``'savgol'`` applies a Savitzky-Golay filter to the
        positions. ``'window'`` uses the custom averaging method described in
        the docs.
    """

    if smoothing and method == "savgol":
        pos = _smooth(pos, window, polyorder)
    elif smoothing and method == "window":
        return _window_speed(pos, times, window)

    dt = np.diff(times)
    dpos = np.diff(pos, axis=0)
    vel = dpos / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    t_mid = times[:-1] + dt / 2
    return speed, t_mid

def compute_angular_speed(quat, times, smoothing=False, window=5, polyorder=2):
    """
    Compute angular speed (rad/s) from quaternion series.
    Returns (ang_speed_array, time_midpoints).
    """
    if smoothing:
        quat = _smooth(quat, window, polyorder)

    # normalize
    q = quat / np.linalg.norm(quat, axis=1)[:, None]
    # inverse of q[i]
    q_inv = np.column_stack([-q[:, :3], q[:, 3]])

    # compute relative quaternion between successive frames
    q_next = q[1:]
    q_cur_inv = q_inv[:-1]

    # Only the w component of the relative quaternion is needed to compute
    # the angular displacement. Compute it directly to avoid unnecessary
    # intermediate arrays.
    x1, y1, z1, w1 = q_cur_inv.T
    x2, y2, z2, w2 = q_next.T
    w_rel = w1*w2 - x1*x2 - y1*y2 - z1*z2

    angle = 2 * np.arccos(np.clip(w_rel, -1.0, 1.0))
    dt    = np.diff(times)
    ang_speed = angle / dt
    t_mid = times[:-1] + dt/2
    return ang_speed, t_mid