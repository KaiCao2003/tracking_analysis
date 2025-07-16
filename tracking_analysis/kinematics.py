import numpy as np
from scipy.signal import savgol_filter

def _smooth(data, window, polyorder):
    if not window or window < 3:
        return data
    # window must be odd
    wl = window if window % 2 == 1 else window + 1
    return savgol_filter(data, wl, polyorder, axis=0)

def compute_linear_velocity(pos, times, smoothing=False, window=5, polyorder=2):
    """
    Compute linear speed from positions.
    Returns (speed_array, time_midpoints).
    """
    if smoothing:
        pos = _smooth(pos, window, polyorder)

    dt   = np.diff(times)
    dpos = np.diff(pos, axis=0)
    vel  = dpos / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    t_mid = times[:-1] + dt/2
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

    # quaternion multiply: q_rel = q_cur_inv * q_next
    x1,y1,z1,w1 = q_cur_inv.T
    x2,y2,z2,w2 = q_next.T

    wr = w1*w2 - x1*x2 - y1*y2 - z1*z2
    xr = w1*x2 + x1*w2 + y1*z2 - z1*y2
    yr = w1*y2 - x1*z2 + y1*w2 + z1*x2
    zr = w1*z2 + x1*y2 - y1*x2 + z1*w2

    angle = 2 * np.arccos(np.clip(wr, -1.0, 1.0))
    dt    = np.diff(times)
    ang_speed = angle / dt
    t_mid = times[:-1] + dt/2
    return ang_speed, t_mid