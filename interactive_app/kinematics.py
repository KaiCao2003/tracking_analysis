"""Kinematics helpers using pluggable smoothing functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from tracking_analysis.angles import unwrap_deg
from tracking_analysis.filtering import filter_no_moving
from .smoothing import apply


def _window_speed(pos: np.ndarray, times: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
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
    pos: np.ndarray,
    times: np.ndarray,
    *,
    smoothing: bool = False,
    window: int = 5,
    polyorder: int = 2,
    method: str = "savgol",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute linear speed from positions with optional smoothing."""
    if smoothing:
        m = method
        if m == "window":
            return _window_speed(pos, times, window)
        pos = apply(m, pos, window=window, polyorder=polyorder)
    dt = np.diff(times)
    dpos = np.diff(pos, axis=0)
    vel = dpos / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    t_mid = times[:-1] + dt / 2
    return speed, t_mid


def compute_angular_speed(
    rot: np.ndarray,
    times: np.ndarray,
    *,
    smoothing: bool = False,
    window: int = 5,
    polyorder: int = 2,
    method: str = "savgol",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angular speed from orientation data."""
    if rot.shape[1] == 3:
        angles = unwrap_deg(rot, axis=0)
        quat = R.from_euler("xyz", angles, degrees=True).as_quat()
    else:
        quat = rot
    if smoothing:
        quat = apply(method, quat, window=window, polyorder=polyorder)
    q = quat / np.linalg.norm(quat, axis=1)[:, None]
    q_inv = np.column_stack([-q[:, :3], q[:, 3]])
    q_next = q[1:]
    q_cur_inv = q_inv[:-1]
    x1, y1, z1, w1 = q_cur_inv.T
    x2, y2, z2, w2 = q_next.T
    w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    angle = 2 * np.arccos(np.clip(w_rel, -1.0, 1.0))
    dt = np.diff(times)
    ang_speed = angle / dt
    t_mid = times[:-1] + dt / 2
    return ang_speed, t_mid


def compute_angular_velocity(
    rot: np.ndarray,
    times: np.ndarray,
    *,
    smoothing: bool = False,
    window: int = 5,
    polyorder: int = 2,
    method: str = "savgol",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-axis angular velocity."""
    if rot.shape[1] == 4:
        angles = R.from_quat(rot).as_euler("xyz", degrees=True)
    else:
        angles = rot
    angles = unwrap_deg(angles, axis=0)
    if smoothing:
        angles = apply(method, angles, window=window, polyorder=polyorder)
    dt = np.diff(times)
    dangles = np.diff(angles, axis=0)
    ang_vel = np.deg2rad(dangles) / dt[:, None]
    t_mid = times[:-1] + dt / 2
    return ang_vel, t_mid


def compute_head_direction(
    rot: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    speed: np.ndarray,
    *,
    nm_window: int = 10,
    nm_after: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute head direction (yaw) with forced no-movement skipping.

    Parameters
    ----------
    rot : ndarray
        Rotation as quaternions or Euler angles for each frame.
    frames : ndarray
        Frame numbers corresponding to ``rot``/``times``.
    times : ndarray
        Time stamp for each frame. Returned unchanged.
    speed : ndarray
        Linear speed array used to detect stationary periods. Should align
        with ``frames`` (``speed[0]`` corresponds to ``frames[0]``).
    nm_window : int, optional
        Minimum length of a zero-speed block.
    nm_after : int, optional
        Number of additional frames removed after a block.

    Returns
    -------
    head_dir : ndarray
        Unwrapped yaw angles in degrees with stationary segments replaced by
        ``1000``.
    times : ndarray
        Same ``times`` array for convenience.
    """

    if rot is None or len(rot) == 0:
        return np.array([]), np.array([])

    if rot.shape[1] == 4:
        angles = R.from_quat(rot).as_euler("xyz", degrees=True)
    else:
        angles = rot
    yaw = unwrap_deg(angles, axis=0)[:, 2]

    start_frame = int(frames[0])
    _, nm_ranges = filter_no_moving(
        speed,
        start_frame,
        window=nm_window,
        after=nm_after,
    )

    head_dir = yaw.copy()
    start_idx = start_frame - 1
    for s, e in nm_ranges:
        i = max(0, s - 1 - start_idx)
        j = max(0, e - start_idx)
        head_dir[i:j] = 1000.0

    return head_dir, times


__all__ = [
    "compute_linear_velocity",
    "compute_angular_speed",
    "compute_angular_velocity",
    "compute_head_direction",
]
