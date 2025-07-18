"""Utility helpers for the Dash web application."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data
from tracking_analysis.grouping import group_entities
from tracking_analysis.filtering import (
    filter_anomalies,
    filter_position,
    apply_ranges,
)
from tracking_analysis.kinematics import (
    compute_linear_velocity,
    compute_angular_speed,
    compute_angular_velocity,
)


def prepare_data(cfg: Config) -> Tuple[Dict[str, dict], List[str]]:
    """Load data and apply the same processing pipeline as the CLI."""
    input_file = cfg.get("input_file")
    if not os.path.exists(input_file):
        fallback = os.path.join("data", "input.csv")
        input_file = fallback if os.path.exists(fallback) else input_file

    df, _, time_col = load_data(input_file)
    groups_all = group_entities(df)

    selected = cfg.get("groups") or list(groups_all.keys())
    selected = [g for g in selected if g in groups_all]

    start_time = cfg.get("interval", "start_time", default=0.0)
    end_time = cfg.get("interval", "end_time")
    times_full = df[time_col].values
    start = int(np.searchsorted(times_full, start_time, side="left"))
    if end_time == float("inf"):
        end = float("inf")
    else:
        end = int(np.searchsorted(times_full, end_time, side="right")) - 1

    kin_cfg = cfg.get("kinematics") or {}
    smoothing = kin_cfg.get("smoothing", False)
    window = kin_cfg.get("smoothing_window", 5)
    polyorder = kin_cfg.get("smoothing_polyorder", 2)
    method = kin_cfg.get("smoothing_method", "savgol")

    filt_cfg = cfg.get("filtering") or {}

    results = {}
    for gid in selected:
        if end == float("inf"):
            sub = df.iloc[start:]
            times = times_full[start:]
        else:
            sub = df.iloc[start : end + 1]
            times = times_full[start : end + 1]

        ent_df = sub.xs(gid, level=0, axis=1)
        if "Position" not in ent_df.columns.get_level_values(1):
            continue

        pos = (
            ent_df.xs("Position", level=1, axis=1)
            .droplevel(0, axis=1)[["X", "Y", "Z"]]
            .values
        )

        rot = None
        if "Rotation" in ent_df.columns.get_level_values(1):
            rot = (
                ent_df.xs("Rotation", level=1, axis=1)
                .droplevel(0, axis=1)[["X", "Y", "Z", "W"]]
                .values
            )

        speed, t_v = compute_linear_velocity(
            pos,
            times,
            smoothing=smoothing,
            window=window,
            polyorder=polyorder,
            method=method,
        )

        if rot is not None:
            ang_speed, t_as = compute_angular_speed(
                rot, times, smoothing=smoothing, window=window, polyorder=polyorder
            )
            ang_vel, t_av = compute_angular_velocity(
                rot, times, smoothing=smoothing, window=window, polyorder=polyorder
            )
        else:
            ang_speed, t_as = np.array([]), np.array([])
            ang_vel, t_av = np.zeros((0, 3)), np.array([])

        if filt_cfg.get("enable"):
            start_frames = start + 1
            speed_ranges: List[Tuple[int, int]] = []
            ang_ranges: List[Tuple[int, int]] = []
            pos_ranges: List[Tuple[int, int]] = []

            if (
                filt_cfg.get("speed_lower") is not None
                or filt_cfg.get("speed_upper") is not None
            ):
                speed, speed_ranges = filter_anomalies(
                    speed,
                    start_frames,
                    filt_cfg.get("speed_lower"),
                    filt_cfg.get("speed_upper"),
                )

            if (
                filt_cfg.get("angular_speed_lower") is not None
                or filt_cfg.get("angular_speed_upper") is not None
            ):
                ang_speed, ang_ranges = filter_anomalies(
                    ang_speed,
                    start_frames,
                    filt_cfg.get("angular_speed_lower"),
                    filt_cfg.get("angular_speed_upper"),
                )

            if any(
                filt_cfg.get(f"{axis}_{b}") is not None
                for axis in ("x", "y", "z")
                for b in ("lower", "upper")
            ):
                pos, pos_ranges = filter_position(
                    pos,
                    start,
                    filt_cfg.get("x_lower"),
                    filt_cfg.get("x_upper"),
                    filt_cfg.get("y_lower"),
                    filt_cfg.get("y_upper"),
                    filt_cfg.get("z_lower"),
                    filt_cfg.get("z_upper"),
                )

            ang_speed = apply_ranges(ang_speed, start_frames, speed_ranges)
            speed = apply_ranges(speed, start_frames, ang_ranges)

            if pos_ranges:
                rng_conv = [(max(start_frames, s), e + 1) for s, e in pos_ranges]
                speed = apply_ranges(speed, start_frames, rng_conv)
                ang_speed = apply_ranges(ang_speed, start_frames, rng_conv)

            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in speed_ranges])
            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in ang_ranges])

        markers = []
        for tm in cfg.get("time_markers") or []:
            idx = int(np.searchsorted(times, tm, side="left"))
            if 0 <= idx < len(times):
                markers.append(idx)

        results[gid] = {
            "pos": pos,
            "times": times,
            "speed": speed,
            "t_speed": t_v,
            "ang_speed": ang_speed,
            "t_ang_speed": t_as,
            "ang_vel": ang_vel,
            "t_ang_vel": t_av,
            "markers": markers,
        }

    return results, selected


def make_figures(
    pos: np.ndarray,
    times: np.ndarray,
    markers: List[int],
    speed: np.ndarray,
    t_speed: np.ndarray,
    ang_speed: np.ndarray,
    t_ang_speed: np.ndarray,
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Create the 3D/2D trajectory and speed plots."""
    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Rainbow"),
            line=dict(color="blue"),
            name="trajectory",
            customdata=times,
        )
    )
    for idx in markers:
        fig3d.add_trace(
            go.Scatter3d(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                z=[pos[idx, 2]],
                mode="markers",
                marker=dict(color="orange", symbol="triangle-down", size=5),
                showlegend=False,
            )
        )

    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Trajectory",
    )

    fig2d = go.Figure()
    fig2d.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Rainbow"),
            name="trajectory",
            customdata=times,
        )
    )
    for idx in markers:
        fig2d.add_trace(
            go.Scatter(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                mode="markers",
                marker=dict(color="orange", symbol="triangle-down", size=8),
                showlegend=False,
            )
        )
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Trajectory")
    fig2d.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_speed = go.Figure()
    fig_speed.add_trace(
        go.Scatter(x=t_speed, y=speed, mode="lines", name="speed", line=dict(color="blue"))
    )
    for tm in markers:
        if tm < len(times):
            fig_speed.add_vline(x=times[tm], line_color="orange", line_dash="dash")
    fig_speed.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Linear Speed",
        title="Linear Speed",
    )

    fig_ang = go.Figure()
    fig_ang.add_trace(
        go.Scatter(x=t_ang_speed, y=ang_speed, mode="lines", name="angular", line=dict(color="blue"))
    )
    for tm in markers:
        if tm < len(times):
            fig_ang.add_vline(x=times[tm], line_color="orange", line_dash="dash")
    fig_ang.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Angular Speed",
        title="Angular Speed",
    )

    return fig3d, fig2d, fig_speed, fig_ang


def slice_range(times: np.ndarray, start: float, end: float) -> slice:
    """Return slice object covering the given time range."""
    i0 = int(np.searchsorted(times, start, side="left"))
    i1 = int(np.searchsorted(times, end, side="right"))
    return slice(i0, i1)


def build_table(d: dict, start: float, end: float) -> List[dict]:
    """Create a list of rows for the DataTable from a time range."""
    sl = slice_range(d["times"], start, end)
    times = d["times"][sl]
    pos = d["pos"][sl]
    iv = np.searchsorted(d["t_speed"], times)
    ia = np.searchsorted(d["t_ang_vel"], times)
    rows = []
    for idx, t in enumerate(times):
        spd = d["speed"][iv[idx]] if iv[idx] < len(d["speed"]) else float("nan")
        ang = d["ang_speed"][ia[idx]] if ia[idx] < len(d["ang_speed"]) else float("nan")
        rows.append(
            {
                "time": float(t),
                "x": float(pos[idx, 0]),
                "y": float(pos[idx, 1]),
                "z": float(pos[idx, 2]),
                "speed": float(spd),
                "angular_speed": float(ang),
            }
        )
    return rows

