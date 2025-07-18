import argparse
import os
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
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


def _prepare_data(cfg):
    """Load data and apply the same processing steps as the CLI."""
    input_file = cfg.get("input_file")
    if not os.path.exists(input_file):
        fallback = os.path.join("data", "input.csv")
        input_file = fallback if os.path.exists(fallback) else input_file

    df, frame_col, time_col = load_data(input_file)
    groups_all = group_entities(df)

    selected = cfg.get("groups") or list(groups_all.keys())
    selected = [g for g in selected if g in groups_all]

    # Time slicing
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

        # Velocities
        speed, t_v = compute_linear_velocity(
            pos, times, smoothing=smoothing, window=window, polyorder=polyorder, method=method
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

        # Optional filtering
        if filt_cfg.get("enable"):
            start_frames = start + 1
            speed_ranges = []
            ang_ranges = []
            pos_ranges = []

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


def _make_figures(pos, times, markers, speed, t_speed, ang_speed, t_ang_speed):
    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Viridis"),
            line=dict(color="blue"),
            name="trajectory",
        )
    )
    for idx in markers:
        fig3d.add_trace(
            go.Scatter3d(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                z=[pos[idx, 2]],
                mode="markers",
                marker=dict(color="red", symbol="triangle-down", size=5),
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
            marker=dict(size=3, color=times, colorscale="Viridis"),
            name="trajectory",
        )
    )
    for idx in markers:
        fig2d.add_trace(
            go.Scatter(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                mode="markers",
                marker=dict(color="red", symbol="triangle-down", size=8),
                showlegend=False,
            )
        )
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Trajectory")

    fig_speed = go.Figure()
    fig_speed.add_trace(
        go.Scatter(x=t_speed, y=speed, mode="lines", name="speed")
    )
    for tm in markers:
        if tm < len(times):
            fig_speed.add_vline(
                x=times[tm],
                line_color="red",
                line_dash="dot",
            )
    fig_speed.update_layout(xaxis_title="Time (s)", yaxis_title="Linear Speed", title="Linear Speed")

    fig_ang = go.Figure()
    fig_ang.add_trace(
        go.Scatter(x=t_ang_speed, y=ang_speed, mode="lines", name="angular")
    )
    for tm in markers:
        if tm < len(times):
            fig_ang.add_vline(
                x=times[tm],
                line_color="red",
                line_dash="dot",
            )
    fig_ang.update_layout(xaxis_title="Time (s)", yaxis_title="Angular Speed", title="Angular Speed")

    return fig3d, fig2d, fig_speed, fig_ang


def create_app(cfg):
    data, groups = _prepare_data(cfg)
    default_gid = groups[0] if groups else None

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Interactive Trajectory Viewer"),
            dcc.Dropdown(
                options=[{"label": g, "value": g} for g in groups],
                value=default_gid,
                id="entity-dropdown",
                clearable=False,
            ),
            dcc.Graph(id="traj3d"),
            dcc.Graph(id="traj2d"),
            dcc.Graph(id="speed"),
            dcc.Graph(id="angular"),
            html.Pre(id="info", children="Click a point on either plot"),
        ]
    )

    @app.callback(
        Output("traj3d", "figure"),
        Output("traj2d", "figure"),
        Output("speed", "figure"),
        Output("angular", "figure"),
        Input("entity-dropdown", "value"),
    )
    def _update_plots(selected_id):
        empty = go.Figure()
        if not selected_id:
            return empty, empty, empty, empty
        d = data[selected_id]
        return _make_figures(
            d["pos"],
            d["times"],
            d["markers"],
            d["speed"],
            d["t_speed"],
            d["ang_speed"],
            d["t_ang_speed"],
        )

    @app.callback(
        Output("info", "children"),
        Input("traj3d", "clickData"),
        Input("traj2d", "clickData"),
        State("entity-dropdown", "value"),
    )
    def _display_info(click3d, click2d, selected_id):
        click = click3d or click2d
        if not selected_id or not click:
            return "Click a point on either plot"
        idx = click["points"][0]["pointIndex"]
        d = data[selected_id]
        t = d["times"][idx]
        p = d["pos"][idx]
        iv = np.searchsorted(d["t_speed"], t)
        ia = np.searchsorted(d["t_ang_vel"], t)
        spd = d["speed"][iv] if iv < len(d["speed"]) else float("nan")
        ang_spd = d["ang_speed"][ia] if ia < len(d["ang_speed"]) else float("nan")
        ang_vec = d["ang_vel"][ia] if ia < len(d["ang_vel"]) else np.array([np.nan] * 3)
        return (
            f"Time: {t:.3f}s\n"
            f"X: {p[0]:.3f}\nY: {p[1]:.3f}\nZ: {p[2]:.3f}\n"
            f"Speed: {spd:.3f}\nAngular Speed: {ang_spd:.3f}\n"
            f"Wx: {ang_vec[0]:.3f}\nWy: {ang_vec[1]:.3f}\nWz: {ang_vec[2]:.3f}"
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive trajectory viewer")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    cfg = Config(args.config)
    port = cfg.get("webapp", "port", default=3010)
    app = create_app(cfg)
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
