import argparse
import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
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


def _make_figures(
    pos,
    times,
    markers,
    speed,
    t_speed,
    ang_speed,
    t_ang_speed,
    highlight=None,
):
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
    if highlight is not None:
        hi = int(np.searchsorted(times, highlight, side="left"))
        if 0 <= hi < len(times):
            fig3d.add_trace(
                go.Scatter3d(
                    x=[pos[hi, 0]],
                    y=[pos[hi, 1]],
                    z=[pos[hi, 2]],
                    mode="markers",
                    marker=dict(color="orange", size=4, symbol="diamond"),
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
    if highlight is not None:
        hi = int(np.searchsorted(times, highlight, side="left"))
        if 0 <= hi < len(times):
            fig2d.add_trace(
                go.Scatter(
                    x=[pos[hi, 0]],
                    y=[pos[hi, 1]],
                    mode="markers",
                    marker=dict(color="orange", size=6, symbol="diamond"),
                    showlegend=False,
                )
            )
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Trajectory")

    fig_speed = go.Figure()
    fig_speed.add_trace(
        go.Scatter(x=t_speed, y=speed, mode="lines", name="speed", line=dict(color="blue"))
    )

    for tm in markers:
        if tm < len(times):
            fig_speed.add_vline(
                x=times[tm],
                line_color="orange",
                line_dash="dash",
            )
    if highlight is not None:
        fig_speed.add_vline(x=highlight, line_color="orange", line_dash="dash")

    fig_speed.update_layout(
        xaxis_title="Time (s)", yaxis_title="Linear Speed", title="Linear Speed"
    )

    fig_ang = go.Figure()
    fig_ang.add_trace(
        go.Scatter(x=t_ang_speed, y=ang_speed, mode="lines", name="angular", line=dict(color="blue"))

    )
    for tm in markers:
        if tm < len(times):
            fig_ang.add_vline(
                x=times[tm],
                line_color="orange",
                line_dash="dash",
            )
    if highlight is not None:
        fig_ang.add_vline(x=highlight, line_color="orange", line_dash="dash")

    fig_ang.update_layout(
        xaxis_title="Time (s)", yaxis_title="Angular Speed", title="Angular Speed"
    )

    return fig3d, fig2d, fig_speed, fig_ang


def _slice_range(times, start, end):
    """Return slice object covering the given time range."""
    i0 = int(np.searchsorted(times, start, side="left"))
    i1 = int(np.searchsorted(times, end, side="right"))
    return slice(i0, i1)


def _build_table(d, start, end):
    """Create list of dicts for Dash DataTable from selected time range."""
    sl = _slice_range(d["times"], start, end)
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


def create_app(cfg):
    data, groups = _prepare_data(cfg)
    default_gid = groups[0] if groups else None

    app = Dash(__name__)
    times_ref = data[default_gid]["times"] if default_gid else np.array([0.0, 1.0])
    t_min, t_max = float(times_ref[0]), float(times_ref[-1])

    app.layout = html.Div(
        [
            html.H2("Interactive Trajectory Viewer"),
            dcc.Dropdown(
                options=[{"label": g, "value": g} for g in groups],
                value=default_gid,
                id="entity-dropdown",
                clearable=False,
            ),
            dcc.RangeSlider(
                id="time-range",
                min=t_min,
                max=t_max,
              
                # step = 1,
                step = max((t_max - t_min) / 50, 0.001),
              
                value=[t_min, t_max],
                allowCross=False,
                tooltip={"placement": "bottom"},
            ),
            html.Button("Play", id="play-btn", n_clicks=0),
            dcc.Interval(id="play-int", interval=1000, disabled=True),
            dcc.Store(id="highlight-time"),
            html.Div(
                [
                    dcc.Graph(id="traj3d", style={"flex": "1 1 45%", "minWidth": "300px"}),
                    dcc.Graph(id="traj2d", style={"flex": "1 1 45%", "minWidth": "300px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    dcc.Graph(id="speed", style={"flex": "1 1 45%", "minWidth": "300px"}),
                    dcc.Graph(id="angular", style={"flex": "1 1 45%", "minWidth": "300px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),

            dash_table.DataTable(
                id="raw-table",
                columns=[
                    {"name": n, "id": n}
                    for n in ["time", "x", "y", "z", "speed", "angular_speed"]
                ],
                page_size=10,
            ),
            html.H3("Edit configuration"),
            dcc.Textarea(
                id="config-editor",
                value=cfg.as_yaml(),
                style={"width": "100%", "height": "200px"},
            ),
            html.Button("Save Config", id="save-config"),
            html.Div(id="save-status"),
            html.Pre(id="info", children="Hover on any plot to highlight; click for details"),

        ]
    )

    @app.callback(
        Output("traj3d", "figure"),
        Output("traj2d", "figure"),
        Output("speed", "figure"),
        Output("angular", "figure"),
        Output("raw-table", "data"),
        Input("entity-dropdown", "value"),
        Input("time-range", "value"),
        Input("highlight-time", "data"),
    )
    def _update_plots(selected_id, t_range, highlight):

        empty = go.Figure()
        if not selected_id:
            return empty, empty, empty, empty, []
        d = data[selected_id]
        start, end = t_range or [d["times"][0], d["times"][-1]]
        sl = _slice_range(d["times"], start, end)
        sl_v = _slice_range(d["t_speed"], start, end)
        sl_a = _slice_range(d["t_ang_speed"], start, end)
        figs = _make_figures(
            d["pos"][sl],
            d["times"][sl],
            [m - sl.start for m in d["markers"] if sl.start <= m < sl.stop],
            d["speed"][sl_v],
            d["t_speed"][sl_v],
            d["ang_speed"][sl_a],
            d["t_ang_speed"][sl_a],
            highlight,

        )
        table = _build_table(d, start, end)
        return (*figs, table)

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

    @app.callback(
        Output("play-int", "disabled"),
        Output("play-btn", "children"),
        Input("play-btn", "n_clicks"),
        State("play-int", "disabled"),
        prevent_initial_call=True,
    )
    def _toggle_play(n, disabled):
        disabled = not disabled
        return disabled, ("Play" if disabled else "Pause")

    @app.callback(
        Output("time-range", "value"),
        Input("play-int", "n_intervals"),
        State("time-range", "value"),
        State("time-range", "max"),
    )
    def _advance(_, val, maximum):
        start, end = val
        
        # step = 1
        step = max((maximum - start) / 50, 0.001)

        if end + step > maximum:
            return [start, maximum]
        return [start + step, end + step]

    @app.callback(
        Output("highlight-time", "data"),
        Input("traj3d", "hoverData"),
        Input("traj2d", "hoverData"),
        Input("speed", "hoverData"),
        Input("angular", "hoverData"),
        State("highlight-time", "data"),
        prevent_initial_call=True,
    )
    def _update_highlight(h3d, h2d, hs, ha, current):
        for hov in (h3d, h2d, hs, ha):
            if hov and hov.get("points"):
                p = hov["points"][0]
                if "customdata" in p:
                    return float(p["customdata"])
                if "x" in p:
                    return float(p["x"])
        raise PreventUpdate

    @app.callback(
        Output("time-range", "value"),
        Input("speed", "relayoutData"),
        Input("angular", "relayoutData"),
        State("time-range", "value"),
        prevent_initial_call=True,
    )
    def _sync_range(r1, r2, cur):
        r = r2 or r1
        if not r:
            raise PreventUpdate
        if "xaxis.range" in r:
            return [float(r["xaxis.range"][0]), float(r["xaxis.range"][1])]
        if "xaxis.range[0]" in r and "xaxis.range[1]" in r:
            return [float(r["xaxis.range[0]"]), float(r["xaxis.range[1]"])]
        raise PreventUpdate

    @app.callback(

        Output("save-status", "children"),
        Input("save-config", "n_clicks"),
        State("config-editor", "value"),
        prevent_initial_call=True,
    )
    def _save_config(_, text):
        try:
            cfg.update_from_yaml(text)
        except Exception as exc:  # noqa: BLE001
            return f"Invalid YAML: {exc}"
        out_dir = cfg.get("output", "output_dir")
        os.makedirs(out_dir, exist_ok=True)
        name = f"web_saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(cfg.as_yaml())
        return f"Saved to {path}"

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
