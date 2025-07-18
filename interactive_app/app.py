import argparse
import os
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data
from tracking_analysis.grouping import group_entities
from tracking_analysis.kinematics import (
    compute_linear_velocity,
    compute_angular_speed,
    compute_angular_velocity,
)


def _prepare_data(cfg):
    input_file = cfg.get("input_file")
    if not os.path.exists(input_file):
        fallback = os.path.join("data", "input.csv")
        input_file = fallback if os.path.exists(fallback) else input_file
    df, frame_col, time_col = load_data(input_file)
    groups = group_entities(df)
    times = df[time_col].values
    data = {}
    for gid in groups.keys():
        ent_df = df.xs(gid, level=0, axis=1)
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
        speed, t_v = compute_linear_velocity(pos, times)
        if rot is not None:
            ang_speed, t_as = compute_angular_speed(rot, times)
            ang_vel, t_av = compute_angular_velocity(rot, times)
        else:
            ang_speed, t_as = np.array([]), np.array([])
            ang_vel, t_av = np.zeros((0, 3)), np.array([])
        data[gid] = {
            "pos": pos,
            "times": times,
            "speed": speed,
            "t_speed": t_v,
            "ang_speed": ang_speed,
            "t_ang_speed": t_as,
            "ang_vel": ang_vel,
            "t_ang_vel": t_av,
        }
    return data, list(groups.keys())


def _make_figures(pos, times):
    fig3d = go.Figure(
        data=go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Viridis"),
            line=dict(color="blue"),
        )
    )
    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Trajectory",
    )

    fig2d = go.Figure(
        data=go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Viridis"),
        )
    )
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Trajectory")
    return fig3d, fig2d


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
            html.Pre(id="info", children="Click a point on either plot"),
        ]
    )

    @app.callback(
        Output("traj3d", "figure"),
        Output("traj2d", "figure"),
        Input("entity-dropdown", "value"),
    )
    def _update_plots(selected_id):
        if not selected_id:
            return go.Figure(), go.Figure()
        d = data[selected_id]
        return _make_figures(d["pos"], d["times"])

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
    app.run_server(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
