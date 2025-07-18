"""Dash-based interactive viewer for tracking analysis."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from tracking_analysis.config import Config

from interactive_app.utils import build_table, make_figures, prepare_data, slice_range


def create_app(cfg: Config) -> Dash:
    """Create Dash application."""
    data, groups = prepare_data(cfg)
    default_gid = groups[0] if groups else None

    app = Dash(__name__)
    times_ref = data[default_gid]["times"] if default_gid else np.array([0.0, 1.0])
    t_min, t_max = float(times_ref[0]), float(times_ref[-1])
    slider_step = round(max((t_max - t_min) / 50, 0.1), 1)

    app.layout = html.Div(
        [
            html.H2("Interactive Trajectory Viewer"),
            dcc.Dropdown(
                options=[{"label": g, "value": g} for g in groups],
                value=default_gid,
                id="entity-dropdown",
                clearable=False,
            ),
            html.Div(
                [
                    dcc.RangeSlider(
                        id="time-range",
                        min=t_min,
                        max=t_max,
                        step=slider_step,
                        value=[t_min, t_max],
                        allowCross=False,
                        tooltip={"placement": "bottom"},
                    ),
                    html.Button("Play", id="play-btn", n_clicks=0),
                    dcc.Interval(id="play-int", interval=1000, disabled=True),
                    html.Div(id="status-bar", children="Ready"),

                  
                ],
                style={"display": "flex", "gap": "10px", "alignItems": "center"},
            ),
            html.Div(
                [
                    dcc.Graph(id="traj3d", style={"flex": "1", "height": "600px"}),

                  dcc.Graph(
                        id="traj2d",
                        style={"flex": "1", "height": "600px", "width": "600px"},
                    ),

                  
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="speed", style={"width": "100%", "height": "400px"}),
            dcc.Graph(id="angular", style={"width": "100%", "height": "400px"}),
            html.Button("Show Table", id="toggle-table", n_clicks=0),
            html.Div(
                dash_table.DataTable(
                    id="raw-table",
                    columns=[
                        {"name": n, "id": n}
                        for n in [
                            "frame",
                            "time",
                            "x",
                            "y",
                            "z",
                            "speed",
                            "angular_speed",
                        ]
                    ],
                    page_size=10,
                ),
                id="table-container",
                style={"display": "none"},
            ),
            html.H3("Edit configuration"),
            dcc.Textarea(
                id="config-editor",
                value=cfg.as_yaml(),
                style={"width": "100%", "height": "200px"},
            ),
            html.Button("Save Config", id="save-config"),
            html.Div(id="save-status"),
            html.Pre(id="info", children="Hover or click on any plot for details"),
        ]
    )

    @app.callback(
        Output("traj3d", "figure"),
        Output("traj2d", "figure"),
        Output("speed", "figure"),
        Output("angular", "figure"),
        Output("raw-table", "data"),
        Output("status-bar", "children"),
        Input("entity-dropdown", "value"),
        Input("time-range", "value"),
    )
    def _update_plots(selected_id, t_range):
        if not selected_id:
            empty = go.Figure()
            return empty, empty, empty, empty, [], "No data"
        d = data[selected_id]
        start, end = t_range or [d["times"][0], d["times"][-1]]
        sl = slice_range(d["times"], start, end)
        sl_v = slice_range(d["t_speed"], start, end)
        sl_a = slice_range(d["t_ang_speed"], start, end)
        figs = make_figures(
            d["pos"][sl],
            d["times"][sl],
            d["frames"][sl],
            [m - sl.start for m in d["markers"] if sl.start <= m < sl.stop],
            d["speed"][sl_v],
            d["t_speed"][sl_v],
            d["frames_speed"][sl_v],
            d["ang_speed"][sl_a],
            d["t_ang_speed"][sl_a],
            d["frames_ang"][sl_a],
        )
        table = build_table(d, start, end)
        return (*figs, table, "Updated")

    @app.callback(
        Output("info", "children"),
        Input("traj3d", "clickData"),
        Input("traj2d", "clickData"),
        Input("speed", "clickData"),
        Input("angular", "clickData"),
        State("entity-dropdown", "value"),
    )
    def _display_info(click3d, click2d, click_speed, click_ang, selected_id):
        ctx = callback_context
        if not selected_id or not ctx.triggered:
            return "Click a point on any plot"

        trigger = ctx.triggered_id
        d = data[selected_id]

        if trigger in {"traj3d", "traj2d"}:
            click = click3d if trigger == "traj3d" else click2d
            if not click:
                raise PreventUpdate
            idx = click["points"][0]["pointIndex"]
            t = d["times"][idx]
            frame = d["frames"][idx]
            p = d["pos"][idx]
            return (
                f"Frame: {int(frame)}\n"

                f"Time: {t:.3f}s\n"
                f"X: {p[0]:.3f}\nY: {p[1]:.3f}\nZ: {p[2]:.3f}"
            )

        if trigger == "speed":
            if not click_speed:
                raise PreventUpdate
            idx = click_speed["points"][0]["pointIndex"]
            t = d["t_speed"][idx]
            frame = d["frames_speed"][idx]
            spd = click_speed["points"][0]["y"]
            return f"Frame: {int(frame)}\nTime: {float(t):.3f}s\nSpeed: {float(spd):.3f}"


        if trigger == "angular":
            if not click_ang:
                raise PreventUpdate
            idx = click_ang["points"][0]["pointIndex"]
            t = d["t_ang_speed"][idx]
            frame = d["frames_ang"][idx]
            ang = click_ang["points"][0]["y"]
            return f"Frame: {int(frame)}\nTime: {float(t):.3f}s\nAngular Speed: {float(ang):.3f}"

        raise PreventUpdate

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
        Input("speed", "relayoutData"),
        Input("angular", "relayoutData"),
        Input("traj3d", "relayoutData"),
        Input("traj2d", "relayoutData"),
        State("time-range", "value"),
        State("time-range", "max"),
        prevent_initial_call=True,
    )
    def _update_range(_, r_speed, r_ang, r3d, r2d, val, maximum):
        """Advance playback or sync slider when plots are zoomed."""
        trigger = callback_context.triggered_id
        if trigger == "play-int":
            start, end = val
            step = round(max((maximum - start) / 50, 0.1), 1)
            if end + step > maximum:
                return [start, maximum]
            return [start + step, end + step]
        r = r_ang or r_speed or r3d or r2d
        if not r:
            raise PreventUpdate
        if "xaxis.range" in r:
            return [float(r["xaxis.range"][0]), float(r["xaxis.range"][1])]
        if "xaxis.range[0]" in r and "xaxis.range[1]" in r:
            return [float(r["xaxis.range[0]"]), float(r["xaxis.range[1]"])]
        raise PreventUpdate

    @app.callback(
        Output("table-container", "style"),
        Input("toggle-table", "n_clicks"),
        State("table-container", "style"),
        prevent_initial_call=True,
    )
    def _toggle_table(n, style):
        disp = style.get("display", "block")
        return {"display": "none" if disp != "none" else "block"}

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


def main() -> None:
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
