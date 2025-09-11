"""Application layout for the Dash viewer."""
from __future__ import annotations

import numpy as np
from dash import dcc, html, dash_table

from .ui_components import build_config_form


def build_layout(cfg, data, groups: list[str]) -> html.Div:
    """Return the full application layout."""
    default_gid = groups[0] if groups else None
    times_ref = data[default_gid]["times"] if default_gid else np.array([0.0, 1.0])
    t_min, t_max = float(times_ref[0]), float(times_ref[-1])
    slider_step = round(max((t_max - t_min) / 50, 0.1), 1)
    config_children = build_config_form(cfg._cfg)

    return html.Div(
        [
            html.H2("Interactive Trajectory Viewer"),
            html.Div(
                [
                    dcc.Dropdown(
                        options=[{"label": g, "value": g} for g in groups],
                        value=default_gid,
                        id="entity-dropdown",
                        clearable=False,
                        style={"minWidth": "200px"},
                    ),
                    html.Button("Show Table", id="toggle-table", n_clicks=0),
                    html.Button("Edit Config", id="toggle-config", n_clicks=0),
                ],
                style={
                    "display": "flex",
                    "gap": "10px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                    "margin": "10px 0",
                },
            ),
            html.Div(
                dcc.RangeSlider(
                    id="time-range",
                    min=t_min,
                    max=t_max,
                    step=slider_step,
                    value=[t_min, t_max],
                    allowCross=False,
                    tooltip={"placement": "bottom"},
                ),
                style={"padding": "0 20px"},
            ),
            html.Div(
                [
                    html.Button("Play", id="play-btn", n_clicks=0),
                    dcc.Interval(id="play-int", interval=1000, disabled=True),
                    html.Div(id="status-bar", children="Ready", style={"marginLeft": "10px"}),
                ],
                style={
                    "display": "flex",
                    "gap": "10px",
                    "alignItems": "center",
                    "margin": "10px 0",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="traj3d", style={"flex": "1", "height": "400px"}),
                            dcc.Graph(id="traj2d", style={"flex": "1", "height": "400px"}),
                        ],
                        style={
                            "display": "flex",
                            "gap": "20px",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="speed", style={"flex": "1", "height": "300px"}),
                            dcc.Graph(id="angular", style={"flex": "1", "height": "300px"}),
                        ],
                        style={
                            "display": "flex",
                            "gap": "20px",
                            "flexWrap": "wrap",
                            "marginTop": "20px",
                        },
                    ),
                ],
            ),
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
                style={"display": "none", "marginTop": "10px"},
            ),
            html.Div(
                [
                    html.Button(
                        "✕",
                        id="close-config",
                        n_clicks=0,
                        style={"float": "right"},
                    ),
                    html.H4("Configuration"),
                    html.Div(id="config-form", children=config_children),
                    html.Button("Save", id="save-config"),
                    html.Div(id="save-status"),
                ],
                id="config-panel",
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": "0",
                    "right": "0",
                    "width": "350px",
                    "height": "100%",
                    "background": "white",
                    "padding": "20px",
                    "fontFamily": "sans-serif",
                    "overflowY": "auto",
                    "boxShadow": "-2px 0 10px rgba(0,0,0,0.3)",
                    "zIndex": "1000",
                },
            ),
            html.Pre(id="info", children="Hover or click on any plot for details"),
            dcc.Store(id="selected-time"),
        ]
    )
