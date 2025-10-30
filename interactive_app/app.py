"""Dash-based interactive viewer for tracking analysis."""
from __future__ import annotations

import argparse

from dash import Dash

from tracking_analysis.config import Config
from .data_utils import prepare_data
from .layout import build_layout
from .callbacks import register_callbacks


def create_app(cfg: Config) -> Dash:
    """Create Dash application."""
    data, groups = prepare_data(cfg)
    app = Dash(__name__)
    app.layout = build_layout(cfg, data, groups)
    register_callbacks(app, cfg, data, groups)
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
