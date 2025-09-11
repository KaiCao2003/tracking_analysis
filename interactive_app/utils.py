"""Compatibility wrapper importing helper functions for the web app."""

from __future__ import annotations

from .data_utils import prepare_data, slice_range, build_table
from .exporting import export_metrics, export_metrics_cfg
from .plotting import make_figures
from .ui_components import build_config_form
from .kinematics import (
    compute_linear_velocity,
    compute_angular_speed,
    compute_angular_velocity,
    compute_head_direction,
)
from .smoothing import apply as apply_smoothing, register

__all__ = [
    "prepare_data",
    "slice_range",
    "build_table",
    "export_metrics",
    "export_metrics_cfg",
    "make_figures",
    "build_config_form",
    "compute_linear_velocity",
    "compute_angular_speed",
    "compute_angular_velocity",
    "compute_head_direction",
    "apply_smoothing",
    "register",
]
