"""Compatibility wrapper importing helper functions for the web app."""

from __future__ import annotations

from .data_utils import apply_filters, prepare_data, slice_range, build_table
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
    "apply_filters",
    "prepare_data",
    "slice_range",
    "build_table",
    "make_figures",
    "build_config_form",
    "compute_linear_velocity",
    "compute_angular_speed",
    "compute_angular_velocity",
    "compute_head_direction",
    "apply_smoothing",
    "register",
]
