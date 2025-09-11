import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from interactive_app.exporting import export_metrics


def _group():
    return {
        "times": np.array([0.0, 1.0], dtype=float),
        "frames": np.array([1, 2], dtype=int),
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        "t_speed": np.array([0.0], dtype=float),
        "t_ang_vel": np.array([0.0], dtype=float),
        "speed": np.array([0.5], dtype=float),
        "ang_speed": np.array([0.2], dtype=float),
    }


def test_export_metrics_csv(tmp_path):
    groups = {"g": _group()}
    paths = export_metrics(groups, out_dir=tmp_path, fmt="csv")
    data = pd.read_csv(paths["g"])  # noqa: S108
    assert list(data.columns) == [
        "frame",
        "time",
        "x",
        "y",
        "z",
        "speed",
        "angular_speed",
    ]


def test_invalid_metric_raises(tmp_path):
    groups = {"g": _group()}
    with pytest.raises(ValueError):
        export_metrics(groups, out_dir=tmp_path, metrics=["speed", "bad"])  # type: ignore[list-item]