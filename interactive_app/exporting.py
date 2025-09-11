"""Export utilities for the Dash application."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from tracking_analysis.config import Config

from .data_utils import build_table


def export_metrics(
    groups: Dict[str, dict],
    out_dir: str | Path = "results",
    fmt: str = "csv",
    metrics: Sequence[str] | None = None,
) -> Dict[str, str]:
    """Export selected kinematic metrics for each group.

    Parameters
    ----------
    groups:
        Mapping of group identifiers to data dictionaries as returned by
        :func:`prepare_data`.
    out_dir:
        Directory where the exported files will be stored. Created if missing.
    fmt:
        Output format: ``"csv"`` or ``"json"``.
    metrics:
        Iterable of metric names to include. ``None`` exports all available
        metrics. Valid values are ``"trajectory"``, ``"speed``" and
        ``"angular_speed"``.

    Returns
    -------
    Dict[str, str]
        Mapping of group identifiers to the written file paths.
    """
    fmt = fmt.lower()
    if fmt not in {"csv", "json"}:
        raise ValueError("fmt must be 'csv' or 'json'")

    allowed = {"trajectory", "speed", "angular_speed"}
    if metrics is None:
        selected = allowed
    else:
        invalid = set(metrics) - allowed
        if invalid:
            inv = ", ".join(sorted(invalid))
            raise ValueError(f"unknown metrics: {inv}")
        selected = set(metrics)

    if not selected:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for gid, data in groups.items():
        rows = build_table(data, float(data["times"][0]), float(data["times"][-1]))
        df = pd.DataFrame(rows)
        cols = ["frame", "time"]
        if "trajectory" in selected:
            cols.extend(["x", "y", "z"])
        if "speed" in selected:
            cols.append("speed")
        if "angular_speed" in selected:
            cols.append("angular_speed")
        df = df[cols]
        out_path = out_dir / f"{gid}.{fmt}"
        if fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            df.to_json(out_path, orient="records", indent=2)
        paths[gid] = str(out_path)

    return paths


def export_metrics_cfg(groups: Dict[str, dict], cfg: Config) -> Dict[str, str]:
    """Export metrics based on configuration options.

    Reads ``output.export_metrics`` from ``cfg``. The sub-keys are:

    ``enable`` (bool): whether exporting is enabled.
    ``metrics`` (list[str]): subset of ``trajectory``, ``speed`` and
    ``angular_speed`` to include.
    ``format`` (str): ``"csv"`` or ``"json"``.
    """
    exp_cfg = cfg.get("output", "export_metrics", default={}) or {}
    if not exp_cfg.get("enable", False):
        return {}

    metrics: Iterable[str] | None = exp_cfg.get("metrics")
    fmt: str = exp_cfg.get("format", "csv")
    out_dir: str | Path = cfg.get("output", "output_dir", default="results")
    return export_metrics(groups, out_dir=out_dir, fmt=fmt, metrics=metrics)


__all__ = ["export_metrics", "export_metrics_cfg"]