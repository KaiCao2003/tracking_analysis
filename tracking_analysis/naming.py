from __future__ import annotations

import os
from datetime import datetime

from tracking_analysis.config import Config


def slugify(value: str) -> str:
    """Return a filesystem-friendly version of ``value``."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(value))


def build_info_suffix(cfg: Config) -> str:
    """Compose a short suffix describing key config options."""
    parts = []
    start = cfg.get("interval", "start_time")
    if start not in (None, 0):
        parts.append(f"s{int(start)}")
    end = cfg.get("interval", "end_time")
    if end not in (None, float("inf")):
        parts.append(f"e{int(end)}")
    filt = cfg.get("filtering", "enable", default=None)
    if filt is not None:
        parts.append(f"flt{'1' if filt else '0'}")
    nm = cfg.get("no_moving", "enable", default=None)
    if nm is not None:
        parts.append(f"nm{'1' if nm else '0'}")
    markers = [m for m in (cfg.get("time_markers") or []) if m is not None]
    if markers:
        parts.append("tm" + "-".join(str(int(m)) for m in markers))
    return "_".join(slugify(p) for p in parts)


def build_run_dir(cfg: Config, base_out: str, prefix: str = "") -> str:
    """Return a run-specific output directory under ``base_out``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = build_info_suffix(cfg)
    name = "_".join(n for n in [prefix, ts, suffix] if n)
    return os.path.join(base_out, name)
