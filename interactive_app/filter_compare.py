"""Generate SVG plots comparing different filters on a signal from the CSV."""


from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, lfilter, firwin
import matplotlib.pyplot as plt
import pandas as pd

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data, preprocess_csv
from tracking_analysis.grouping import group_entities
from tracking_analysis.kinematics import compute_linear_velocity, compute_angular_speed


def _load_signal(cfg: Config) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load a 1-D signal from the configured CSV.

    Returns (times, signal, fs).
    """
    in_path = cfg.get("input_file")
    pre_cfg = cfg.get("preprocess") or {}
    data_path = in_path
    if pre_cfg.get("enable"):
        out_file = pre_cfg.get(
            "output_file",
            os.path.join(cfg.get("output", "output_dir"), "trimmed.csv"),
        )
        summary = pre_cfg.get(
            "summary_file",
            os.path.join(cfg.get("output", "output_dir"), "summary.txt"),
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        os.makedirs(os.path.dirname(summary), exist_ok=True)
        preprocess_csv(in_path, out_file, summary)
        data_path = out_file

    try:
        df, frame_col, time_col = load_data(data_path)
    except Exception:
        df = pd.read_csv(data_path, skiprows=[0, 1, 2, 5], header=[0, 1, 2, 3])
        frame_col = next(c for c in df.columns if c[3] == "Frame")
        time_col = next(c for c in df.columns if c[3] == "Time (Seconds)")

    groups = group_entities(df)
    group = cfg.get("filter_test", "group")
    if not group or group not in groups:
        group = next(iter(groups)) if groups else None
    if group is None:
        raise RuntimeError("No valid group found in data")

    src = cfg.get("filter_test", "source", default="speed").lower()

    start = cfg.get(
        "filter_test",
        "start_time",
        default=cfg.get("interval", "start_time", default=0.0),
    )
    end = cfg.get(
        "filter_test",
        "end_time",
        default=cfg.get("interval", "end_time", default=float("inf")),
    )

    times_all = df[time_col].values
    i0 = int(np.searchsorted(times_all, start, side="left"))
    if end == float("inf"):
        i1 = len(times_all)
    else:
        i1 = int(np.searchsorted(times_all, end, side="right"))

    ent_df = df.xs(group, level=0, axis=1)
    pos = (
        ent_df.xs("Position", level=1, axis=1)
        .droplevel(0, axis=1)[["X", "Y", "Z"]]
        .values
    )[i0:i1]
    times = times_all[i0:i1]

    if src == "speed":
        signal, t = compute_linear_velocity(pos, times)
    elif src == "angular_speed":
        if "Rotation" not in ent_df.columns.get_level_values(1):
            raise ValueError("No rotation data available for angular speed")
        rot = (
                  ent_df.xs("Rotation", level=1, axis=1)
                  .droplevel(0, axis=1)[["X", "Y", "Z", "W"]]
                  .values
              )[i0:i1]
        signal, t = compute_angular_speed(rot, times)
    else:
        comp = {"position_x": 0, "position_y": 1, "position_z": 2}.get(src)
        if comp is None:
            raise ValueError(f"Unknown source '{src}'")
        signal = pos[:, comp]
        t = times

    fs = 1.0 / float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    return t, signal, fs


def _apply_filters(signal: np.ndarray, fs: float, filters: Iterable[dict]) -> Dict[str, np.ndarray]:
    """Apply a series of filters to the signal."""
    results: Dict[str, np.ndarray] = {}
    for idx, cfg in enumerate(filters):
        ftype = cfg.get("type")
        name = cfg.get("name", ftype or f"f{idx}")
        if not ftype:
            continue
        if ftype == "moving_average":
            window = max(1, int(cfg.get("window", 5)))
            kernel = np.ones(window) / window
            filt = np.convolve(signal, kernel, mode="same")
        elif ftype == "ema":
            alpha = float(cfg.get("alpha", 0.3))
            filt = np.empty_like(signal)
            filt[0] = signal[0]
            for i in range(1, len(signal)):
                filt[i] = alpha * signal[i] + (1 - alpha) * filt[i - 1]
        elif ftype == "butterworth":
            order = int(cfg.get("order", 3))
            cutoff_hz = float(cfg.get("cutoff", 1.0))
            nyq = 0.5 * fs
            norm_cutoff = min(cutoff_hz / nyq, 0.99)
            b, a = butter(order, norm_cutoff, btype="low")
            padlen = 3 * max(len(a), len(b))
            if len(signal) <= padlen:
                filt = lfilter(b, a, signal)
            else:
                filt = filtfilt(b, a, signal)

        elif ftype == "savgol":
            window = int(cfg.get("window", 5))
            if window % 2 == 0:
                window += 1
            poly = int(cfg.get("polyorder", 2))
            filt = savgol_filter(signal, window, poly)
        elif ftype == "window":
            window = int(cfg.get("window", 10))
            if window < 2:
                window = 2
            if window % 2 != 0:
                window += 1
            half = window // 2
            padded = np.pad(signal, (half, half), mode="edge")
            filt = np.empty_like(signal, dtype=float)
            for i in range(len(signal)):
                seg = padded[i : i + window]
                m1 = np.mean(seg[:half])
                m2 = np.mean(seg[half:])
                filt[i] = (m1 + m2) / 2
        elif ftype == "decimal_removal":
            digits = int(cfg.get("digits", 1))
            digits = max(0, digits)
            factor = 10 ** digits
            scaled = signal / 180.0
            scaled = np.trunc(scaled * factor) / factor
            filt = scaled * 180.0
        elif ftype == "fir":
            taps = int(cfg.get("numtaps", 21))
            cutoff_hz = float(cfg.get("cutoff", 1.0))
            nyq = 0.5 * fs
            filt = lfilter(firwin(taps, cutoff_hz / nyq), [1.0], signal)
        else:
            continue
        results[name] = filt
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filter comparison plots")
    parser.add_argument("-c", "--config", default="config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = Config(args.config)
    if not cfg.get("filter_test", "enable", default=False):
        print("filter_test.enable is not set")
        return

    t, base, fs = _load_signal(cfg)

    filters = cfg.get("filter_test", "filters", default=[]) or []
    results = _apply_filters(base, fs, filters)


    out_dir = cfg.get("output", "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(out_dir, f"filter_test_{stamp}")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(100, 12))
    plt.plot(t, base, label="original")
    for name, arr in results.items():
        plt.plot(t, arr, label=name)
    plt.xlabel("Time (s)")
    plt.title("Filter Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison.svg"))
    plt.close()

    for name, arr in results.items():
        plt.figure(figsize=(18, 10))
        plt.plot(t, base, label="original", alpha=0.5)
        plt.plot(t, arr, label=name)
        plt.xlabel("Time (s)")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name}.svg"))
        plt.close()

    print(f"Plots written to {plot_dir}")


if __name__ == "__main__":
    main()
