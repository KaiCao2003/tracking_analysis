"""Generate SVG plots comparing different filters on a synthetic signal."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Dict

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, lfilter, firwin
import matplotlib.pyplot as plt

from tracking_analysis.config import Config


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
            filt = filtfilt(b, a, signal)
        elif ftype == "savgol":
            window = int(cfg.get("window", 5))
            if window % 2 == 0:
                window += 1
            poly = int(cfg.get("polyorder", 2))
            filt = savgol_filter(signal, window, poly)
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

    duration = float(cfg.get("filter_test", "duration", default=5.0))
    sample_rate = float(cfg.get("filter_test", "sample_rate", default=100.0))
    freq = float(cfg.get("filter_test", "signal_freq", default=1.0))
    noise = float(cfg.get("filter_test", "noise_level", default=0.2))

    t = np.arange(0.0, duration, 1.0 / sample_rate)
    base = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))

    filters = cfg.get("filter_test", "filters", default=[]) or []
    results = _apply_filters(base, sample_rate, filters)

    out_dir = cfg.get("output", "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(out_dir, f"filter_test_{stamp}")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
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
        plt.figure(figsize=(10, 6))
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
