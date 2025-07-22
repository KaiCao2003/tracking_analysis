"""Collection of smoothing functions used by the web app."""

from __future__ import annotations

from typing import Callable, Dict
import inspect
import numpy as np

SMOOTHING_FUNCS: Dict[str, Callable[..., np.ndarray]] = {}

def register(name: str) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator to register a smoothing function."""
    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        SMOOTHING_FUNCS[name] = func
        return func
    return decorator


def apply(method: str, data: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a registered smoothing ``method`` to ``data``."""
    func = SMOOTHING_FUNCS.get(method)
    if func is None:
        raise ValueError(f"Unknown smoothing method '{method}'")
    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(data, **filtered)


@register("savgol")
def savgol(data: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    from scipy.signal import savgol_filter

    if window % 2 == 0:
        window += 1
    return savgol_filter(data, window, polyorder, axis=0)


@register("ema")
def ema(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.empty_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


@register("window")
def window_avg(data: np.ndarray, window: int = 4) -> np.ndarray:
    """Simple windowed average smoothing."""
    if window < 2:
        return data
    if window % 2 != 0:
        window += 1
    half = window // 2
    if data.ndim == 1:
        padded = np.pad(data, (half, half), mode="edge")
    else:
        padded = np.pad(data, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(data, dtype=float)
    for i in range(len(data)):
        segment = padded[i : i + window]
        out[i] = np.mean(segment, axis=0)
    return out