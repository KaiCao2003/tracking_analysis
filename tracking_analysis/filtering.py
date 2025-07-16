# File: tracking_analysis/filtering.py
import pandas as pd
import numpy as np

def filter_missing(df, start_frame, end_frame):
    """
    Identify entities whose X coordinate never changes in the given frame interval.
    Handles end_frame = float('inf') as “up through the last frame.”
    Returns a list of entity names whose Position→X is static.
    """
    # Slice the DataFrame by frame index
    if end_frame == float('inf'):
        sub = df.iloc[start_frame:]
    else:
        sub = df.iloc[start_frame:end_frame + 1]

    missing = []
    # Top-level entity names
    entities = df.columns.get_level_values(0).unique()

    for entity in entities:
        try:
            # 1) select only this entity (drops level 0)
            ent_df    = sub.xs(entity, level=0, axis=1)
            # 2) select the 'Position' measurement (now level 1 of ent_df)
            pos_block = ent_df.xs('Position', level=1, axis=1)
            # 3) drop the ID-level (first level) to leave components X,Y,Z
            pos_df    = pos_block.droplevel(0, axis=1)
            # 4) extract the X-coordinate series
            x_series  = pos_df['X']
        except KeyError:
            # no Position→X for this entity
            continue

        # Robust unique count handling, even if x_series is accidentally a
        # DataFrame due to unexpected column structure. Flatten the values to a
        # 1D array, drop NaNs, and measure the number of unique entries.
        vals = pd.unique(pd.Series(x_series.values.ravel()).dropna())
        if len(vals) <= 1:
            missing.append(entity)

    return missing


def filter_anomalies(values, start_frame, low=None, high=None):
    """Replace values outside the [low, high] range with NaN.

    Parameters
    ----------
    values : array_like
        Numeric data to filter.
    start_frame : int
        Frame index corresponding to ``values[0]``.
    low : float, optional
        Values ``<= low`` are removed when specified.
    high : float, optional
        Values ``>= high`` are removed when specified.

    Returns
    -------
    filtered : ndarray
        Array with out-of-range entries replaced by ``NaN``.
    ranges : list of tuple
        List of ``(start_frame, end_frame)`` ranges for removed sections.
    """
    arr = np.asarray(values, dtype=float)
    filtered = arr.copy()
    ranges = []
    i = 0
    n = len(arr)
    while i < n:
        cond = False
        if low is not None and arr[i] <= low:
            cond = True
        if high is not None and arr[i] >= high:
            cond = True
        if not cond:
            i += 1
            continue

        j = i
        while j < n:
            violate = False
            if low is not None and arr[j] <= low:
                violate = True
            if high is not None and arr[j] >= high:
                violate = True
            if not violate:
                break
            filtered[j] = np.nan
            j += 1
        ranges.append((int(start_frame + i), int(start_frame + j)))
        i = j

    return filtered, ranges


def apply_ranges(values, start_frame, ranges):
    """Apply ``ranges`` of frame indices as ``NaN`` to ``values``.

    Parameters
    ----------
    values : array_like
        Array of numeric values. Modified copy is returned.
    start_frame : int
        Frame index corresponding to ``values[0]``.
    ranges : iterable of tuple
        ``(start_frame, end_frame)`` pairs as produced by :func:`filter_anomalies`.

    Returns
    -------
    ndarray
        Array with selected ranges replaced by ``NaN``.
    """
    arr = np.asarray(values, dtype=float).copy()
    for a, b in ranges:
        i = max(0, int(a - start_frame))
        j = max(0, int(b - start_frame))
        if arr.ndim == 1:
            arr[i:j] = np.nan
        else:
            arr[i:j, ...] = np.nan
    return arr


def compute_stats(values, frames, times):
    """Compute summary statistics for a time series."""
    vals = np.asarray(values, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return {}
    v = vals[mask]
    f = np.asarray(frames)[mask]
    t = np.asarray(times)[mask]
    stats = {
        'mean': float(v.mean()),
        'sd': float(v.std()),
        'q25': float(np.quantile(v, 0.25)),
        'q75': float(np.quantile(v, 0.75)),
        'q90': float(np.quantile(v, 0.90)),
        'q95': float(np.quantile(v, 0.95)),
        'max': float(v.max()),
        'max_frame': int(f[np.argmax(v)]),
        'max_time': float(t[np.argmax(v)]),
        'min': float(v.min()),
        'min_frame': int(f[np.argmin(v)]),
        'min_time': float(t[np.argmin(v)]),
    }
    return stats
