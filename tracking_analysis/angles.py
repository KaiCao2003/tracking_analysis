import numpy as np


def unwrap_deg(angles_deg, cutoff_deg=180.0, axis=-1):
    """Unwrap angle array given in degrees.

    Parameters
    ----------
    angles_deg : array_like
        Angle values in degrees.
    cutoff_deg : float, optional
        Discontinuity threshold in degrees. Values above this trigger an
        unwrap step. Defaults to 180 degrees.
    axis : int, optional
        Axis along which to unwrap. Defaults to last axis.

    Returns
    -------
    ndarray
        Unwrapped angles in degrees.
    """
    angles_rad = np.deg2rad(angles_deg)
    unwrapped = np.unwrap(angles_rad, discont=np.deg2rad(cutoff_deg), axis=axis)
    return np.rad2deg(unwrapped)
