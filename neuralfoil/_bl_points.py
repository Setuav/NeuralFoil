"""
Where boundary layer quantities are sampled along each airfoil surface.

Shared by inference (`neuralfoil.main`) and training (`neuralfoil._basic_data_type.Data`), so that
both always agree. NumPy only, so that inference does not need AeroSandbox.
"""

import numpy as np

N: int = 32
"""Number of boundary layer evaluation points per surface."""


def compute_optimal_x_points(n_points: int) -> np.ndarray:
    """
    Compute x-coordinates for the boundary layer evaluation points ("sensors").

    Creates evenly spaced midpoints between 0 and 1, avoiding the exact edges.
    (Uniform spacing is used for simplicity; see the compressed-sensing study in
    `training/optimal_sensor_placement/` for non-uniform alternatives considered.)

    Args:
        n_points: Number of points to generate.

    Returns:
        Array of x-coordinates of shape (n_points,).
    """
    s = np.linspace(0, 1, n_points + 1)
    return (s[1:] + s[:-1]) / 2


bl_x_points: np.ndarray = compute_optimal_x_points(n_points=N)
"""x/c of the boundary layer evaluation points."""
