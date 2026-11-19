"""Normalizing airfoil coordinates to a unit chord from (0, 0) to (1, 0)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalization:
    """
    Coordinates moved to a unit chord, and the changes that were needed to get there.

    Attributes:
        coordinates: Normalized coordinates, leading edge at (0, 0), trailing edge at (1, 0).
        x_translation: Translation of the leading edge in x to reach the origin.
        y_translation: Translation of the leading edge in y to reach the origin.
        scale_factor: Scaling applied to reach a unit chord (>1 if the airfoil had to grow).
        rotation_angle: Counter-clockwise rotation about the leading edge, in degrees.
    """

    coordinates: np.ndarray
    x_translation: float
    y_translation: float
    scale_factor: float
    rotation_angle: float


def normalize(coordinates: np.ndarray) -> Normalization:
    """
    Moves airfoil coordinates to a unit chord from (0, 0) to (1, 0).

    The trailing edge is the midpoint of the first and last points; the leading edge is the point
    farthest from it. The airfoil is translated so the leading edge is at the origin, scaled to a
    unit chord, and rotated about the leading edge so the trailing edge lies on the x-axis.

    This matches `aerosandbox.Airfoil.normalize()`.

    Args:
        coordinates: Airfoil coordinates in Selig order, shape (N_points, 2).

    Returns: A `Normalization`.
    """
    coordinates = np.array(coordinates, dtype=float)
    x, y = coordinates[:, 0], coordinates[:, 1]

    x_te = (x[0] + x[-1]) / 2
    y_te = (y[0] + y[-1]) / 2
    distance_to_te = ((x - x_te) ** 2 + (y - y_te) ** 2) ** 0.5
    le_index = np.argmax(distance_to_te)

    x_translation = -x[le_index]
    y_translation = -y[le_index]
    scale_factor = 1 / distance_to_te[le_index]
    coordinates = (coordinates + np.array([x_translation, y_translation])) * scale_factor

    x_te = (coordinates[0, 0] + coordinates[-1, 0]) / 2
    y_te = (coordinates[0, 1] + coordinates[-1, 1]) / 2
    rotation_angle = -np.arctan2(y_te, x_te)
    sin, cos = np.sin(rotation_angle), np.cos(rotation_angle)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    coordinates = (rotation_matrix @ coordinates.T).T

    return Normalization(
        coordinates=coordinates,
        x_translation=float(x_translation),
        y_translation=float(y_translation),
        scale_factor=float(scale_factor),
        rotation_angle=float(np.degrees(rotation_angle)),
    )
