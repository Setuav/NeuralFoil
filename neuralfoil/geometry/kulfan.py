"""
Kulfan (CST) airfoil parameterization, "CST with leading-edge modification" as in AeroSandbox.

Each surface is y = C(x) S(x) +/- x t / 2 + w_LE x (1 - x)^(n + 0.5), where C(x) = x^N1 (1 - x)^N2
is the class function, S(x) a weighted sum of Bernstein polynomials of order n - 1 (n weights per
surface), t the trailing-edge thickness and w_LE the leading-edge weight.

References:
    Kulfan, B. "Universal Parametric Geometry Representation Method", Journal of Aircraft, 2008.
    Kulfan, B. "Modification of CST Airfoil Representation Methodology", 2020.
"""

from math import comb

import numpy as np

from neuralfoil.geometry.normalization import normalize


def get_kulfan_parameters(
    coordinates: np.ndarray,
    n_weights_per_side: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
    normalize_coordinates: bool = True,
    use_leading_edge_modification: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Fits Kulfan (CST) parameters to airfoil coordinates by linear least squares.

    The surface equation is linear in all unknowns (the weights of both surfaces, the leading-edge
    weight and the trailing-edge thickness), so one least-squares solve over the given points finds
    them. If the fitted trailing-edge thickness is negative, it is fixed at zero and the rest refit.
    Points up to the one with the smallest x belong to the upper surface.

    This matches `aerosandbox.get_kulfan_parameters(..., method="least_squares")`.

    Args:
        coordinates: Airfoil coordinates in Selig order, shape (N_points, 2).
        n_weights_per_side: Number of CST weights per surface. NeuralFoil's networks expect 8.
        N1: Class function exponent at the leading edge (0.5 for a round leading edge).
        N2: Class function exponent at the trailing edge (1.0 for a sharp trailing edge).
        normalize_coordinates: Whether to first move the coordinates to a unit chord with
            `neuralfoil.geometry.normalize()`.
        use_leading_edge_modification: Whether to fit the leading-edge weight; otherwise it is 0.

    Returns: A dictionary with "lower_weights", "upper_weights", "leading_edge_weight" and
        "TE_thickness".
    """
    if normalize_coordinates:
        coordinates = normalize(coordinates).coordinates
    coordinates = np.asarray(coordinates, dtype=float)
    x, y = coordinates[:, 0], coordinates[:, 1]
    is_upper = (np.arange(len(x)) <= np.argmin(x))[:, None]

    surface = (_class_function(x, N1, N2) * _bernstein_basis(x, n_weights_per_side)).T
    columns = [
        np.where(is_upper, 0, surface),  # lower weights
        np.where(is_upper, surface, 0),  # upper weights
    ]
    if use_leading_edge_modification:
        columns.append(_leading_edge_mode(x, n_weights_per_side)[:, None])
    columns.append(np.where(is_upper[:, 0], x / 2, -x / 2)[:, None])  # trailing-edge thickness
    A = np.concatenate(columns, axis=1)

    solution = np.linalg.lstsq(A, y, rcond=None)[0]
    trailing_edge_thickness = solution[-1]
    if trailing_edge_thickness < 0:
        solution = np.append(np.linalg.lstsq(A[:, :-1], y, rcond=None)[0], 0.0)
        trailing_edge_thickness = 0.0

    n = n_weights_per_side
    return {
        "lower_weights": solution[:n],
        "upper_weights": solution[n : 2 * n],
        "leading_edge_weight": solution[2 * n] if use_leading_edge_modification else 0.0,
        "TE_thickness": trailing_edge_thickness,
    }


def get_kulfan_coordinates(
    lower_weights: np.ndarray,
    upper_weights: np.ndarray,
    leading_edge_weight: float = 0.0,
    TE_thickness: float = 0.0,
    n_points_per_side: int = 200,
    N1: float = 0.5,
    N2: float = 1.0,
) -> np.ndarray:
    """
    Airfoil coordinates from Kulfan (CST) parameters; the inverse of `get_kulfan_parameters()`.

    Points are cosine-spaced in x. This matches `aerosandbox.get_kulfan_coordinates()`.

    Returns: Coordinates in Selig order, shape (2 * n_points_per_side - 1, 2).
    """
    x = cosine_spacing(n_points_per_side)
    C = _class_function(x, N1, N2)
    y_lower = C * (np.asarray(lower_weights) @ _bernstein_basis(x, len(lower_weights)))
    y_upper = C * (np.asarray(upper_weights) @ _bernstein_basis(x, len(upper_weights)))
    y_lower -= x * TE_thickness / 2
    y_upper += x * TE_thickness / 2
    y_lower += leading_edge_weight * _leading_edge_mode(x, len(lower_weights))
    y_upper += leading_edge_weight * _leading_edge_mode(x, len(upper_weights))

    x_all = np.concatenate((x[::-1], x[1:]))
    y_all = np.concatenate((y_upper[::-1], y_lower[1:]))
    return np.stack((x_all, y_all), axis=1)


def cosine_spacing(n_points: int) -> np.ndarray:
    """`n_points` values from 0 to 1, clustered at both ends; matches `aerosandbox.numpy.cosspace`."""
    x = 0.5 + 0.5 * np.cos(np.linspace(np.pi, 0, n_points))
    x[0], x[-1] = 0.0, 1.0
    return x


def _class_function(x: np.ndarray, N1: float, N2: float) -> np.ndarray:
    return x**N1 * (1 - x) ** N2


def _bernstein_basis(x: np.ndarray, n_weights: int) -> np.ndarray:
    """Bernstein polynomials of order n_weights - 1 at x, shape (n_weights, len(x))."""
    order = n_weights - 1
    i = np.arange(n_weights)[:, None]
    K = np.array([comb(order, k) for k in range(n_weights)], dtype=float)[:, None]
    return K * x[None, :] ** i * (1 - x[None, :]) ** (order - i)


def _leading_edge_mode(x: np.ndarray, n_weights: int) -> np.ndarray:
    return x * np.maximum(1 - x, 0) ** (n_weights + 0.5)
