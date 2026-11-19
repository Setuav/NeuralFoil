"""
NACA 4-digit and 5-digit airfoils generated from their designation.

Both families share the NACA thickness distribution (open trailing edge, as in the original
definition) laid perpendicular to a camber line; they differ only in the camber line.

References:
    Abbott, I. H. and von Doenhoff, A. E. "Theory of Wing Sections", Dover, 1959, sections 6.4-6.6.
    Jacobs, E. N. and Pinkerton, R. M. "Tests in the Variable-Density Wind Tunnel of Related
    Airfoils Having the Maximum Camber Unusually Far Forward", NACA Report 537, 1935.
"""

import numpy as np

from .kulfan import cosine_spacing

_DEFAULT_N_POINTS_PER_SIDE = 200


def naca(designation: str, n_points_per_side: int = _DEFAULT_N_POINTS_PER_SIDE) -> np.ndarray:
    """
    Coordinates of a NACA 4-digit or 5-digit airfoil from its designation.

    Args:
        designation: For example "2412", "NACA 0012", "naca23012" or "23112". Letters, spaces and
            dashes are ignored; the remaining digits select the family.
        n_points_per_side: Number of cosine-spaced points on each surface.

    Returns: Coordinates in Selig order, shape (2 * n_points_per_side - 1, 2).
    """
    digits = "".join(c for c in designation if c.isdigit())
    if len(digits) == 4:
        return naca4(
            max_camber=int(digits[0]) / 100,
            max_camber_location=int(digits[1]) / 10,
            thickness=int(digits[2:]) / 100,
            n_points_per_side=n_points_per_side,
        )
    if len(digits) == 5:
        if digits[2] not in "01":
            raise ValueError(
                f"{designation!r}: the third digit of a NACA 5-digit airfoil must be 0 (standard) "
                f"or 1 (reflexed camber line)."
            )
        return naca5(
            design_lift_coefficient=int(digits[0]) * 0.15,
            max_camber_location=int(digits[1]) / 20,
            thickness=int(digits[3:]) / 100,
            reflexed=digits[2] == "1",
            n_points_per_side=n_points_per_side,
        )
    raise ValueError(f"{designation!r} is not a NACA 4-digit or 5-digit designation.")


def naca4(
    max_camber: float,
    max_camber_location: float,
    thickness: float,
    n_points_per_side: int = _DEFAULT_N_POINTS_PER_SIDE,
) -> np.ndarray:
    """
    Coordinates of a NACA 4-digit airfoil, e.g. NACA 2412: max_camber=0.02,
    max_camber_location=0.4, thickness=0.12. Matches `aerosandbox.Airfoil("naca2412")`.

    The camber line is two parabolas that meet at their common maximum.
    """
    m, p = max_camber, max_camber_location
    if p == 0:
        p = 0.5  # Any location works for a symmetric airfoil; avoids dividing by zero.
    x = cosine_spacing(n_points_per_side)
    camber = np.where(
        x <= p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    slope = np.where(
        x <= p,
        2 * m / p**2 * (p - x),
        2 * m / (1 - p) ** 2 * (p - x),
    )
    return _wrap_thickness(x, camber, slope, thickness)


def naca5(
    design_lift_coefficient: float,
    max_camber_location: float,
    thickness: float,
    reflexed: bool = False,
    n_points_per_side: int = _DEFAULT_N_POINTS_PER_SIDE,
) -> np.ndarray:
    """
    Coordinates of a NACA 5-digit airfoil, e.g. NACA 23012: design_lift_coefficient=0.3,
    max_camber_location=0.15, thickness=0.12; NACA 23112 is the same with reflexed=True.

    The camber line is a cubic ahead of x = m and a straight line (standard) or a second cubic
    (reflexed, giving zero pitching moment) behind it. The constants m, k1 and k2/k1 come from
    `naca5_camber_constants()`.
    """
    m, k1, r = naca5_camber_constants(design_lift_coefficient, max_camber_location, reflexed)
    x = cosine_spacing(n_points_per_side)
    camber, slope = _naca5_camber(x, m, k1, r, reflexed)
    return _wrap_thickness(x, camber, slope, thickness)


def naca5_camber_constants(
    design_lift_coefficient: float,
    max_camber_location: float,
    reflexed: bool = False,
) -> tuple[float, float, float]:
    """
    Camber line constants (m, k1, k2/k1) of a NACA 5-digit airfoil, computed from their definitions
    instead of read from a table, so that any digits work:

    - The camber line has its maximum at x = max_camber_location. For the standard line this gives
      m in closed form; for the reflexed line it gives k2/k1 as a function of m.
    - The reflexed line has zero pitching moment about the quarter chord in thin-airfoil theory,
      which fixes m.
    - k1 scales the line to the design (ideal) lift coefficient, pi * A1 in thin-airfoil theory.

    For a design lift coefficient of 0.3, these reproduce the tabulated values of Abbott and
    von Doenhoff, e.g. m = 0.2025, k1 = 15.957 for NACA 230xx.

    Returns: (m, k1, k2/k1); k2/k1 is 0 for the standard camber line.
    """
    p = max_camber_location
    if not 0 < p < 0.5:
        raise ValueError(f"max_camber_location must be between 0 and 0.5, got {p}.")
    if reflexed:
        m = _bisect(lambda m: _a1_minus_a2(m, _reflex_ratio(m, p), True), p, 1 - 1e-9)
        r = _reflex_ratio(m, p)
    else:
        m = _bisect(lambda m: m * (1 - (m / 3) ** 0.5) - p, p, 1.0)
        r = 0.0
    k1 = design_lift_coefficient / (np.pi * _fourier_coefficient(m, 1.0, r, reflexed, 1))
    return float(m), float(k1), float(r)


def _naca5_camber(x, m, k1, r, reflexed):
    """Camber line and its slope."""
    if reflexed:
        front = k1 / 6 * ((x - m) ** 3 - r * (1 - m) ** 3 * x - m**3 * x + m**3)
        back = k1 / 6 * (r * (x - m) ** 3 - r * (1 - m) ** 3 * x - m**3 * x + m**3)
        front_slope = k1 / 6 * (3 * (x - m) ** 2 - r * (1 - m) ** 3 - m**3)
        back_slope = k1 / 6 * (3 * r * (x - m) ** 2 - r * (1 - m) ** 3 - m**3)
    else:
        front = k1 / 6 * (x**3 - 3 * m * x**2 + m**2 * (3 - m) * x)
        back = k1 / 6 * m**3 * (1 - x)
        front_slope = k1 / 6 * (3 * x**2 - 6 * m * x + m**2 * (3 - m))
        back_slope = -k1 / 6 * m**3 * np.ones_like(x)
    return np.where(x < m, front, back), np.where(x < m, front_slope, back_slope)


def _reflex_ratio(m: float, p: float) -> float:
    """k2/k1 that puts the maximum of the reflexed camber line at x = p."""
    return (3 * (m - p) ** 2 - m**3) / (1 - m) ** 3


def _a1_minus_a2(m, r, reflexed):
    """Proportional to the thin-airfoil pitching moment about the quarter chord."""
    return _fourier_coefficient(m, 1.0, r, reflexed, 1) - _fourier_coefficient(m, 1.0, r, reflexed, 2)


def _fourier_coefficient(m, k1, r, reflexed, n, n_nodes: int = 64) -> float:
    """
    Thin-airfoil coefficient A_n = 2/pi * integral over theta in [0, pi] of
    dyc/dx * cos(n theta), with x = (1 - cos theta) / 2. Gauss-Legendre on each side of x = m,
    where the slope is polynomial in cos(theta).
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    theta_m = np.arccos(1 - 2 * m)
    total = 0.0
    for a, b in ((0.0, theta_m), (theta_m, np.pi)):
        theta = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        x = (1 - np.cos(theta)) / 2
        _, slope = _naca5_camber(x, m, k1, r, reflexed)
        total += 0.5 * (b - a) * np.sum(weights * slope * np.cos(n * theta))
    return 2 / np.pi * total


def _bisect(f, low: float, high: float, tolerance: float = 1e-14) -> float:
    f_low, f_high = f(low), f(high)
    if np.sign(f_low) == np.sign(f_high):
        raise ValueError("No NACA 5-digit camber line exists for these digits.")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        f_middle = f(middle)
        if np.sign(f_middle) == np.sign(f_low):
            low, f_low = middle, f_middle
        else:
            high = middle
    return 0.5 * (low + high)


def _wrap_thickness(x, camber, slope, thickness) -> np.ndarray:
    """Lays the NACA thickness distribution perpendicular to the camber line; Selig order."""
    half_thickness = (
        5
        * thickness
        * (
            +0.2969 * x**0.5
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4  # The original definition, with an open trailing edge.
        )
    )
    theta = np.arctan(slope)
    x_upper = x - half_thickness * np.sin(theta)
    x_lower = x + half_thickness * np.sin(theta)
    y_upper = camber + half_thickness * np.cos(theta)
    y_lower = camber - half_thickness * np.cos(theta)
    return np.stack(
        (
            np.concatenate((x_upper[::-1], x_lower[1:])),
            np.concatenate((y_upper[::-1], y_lower[1:])),
        ),
        axis=1,
    )
