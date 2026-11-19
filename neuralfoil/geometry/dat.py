"""Reading airfoil coordinate files (`*.dat`) in Selig or Lednicer format."""

from os import PathLike

import numpy as np


def read_dat(filename: str | PathLike) -> np.ndarray:
    """
    Reads an airfoil coordinate file in Selig or Lednicer format.

    - Selig: an optional name line, then one `x y` pair per line, from the upper-surface trailing
      edge, around the leading edge, to the lower-surface trailing edge.
    - Lednicer: a name line, a line with the number of upper- and lower-surface points (e.g.
      `61. 61.`), then the upper surface and the lower surface, each from the leading edge to the
      trailing edge, usually separated by blank lines.

    Lines that are not a pair of numbers (names, comments, blank lines) are skipped; values may be
    separated by whitespace, commas, or semicolons.

    Args:
        filename: Path to the file.

    Returns: The coordinates in Selig order, as an array of shape (N_points, 2).
    """
    pairs = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            pair = _parse_pair(line)
            if pair is not None:
                pairs.append(pair)
    if not pairs:
        raise ValueError(f"{filename}: no coordinate pairs found.")

    n_upper, n_lower = pairs[0]
    if n_upper > 1 and n_lower > 1:  # A Lednicer header: point counts, not coordinates.
        return _lednicer_to_selig(pairs[1:], int(n_upper), int(n_lower), filename)
    return np.array(pairs)


def _parse_pair(line: str) -> tuple[float, float] | None:
    tokens = line.replace(",", " ").replace(";", " ").split()
    if len(tokens) != 2:
        return None
    try:
        return float(tokens[0]), float(tokens[1])
    except ValueError:
        return None


def _lednicer_to_selig(
    pairs: list[tuple[float, float]], n_upper: int, n_lower: int, filename
) -> np.ndarray:
    if len(pairs) != n_upper + n_lower:
        raise ValueError(
            f"{filename}: the Lednicer header gives {n_upper} + {n_lower} points, "
            f"but the file has {len(pairs)}."
        )
    upper = np.array(pairs[:n_upper])
    lower = np.array(pairs[n_upper:])
    if np.array_equal(upper[0], lower[0]):  # The leading edge appears in both surfaces.
        lower = lower[1:]
    return np.concatenate([upper[::-1], lower])
