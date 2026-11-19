"""
Array backend for inference.

With AeroSandbox installed, inference runs on `aerosandbox.numpy`, which also accepts CasADi
symbolic inputs (for use inside AeroSandbox optimization problems). Without it, plain NumPy is
used, with the few AeroSandbox helpers that inference needs re-implemented identically below.
"""

try:
    import aerosandbox.numpy as np
    from aerosandbox.numpy import cosd, length, sind, swish
except ModuleNotFoundError as error:
    if error.name is None or error.name.split(".")[0] != "aerosandbox":
        raise  # AeroSandbox is installed, but one of its own dependencies is missing.

    import numpy as np

    _DEG2RAD = np.pi / 180.0

    def sind(x):
        """Sine of an angle given in degrees."""
        return np.sin(np.asarray(x * _DEG2RAD))

    def cosd(x):
        """Cosine of an angle given in degrees."""
        return np.cos(np.asarray(x * _DEG2RAD))

    def swish(x, beta: float = 1.0):
        """Swish activation, x / (1 + exp(-beta x))."""
        return x / (1 + np.exp(-beta * x))

    def length(a) -> int:
        """Length of an array; 1 for a scalar."""
        try:
            return len(a)
        except TypeError:
            return 1


__all__ = ["np", "sind", "cosd", "swish", "length"]
