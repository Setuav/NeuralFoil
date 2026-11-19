from neuralfoil.main import (
    get_aero_from_kulfan_parameters,
    get_aero_from_airfoil,
    get_aero_from_coordinates,
    get_aero_from_dat_file,
    get_aero_from_naca,
    bl_x_points,
)
from neuralfoil import geometry

__all__ = [
    "get_aero_from_kulfan_parameters",
    "get_aero_from_airfoil",
    "get_aero_from_coordinates",
    "get_aero_from_dat_file",
    "get_aero_from_naca",
    "bl_x_points",
    "geometry",
]

try:
    from importlib.metadata import version

    __version__ = version("NeuralFoil")
except Exception:
    __version__ = "unknown"
