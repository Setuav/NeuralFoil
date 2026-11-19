"""
Airfoil geometry tools that NeuralFoil needs, in plain NumPy: reading coordinate files, generating
NACA airfoils, normalizing coordinates, and fitting Kulfan (CST) parameters.

All coordinates are (N, 2) arrays in Selig order: from the upper-surface trailing edge, around the
leading edge, to the lower-surface trailing edge.
"""

from neuralfoil.geometry.dat import read_dat
from neuralfoil.geometry.kulfan import get_kulfan_coordinates, get_kulfan_parameters
from neuralfoil.geometry.naca_airfoils import naca, naca4, naca5, naca5_camber_constants
from neuralfoil.geometry.normalization import Normalization, normalize

__all__ = [
    "read_dat",
    "naca",
    "naca4",
    "naca5",
    "naca5_camber_constants",
    "normalize",
    "Normalization",
    "get_kulfan_parameters",
    "get_kulfan_coordinates",
]
