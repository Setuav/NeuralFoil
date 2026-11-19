"""
Tests that NeuralFoil runs on NumPy alone, and that its own geometry entry points (NACA designations,
Selig and Lednicer files) agree with the coordinate entry point.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

import neuralfoil as nf
from neuralfoil import geometry

KULFAN = dict(
    upper_weights=[0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20],
    lower_weights=[-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10],
    leading_edge_weight=0.15,
    TE_thickness=0.002,
)
ALPHA = np.array([-4.0, 0.0, 4.0, 8.0, 12.0])
SCALAR_KEYS = ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]

_WITHOUT_AEROSANDBOX = """
import json, sys
sys.modules["aerosandbox"] = None  # As if AeroSandbox were not installed.
import numpy as np
import neuralfoil as nf
kulfan = {key: np.asarray(value) for key, value in json.loads(sys.argv[1]).items()}
aero = nf.get_aero_from_kulfan_parameters(kulfan, alpha=np.array(json.loads(sys.argv[2])), Re=2e5)
nf.get_aero_from_naca("23012", alpha=4.0, Re=2e5)
print(json.dumps({
    "loaded": sorted(m for m in ("aerosandbox", "casadi", "scipy") if sys.modules.get(m) is not None),
    "aero": {key: aero[key].tolist() for key in %r},
}))
""" % SCALAR_KEYS


def _run_without_aerosandbox() -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_AEROSANDBOX, json.dumps(KULFAN), json.dumps(ALPHA.tolist())],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_runs_without_loading_aerosandbox_casadi_or_scipy():
    assert _run_without_aerosandbox()["loaded"] == []


def test_numpy_backend_gives_the_same_numbers_as_the_default_backend():
    """With AeroSandbox installed, this compares its backend against plain NumPy, bit for bit."""
    kulfan = {key: np.asarray(value) for key, value in KULFAN.items()}
    expected = nf.get_aero_from_kulfan_parameters(kulfan, alpha=ALPHA, Re=2e5)

    actual = _run_without_aerosandbox()["aero"]

    for key in SCALAR_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("designation", ["0012", "2412", "23012", "23112"])
def test_naca_entry_point_matches_coordinates(designation):
    aero_naca = nf.get_aero_from_naca(designation, alpha=ALPHA, Re=5e5)
    aero_coordinates = nf.get_aero_from_coordinates(geometry.naca(designation), alpha=ALPHA, Re=5e5)

    for key in SCALAR_KEYS:
        np.testing.assert_array_equal(aero_naca[key], aero_coordinates[key], err_msg=key)


def test_naca_entry_point_matches_aerosandbox_airfoil():
    asb = pytest.importorskip("aerosandbox")

    aero_naca = nf.get_aero_from_naca("NACA 4412", alpha=ALPHA, Re=5e5)
    aero_airfoil = nf.get_aero_from_airfoil(asb.Airfoil("naca4412"), alpha=ALPHA, Re=5e5)

    for key in SCALAR_KEYS:
        np.testing.assert_allclose(aero_naca[key], aero_airfoil[key], rtol=1e-10, err_msg=key)


def test_lednicer_and_selig_files_give_the_same_answer(tmp_path):
    coordinates = geometry.naca("2412", n_points_per_side=61)
    le = np.argmin(coordinates[:, 0])
    upper, lower = coordinates[: le + 1][::-1], coordinates[le:]
    line = lambda point: f"{point[0]:.10f} {point[1]:.10f}"  # noqa: E731

    selig = tmp_path / "selig.dat"
    selig.write_text("NACA 2412\n" + "\n".join(map(line, coordinates)) + "\n")
    lednicer = tmp_path / "lednicer.dat"
    lednicer.write_text(
        f"NACA 2412\n{len(upper)}. {len(lower)}.\n\n"
        + "\n".join(map(line, upper))
        + "\n\n"
        + "\n".join(map(line, lower))
        + "\n"
    )

    aero_selig = nf.get_aero_from_dat_file(selig, alpha=ALPHA, Re=5e5)
    aero_lednicer = nf.get_aero_from_dat_file(lednicer, alpha=ALPHA, Re=5e5)

    for key in SCALAR_KEYS:
        np.testing.assert_array_equal(aero_lednicer[key], aero_selig[key], err_msg=key)
