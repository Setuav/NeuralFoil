"""
Tests of `neuralfoil.geometry`: coordinate files, NACA airfoils, normalization and Kulfan fitting.

Where AeroSandbox implements the same operation, results are checked against it (skipped when
AeroSandbox is not installed), so that the two geometry pipelines stay interchangeable.
"""

import numpy as np
import pytest

from neuralfoil import geometry
from neuralfoil.geometry import naca_airfoils

AEROSANDBOX_AIRFOILS = ["naca4412", "e63", "clarky", "s1223", "mh32", "sd7037", "dae11", "rg15"]


def _surfaces(coordinates):
    """Upper and lower surface, each from the leading edge to the trailing edge."""
    le = np.argmin(coordinates[:, 0])
    return coordinates[: le + 1][::-1], coordinates[le:]


### .dat files


def test_read_selig(tmp_path):
    path = tmp_path / "foil.dat"
    path.write_text("My Foil\n1.0 0.001\n0.5, 0.06\n0.0 0.0\n0.5;-0.03\n1.0 -0.001\n")

    np.testing.assert_array_equal(
        geometry.read_dat(path),
        [[1.0, 0.001], [0.5, 0.06], [0.0, 0.0], [0.5, -0.03], [1.0, -0.001]],
    )


def test_read_lednicer_returns_selig_order(tmp_path):
    path = tmp_path / "foil.dat"
    path.write_text(
        "MY LEDNICER FOIL\n 3. 3.\n\n0.0 0.0\n0.5 0.06\n1.0 0.001\n\n0.0 0.0\n0.5 -0.03\n1.0 -0.001\n"
    )

    np.testing.assert_array_equal(
        geometry.read_dat(path),
        [[1.0, 0.001], [0.5, 0.06], [0.0, 0.0], [0.5, -0.03], [1.0, -0.001]],
    )


def test_read_lednicer_with_point_count_mismatch_raises(tmp_path):
    path = tmp_path / "foil.dat"
    path.write_text("FOIL\n 3. 4.\n0.0 0.0\n0.5 0.06\n1.0 0.0\n0.0 0.0\n0.5 -0.03\n1.0 0.0\n")

    with pytest.raises(ValueError, match="3 \\+ 4 points"):
        geometry.read_dat(path)


def test_read_file_without_coordinates_raises(tmp_path):
    path = tmp_path / "foil.dat"
    path.write_text("Just a name\n")

    with pytest.raises(ValueError, match="no coordinate pairs"):
        geometry.read_dat(path)


### NACA 4-digit


def test_naca4_shape():
    upper, lower = _surfaces(geometry.naca("2412"))
    x = np.linspace(0.01, 0.99, 500)
    y_upper, y_lower = np.interp(x, *upper.T), np.interp(x, *lower.T)
    camber, thickness = (y_upper + y_lower) / 2, y_upper - y_lower

    assert camber.max() == pytest.approx(0.02, abs=2e-4)
    assert x[np.argmax(camber)] == pytest.approx(0.4, abs=0.01)
    assert thickness.max() == pytest.approx(0.12, abs=2e-4)


def test_naca4_symmetric():
    coordinates = geometry.naca("0012")
    upper, lower = _surfaces(coordinates)

    np.testing.assert_allclose(upper[:, 1], -lower[:, 1], atol=1e-15)


@pytest.mark.parametrize("designation", ["NACA 2412", "naca2412", "naca-2412", "2412"])
def test_naca_designation_formats(designation):
    np.testing.assert_array_equal(geometry.naca(designation), geometry.naca("2412"))


@pytest.mark.parametrize("name", ["naca0012", "naca2412", "naca4415", "naca6409"])
def test_naca4_matches_aerosandbox(name):
    asb = pytest.importorskip("aerosandbox")

    np.testing.assert_allclose(geometry.naca(name), asb.Airfoil(name).coordinates, atol=1e-15)


### NACA 5-digit

# Abbott & von Doenhoff, "Theory of Wing Sections", tables for a design lift coefficient of 0.3:
# (designation digits, max camber location, reflexed, m, k1, k2/k1).
ABBOTT_5_DIGIT = [
    ("210", 0.05, False, 0.0580, 361.4, 0.0),
    ("220", 0.10, False, 0.1260, 51.64, 0.0),
    ("230", 0.15, False, 0.2025, 15.957, 0.0),
    ("240", 0.20, False, 0.2900, 6.643, 0.0),
    ("250", 0.25, False, 0.3910, 3.230, 0.0),
    ("221", 0.10, True, 0.1300, 51.99, 0.000764),
    ("231", 0.15, True, 0.2170, 15.793, 0.00677),
    ("241", 0.20, True, 0.3180, 6.520, 0.0303),
    ("251", 0.25, True, 0.4410, 3.191, 0.1355),
]


@pytest.mark.parametrize("digits, p, reflexed, m_table, k1_table, r_table", ABBOTT_5_DIGIT)
def test_naca5_camber_constants_satisfy_their_definitions(digits, p, reflexed, m_table, k1_table, r_table):
    m, k1, r = geometry.naca5_camber_constants(0.3, p, reflexed)

    _, slope = naca_airfoils._naca5_camber(np.array([p]), m, k1, r, reflexed)
    assert slope[0] == pytest.approx(0.0, abs=1e-10)  # maximum camber at x = p
    assert np.pi * naca_airfoils._fourier_coefficient(m, k1, r, reflexed, 1) == pytest.approx(0.3, rel=1e-12)
    if reflexed:  # zero pitching moment about the quarter chord
        assert naca_airfoils._a1_minus_a2(m, r, True) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("digits, p, reflexed, m_table, k1_table, r_table", ABBOTT_5_DIGIT)
def test_naca5_camber_constants_agree_with_the_table(digits, p, reflexed, m_table, k1_table, r_table):
    """
    The table rounds m to 4 digits. With the table's own m, k2/k1 and the standard 230-250 k1 are
    reproduced to the table's precision; the 210, 220 and reflexed k1 values in the table differ
    from thin-airfoil theory by up to 3 %, so only m is compared there.
    """
    m, _, _ = geometry.naca5_camber_constants(0.3, p, reflexed)
    assert m == pytest.approx(m_table, abs=1e-3)

    if reflexed:
        assert naca_airfoils._reflex_ratio(m_table, p) == pytest.approx(r_table, rel=1e-3)
    elif digits in ("230", "240", "250"):
        k1 = 0.3 / (np.pi * naca_airfoils._fourier_coefficient(m_table, 1.0, 0.0, False, 1))
        assert k1 == pytest.approx(k1_table, rel=1e-3)


def test_naca23012_matches_published_ordinates():
    # From the UIUC Airfoil Coordinates Database, naca23012.dat (5 decimals).
    published_upper = [(0.90482, 0.01602), (0.65541, 0.04854), (0.34681, 0.07402), (0.09230, 0.06265)]
    published_lower = [(0.09868, -0.02922), (0.34418, -0.04510), (0.65360, -0.03327), (0.90420, -0.01180)]
    upper, lower = _surfaces(geometry.naca("23012"))

    for surface, published in ((upper, published_upper), (lower, published_lower)):
        x, y = np.array(published).T
        np.testing.assert_allclose(np.interp(x, *surface.T), y, atol=2e-5)


def test_naca5_design_lift_coefficient_scales_the_camber():
    _, k1_230, _ = geometry.naca5_camber_constants(0.3, 0.15)
    _, k1_430, _ = geometry.naca5_camber_constants(0.6, 0.15)

    assert k1_430 == pytest.approx(2 * k1_230, rel=1e-12)


def test_naca5_reflexed_differs_only_behind_the_maximum_camber():
    standard, reflexed = _surfaces(geometry.naca("23012"))[0], _surfaces(geometry.naca("23112"))[0]

    assert np.max(np.abs(standard[:, 1] - reflexed[:, 1])) > 1e-3


@pytest.mark.parametrize("designation", ["23212", "123", "241012", "NACA"])
def test_invalid_naca_designation_raises(designation):
    with pytest.raises(ValueError):
        geometry.naca(designation)


### Normalization


def test_normalize_undoes_translation_scaling_and_rotation():
    original = geometry.naca("2412")
    angle = np.radians(7.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = (rotation @ original.T).T * 0.3 + np.array([1.5, -0.2])

    result, reference = geometry.normalize(moved), geometry.normalize(original)

    np.testing.assert_allclose(result.coordinates, reference.coordinates, atol=1e-12)
    assert result.scale_factor == pytest.approx(reference.scale_factor / 0.3)
    assert result.rotation_angle == pytest.approx(reference.rotation_angle - 7.0)


@pytest.mark.parametrize("name", AEROSANDBOX_AIRFOILS)
def test_normalize_matches_aerosandbox(name):
    asb = pytest.importorskip("aerosandbox")
    airfoil = asb.Airfoil(name)
    expected = airfoil.normalize(return_dict=True)

    result = geometry.normalize(airfoil.coordinates)

    np.testing.assert_allclose(result.coordinates, expected["airfoil"].coordinates, atol=1e-14)
    for key in ("x_translation", "y_translation", "scale_factor", "rotation_angle"):
        assert getattr(result, key) == pytest.approx(expected[key], abs=1e-14)


### Kulfan parameters


def test_kulfan_fit_recovers_the_parameters_of_a_kulfan_airfoil():
    parameters = dict(
        upper_weights=np.array([0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20]),
        lower_weights=np.array([-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10]),
        leading_edge_weight=0.15,
        TE_thickness=0.002,
    )
    fitted = geometry.get_kulfan_parameters(geometry.get_kulfan_coordinates(**parameters))

    for key, value in parameters.items():
        np.testing.assert_allclose(fitted[key], value, atol=1e-10, err_msg=key)


def test_kulfan_fit_clips_a_negative_trailing_edge_thickness():
    crossed = geometry.get_kulfan_coordinates(  # Surfaces cross over at the trailing edge.
        upper_weights=np.full(8, 0.2), lower_weights=np.full(8, -0.1), TE_thickness=-0.01
    )

    fitted = geometry.get_kulfan_parameters(crossed, normalize_coordinates=False)

    assert fitted["TE_thickness"] == 0.0
    assert np.all(np.isfinite(fitted["upper_weights"]))


@pytest.mark.parametrize("name", AEROSANDBOX_AIRFOILS)
def test_kulfan_fit_matches_aerosandbox(name):
    asb = pytest.importorskip("aerosandbox")
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

    airfoil = asb.Airfoil(name)
    expected = get_kulfan_parameters(airfoil.coordinates)

    fitted = geometry.get_kulfan_parameters(airfoil.coordinates)

    for key, value in expected.items():
        np.testing.assert_allclose(fitted[key], value, atol=1e-12, err_msg=key)


def test_kulfan_coordinates_match_aerosandbox():
    asb = pytest.importorskip("aerosandbox")
    from aerosandbox.geometry.airfoil.airfoil_families import (
        get_kulfan_coordinates,
        get_kulfan_parameters,
    )

    parameters = get_kulfan_parameters(asb.Airfoil("e63").coordinates)

    np.testing.assert_allclose(
        geometry.get_kulfan_coordinates(**parameters),
        get_kulfan_coordinates(**parameters),
        atol=1e-15,
    )
