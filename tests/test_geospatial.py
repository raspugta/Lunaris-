"""CRS / pixel-world transforms checked against analytic control points."""
import math

import numpy as np
import pytest

from core.geospatial.grid import (GridSpec, MOON_RADIUS_M, max_scale_error, polar_stereo_south_forward,
                                  polar_stereo_south_inverse, scale_factor)


def test_pole_maps_to_origin():
    x, y = polar_stereo_south_forward(-90.0, 123.0)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6


@pytest.mark.parametrize("lat,lon", [(-89.0, 0.0), (-89.0, 90.0), (-88.5, 180.0), (-87.5, 270.0), (-89.9, 45.0)])
def test_control_points_analytic(lat, lon):
    # rho = 2R tan(pi/4 + lat/2); x = rho sin(lon), y = rho cos(lon)
    rho = 2 * MOON_RADIUS_M * math.tan(math.pi / 4 + math.radians(lat) / 2)
    x, y = polar_stereo_south_forward(lat, lon)
    assert float(x) == pytest.approx(rho * math.sin(math.radians(lon)), abs=1e-6)
    assert float(y) == pytest.approx(rho * math.cos(math.radians(lon)), abs=1e-6)


def test_one_degree_from_pole_is_about_30km():
    # arc length of 1 deg on R=1737.4 km is 30.32 km; stereographic rho is within 0.01 % of it
    x, y = polar_stereo_south_forward(-89.0, 0.0)
    assert float(np.hypot(x, y)) == pytest.approx(MOON_RADIUS_M * math.radians(1.0), rel=1e-4)


def test_forward_inverse_roundtrip():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-90, -80, 500)
    lon = rng.uniform(0, 360, 500)
    x, y = polar_stereo_south_forward(lat, lon)
    lat2, lon2 = polar_stereo_south_inverse(x, y)
    assert np.allclose(lat, lat2, atol=1e-9)
    ok = lat < -90 + 1e-7
    dlon = (lon - lon2 + 180) % 360 - 180
    assert np.allclose(dlon[~ok], 0, atol=1e-7)


def test_pixel_world_roundtrip_and_centres():
    g = GridSpec(100, 80, (20.0, 0.0, -1000.0, 0.0, -20.0, 800.0), polar_stereo_south=True, calibrated=True)
    x, y = g.pixel_to_world(0, 0)
    assert (float(x), float(y)) == (-990.0, 790.0)  # centre of top-left cell
    r, c = g.world_to_pixel(x, y)
    assert float(r) == pytest.approx(0.0) and float(c) == pytest.approx(0.0)
    rr, cc = np.meshgrid(np.arange(80), np.arange(100), indexing="ij")
    X, Y = g.pixel_to_world(rr, cc)
    r2, c2 = g.world_to_pixel(X, Y)
    assert np.allclose(r2, rr) and np.allclose(c2, cc)


def test_downsampled_grid_keeps_origin_and_scales_res():
    g = GridSpec(1600, 1600, (20.0, 0, -12000.0, 0, -20.0, 16000.0), calibrated=True)
    d = g.downsampled(2)
    assert d.width == 800 and d.res_m == 40.0 and d.bounds() == g.bounds()


def test_scale_error_small_near_pole():
    assert scale_factor(-90.0) == pytest.approx(1.0)
    g = GridSpec(1600, 1600, (20.0, 0, -12000.0, 0, -20.0, 16000.0), polar_stereo_south=True, calibrated=True)
    assert max_scale_error(g) < 1e-4


def test_uncalibrated_grid_has_no_resolution():
    assert GridSpec(10, 10, (1, 0, 0, 0, -1, 0)).res_m is None
