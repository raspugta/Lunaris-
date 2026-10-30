import math

import numpy as np
import pytest

from core.geospatial import terrain


def test_slope_of_plane(plane):
    s = terrain.slope_degrees(plane, 40.0)
    inner = s[1:-1, 1:-1]
    assert np.allclose(inner, math.degrees(math.atan(0.1)), atol=1e-4)
    # Edges have no full 3x3 stencil -> NaN, never silently flat.
    assert np.isnan(s[0]).all() and np.isnan(s[:, -1]).all()


def test_slope_nodata_propagates(plane):
    z = plane.copy()
    z[10, 10] = np.nan
    s = terrain.slope_degrees(z, 40.0)
    assert np.isnan(s[9:12, 9:12]).all()
    assert np.isfinite(s[20, 20])


def test_slope_scales_with_cell_size(plane):
    s40 = terrain.slope_degrees(plane, 40.0)[5, 5]
    s80 = terrain.slope_degrees(plane, 80.0)[5, 5]
    assert math.tan(math.radians(s40)) == pytest.approx(2 * math.tan(math.radians(s80)), rel=1e-6)


def test_roughness_zero_on_plane(plane):
    r = terrain.roughness_plane_rms(plane, 5, 40.0)
    assert np.nanmax(r[2:-2, 2:-2]) < 1e-3


def test_roughness_detects_noise_and_matches_known_value():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 2.0, (200, 200))
    r = terrain.roughness_plane_rms(noise, 7, 10.0)
    # Plane removal consumes 3 dof of 49: E[RMS^2] = sigma^2 (n-3)/n
    assert np.nanmedian(r) == pytest.approx(2.0 * math.sqrt(46 / 49), rel=0.05)


def test_roughness_nodata_window():
    z = np.zeros((30, 30), np.float32)
    z[15, 15] = np.nan
    r = terrain.roughness_plane_rms(z, 5, 10.0)
    assert np.isnan(r[13:18, 13:18]).all() and np.isfinite(r[5, 5])


def test_roughness_rejects_even_window():
    with pytest.raises(ValueError):
        terrain.roughness_plane_rms(np.zeros((10, 10)), 4, 1.0)


def test_block_reduce_mean_and_nodata_threshold():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = terrain.block_reduce_mean(a, 2)
    assert out.tolist() == [[2.5, 4.5], [10.5, 12.5]]
    a[0, 0] = a[0, 1] = np.nan  # 50 % valid < 75 % -> NaN
    out = terrain.block_reduce_mean(a, 2)
    assert np.isnan(out[0, 0]) and np.isfinite(out[1, 1])


def test_hazard_unknown_is_never_safe(cfg):
    slope = np.array([[5.0, np.nan, 12.0, 30.0]], np.float32)
    rough = np.array([[0.5, 0.5, np.nan, 0.5]], np.float32)
    h = terrain.classify_hazard(slope, rough, cfg["hazard"])
    assert h.tolist() == [[terrain.HAZARD_LOW, terrain.HAZARD_UNKNOWN, terrain.HAZARD_UNKNOWN, terrain.HAZARD_BLOCKED]]


def test_hazard_uncertain_raises_class(cfg):
    slope = np.array([[5.0, 5.0]], np.float32)
    rough = np.array([[0.5, 0.5]], np.float32)
    h = terrain.classify_hazard(slope, rough, cfg["hazard"], uncertain=np.array([[False, True]]))
    assert h[0, 1] == h[0, 0] + 1
