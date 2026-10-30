"""Depth comparison semantics, tested without the network or the model (synthetic inputs)."""
import numpy as np
from scipy import ndimage

from core.reconstruction.compare import compare_depth_to_elevation, estimate_offset
from core.reconstruction.depth_adapter import SEMANTICS, normalise


def terrain(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 6) * 500


def test_semantics_text_forbids_metric_interpretation():
    assert "Not elevation" in SEMANTICS and "not metres" in SEMANTICS


def test_normalise_range_and_nan():
    a = np.array([[0.0, 1.0], [np.nan, 10.0]])
    n = normalise(a)
    assert np.isnan(n[1, 0]) and np.nanmin(n) >= 0 and np.nanmax(n) <= 1


def test_perfect_affine_relation_gives_high_correlation_and_good_oos():
    z = terrain()
    d = (z - z.min()) / (z.max() - z.min())
    img = np.ones_like(z)
    r = compare_depth_to_elevation(d, z, img, 20.0, 0.02)
    assert r["pearson_r_full"] > 0.999 and r["affine_calibration_oos"]["rmse_m"] < 1.0
    assert r["calibration_status"].startswith("uncalibrated")
    assert r["alignment"]["best_integer_shift_px"] == [0, 0]


def test_unrelated_prediction_scores_low():
    z, d = terrain(seed=1), normalise(terrain(seed=2))
    r = compare_depth_to_elevation(d, z, np.ones_like(z), 20.0, 0.02)
    assert abs(r["pearson_r_full"]) < 0.3
    assert r["affine_calibration_oos"]["skill_vs_mean"] < 0.2


def test_uncalibrated_grid_gets_no_metric_errors():
    z = terrain()
    r = compare_depth_to_elevation(normalise(z), z, np.ones_like(z), None, 0.02)
    assert r["affine_calibration_oos"] is None


def test_insufficient_overlap_and_shape_mismatch():
    z = terrain()
    img = np.full_like(z, np.nan)
    img[:10, :10] = 1
    assert compare_depth_to_elevation(normalise(z), z, img, 20.0, 0.02)["status"] == "insufficient_overlap"
    assert compare_depth_to_elevation(np.zeros((5, 5)), z, z, 20.0, 0.02)["status"] == "not_comparable"


def test_offset_estimator_recovers_shift():
    z = terrain(seed=3)
    shifted = np.roll(np.roll(z, -2, 0), 3, 1)
    dr, dc, r = estimate_offset(z, shifted, np.ones_like(z, bool))
    assert (dr, dc) == (2, -3) and r > 0.8  # < 1 because np.roll wraps the edges
