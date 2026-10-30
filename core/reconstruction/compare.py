"""Compare monocular relative depth with LOLA reference elevation.

Because relative depth has no units, the primary metrics are *unit-free*
(Pearson r, Spearman rho) on the full field and on a high-pass (local relief)
version. A metric-unit RMSE is reported **only** after an affine calibration
fitted to LOLA, and only out-of-sample (fit on one spatial half, evaluate on
the other) so it is not an in-sample tautology. This calibration uses the
reference itself and is labelled as such: it measures how *usable* the
predicted structure would be if a few control heights were known; it is not
an independent accuracy score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, stats


def _hp(a: np.ndarray, sigma: float) -> np.ndarray:
    fill = np.where(np.isfinite(a), a, np.nanmean(a))
    return fill - ndimage.gaussian_filter(fill, sigma)


def estimate_offset(a: np.ndarray, b: np.ndarray, valid: np.ndarray, max_shift: int = 6) -> tuple[int, int, float]:
    """Integer (drow, dcol) maximising high-pass correlation of b shifted onto a."""
    ah, bh = _hp(a, 4), _hp(b, 4)
    best = (0, 0, -2.0)
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            bs = np.roll(np.roll(bh, dr, 0), dc, 1)
            vs = valid & np.roll(np.roll(valid, dr, 0), dc, 1)
            if vs.sum() < 500:
                continue
            r = np.corrcoef(ah[vs], bs[vs])[0, 1]
            if r > best[2]:
                best = (dr, dc, float(r))
    return best


def compare_depth_to_elevation(rel_depth: np.ndarray, elevation: np.ndarray, image: np.ndarray,
                               res_m: float | None, min_signal: float, dem_effres_m: float | None = None,
                               sample: int = 200_000, seed: int = 0) -> dict[str, Any]:
    """Return a dictionary of honest comparison diagnostics (see module docstring)."""
    out: dict[str, Any] = {"status": "ok", "notes": []}
    if rel_depth.shape != elevation.shape:
        return {"status": "not_comparable", "reason": "depth and elevation grids differ; co-registration required first"}
    sig = np.isfinite(image) & (image > min_signal * np.nanpercentile(image, 99))
    valid = np.isfinite(rel_depth) & np.isfinite(elevation) & sig
    out["coverage_fraction"] = float(valid.mean())
    out["image_footprint_fraction"] = float(np.isfinite(image).mean())
    out["low_signal_excluded_fraction"] = float((np.isfinite(image) & ~sig).mean())
    if valid.sum() < 1000:
        return out | {"status": "insufficient_overlap", "reason": f"only {int(valid.sum())} co-valid pixels"}
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid.ravel())
    if idx.size > sample:
        idx = rng.choice(idx, sample, replace=False)
    d, z = rel_depth.ravel()[idx], elevation.ravel()[idx]
    pr = float(np.corrcoef(d, z)[0, 1])
    sr = float(stats.spearmanr(d, z).statistic)
    out["pearson_r_full"] = pr
    out["spearman_rho_full"] = sr
    out["sign_convention"] = "larger depth-output ~ higher terrain" if pr >= 0 else "larger depth-output ~ lower terrain"
    sigma = 6.0
    dh, zh = _hp(np.where(valid, rel_depth, np.nan), sigma), _hp(np.where(valid, elevation, np.nan), sigma)
    inner = ndimage.binary_erosion(valid, iterations=int(2 * sigma))
    if inner.sum() > 1000:
        out["pearson_r_local_relief"] = float(np.corrcoef(dh[inner], zh[inner])[0, 1])
        out["local_relief_scale"] = f"Gaussian high-pass sigma={sigma:.0f} px" + (f" (~{sigma * res_m:.0f} m)" if res_m else "")
    dr, dc, r_off = estimate_offset(np.where(valid, elevation, np.nan), np.where(valid, rel_depth, np.nan), valid)
    out["alignment"] = {"best_integer_shift_px": [dr, dc], "hp_corr_at_best_shift": r_off,
                        "interpretation": "0,0 means no detectable mis-registration at pixel level" if (dr, dc) == (0, 0)
                        else "non-zero shift: residual mis-registration or model-induced displacement (shading bias)"}
    # Out-of-sample affine calibration: split by column median of valid pixels.
    rows, cols = np.nonzero(valid)
    cmid = np.median(cols)
    a_mask = cols < cmid
    da, za = rel_depth[rows[a_mask], cols[a_mask]], elevation[rows[a_mask], cols[a_mask]]
    db, zb = rel_depth[rows[~a_mask], cols[~a_mask]], elevation[rows[~a_mask], cols[~a_mask]]
    if len(da) > 500 and len(db) > 500 and res_m:
        A = np.c_[da, np.ones_like(da)]
        coef, *_ = np.linalg.lstsq(A, za, rcond=None)
        pred = coef[0] * db + coef[1]
        rmse = float(np.sqrt(np.mean((pred - zb) ** 2)))
        mae = float(np.mean(np.abs(pred - zb)))
        out["affine_calibration_oos"] = {
            "rmse_m": rmse, "mae_m": mae, "test_elev_std_m": float(np.std(zb)),
            "skill_vs_mean": float(1 - rmse / max(np.std(zb), 1e-9)),
            "label": "Out-of-sample after affine fit to LOLA on the other spatial half. "
                     "Calibration uses the reference; NOT an independent accuracy.",
        }
    else:
        out["affine_calibration_oos"] = None
        out["notes"].append("Metric errors not computed: grid not calibrated or too little overlap.")
    out["resolution"] = {"image_grid_m": res_m, "dem_effective_resolution_median_m": dem_effres_m}
    out["calibration_status"] = "uncalibrated (relative depth)"
    return out
