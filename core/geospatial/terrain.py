"""Terrain derivatives from *calibrated* elevation rasters.

All functions are NaN-aware: NaN marks no-data and propagates so that a cell
whose neighbourhood touches a gap is itself unknown (never silently "flat").
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

HAZARD_UNKNOWN, HAZARD_LOW, HAZARD_MODERATE, HAZARD_HIGH, HAZARD_BLOCKED = 0, 1, 2, 3, 4
HAZARD_LEGEND = {
    HAZARD_UNKNOWN: {"label": "Unknown / insufficient evidence", "color": "#8a8f98"},
    HAZARD_LOW: {"label": "Low", "color": "#2f9e6f"},
    HAZARD_MODERATE: {"label": "Moderate", "color": "#d9b43a"},
    HAZARD_HIGH: {"label": "High", "color": "#e0752d"},
    HAZARD_BLOCKED: {"label": "Blocked (impassable)", "color": "#c22f3d"},
}


def block_reduce_mean(a: np.ndarray, factor: int, min_valid_fraction: float = 0.75) -> np.ndarray:
    """Block-average by an integer factor, ignoring NaN.

    A coarse cell is NaN unless at least ``min_valid_fraction`` of its
    ``factor x factor`` source cells are valid. Trailing rows/cols that do not
    fill a whole block are dropped (the grid transform must be updated to
    match, see :meth:`GridSpec.downsampled`).
    """
    if factor == 1:
        return a.astype(np.float32, copy=True)
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    blk = a[:h, :w].reshape(h // factor, factor, w // factor, factor).astype(np.float64)
    valid = np.isfinite(blk)
    n = valid.sum(axis=(1, 3))
    s = np.where(valid, blk, 0.0).sum(axis=(1, 3))
    out = np.full(n.shape, np.nan)
    ok = n >= min_valid_fraction * factor * factor
    out[ok] = s[ok] / n[ok]
    return out.astype(np.float32)


def block_reduce_mode(a: np.ndarray, factor: int) -> np.ndarray:
    """Block-reduce a binary {0,1,NaN} mask: fraction of valid sub-cells that are 1."""
    return block_reduce_mean(a, factor, min_valid_fraction=0.5)


def slope_degrees(z: np.ndarray, dx: float, dy: float | None = None) -> np.ndarray:
    """Slope (degrees) using Horn's (1981) 3x3 finite-difference operator.

    ``dx``/``dy`` are ground distances (m) between cell centres; ``z`` in m.
    Edge cells and cells adjacent to NaN are NaN.
    """
    dy = dx if dy is None else dy
    zp = np.pad(z.astype(np.float64), 1, constant_values=np.nan)
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
    out = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
    out[~np.isfinite(z)] = np.nan  # Horn's stencil ignores the centre; a no-data cell must stay unknown
    return out


def aspect_degrees(z: np.ndarray, dx: float) -> np.ndarray:
    """Downslope direction, degrees clockwise from grid-up (+y). Illustrative use only."""
    zp = np.pad(z.astype(np.float64), 1, constant_values=np.nan)
    dzdx = ((zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])) / (8 * dx)
    dzdy_up = ((zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:]) - (zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:])) / (8 * dx)
    return (np.degrees(np.arctan2(-dzdx, -dzdy_up)) % 360).astype(np.float32)


def roughness_plane_rms(z: np.ndarray, window: int, dx: float) -> np.ndarray:
    """RMS residual (m) of a least-squares plane fitted in a ``window x window`` neighbourhood.

    Removing the local plane separates roughness from regional slope. For a
    complete window with centred coordinates (u, v):

        RSS = sum(z^2) - n*mean(z)^2 - (sum(u z))^2 / sum(u^2) - (sum(v z))^2 / sum(v^2)

    computed for every cell with separable correlations. Any NaN in the window
    yields NaN. Scale: the result depends on ``window * dx`` (baseline length),
    which must be reported alongside the layer.
    """
    if window % 2 != 1 or window < 3:
        raise ValueError("window must be an odd integer >= 3")
    k = window // 2
    zz = z.astype(np.float64)
    nanmask = ~np.isfinite(zz)
    zz = np.where(nanmask, 0.0, zz)
    ones = np.ones((window, window))
    u = np.tile(np.arange(-k, k + 1, dtype=float) * dx, (window, 1))
    v = u.T.copy()
    n = window * window
    s_z = ndimage.correlate(zz, ones, mode="constant", cval=0.0)
    s_zz = ndimage.correlate(zz * zz, ones, mode="constant", cval=0.0)
    s_uz = ndimage.correlate(zz, u, mode="constant", cval=0.0)
    s_vz = ndimage.correlate(zz, v, mode="constant", cval=0.0)
    suu = float((u * u).sum())
    rss = s_zz - s_z * s_z / n - s_uz * s_uz / suu - s_vz * s_vz / suu
    rms = np.sqrt(np.clip(rss, 0.0, None) / n)
    bad = ndimage.maximum_filter(nanmask.astype(np.uint8), size=window, mode="constant", cval=1) > 0
    rms[bad] = np.nan
    return rms.astype(np.float32)


def hillshade(z: np.ndarray, dx: float, azimuth_deg: float = 315.0, altitude_deg: float = 30.0) -> np.ndarray:
    """Lambertian hillshade in [0, 1] (illustrative only; not a lighting simulation)."""
    zp = np.pad(z.astype(np.float64), 1, mode="edge")
    zp = np.where(np.isfinite(zp), zp, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    dzdx = ((zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])) / (8 * dx)
    dzdy = ((zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:]) - (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:])) / (8 * dx)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(hs, 0, 1).astype(np.float32)
    out[~np.isfinite(z)] = np.nan
    return out


def classify_hazard(slope: np.ndarray | None, rough: np.ndarray | None, hazard_cfg: dict,
                    uncertain: np.ndarray | None = None) -> np.ndarray:
    """Combine slope and roughness into ordinal hazard classes (see ``HAZARD_LEGEND``).

    The class is the *worst* of the slope class and roughness class. Cells
    lacking either required input are ``HAZARD_UNKNOWN`` — unknown is never safe.
    Cells flagged ``uncertain`` (e.g. large DEM error) are raised by one class.
    """
    if slope is None:
        shape = rough.shape if rough is not None else None
        if shape is None:
            raise ValueError("need at least one input")
        return np.full(shape, HAZARD_UNKNOWN, dtype=np.uint8)
    e1, e2, e3 = hazard_cfg["slope_classes_deg"]
    s_cls = np.full(slope.shape, HAZARD_UNKNOWN, dtype=np.uint8)
    fin = np.isfinite(slope)
    s_cls[fin & (slope < e1)] = HAZARD_LOW
    s_cls[fin & (slope >= e1) & (slope < e2)] = HAZARD_MODERATE
    s_cls[fin & (slope >= e2) & (slope < e3)] = HAZARD_HIGH
    s_cls[fin & (slope >= hazard_cfg["slope_blocked_deg"])] = HAZARD_BLOCKED
    out = s_cls
    if rough is not None:
        r_cls = np.full(rough.shape, HAZARD_UNKNOWN, dtype=np.uint8)
        rf = np.isfinite(rough)
        r_cls[rf] = HAZARD_LOW
        r_cls[rf & (rough >= hazard_cfg["roughness_high_m"])] = HAZARD_HIGH
        r_cls[rf & (rough >= hazard_cfg["roughness_blocked_m"])] = HAZARD_BLOCKED
        out = np.where((s_cls == HAZARD_UNKNOWN) | (r_cls == HAZARD_UNKNOWN), HAZARD_UNKNOWN,
                       np.maximum(s_cls, r_cls)).astype(np.uint8)
    if uncertain is not None:
        bump = uncertain & (out != HAZARD_UNKNOWN) & (out < HAZARD_BLOCKED)
        out = out.copy()
        out[bump] += 1
    return out
