"""Optional feature detection — baseline only, unvalidated.

Boulders relevant to rover safety are ~0.3–10 m across. At 20–40 m/px they
are sub-pixel and cannot be detected; :func:`boulder_detection_feasibility`
says so explicitly. What *can* be found on a DEM at this scale are compact
topographic anomalies (small craters, mounds). :func:`detect_topographic_anomalies`
is a transparent Laplacian-of-Gaussian baseline with no training and no
evaluation set — its output is labelled "candidate anomalies (unvalidated)".
Manual annotations are stored separately and never mixed with detector output.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

MIN_BOULDER_RES_M = 1.0  # need ~2+ px across a 2 m boulder


def boulder_detection_feasibility(res_m: float | None) -> dict[str, Any]:
    if res_m is None:
        return {"feasible": False, "reason": "uncalibrated grid: object sizes unknown"}
    if res_m > MIN_BOULDER_RES_M:
        return {"feasible": False, "reason": f"grid is {res_m:.0f} m/px; rover-scale boulders (<10 m) are sub-pixel. "
                "Requires native-resolution NAC/ShadowCam imagery (~0.5–2 m/px) and a labelled training/evaluation set."}
    return {"feasible": True, "reason": "resolution adequate; a labelled dataset is still required before any detector is trusted"}


def detect_topographic_anomalies(z: np.ndarray, res_m: float, sigma_px: float = 2.0, k: float = 4.0,
                                 max_features: int = 200) -> list[dict[str, Any]]:
    """Return compact pits/mounds where |LoG| exceeds k robust-sigma. Unvalidated baseline."""
    fin = np.isfinite(z)
    if fin.sum() < 100:
        return []
    zf = np.where(fin, z, np.nanmedian(z))
    log = -ndimage.gaussian_laplace(zf, sigma_px) * sigma_px ** 2
    log[~ndimage.binary_erosion(fin, iterations=int(3 * sigma_px))] = 0
    mad = np.median(np.abs(log[fin] - np.median(log[fin]))) * 1.4826 or 1e-9
    feats = []
    for sign, kind in ((1, "mound"), (-1, "pit")):
        resp = sign * log
        peaks = (resp == ndimage.maximum_filter(resp, size=int(4 * sigma_px) + 1)) & (resp > k * mad)
        for r, c in zip(*np.nonzero(peaks)):
            feats.append({"row": int(r), "col": int(c), "kind": kind, "strength_sigma": float(resp[r, c] / mad),
                          "scale_m": float(2 * 1.414 * sigma_px * res_m)})
    feats.sort(key=lambda f: -f["strength_sigma"])
    return feats[:max_features]
