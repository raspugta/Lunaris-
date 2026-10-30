"""Traversal cost surface.

Per-cell cost multiplier ``c >= 1`` (cost per metre of travel):

    c = 1 + w_s (slope / slope_blocked)^2
          + w_r (roughness / roughness_blocked)
          + w_u [uncertain]
          + w_p [psr]

``c = inf`` (blocked) where slope >= slope_blocked, roughness >= roughness_blocked,
or the elevation is no-data (unknown is not safe). Missing *optional* inputs
(e.g. no PSR map) simply drop their term; missing *slope* makes every cell
unknown/blocked because traversability cannot be assessed.
The floor of 1.0 per unit distance keeps the octile heuristic admissible.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def build_cost_surface(slope: np.ndarray | None, rough: np.ndarray | None, uncertain: np.ndarray | None,
                       psr: np.ndarray | None, hazard_cfg: dict, nav_cfg: dict,
                       shape: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    w = nav_cfg["weights"]
    if slope is None:
        shp = shape or (rough.shape if rough is not None else None)
        return np.full(shp, np.inf, dtype=np.float32), {"terms": [], "reason": "no calibrated slope: traversability not assessable"}
    cost = np.ones(slope.shape, dtype=np.float64)
    terms = []
    sb = hazard_cfg["slope_blocked_deg"]
    s = np.where(np.isfinite(slope), slope, np.nan)
    cost += w["slope"] * np.clip(s / sb, 0, None) ** 2
    terms.append(f"slope: {w['slope']} * (slope/{sb} deg)^2")
    blocked = ~np.isfinite(s) | (s >= sb)
    if rough is not None:
        rb = hazard_cfg["roughness_blocked_m"]
        cost += w["roughness"] * np.where(np.isfinite(rough), rough / rb, 0)
        blocked |= ~np.isfinite(rough) | (rough >= rb)
        terms.append(f"roughness: {w['roughness']} * (rms/{rb} m)")
    if uncertain is not None:
        cost += w["uncertainty"] * uncertain.astype(np.float64)
        terms.append(f"uncertain elevation: +{w['uncertainty']}")
    if psr is not None:
        cost += w["psr"] * np.where(np.isfinite(psr), psr >= 0.5, 0)
        terms.append(f"inside PSR: +{w['psr']}")
    if nav_cfg.get("block_nodata", True):
        cost[blocked] = np.inf
    return cost.astype(np.float32), {"terms": terms, "blocked_fraction": float(blocked.mean()),
                                     "floor": 1.0, "formula": "c = 1 + " + " + ".join(terms)}
