"""Route planning service: candidate route + rejected alternatives, independent of the UI."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.config import deep_merge
from core.geospatial.terrain import HAZARD_LEGEND
from core.navigation.astar import PlanningError, astar, bresenham, summarise_path
from core.navigation.cost import build_cost_surface

COST_SEMANTICS = ("Path cost = sum over moves of step length x mean per-metre cost multiplier of the two cells; "
                  "a value equal to the length means flat, certain, low-hazard terrain all the way. "
                  "Relative units — not energy, time or risk probability.")


def plan_route(an, start: tuple[int, int], goal: tuple[int, int], cfg: dict,
               nav_overrides: dict | None = None) -> dict[str, Any]:
    """Plan a hazard-weighted A* route and compare with distance-only A* and a direct line."""
    nav = deep_merge(cfg["navigation"], nav_overrides or {})
    A = an.arrays
    h, w = A["cost"].shape
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not (0 <= r < h and 0 <= c < w):
            raise PlanningError(f"{name} ({r}, {c}) is outside the analysis grid ({h} x {w} cells)")
    cost = A["cost"]
    if nav_overrides:
        unc = A.get("uncertain")
        cost, _ = build_cost_surface(A.get("slope"), A.get("roughness"), unc > 0.5 if unc is not None else None,
                                     A.get("psr"), cfg["hazard"], nav, shape=cost.shape)
    res = an.grid.res_m
    cs = res or 1.0
    common = dict(slope=A.get("slope"), hazard=A.get("hazard"), psr=A.get("psr"), elevation=A.get("elevation"),
                  hazard_legend=HAZARD_LEGEND)
    out: dict[str, Any] = {"start": list(start), "goal": list(goal), "label": "Simulated candidate route — not mission-certified",
                           "cost_semantics": COST_SEMANTICS, "weights": nav["weights"], "grid_res_m": res,
                           "distance_calibrated": res is not None,
                           "start_latlon": an.lat_lon(*start), "goal_latlon": an.lat_lon(*goal)}
    main = astar(cost, start, goal, cs)
    out["found"] = main.found
    out["expanded_nodes"] = main.expanded
    if not main.found:
        out["reason"] = main.reason
        for name, cell in (("start", start), ("goal", goal)):
            if not np.isfinite(cost[cell]):
                out[f"nearest_traversable_{name}"] = nearest_traversable(cost, cell)
        return out
    out["cells"] = [list(c) for c in main.cells]
    out["summary"] = summarise_path(main.cells, cost, res, **common)
    # Validation: route must stay on traversable cells and reproduce its cost.
    vals = cost[tuple(np.array(main.cells).T)]
    out["validation"] = {"all_cells_traversable": bool(np.isfinite(vals).all()),
                         "recomputed_cost_matches": bool(abs(out["summary"]["cost"] - main.cost) < 1e-6 * max(main.cost, 1))}
    # Alternative 1: shortest traversable path ignoring hazard weighting.
    flat = np.where(np.isfinite(cost), 1.0, np.inf).astype(np.float32)
    d_only = astar(flat, start, goal, cs)
    alts: dict[str, Any] = {}
    paths: dict[str, list] = {}
    if d_only.found:
        s = summarise_path(d_only.cells, cost, res, **common)
        s["why_rejected"] = (f"{(s['cost'] / out['summary']['cost'] - 1) * 100:.0f}% higher hazard-weighted cost"
                             if s["cost"] > out["summary"]["cost"] else "equal cost")
        alts["distance_only"] = s
        paths["distance_only"] = [list(c) for c in d_only.cells]
    line = bresenham(start, goal)
    s = summarise_path(line, cost, res, **common)
    s["why_rejected"] = (f"crosses {s['blocked_cells_crossed']} blocked/unknown cells" if s["blocked_cells_crossed"]
                         else "traversable but higher cost" if s["cost"] > out["summary"]["cost"] else "equivalent")
    if s["cost"] is None:
        s["cost_note"] = "impassable (infinite cost)"
    alts["direct_line"] = s
    paths["direct_line"] = [list(c) for c in line]
    out["alternatives"] = alts
    out["alternative_paths"] = paths
    return out


def nearest_traversable(cost: np.ndarray, cell: tuple[int, int], max_radius: int = 50) -> tuple[int, int] | None:
    """Closest finite-cost cell to ``cell`` (Euclidean, within max_radius), or None."""
    r, c = cell
    h, w = cost.shape
    r0, r1, c0, c1 = max(0, r - max_radius), min(h, r + max_radius + 1), max(0, c - max_radius), min(w, c + max_radius + 1)
    win = np.isfinite(cost[r0:r1, c0:c1])
    if not win.any():
        return None
    rr, cc = np.nonzero(win)
    k = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
    return int(rr[k] + r0), int(cc[k] + c0)


def suggest_points(cost: np.ndarray, frac_a=(0.25, 0.2), frac_b=(0.8, 0.85)) -> dict[str, Any]:
    """Two cells in the largest 8-connected traversable component, near the given fractional positions."""
    from scipy import ndimage

    lab, n = ndimage.label(np.isfinite(cost), structure=np.ones((3, 3)))
    if n == 0:
        return {"ok": False, "reason": "no traversable cells"}
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == int(np.argmax(sizes))
    rr, cc = np.nonzero(comp)
    h, w = cost.shape
    pts = []
    for fr, fc in (frac_a, frac_b):
        k = int(np.argmin((rr - fr * h) ** 2 + (cc - fc * w) ** 2))
        pts.append([int(rr[k]), int(cc[k])])
    return {"ok": True, "start": pts[0], "goal": pts[1], "component_fraction": float(comp.mean()),
            "note": "Both points lie in the largest connected traversable region of the current cost map."}
