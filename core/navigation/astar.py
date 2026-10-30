"""Deterministic grid A* planner over a per-cell cost surface.

Movement model
--------------
8-connected grid. Moving between adjacent cells ``a -> b`` costs

    step(a, b) * (c[a] + c[b]) / 2

where ``step`` is the centre-to-centre distance (``d`` or ``sqrt(2) d``) and
``c`` the per-metre cost multiplier (>= 1, ``inf`` = blocked). Diagonal moves
are forbidden when either orthogonal neighbour is blocked (no corner cutting
through obstacles).

Heuristic
---------
Octile distance ``d * (max(dx,dy) + (sqrt2 - 1) * min(dx,dy))`` multiplied by
the global minimum cost multiplier (1.0). Because every edge costs at least
its octile length, the heuristic never overestimates: it is admissible and
consistent, so A* returns an optimal path on this graph.

Determinism
-----------
Ties are broken by (f, h, insertion counter) so identical inputs give an
identical path.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SQRT2 = math.sqrt(2.0)
_NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


class PlanningError(ValueError):
    pass


@dataclass
class PathResult:
    found: bool
    cells: list[tuple[int, int]] = field(default_factory=list)
    cost: float = math.inf
    expanded: int = 0
    reason: str | None = None


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cell_size: float = 1.0,
          max_expansions: int | None = None) -> PathResult:
    h, w = cost.shape
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not (0 <= r < h and 0 <= c < w):
            raise PlanningError(f"{name} {r, c} is outside the grid {h}x{w}")
        if not math.isfinite(float(cost[r, c])):
            return PathResult(False, reason=f"{name} cell is blocked or unknown")
    cmin = float(np.min(cost[np.isfinite(cost)]))
    if cmin < 1.0 - 1e-9:
        raise PlanningError("cost multipliers must be >= 1 for the heuristic to remain admissible")
    flat = cost.astype(np.float64).ravel().tolist()
    passable = [math.isfinite(v) for v in flat]
    gr, gc = goal
    goal_i = gr * w + gc
    start_i = start[0] * w + start[1]
    d = float(cell_size)
    k = SQRT2 - 1.0

    def heur(i: int) -> float:
        dr = abs(i // w - gr)
        dc = abs(i % w - gc)
        return d * (max(dr, dc) + k * min(dr, dc))

    g = {start_i: 0.0}
    parent = {start_i: -1}
    closed = bytearray(h * w)
    counter = 0
    h0 = heur(start_i)
    openq = [(h0, h0, counter, start_i)]
    expanded = 0
    limit = max_expansions or h * w
    while openq:
        f, hh, _, cur = heapq.heappop(openq)
        if closed[cur]:
            continue
        closed[cur] = 1
        expanded += 1
        if cur == goal_i:
            path = []
            while cur != -1:
                path.append((cur // w, cur % w))
                cur = parent[cur]
            path.reverse()
            return PathResult(True, path, g[goal_i], expanded)
        if expanded > limit:
            return PathResult(False, expanded=expanded, reason="expansion limit reached")
        r, c = divmod(cur, w)
        gcur = g[cur]
        ccur = flat[cur]
        for dr, dc, step in _NEIGH:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            ni = nr * w + nc
            if not passable[ni] or closed[ni]:
                continue
            if dr and dc and (not passable[r * w + nc] or not passable[nr * w + c]):
                continue
            ng = gcur + d * step * 0.5 * (ccur + flat[ni])
            if ng < g.get(ni, math.inf):
                g[ni] = ng
                parent[ni] = cur
                counter += 1
                hn = heur(ni)
                heapq.heappush(openq, (ng + hn, hn, counter, ni))
    return PathResult(False, expanded=expanded, reason="no traversable path between start and goal")


def path_cost(cost: np.ndarray, cells: list[tuple[int, int]], cell_size: float = 1.0) -> float:
    """Recompute the cost of a path under the planner's edge model (for validation)."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        if max(abs(r1 - r0), abs(c1 - c0)) != 1:
            raise PlanningError("path cells are not 8-adjacent")
        step = SQRT2 if (r1 != r0 and c1 != c0) else 1.0
        total += cell_size * step * 0.5 * (float(cost[r0, c0]) + float(cost[r1, c1]))
    return total


def bresenham(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells on the straight segment between two cells (8-connected)."""
    (r0, c0), (r1, c1) = start, goal
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    out = [(r0, c0)]
    while (r0, c0) != (r1, c1):
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
        out.append((r0, c0))
    return out


def summarise_path(cells: list[tuple[int, int]], cost: np.ndarray, cell_size: float | None,
                   slope: np.ndarray | None, hazard: np.ndarray | None, psr: np.ndarray | None,
                   elevation: np.ndarray | None, hazard_legend: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Length, cost, and hazard exposure along a path (length only if the grid is calibrated)."""
    if not cells:
        return {}
    rr = np.array([c[0] for c in cells])
    cc = np.array([c[1] for c in cells])
    steps = np.hypot(np.diff(rr), np.diff(cc))
    cs = cell_size or 1.0
    vals = cost[rr, cc]
    blocked = int((~np.isfinite(vals)).sum())
    s: dict[str, Any] = {
        "n_cells": len(cells),
        "length_m": float(steps.sum() * cell_size) if cell_size else None,
        "length_cells": float(steps.sum()),
        "blocked_cells_crossed": blocked,
        # A path through blocked/unknown cells has no finite cost (impassable).
        "cost": None if blocked else float(path_cost(cost, cells, cs)),
    }
    if slope is not None:
        sv = slope[rr, cc]
        s["max_slope_deg"] = float(np.nanmax(sv)) if np.isfinite(sv).any() else None
        s["mean_slope_deg"] = float(np.nanmean(sv)) if np.isfinite(sv).any() else None
    if hazard is not None:
        hv = hazard[rr, cc]
        s["hazard_exposure"] = {hazard_legend[k]["label"]: int((hv == k).sum()) for k in hazard_legend}
    if psr is not None:
        pv = psr[rr, cc]
        s["cells_in_psr"] = int(np.nansum(pv >= 0.5))
    if elevation is not None and cell_size:
        ev = elevation[rr, cc]
        de = np.diff(ev)
        s["total_ascent_m"] = float(np.nansum(np.clip(de, 0, None)))
        s["total_descent_m"] = float(-np.nansum(np.clip(de, None, 0)))
    return s
