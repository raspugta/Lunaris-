import math

import numpy as np
import pytest

from core.navigation.astar import PlanningError, astar, bresenham, path_cost
from core.navigation.cost import build_cost_surface


def brute_force_dijkstra(cost, start, goal, d=1.0):
    import heapq
    h, w = cost.shape
    dist = {start: 0.0}
    q = [(0.0, start)]
    seen = set()
    while q:
        g, (r, c) = heapq.heappop(q)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if (r, c) == goal:
            return g
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not dr and not dc:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not np.isfinite(cost[nr, nc]):
                    continue
                if dr and dc and (not np.isfinite(cost[r, nc]) or not np.isfinite(cost[nr, c])):
                    continue
                step = math.sqrt(2) if dr and dc else 1.0
                ng = g + d * step * 0.5 * (cost[r, c] + cost[nr, nc])
                if ng < dist.get((nr, nc), math.inf):
                    dist[(nr, nc)] = ng
                    heapq.heappush(q, (ng, (nr, nc)))
    return math.inf


def test_straight_and_diagonal_costs():
    c = np.ones((10, 10), np.float32)
    r = astar(c, (0, 0), (0, 9), 40.0)
    assert r.found and r.cost == pytest.approx(9 * 40.0)
    r = astar(c, (0, 0), (9, 9), 40.0)
    assert r.cost == pytest.approx(9 * math.sqrt(2) * 40.0)


def test_avoids_blocked_wall_and_stays_valid():
    c = np.ones((30, 30), np.float32)
    c[0:25, 15] = np.inf
    r = astar(c, (5, 5), (5, 25))
    assert r.found
    assert all(np.isfinite(c[p]) for p in r.cells)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(r.cells, r.cells[1:]))
    assert path_cost(c, r.cells) == pytest.approx(r.cost)


def test_no_corner_cutting():
    c = np.ones((3, 3), np.float32)
    c[0, 1] = np.inf
    c[1, 0] = np.inf
    r = astar(c, (0, 0), (1, 1))
    assert not r.found  # only exit would be a diagonal squeeze between two blocked cells


def test_unreachable_and_blocked_endpoints():
    c = np.ones((10, 10), np.float32)
    c[:, 5] = np.inf
    assert not astar(c, (0, 0), (0, 9)).found
    assert astar(c, (0, 5), (0, 9)).reason.startswith("start")
    with pytest.raises(PlanningError):
        astar(c, (0, 0), (20, 20))


def test_optimal_against_dijkstra_random_maps():
    rng = np.random.default_rng(7)
    for _ in range(15):
        c = (1 + rng.random((25, 25)) * 9).astype(np.float32)
        c[rng.random((25, 25)) < 0.2] = np.inf
        c[0, 0] = c[24, 24] = 1.0
        r = astar(c, (0, 0), (24, 24), 2.0)
        ref = brute_force_dijkstra(c, (0, 0), (24, 24), 2.0)
        if math.isinf(ref):
            assert not r.found
        else:
            assert r.found and r.cost == pytest.approx(ref, rel=1e-6)  # float32 vs float64 accumulation


def test_reproducible():
    rng = np.random.default_rng(3)
    c = (1 + rng.random((60, 60)) * 3).astype(np.float32)
    a = astar(c, (0, 0), (59, 40))
    b = astar(c, (0, 0), (59, 40))
    assert a.cells == b.cells and a.cost == b.cost


def test_rejects_costs_below_one():
    with pytest.raises(PlanningError):
        astar(np.full((5, 5), 0.5, np.float32), (0, 0), (4, 4))


def test_cost_surface_blocks_nodata_and_steep(cfg):
    slope = np.array([[5.0, np.nan, 30.0, 10.0]], np.float32)
    rough = np.array([[0.5, 0.5, 0.5, 0.5]], np.float32)
    unc = np.array([[False, False, False, True]])
    c, meta = build_cost_surface(slope, rough, unc, None, cfg["hazard"], cfg["navigation"])
    assert np.isfinite(c[0, 0]) and c[0, 0] >= 1.0
    assert np.isinf(c[0, 1]) and np.isinf(c[0, 2])
    assert c[0, 3] > c[0, 0]  # uncertainty penalty
    assert meta["blocked_fraction"] == pytest.approx(0.5)


def test_cost_surface_without_slope_is_all_blocked(cfg):
    c, meta = build_cost_surface(None, None, None, None, cfg["hazard"], cfg["navigation"], shape=(4, 4))
    assert np.isinf(c).all() and "not assessable" in meta["reason"]


def test_bresenham_endpoints():
    line = bresenham((0, 0), (5, 9))
    assert line[0] == (0, 0) and line[-1] == (5, 9)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(line, line[1:]))
