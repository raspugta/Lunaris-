"""Scale a planned lunar route into a 4 ft x 4 ft arena and emit an Arduino Alvik MicroPython program.

Alvik facts used (from the official arduino-alvik-mpy library, robot_definitions.py + examples):
  * ``ArduinoAlvik().begin()``; ``move(distance, 'mm', blocking=False)``; ``rotate(angle, 'deg', blocking=False)``;
    ``is_target_reached()``; ``brake()``; ``stop()``; ``get_touch_ok()``; ``get_touch_cancel()``;
    ``left_led/right_led.set_color(r, g, b)``; ``reset_pose()`` / ``get_pose()``.
  * Positive ``rotate`` angles turn LEFT (counter-clockwise): the official touch_move example maps
    'left' -> rotate(90) and 'right' -> rotate(-90).
  * Position-control speeds: MOTOR_CONTROL_MM_S = 100, MOTOR_CONTROL_DEG_S = 100; the library sleeps
    200 ms before each move/rotate command. These give the time estimate below.
  * Wheel track 89 mm, wheel diameter 34 mm.

Arena frame: origin at the arena's bottom-left corner as seen by the operator, x to the right, y away
from the operator ("up"), millimetres. Map/grid up (the 0°E meridian direction on the polar grid) is drawn
as arena "up", so the arena is a north-up miniature of the 2D map.

This is a *kinematic* replay of the route geometry at a fixed scale. Wheel slip, slopes and terrain
physics from the simulation are NOT reproduced by the robot. The LEDs only *signal* the simulated
conditions of each segment.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

ARENA_MM_4FT = 4 * 12 * 25.4  # 1219.2 mm
ALVIK = {"track_mm": 89.0, "wheel_d_mm": 34.0, "mm_s": 100.0, "deg_s": 100.0, "cmd_overhead_s": 0.2,
         "footprint_mm": 100.0}  # body ~10 cm (half-footprint used for the safety margin)
LED = {"go": (0, 1, 0), "psr": (0, 0, 1), "slip": (1, 1, 0), "fail": (1, 0, 0), "idle": (0, 0, 0)}


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification (iterative)."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = points[a], points[b]
        seg = q - p
        L = np.hypot(*seg)
        mid = points[a + 1:b]
        if L == 0:
            d = np.hypot(*(mid - p).T)
        else:
            d = np.abs(seg[0] * (mid[:, 1] - p[1]) - seg[1] * (mid[:, 0] - p[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            i = a + 1 + k
            keep[i] = True
            stack += [(a, i), (i, b)]
    return points[keep]


def build_program(route_xy: np.ndarray, *, arena_mm: float = ARENA_MM_4FT, margin_mm: float = 120.0,
                  tolerance_mm: float = 6.0, min_segment_mm: float = 15.0, max_segments: int = 80,
                  frame_s: np.ndarray | None = None, frame_tags: list[str] | None = None,
                  stop_at_s: float | None = None) -> dict[str, Any]:
    """Fit a world route (metres, polar stereographic; +y = grid up) into the arena and make move/rotate steps.

    ``frame_s``/``frame_tags`` (distance along route in metres and a tag per sim frame: go/psr/slip/fail) colour
    the LEDs per segment. ``stop_at_s`` truncates the program where the simulation failed, so the robot stops
    where the simulated rover stopped.
    """
    xy = np.asarray(route_xy, float)
    if len(xy) < 2:
        raise ValueError("route needs at least two points")
    # Scale/offset come from the FULL route so a truncated (failed) run stops at the right spot on the same map.
    full_lo, full_hi = xy.min(0), xy.max(0)
    seglen = np.hypot(*np.diff(xy, axis=0).T)
    s_world = np.concatenate([[0], np.cumsum(seglen)])
    if stop_at_s is not None and stop_at_s < s_world[-1]:
        cut = np.searchsorted(s_world, stop_at_s)
        t = (stop_at_s - s_world[cut - 1]) / max(seglen[cut - 1], 1e-9)
        end = xy[cut - 1] + t * (xy[cut] - xy[cut - 1])
        xy = np.vstack([xy[:cut], end])
        s_world = np.concatenate([s_world[:cut], [stop_at_s]])
    usable = arena_mm - 2 * margin_mm
    if usable <= 50:
        raise ValueError("margin too large for the arena")
    lo, hi = full_lo, full_hi
    span = float(max(hi - lo))
    if span <= 0:
        raise ValueError("route has zero extent")
    k = usable / span  # mm per world metre
    centre_off = (usable - (hi - lo) * k) / 2
    arena = (xy - lo) * k + margin_mm + centre_off  # arena mm, +y up
    # Simplify, then enforce minimum segment length / max segment count by raising tolerance if needed.
    eps = tolerance_mm
    for _ in range(12):
        idx_keep = _rdp_indices(arena, eps)
        pts = arena[idx_keep]
        d = np.hypot(*np.diff(pts, axis=0).T)
        if (d.min() if len(d) else 1e9) >= min_segment_mm * 0.999 and len(pts) - 1 <= max_segments:
            break
        # merge too-short segments
        merged = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[merged[-1]])) >= min_segment_mm or i == len(pts) - 1:
                merged.append(i)
        idx_keep = idx_keep[merged]
        pts = arena[idx_keep]
        if len(pts) - 1 <= max_segments:
            break
        eps *= 1.5
    s_keep = s_world[idx_keep]
    # Max deviation of the simplified path from the full-resolution scaled path
    dev = _max_deviation(arena, pts)
    heading0 = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))  # math angle, CCW from +x
    steps = []
    cur = heading0
    for i in range(len(pts) - 1):
        dx, dy = pts[i + 1] - pts[i]
        h = math.degrees(math.atan2(dy, dx))
        # Turn relative to the *commanded* (rounded) heading, so 0.1-degree rounding never accumulates.
        turn = round(((h - cur + 180) % 360) - 180, 1) if i else 0.0  # + = left (CCW), matches Alvik rotate()
        dist = float(math.hypot(dx, dy))
        tag = "go"
        if frame_s is not None and frame_tags is not None:
            m = (frame_s >= s_keep[i]) & (frame_s <= s_keep[i + 1])
            tags = {frame_tags[j] for j in np.flatnonzero(m)}
            tag = "fail" if "fail" in tags else "slip" if "slip" in tags else "psr" if "psr" in tags else "go"
        steps.append({"turn_deg": turn, "move_mm": round(dist, 1), "led": tag,
                      "world_from_m": float(s_keep[i]), "world_to_m": float(s_keep[i + 1])})
        cur = cur + turn if i else h
    n_turns = sum(1 for s in steps if abs(s["turn_deg"]) >= 0.5)
    t_est = sum(s["move_mm"] for s in steps) / ALVIK["mm_s"] + sum(abs(s["turn_deg"]) for s in steps) / ALVIK["deg_s"] \
        + ALVIK["cmd_overhead_s"] * (len(steps) + n_turns)
    inside = bool(((pts >= margin_mm - 1e-6) & (pts <= arena_mm - margin_mm + 1e-6)).all())
    start_heading_compass = (90.0 - heading0) % 360  # 0 = facing arena "up", clockwise
    return {
        "arena_mm": arena_mm, "margin_mm": margin_mm, "scale_mm_per_m": k, "scale_ratio": f"1 : {1000.0 / k:,.0f}",
        "world_length_m": float(s_world[-1]), "arena_length_mm": float(sum(s["move_mm"] for s in steps)),
        "start_mm": [float(pts[0][0]), float(pts[0][1])], "start_heading_deg_from_up_cw": round(start_heading_compass, 1),
        "goal_mm": [float(pts[-1][0]), float(pts[-1][1])],
        "waypoints_mm": pts.round(1).tolist(), "full_path_mm": arena[:: max(1, len(arena) // 600)].round(1).tolist(),
        "steps": steps, "n_steps": len(steps), "max_deviation_mm": float(dev), "tolerance_mm": eps,
        "inside_safe_area": inside, "estimated_run_s": round(t_est, 1),
        "truncated_at_failure_m": stop_at_s,
        "semantics": "Kinematic miniature of the route geometry only; terrain physics are not reproduced. "
                     "LED colours signal the simulated condition of each segment.",
    }


def _rdp_indices(points: np.ndarray, eps: float) -> np.ndarray:
    idx = np.arange(len(points))
    pts = np.c_[points, idx]
    out = rdp_with_index(pts, eps)
    return out[:, 2].astype(int)


def rdp_with_index(pts: np.ndarray, eps: float) -> np.ndarray:
    simplified = rdp(pts[:, :2], eps)
    # map back to original indices (points are unique along a route)
    keys = {tuple(p): i for i, p in enumerate(pts[:, :2].tolist())}
    return np.array([[*p, keys[tuple(p)]] for p in simplified.tolist()])


def _max_deviation(full: np.ndarray, simp: np.ndarray) -> float:
    best = np.full(len(full), np.inf)
    for a, b in zip(simp[:-1], simp[1:]):
        ab = b - a
        L2 = float(ab @ ab) or 1e-12
        t = np.clip(((full - a) @ ab) / L2, 0, 1)
        d = np.hypot(*(full - (a + t[:, None] * ab)).T)
        best = np.minimum(best, d)
    return float(best.max())


def micropython_source(prog: dict[str, Any], meta: dict[str, Any]) -> str:
    """Render a standalone main.py for the Alvik (MicroPython; no f-strings, for older firmware)."""
    steps = ",\n".join(f"    ({s['turn_deg']:.1f}, {s['move_mm']:.1f}, '{s['led']}')" for s in prog["steps"])
    sx, sy = prog["start_mm"]
    hdr = [
        "# LUNARIS -> Arduino Alvik mission program (auto-generated, do not edit by hand)",
        f"# Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"# Dataset   : {meta.get('dataset_title', '')}",
        f"# Route     : {prog['world_length_m'] / 1000:.2f} km on the Moon -> {prog['arena_length_mm'] / 10:.1f} cm in the arena "
        f"(scale {prog['scale_ratio']})",
        f"# Rover sim : {meta.get('sim', 'not simulated')}",
        f"# Arena     : {prog['arena_mm']:.0f} mm square (4 ft x 4 ft), safety margin {prog['margin_mm']:.0f} mm",
        "#",
        "# PLACE THE ALVIK:",
        f"#   centre of the robot at x = {sx / 10:.1f} cm from the LEFT edge, y = {sy / 10:.1f} cm from the NEAR edge,",
        f"#   facing {prog['start_heading_deg_from_up_cw']:.0f} deg clockwise from straight 'up' (away from you).",
        "# Press OK (check) to start. Press CANCEL (x) at any time to stop; press OK again to restart.",
        "# LEDs: green = normal, blue = inside a permanently shadowed region, yellow = high wheel slip in the",
        "#       simulation, red = segment where the simulated rover failed. This is a geometric replay only.",
    ]
    return "\n".join(hdr) + f'''

from arduino_alvik import ArduinoAlvik
from time import sleep_ms

# (turn_deg [+ = left/CCW], move_mm, led) per step
STEPS = [
{steps}
]

COLORS = {{'go': (0, 1, 0), 'psr': (0, 0, 1), 'slip': (1, 1, 0), 'fail': (1, 0, 0)}}


def leds(alvik, rgb):
    alvik.left_led.set_color(*rgb)
    alvik.right_led.set_color(*rgb)


def wait_target(alvik):
    # Non-blocking wait so CANCEL can abort mid-motion (pattern from the official examples).
    while not alvik.is_target_reached():
        if alvik.get_touch_cancel():
            alvik.brake()
            return False
        sleep_ms(20)
    return True


def run(alvik):
    alvik.reset_pose(0, 0, 0)
    for i in range(len(STEPS)):
        turn, dist, tag = STEPS[i]
        leds(alvik, COLORS.get(tag, (0, 1, 0)))
        print('step %d/%d  turn %.1f deg  move %.1f mm  [%s]' % (i + 1, len(STEPS), turn, dist, tag))
        if abs(turn) >= 0.5:
            alvik.rotate(turn, 'deg', blocking=False)
            if not wait_target(alvik):
                return False
        alvik.move(dist, 'mm', blocking=False)
        if not wait_target(alvik):
            return False
    x, y, theta = alvik.get_pose()
    print('finished: odometry pose x=%s cm y=%s cm theta=%s deg' % (x, y, theta))
    return True


alvik = ArduinoAlvik()
alvik.begin()
try:
    while True:
        # Ready: blink green until OK is pressed.
        on = True
        while not alvik.get_touch_ok():
            leds(alvik, (0, 1, 0) if on else (0, 0, 0))
            on = not on
            sleep_ms(300)
        while alvik.get_touch_ok():
            sleep_ms(50)
        sleep_ms(700)  # hands off
        ok = run(alvik)
        alvik.brake()
        for _ in range(6):  # finished = white flashes, aborted = red flashes
            leds(alvik, (1, 1, 1) if ok else (1, 0, 0))
            sleep_ms(200)
            leds(alvik, (0, 0, 0))
            sleep_ms(200)
        while alvik.get_touch_cancel():
            sleep_ms(50)
except KeyboardInterrupt:
    pass
finally:
    alvik.stop()
'''
