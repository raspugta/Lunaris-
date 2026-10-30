"""Alvik program generation + execution of the generated main.py against a mock Alvik (offline).

The mock follows the official arduino-alvik-mpy semantics: rotate(+angle) turns LEFT (CCW),
move(distance, 'mm') drives forward along the current heading, is_target_reached() polling,
OK/CANCEL touch buttons. Running the generated code and integrating the commanded motions must
reproduce the planned arena waypoints.
"""
import math
import sys
import time
import types

import numpy as np
import pytest

from core.hardware.alvik import ARENA_MM_4FT, build_program, micropython_source


def zigzag_route():
    t = np.linspace(0, 1, 400)
    x = 30000 * t                      # 30 km east
    y = 6000 * np.sin(4 * np.pi * t)   # +/- 6 km north-south wiggle
    return np.c_[x, y]


class MockAlvik:
    instances = []

    def __init__(self):
        self.x = self.y = 0.0
        self.th = 0.0  # deg, CCW, 0 = initial heading
        self.trace = [(0.0, 0.0)]
        self.ok_presses = 1
        self.cancel_at_step = None
        self.cmds = 0
        self.leds = []
        led = types.SimpleNamespace(set_color=lambda r, g, b: self.leds.append((r, g, b)))
        self.left_led = self.right_led = led
        MockAlvik.instances.append(self)

    def begin(self): return 0
    def reset_pose(self, x, y, th): self.x = self.y = self.th = 0.0
    def is_target_reached(self):
        # like the real robot: not reached on the first poll after a command
        self._polls = getattr(self, "_polls", 0) + 1
        return self._polls % 2 == 0
    def brake(self): pass
    def stop(self): pass
    def get_pose(self): return self.x / 10, self.y / 10, self.th

    def rotate(self, angle, unit="deg", blocking=True):
        assert unit == "deg"
        self.th += angle
        self.cmds += 1

    def move(self, distance, unit="cm", blocking=True):
        assert unit == "mm"
        self.x += distance * math.cos(math.radians(self.th))
        self.y += distance * math.sin(math.radians(self.th))
        self.trace.append((self.x, self.y))
        self.cmds += 1

    def get_touch_ok(self):
        # first poll -> pressed; release -> run; afterwards end the session via KeyboardInterrupt
        if self.ok_presses == 1:
            self.ok_presses = 0
            return True
        if self.ok_presses == 0:
            self.ok_presses = -1
            return False
        raise KeyboardInterrupt

    def get_touch_cancel(self):
        # a single press: reads True once when the trigger step is reached, then released
        if self.cancel_at_step is not None and self.cmds >= self.cancel_at_step:
            self.cancel_at_step = None
            return True
        return False


def run_generated(src, monkeypatch, cancel_at=None):
    MockAlvik.instances.clear()
    mod = types.ModuleType("arduino_alvik")

    class A(MockAlvik):
        def __init__(self):
            super().__init__()
            self.cancel_at_step = cancel_at
    mod.ArduinoAlvik = A
    monkeypatch.setitem(sys.modules, "arduino_alvik", mod)
    monkeypatch.setattr(time, "sleep_ms", lambda ms: None, raising=False)
    exec(compile(src, "main.py", "exec"), {"__name__": "__main__"})
    return MockAlvik.instances[0]


def test_program_fits_arena_and_tracks_route():
    p = build_program(zigzag_route(), margin_mm=120)
    pts = np.array(p["waypoints_mm"])
    assert p["inside_safe_area"] and pts.min() >= 120 - 1e-6 and pts.max() <= ARENA_MM_4FT - 120 + 1e-6
    assert p["max_deviation_mm"] <= p["tolerance_mm"] + 1e-6
    assert all(s["move_mm"] >= 15 - 0.05 for s in p["steps"][:-1])
    # scale keeps aspect: the longest side uses the full usable width
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(ARENA_MM_4FT - 240, rel=1e-6)


def test_generated_micropython_traces_the_waypoints(monkeypatch):
    p = build_program(zigzag_route())
    src = micropython_source(p, {"dataset_title": "test"})
    bot = run_generated(src, monkeypatch)
    # Transform the robot's local trace (start at origin, heading = first segment) into arena coords.
    w = np.array(p["waypoints_mm"])
    h0 = math.atan2(w[1, 1] - w[0, 1], w[1, 0] - w[0, 0])
    R = np.array([[math.cos(h0), -math.sin(h0)], [math.sin(h0), math.cos(h0)]])
    trace = np.array(bot.trace) @ R.T + w[0]
    assert trace.shape == w.shape
    assert np.abs(trace - w).max() < 0.5  # mm (only the 0.1 mm / 0.1 deg rounding in the source)


def test_left_turn_sign_matches_alvik_convention():
    # Route east then north = a LEFT turn -> positive rotate() per the official examples.
    route = np.array([[0, 0], [1000, 0], [1000, 1000]], float)
    p = build_program(route, tolerance_mm=1, min_segment_mm=5)
    assert p["steps"][1]["turn_deg"] == pytest.approx(90.0, abs=0.1)
    assert p["start_heading_deg_from_up_cw"] == pytest.approx(90.0, abs=0.1)  # facing right (east)


def test_cancel_button_aborts(monkeypatch):
    p = build_program(zigzag_route())
    bot = run_generated(micropython_source(p, {}), monkeypatch, cancel_at=3)
    assert len(bot.trace) < len(p["waypoints_mm"])  # stopped early
    assert (1, 0, 0) in bot.leds  # red abort flash


def test_truncation_at_simulated_failure():
    route = zigzag_route()
    full = build_program(route)
    half = build_program(route, stop_at_s=full["world_length_m"] / 2)
    assert half["world_length_m"] == pytest.approx(full["world_length_m"] / 2, rel=1e-6)
    assert half["arena_length_mm"] < full["arena_length_mm"]
    assert half["scale_mm_per_m"] == pytest.approx(full["scale_mm_per_m"])  # same map scale
    assert half["start_mm"] == pytest.approx(full["start_mm"])
