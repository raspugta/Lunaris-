"""Rover simulator checked against closed-form results on synthetic planes (offline)."""
import copy
import math

import numpy as np
import pytest

from core.rover.sim import Sampler, bekker_wheel, load_rover_db, pv, simulate, solve_slip, thrust

DB = load_rover_db()
VIPER = DB["rovers"]["viper"]
SOIL = DB["soils"]["apollo_lrv_nominal"]
G = DB["environment"]["g_moon"]
RES = 20.0
TR = (RES, 0.0, -2000.0, 0.0, -RES, 2000.0)


def plane(gx=0.0, gy=0.0, n=200):
    """z = gx * x + gy * y on a 4 km grid centred on the origin."""
    c = -2000 + (np.arange(n) + 0.5) * RES
    X, Y = np.meshgrid(c, 2000 - (np.arange(n) + 0.5) * RES)
    return Sampler(gx * X + gy * Y, TR)


def path_north(length=200.0):
    return np.array([[0.0, -length / 2], [0.0, length / 2]])


def test_viper_parameters_match_published_values():
    p = VIPER["params"]
    assert pv(p, "mass_kg") == 447 and pv(p, "wheel_diameter_m") == 0.5 and pv(p, "battery_wh") == 5420
    assert pv(p, "speed_nominal_mps") == 0.10 and pv(p, "max_slope_req_deg") == 15 and pv(p, "psr_ops_limit_h") == 9
    assert p["cg_height_m"]["tag"] == "assumed"  # unpublished values must stay flagged


def test_flat_ground_analytic_sinkage_and_no_tilt():
    r = simulate(path_north(), plane(), VIPER, SOIL, G)
    f = r["frames"]
    W = pv(VIPER["params"], "mass_kg") * G / 4
    n, b, D = SOIL["n"], 0.2, 0.5
    z0 = (3 * W / ((3 - n) * (SOIL["k_c"] + b * SOIL["k_phi"]) * math.sqrt(D))) ** (2 / (2 * n + 1))
    assert np.allclose(f["sinkage_mm"], 1000 * z0, rtol=1e-6)
    assert np.allclose(f["pitch"], 0, atol=1e-9) and np.allclose(f["roll"], 0, atol=1e-9)
    assert r["summary"]["verdict"] == "PASS"
    assert r["summary"]["distance_m"] == pytest.approx(200.0)


def test_pitch_equals_grade_uphill_and_roll_on_cross_slope():
    g = math.tan(math.radians(8))
    up = simulate(path_north(), plane(gy=g), VIPER, SOIL, G)["frames"]
    assert np.allclose(up["pitch"][2:-2], 8.0, atol=1e-6) and np.allclose(up["roll"][2:-2], 0, atol=1e-6)
    # Driving north with ground rising to the east (+x = right side) -> left side down -> negative roll.
    side = simulate(path_north(), plane(gx=g), VIPER, SOIL, G)["frames"]
    assert np.allclose(side["roll"][2:-2], -8.0, atol=1e-6) and np.allclose(side["pitch"][2:-2], 0, atol=1e-6)


def test_time_and_speed_follow_slip():
    g = math.tan(math.radians(10))
    r = simulate(path_north(), plane(gy=g), VIPER, SOIL, G)
    f = r["frames"]
    assert np.allclose(f["speed"], 0.10 * (1 - f["slip"]), rtol=1e-9)
    expected_t = 200.0 / (0.10 * (1 - f["slip"][5]))
    assert r["summary"]["drive_time_h"] * 3600 == pytest.approx(expected_t, rel=0.01)


def test_slip_solution_satisfies_force_balance():
    W = np.full((1, 4), 447 * G / 4 * math.cos(math.radians(12)))
    z0, Rc, l, hmax = bekker_wheel(W, SOIL, 0.2, 0.5)
    req = Rc.sum(1) + 447 * G * math.sin(math.radians(12))
    i, ok = solve_slip(req, hmax, l, SOIL["K_m"])
    assert ok[0] and 0 < i[0] < 1
    assert thrust(i, hmax, l, SOIL["K_m"]).sum() == pytest.approx(req[0], rel=1e-6)


def test_climb_limit_immobilises_and_is_physically_bounded():
    # Net traction coefficient tops out near 0.44 (sin 26 deg) on LRV soil; 30 deg is beyond it but
    # below the 33.7 deg longitudinal tip angle, so the failure must be immobilisation, not tip-over.
    steep = simulate(path_north(), plane(gy=math.tan(math.radians(30))), VIPER, SOIL, G)
    assert steep["summary"]["verdict"] == "FAIL" and "immobilised" in steep["summary"]["failure"]
    ok = simulate(path_north(), plane(gy=math.tan(math.radians(5))), VIPER, SOIL, G)
    assert ok["summary"]["completed_fraction"] == pytest.approx(1.0)


def test_design_slope_warning_between_15_and_climb_limit():
    r = simulate(path_north(), plane(gy=math.tan(math.radians(17))), VIPER, SOIL, G)
    assert any(e["type"] == "design_slope" for e in r["events"])


def test_static_tip_over_with_high_cg():
    rv = copy.deepcopy(VIPER)
    rv["params"]["cg_height_m"]["value"] = 3.0  # lateral tip angle = atan(0.65/3) = 12.2 deg
    r = simulate(path_north(), plane(gx=math.tan(math.radians(15))), rv, SOIL, G)
    assert r["summary"]["verdict"] == "FAIL" and "tip-over" in r["summary"]["failure"]


def test_psr_time_limit_is_enforced():
    psr = Sampler(np.ones((200, 200)), TR)
    long_path = np.array([[0.0, -1900.0], [0.0, 1900.0]])  # 3.8 km at ~10 cm/s ~ 10.6 h > 9 h
    r = simulate(long_path, plane(), VIPER, SOIL, G, psr=psr)
    assert r["summary"]["verdict"] == "FAIL"
    assert any(e["type"] == "psr_limit" for e in r["events"])
    hotel = r["summary"]["hotel_power_shadow_w"]
    assert hotel + r["summary"]["mobility_energy_wh"] * 0 > 0
    # Derived load reproduces the published endurance: 9 h of flat driving uses the full 5420 Wh.
    W = np.full(4, 447 * G / 4)
    _, Rc, _, _ = bekker_wheel(W, SOIL, 0.2, 0.5)
    p_flat = Rc.sum() * 0.10 / pv(VIPER["params"], "drive_efficiency")
    assert (hotel + p_flat) * 9 == pytest.approx(5420, rel=1e-9)


def test_deterministic():
    a = simulate(path_north(), plane(gy=0.1), VIPER, SOIL, G)
    b = simulate(path_north(), plane(gy=0.1), VIPER, SOIL, G)
    assert a["summary"] == b["summary"]


def test_perseverance_profile_matches_published_values():
    p = DB["rovers"]["perseverance"]["params"]
    assert pv(p, "mass_kg") == 1025 and pv(p, "wheel_diameter_m") == 0.525 and pv(p, "n_wheels") == 6
    assert pv(p, "speed_max_mps") == pytest.approx(152 / 3600, rel=0.01)  # 152 m/h published
    r = simulate(path_north(), plane(), DB["rovers"]["perseverance"], SOIL, G)
    # CG height is derived from the published 45 deg tilt rating, so the lateral tip angle must reproduce it.
    assert r["summary"]["static_tip_angles_deg"]["lateral"] == pytest.approx(45.0, abs=0.01)
    assert r["summary"]["energy_model"].startswith("not modelled")  # battery capacity unpublished -> no fake energy
    assert r["summary"]["verdict"] == "PASS"
