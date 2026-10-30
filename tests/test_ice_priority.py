import numpy as np
import pytest

from core.ice_priority.evidence import NOT_ASSESSED, PRIORITY_HIGH, PRIORITY_LOW, compute_priority, explain_cell, subscores


def test_transfer_functions(cfg):
    ic = cfg["ice_priority"]
    s = subscores({"psr": np.array([1.0, 0.0]), "thermal_max": np.array([90.0, 140.0]),
                   "ice_stability_depth": np.array([0.0, 999.0])}, ic)
    assert s["psr"].tolist() == [1.0, 0.0]
    assert s["thermal_max"].tolist() == [1.0, 0.0]
    assert s["ice_stability_depth"].tolist() == [1.0, 0.0]


def test_not_assessed_without_required_thermal(cfg):
    """PSR membership alone (a darkness-like proxy) must never produce a priority."""
    r = compute_priority({"psr": np.ones((4, 4), np.float32)}, cfg["ice_priority"])
    assert np.isnan(r.score).all()
    assert (r.classes == NOT_ASSESSED).all()
    assert "thermal_max" in r.meta["layers_missing"]


def test_missing_cells_are_not_assessed_not_low(cfg):
    t = np.array([[80.0, np.nan]], np.float32)
    r = compute_priority({"psr": np.ones((1, 2), np.float32), "thermal_max": t}, cfg["ice_priority"])
    assert r.classes[0, 0] == PRIORITY_HIGH
    assert r.classes[0, 1] == NOT_ASSESSED


def test_score_is_weighted_mean_of_available(cfg):
    ic = cfg["ice_priority"]
    r = compute_priority({"psr": np.array([[1.0]]), "thermal_max": np.array([[130.0]])}, ic)
    w = ic["weights"]
    expected = (w["psr"] * 1.0 + w["thermal_max"] * 0.0) / (w["psr"] + w["thermal_max"])
    assert float(r.score[0, 0]) == pytest.approx(expected)
    assert float(r.coverage[0, 0]) == pytest.approx((w["psr"] + w["thermal_max"]) / sum(w.values()))


def test_warm_sunlit_is_low(cfg):
    r = compute_priority({"psr": np.zeros((1, 1)), "thermal_max": np.array([[300.0]]),
                          "ice_stability_depth": np.array([[999.0]])}, cfg["ice_priority"])
    assert r.classes[0, 0] == PRIORITY_LOW


def test_sensitivity_reported(cfg):
    rng = np.random.default_rng(0)
    inp = {"psr": rng.random((20, 20)), "thermal_max": rng.uniform(60, 160, (20, 20)),
           "ice_stability_depth": rng.uniform(0, 2, (20, 20))}
    r = compute_priority(inp, cfg["ice_priority"])
    assert r.meta["sensitivity"]["scenarios"] == 9
    assert 0.0 <= r.class_change_fraction <= 1.0
    assert np.isfinite(r.sensitivity_std).all()


def test_explain_cell_contributions_sum_to_score(cfg):
    inp = {"psr": np.array([[0.5]]), "thermal_max": np.array([[110.0]]), "ice_stability_depth": np.array([[0.2]]),
           "hydrogen": None}
    r = compute_priority(inp, cfg["ice_priority"])
    e = explain_cell(r, inp, 0, 0, cfg["ice_priority"])
    tot = sum(x["contribution"] for x in e["evidence"] if x["contribution"] is not None)
    assert tot == pytest.approx(e["score"])
    assert [x for x in e["evidence"] if x["layer"] == "hydrogen"][0]["state"] == "not ingested"
    assert "Not a detection" in e["caveat"]
