"""Crater library: gazetteer-derived coverage and window planning (offline)."""
import sys

import pytest

from core.config import ROOT
from core.craters import crater_specs, features_in_bounds, gazetteer
from core.geospatial.grid import polar_stereo_south_forward

sys.path.insert(0, str(ROOT / "scripts"))


def test_gazetteer_has_named_psr_craters():
    names = {f["name"] for f in gazetteer()["features"]}
    for n in ("Shackleton", "Haworth", "Shoemaker", "Faustini", "de Gerlache", "Cabeus", "Nobile"):
        assert n in names


def test_shackleton_window_covers_shackleton_by_computation():
    # Our original 32 km window: x -12..20 km, y -16..16 km
    feats = {f["name"]: f for f in features_in_bounds((-12000, -16000, 20000, 16000))}
    # Shackleton's centre is inside, but its 20.9 km rim circle extends ~0.9 km past y = -16 km.
    assert "Shackleton" in feats and feats["Shackleton"]["coverage"].startswith("partial")
    assert "Haworth" not in feats  # never claimed unless its centre is inside the window


def test_partial_coverage_flag():
    g = {f["name"]: f for f in gazetteer()["features"]}["Shackleton"]
    x, y = polar_stereo_south_forward(g["lat"], g["lon"])
    small = (float(x) - 2000, float(y) - 2000, float(x) + 2000, float(y) + 2000)  # 4 km box inside a 21 km crater
    f = next(f for f in features_in_bounds(small) if f["name"] == "Shackleton")
    assert f["coverage"].startswith("partial")


def test_window_plan_centred_and_sized():
    import yaml
    from fetch_crater import plan
    cfg = yaml.safe_load((ROOT / "configs/craters.yaml").read_text())
    for c in cfg["craters"]:
        p = plan(c, cfg["window_factor"])
        b, (cx, cy) = p["bounds"], p["centre_xy"]
        assert b[0] < cx < b[2] and b[1] < cy < b[3]
        side = b[2] - b[0]
        assert side >= cfg["window_factor"] * p["gazetteer"]["diameter_km"] * 1000 - 2 * p["res_m"]
        assert side / p["res_m"] <= 2100  # keeps analysis grids tractable
        assert b[0] % p["res_m"] == 0 and b[1] % p["res_m"] == 0  # snapped to the product grid


def test_crater_specs_have_provenance_for_every_layer():
    specs = crater_specs()
    assert len(specs) >= 9
    for s in specs:
        assert s["kind"] == "real" and s["crater"]["link"].startswith("http")
        for role, l in s["layers"].items():
            assert l["source_url"] and l["units"] and l["status"] in ("measured", "derived")


def test_api_lists_craters_and_handles_undownloaded(cfg):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    c = TestClient(appmod.app)
    r = c.get("/api/craters").json()
    names = {x["name"]: x for x in r["craters"]}
    assert "Shackleton" in names and "Cabeus" in names
    missing = [x for x in r["craters"] if not x["available"]]
    for m in missing:
        assert m["thumb"] is None and "fetch_crater.py" in m["fetch"]
        assert c.get(f"/api/datasets/{m['id']}/thumb.png").status_code == 404
