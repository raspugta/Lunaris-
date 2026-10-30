"""HTTP API smoke test on the synthetic fixture (offline)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(demo_dir):
    import backend.app as appmod
    appmod.UPLOAD_SPECS["demo_api"] = {
        "id": "demo_api", "title": "SYNTHETIC api", "kind": "synthetic", "layers": {
            "elevation": {"file": str(demo_dir / "synthetic_dem.tif"), "status": "synthetic"},
            "psr_mask": {"file": str(demo_dir / "synthetic_psr.tif"), "status": "synthetic"},
            "thermal_max": {"file": str(demo_dir / "synthetic_tmax.tif"), "status": "synthetic"}}}
    return TestClient(appmod.app)


def test_full_workflow(client):
    s = client.post("/api/datasets/demo_api/load").json()
    assert s["dataset"]["kind"] == "synthetic" and s["layers"]["slope"]["status"] == "synthetic"
    png = client.get("/api/datasets/demo_api/layers/hazard.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get("/api/datasets/demo_api/layers/nope.png").status_code == 404
    mesh = client.get("/api/datasets/demo_api/mesh").json()
    assert mesh["calibrated"] and mesh["width"] > 10
    insp = client.get("/api/datasets/demo_api/inspect", params={"row": 50, "col": 50}).json()
    assert "priority" in insp and insp["lat_lon"] is not None
    pts = client.get("/api/datasets/demo_api/suggest_points").json()
    r = client.post("/api/datasets/demo_api/route", json={"start": pts["start"], "goal": pts["goal"]}).json()
    assert r["found"] and r["validation"]["all_cells_traversable"]
    sim = client.post("/api/datasets/demo_api/simulate", json={"rover": "viper"}).json()
    assert sim["summary"]["verdict"] in ("PASS", "MARGINAL", "FAIL") and len(sim["frames"]["t"]) > 10
    assert sim["soil_sensitivity"]["soil"] == "loose_psr_hypothetical"
    pat = client.get("/api/datasets/demo_api/terrain_patch", params={"x": 0, "y": 0, "size": 200, "spacing": 4}).json()
    assert pat["n"] == 51 and "interpolation" in pat["note"].lower()
    assert client.post("/api/datasets/demo_api/simulate", json={"rover": "nope"}).status_code == 400
    pr = client.post("/api/datasets/demo_api/priority", json={"weights": {"psr": 0.0, "thermal_max": 1.0, "ice_stability_depth": 0.0}}).json()
    assert pr["weights"]["psr"] == 0.0
    a = client.post("/api/datasets/demo_api/annotations", json={"row": 1, "col": 2, "label": "check"}).json()
    assert a["source"].startswith("manual")
    j = client.get("/api/datasets/demo_api/report.json").json()
    assert j["route"]["found"] and j["manual_annotations"] and j["rover_simulation"]["summary"]["verdict"]
    pdf = client.get("/api/datasets/demo_api/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"


def test_errors_are_actionable(client):
    r = client.get("/api/datasets/never_loaded/summary")
    assert r.status_code == 404 and "load" in r.json()["detail"].lower()
    r = client.post("/api/datasets/demo_api/route", json={"start": [0, 0], "goal": [9999, 9999]})
    assert r.status_code == 400
    up = client.post("/api/upload", files={"file": ("x.txt", b"hello")}, data={"role": "elevation"})
    assert up.status_code == 415
