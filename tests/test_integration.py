"""Integration: ingest a small known raster -> process -> layers -> route -> report (offline, CPU)."""
import json

import numpy as np
import pytest

from core.navigation.planner import plan_route, suggest_points
from core.pipeline import analyse, build_dataset
from core.reporting.report import DISCLAIMER, build_pdf, build_summary


@pytest.fixture(scope="module")
def analysis(demo_dir):
    from core.config import load_config
    cfg = load_config()
    spec = {"id": "demo_test", "title": "SYNTHETIC test", "kind": "synthetic", "layers": {
        "elevation": {"file": str(demo_dir / "synthetic_dem.tif"), "status": "synthetic", "source_name": "synthetic"},
        "psr_mask": {"file": str(demo_dir / "synthetic_psr.tif"), "status": "synthetic"},
        "thermal_max": {"file": str(demo_dir / "synthetic_tmax.tif"), "status": "synthetic"},
    }}
    ds = build_dataset(spec, cfg)
    return analyse(ds, cfg), cfg


def test_layers_created_and_labelled_synthetic(analysis):
    an, _ = analysis
    for k in ("elevation", "slope", "roughness", "hazard", "cost", "psr", "priority_score"):
        assert k in an.arrays and an.arrays[k].shape == (an.grid.height, an.grid.width)
    assert an.meta["slope"].status.value == "synthetic"
    assert an.meta["priority_score"].status.value == "synthetic"
    assert not an.meta["hydrogen"].available


def test_nodata_gap_is_unknown_and_blocked(analysis):
    an, _ = analysis
    gap = ~np.isfinite(an.arrays["elevation"])
    assert gap.any()
    assert np.isinf(an.arrays["cost"][gap]).all()
    assert (an.arrays["hazard"][gap] == 0).all()


def test_route_end_to_end(analysis):
    an, cfg = analysis
    pts = suggest_points(an.arrays["cost"])
    r = plan_route(an, tuple(pts["start"]), tuple(pts["goal"]), cfg)
    assert r["found"]
    assert r["validation"]["all_cells_traversable"] and r["validation"]["recomputed_cost_matches"]
    assert r["summary"]["length_m"] > 0 and r["summary"]["blocked_cells_crossed"] == 0
    assert r["alternatives"]["distance_only"]["cost"] >= r["summary"]["cost"] - 1e-6
    assert "not mission-certified" in r["label"]
    r2 = plan_route(an, tuple(pts["start"]), tuple(pts["goal"]), cfg)
    assert r2["cells"] == r["cells"]


def test_report_json_and_pdf(analysis):
    an, cfg = analysis
    pts = suggest_points(an.arrays["cost"])
    route = plan_route(an, tuple(pts["start"]), tuple(pts["goal"]), cfg)
    s = build_summary(an, route, None, cfg)
    json.dumps(s, allow_nan=False)  # must be strict JSON: no NaN/Infinity values anywhere
    assert s["disclaimer"] == DISCLAIMER
    assert any("SYNTHETIC" in l for l in s["limitations"])
    pdf = build_pdf(s, an, route["cells"], route["alternative_paths"])
    assert pdf[:5] == b"%PDF-" and len(pdf) > 20_000


def test_image_only_dataset_has_no_physical_layers(tmp_path):
    from PIL import Image

    from core.config import load_config
    cfg = load_config()
    p = tmp_path / "img.png"
    Image.fromarray((np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)).save(p)
    ds = build_dataset({"id": "u", "title": "u", "kind": "upload", "layers": {"image": {"file": str(p)}}}, cfg)
    an = analyse(ds, cfg)
    assert not an.meta["slope"].available and np.isinf(an.arrays["cost"]).all()
    assert an.stats["priority"]["status"] == "not_assessed"
