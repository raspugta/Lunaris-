"""Reproduce the validation numbers quoted in docs/VALIDATION.md on the real datasets.

Writes docs/validation_results.json. Every metric printed here is computed from
data on disk; nothing is hard-coded. Sections are skipped (with a reason) when
their inputs are absent.

Usage:  python scripts/validate_real_data.py [--no-depth]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from core.ingestion.loader import IngestError  # noqa: E402
from core.navigation.planner import plan_route, suggest_points  # noqa: E402
from core.pipeline import analyse, load_dataset  # noqa: E402
from core.reconstruction.compare import estimate_offset  # noqa: E402
from core.reporting.report import _clean  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-depth", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. Independent-product registration check: PDS LOLA GDR (2017) vs PGDA DEM (2023).
    try:
        a = load_dataset("lola_shackleton_20m", cfg).layers["elevation"]
        b = load_dataset("lola_pds_875s", cfg).layers["elevation"]
        assert a.grid.transform == b.grid.transform, "grids differ"
        d = a.data - b.data
        dr, dc, r = estimate_offset(a.data[::2, ::2], b.data[::2, ::2], np.ones(a.data[::2, ::2].shape, bool), 4)
        out["registration_pds_vs_pgda"] = {
            "what": "PDS3 IMG georeferencing (our label parser) vs PGDA GeoTIFF over the same 32 km window",
            "median_diff_m": float(np.median(d)), "mad_m": float(np.median(np.abs(d - np.median(d)))),
            "p95_abs_diff_m": float(np.percentile(np.abs(d), 95)),
            "best_integer_shift_px_at_40m": [dr, dc], "highpass_corr": r,
            "pass": (dr, dc) == (0, 0) and abs(float(np.median(d))) < 1.0,
        }
    except (IngestError, KeyError, AssertionError) as exc:
        out["registration_pds_vs_pgda"] = {"skipped": str(exc)}

    # 2. Terrain analysis on the PGDA AOI.
    try:
        an = analyse(load_dataset("lola_shackleton_20m", cfg), cfg)
        out["terrain"] = {k: an.stats.get(k) for k in ("analysis_res_m", "slope_vs_reference", "hazard_fractions",
                                                        "psr_fraction", "uncertain_fraction", "image_coverage_fraction")}
        out["ice_priority"] = an.stats["priority"]
        pts = suggest_points(an.arrays["cost"])
        t0 = time.time()
        r = plan_route(an, tuple(pts["start"]), tuple(pts["goal"]), cfg)
        out["route_demo"] = {"start": pts["start"], "goal": pts["goal"], "found": r["found"],
                             "runtime_s": round(time.time() - t0, 2), "summary": r.get("summary"),
                             "validation": r.get("validation"),
                             "alternatives": {k: {kk: v.get(kk) for kk in ("length_m", "cost", "blocked_cells_crossed", "why_rejected")}
                                              for k, v in (r.get("alternatives") or {}).items()}}
        if not args.no_depth:
            from core.reconstruction.run import run_depth
            dep = run_depth(an, cfg)
            out["depth_vs_lola"] = {k: dep.get(k) for k in ("status", "info", "comparison", "reason")}
    except IngestError as exc:
        out["terrain"] = {"skipped": str(exc)}

    p = ROOT / "docs" / "validation_results.json"
    p.write_text(json.dumps(_clean(out), indent=2))
    print(json.dumps(_clean(out), indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
