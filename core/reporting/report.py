"""Mission-brief export: machine-readable JSON and human-readable PDF."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.reporting.figures import map_figure

DISCLAIMER = ("Scope: analyses are derived from orbital remote-sensing and model products. Traverses are simulated "
              "candidates on a documented cost model; water-ice results are relative investigation priorities, "
              "not detections. Results support, and do not replace, mission certification and in-situ confirmation.")


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def build_summary(an, route: dict[str, Any] | None, depth: dict[str, Any] | None, cfg: dict,
                  annotations: list[dict] | None = None) -> dict[str, Any]:
    ds = an.dataset
    layers = {k: {"title": m.title, "status": m.status.value, "units": m.units, "available": m.available,
                  "method": m.method, "limitations": m.limitations, "resolution_m": m.resolution_m,
                  "unavailable_reason": m.unavailable_reason} for k, m in an.meta.items()}
    return _clean({
        "schema": "lunar-psr-intelligence/analysis-summary/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "dataset": ds.manifest(),
        "analysis_grid": an.grid.to_dict(),
        "config_hash": an.config_hash,
        "configuration": {k: cfg[k] for k in ("analysis", "hazard", "navigation", "ice_priority")},
        "processing_log": an.processing_log,
        "layers": layers,
        "terrain_findings": {k: an.stats.get(k) for k in ("hazard_fractions", "slope_vs_reference", "psr_fraction",
                                                          "uncertain_fraction", "image_coverage_fraction", "boulder_detection")},
        "ice_investigation_priority": an.stats.get("priority"),
        "route": route,
        "depth_reconstruction": depth,
        "manual_annotations": annotations or [],
        "baseline_feature_candidates": {"count": len(an.features), "label": "unvalidated LoG topographic anomalies"},
        "assumptions": assumptions(cfg),
        "limitations": limitations(an),
    })


def assumptions(cfg: dict) -> list[str]:
    h, n = cfg["hazard"], cfg["navigation"]
    return [
        f"Slope classes {h['slope_classes_deg']} deg and blocked >= {h['slope_blocked_deg']} deg are prototype assumptions.",
        f"Roughness high >= {h['roughness_high_m']} m, blocked >= {h['roughness_blocked_m']} m RMS (prototype).",
        f"Cost weights {n['weights']} are prototype choices; cost is relative, not energy or time.",
        f"Ice-priority weights {cfg['ice_priority']['weights']} with sensitivity +/-{100 * cfg['ice_priority']['sensitivity_perturbation']:.0f}%.",
        "Ground distance equals grid distance (polar stereographic scale error < 0.04 % within 2 deg of the pole).",
        "No-data and unknown cells are treated as blocked for routing.",
    ]


def limitations(an) -> list[str]:
    out = [
        f"Analysis grid {an.grid.res_m or 'uncalibrated'} m/px: rover-scale obstacles (boulders, small craters) are not resolved.",
        "PSR map, temperatures and ice-stability depth are model products, each with its own epoch and resolution.",
        "No LEND hydrogen layer was used; no direct measurement of water ice is included.",
        "Illumination, communications line-of-sight and thermal survival are not modelled in the route.",
    ]
    for k, m in an.meta.items():
        if not m.available:
            out.append(f"Layer '{m.title}' unavailable: {m.unavailable_reason}")
    if an.dataset.kind == "synthetic":
        out.insert(0, "SYNTHETIC DATA: this report does not describe any real lunar location.")
    return out


from core.reporting.pdf_report import build_pdf  # noqa: E402,F401  (professional report layout)
