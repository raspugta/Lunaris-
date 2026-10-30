"""Transparent water-ice *investigation-priority* model.

This is NOT an ice detector. It ranks cells for further investigation by
combining independently documented evidence layers with explicit weights:

=====================  =============================================  =====================
Layer                  Sub-score in [0, 1]                            Why it is relevant
=====================  =============================================  =====================
psr                    fraction of the cell inside a LOLA-modelled    No direct sunlight over
                       PSR (illumination modelling on the DEM —       the modelled epoch
                       *not* image darkness)
thermal_max            1 at <= T_full, linear to 0 at >= T_zero        Surface ice is stable to
                       (defaults 100 K / 130 K; ~110 K threshold)      sublimation below ~110 K
ice_stability_depth    1 at 0 m, linear to 0 at >= D_zero (1 m);       Depth at which ice would
                       ">2.87 m" (999) -> 0                            be stable (thermal model)
hydrogen (LEND)        not ingested: LEND footprints are ~10+ km;      Not usable per pixel
                       never treated as pixel-level ice
=====================  =============================================  =====================

Score = sum_i w_i s_i / sum_i w_i over layers *available at the cell*.
Evidence coverage = sum of available weights / sum of all configured weights.
A cell is **not assessed** when coverage < ``min_evidence_fraction`` or a
``require_layers`` input is missing there. Low score is "low relative
priority", never "no ice"; high score is "candidate for further investigation",
never "ice detected".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

NOT_ASSESSED, PRIORITY_LOW, PRIORITY_MED, PRIORITY_HIGH = 0, 1, 2, 3
PRIORITY_LEGEND = {
    NOT_ASSESSED: {"label": "Not assessed (insufficient evidence)", "color": "#5b616b"},
    PRIORITY_LOW: {"label": "Lower relative priority", "color": "#2b4a7a"},
    PRIORITY_MED: {"label": "Moderate relative priority", "color": "#3f8fc0"},
    PRIORITY_HIGH: {"label": "Candidate for further investigation", "color": "#9fe3ff"},
}
LAYER_INFO = {
    "psr": "Permanently shadowed (LOLA illumination model)",
    "thermal_max": "Diviner annual max surface temperature",
    "ice_stability_depth": "Diviner modelled ice-stability depth",
    "hydrogen": "LEND epithermal-neutron hydrogen (not ingested)",
}


def subscores(inputs: dict[str, np.ndarray | None], cfg: dict) -> dict[str, np.ndarray]:
    """Transform raw evidence layers into [0, 1] sub-scores (NaN = missing)."""
    out: dict[str, np.ndarray] = {}
    if inputs.get("psr") is not None:
        out["psr"] = np.clip(inputs["psr"].astype(np.float32), 0, 1)
    if inputs.get("thermal_max") is not None:
        t = inputs["thermal_max"].astype(np.float32)
        tf, tz = cfg["thermal_full_score_K"], cfg["thermal_zero_score_K"]
        out["thermal_max"] = np.clip((tz - t) / (tz - tf), 0, 1)
    if inputs.get("ice_stability_depth") is not None:
        dd = inputs["ice_stability_depth"].astype(np.float32)
        dz = cfg["stability_depth_zero_score_m"]
        out["ice_stability_depth"] = np.clip(1 - dd / dz, 0, 1)
    return out


@dataclass
class PriorityResult:
    score: np.ndarray            # [0, 1], NaN = not assessed
    classes: np.ndarray          # uint8 PRIORITY_* codes
    coverage: np.ndarray         # fraction of configured weight available
    subs: dict[str, np.ndarray]
    sensitivity_std: np.ndarray  # std of score across weight perturbations
    class_change_fraction: float
    meta: dict[str, Any]


def _combine(subs: dict[str, np.ndarray], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    shape = next(iter(subs.values())).shape
    num = np.zeros(shape, np.float64)
    den = np.zeros(shape, np.float64)
    for k, s in subs.items():
        wk = float(weights.get(k, 0.0))
        if wk <= 0:
            continue
        ok = np.isfinite(s)
        num[ok] += wk * s[ok]
        den[ok] += wk
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(den > 0, num / den, np.nan)
    return score, den


def classify(score: np.ndarray) -> np.ndarray:
    c = np.full(score.shape, NOT_ASSESSED, np.uint8)
    f = np.isfinite(score)
    c[f & (score < 1 / 3)] = PRIORITY_LOW
    c[f & (score >= 1 / 3) & (score < 2 / 3)] = PRIORITY_MED
    c[f & (score >= 2 / 3)] = PRIORITY_HIGH
    return c


def compute_priority(inputs: dict[str, np.ndarray | None], cfg: dict, weights: dict[str, float] | None = None,
                     shape: tuple[int, int] | None = None) -> PriorityResult:
    weights = dict(weights or cfg["weights"])
    subs = subscores(inputs, cfg)
    if not subs:
        shape = shape or next((v.shape for v in inputs.values() if v is not None), (1, 1))
        nan = np.full(shape, np.nan, np.float32)
        return PriorityResult(nan, np.zeros(shape, np.uint8), np.zeros(shape, np.float32), {}, nan, 0.0,
                              {"status": "not_assessed", "reason": "no evidence layers available", "weights": weights,
                               "layers_used": [], "layers_missing": list(LAYER_INFO), "assessed_fraction": 0.0,
                               "wording": "Not assessed: no documented evidence layers are available for this dataset."})
    total_w = sum(float(v) for v in weights.values() if v > 0)
    score, den = _combine(subs, weights)
    coverage = (den / total_w).astype(np.float32) if total_w else np.zeros_like(den, np.float32)
    assessed = coverage >= cfg["min_evidence_fraction"] - 1e-9
    for req in cfg.get("require_layers", []):
        assessed &= np.isfinite(subs[req]) if req in subs else False
    score = np.where(assessed, score, np.nan).astype(np.float32)
    classes = classify(score)

    # One-at-a-time sensitivity: each available weight x (1 +/- p), plus drop-one.
    p = cfg.get("sensitivity_perturbation", 0.5)
    scenarios = []
    for k in subs:
        for fac in (1 - p, 1 + p, 0.0):
            w2 = dict(weights)
            w2[k] = weights.get(k, 0.0) * fac
            if sum(v for v in w2.values() if v > 0) == 0:
                continue
            s2, _ = _combine(subs, w2)
            scenarios.append(np.where(assessed, s2, np.nan))
    if scenarios:
        stack = np.stack(scenarios)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns = not-assessed cells
            sens = np.nanstd(np.concatenate([stack, score[None]]), axis=0).astype(np.float32)
        changed = np.zeros(score.shape, bool)
        for s2 in scenarios:
            changed |= (classify(s2) != classes) & assessed
        ccf = float(changed[assessed].mean()) if assessed.any() else 0.0
    else:
        sens, ccf = np.full(score.shape, np.nan, np.float32), 0.0
    missing = [k for k in LAYER_INFO if k not in subs]
    meta = {
        "status": "ok" if assessed.any() else "not_assessed",
        "weights": weights, "layers_used": list(subs), "layers_missing": missing,
        "assessed_fraction": float(assessed.mean()),
        "class_fractions": {PRIORITY_LEGEND[k]["label"]: float((classes == k).mean()) for k in PRIORITY_LEGEND},
        "sensitivity": {"perturbation": p, "scenarios": len(scenarios),
                        "fraction_of_assessed_cells_changing_class": ccf,
                        "median_score_std": float(np.nanmedian(sens)) if np.isfinite(sens).any() else None},
        "wording": "Relative investigation priority from available evidence. Not a detection of water ice.",
    }
    return PriorityResult(score, classes, coverage, subs, sens, ccf, meta)


def explain_cell(res: PriorityResult, inputs: dict[str, np.ndarray | None], r: int, c: int, cfg: dict) -> dict[str, Any]:
    """Per-cell evidence breakdown for the UI inspector."""
    weights = res.meta.get("weights", cfg["weights"])
    rows = []
    den = sum(weights.get(k, 0) for k in res.subs if np.isfinite(res.subs[k][r, c]))
    for k in LAYER_INFO:
        raw = inputs.get(k)
        sub = res.subs.get(k)
        sv = float(sub[r, c]) if sub is not None and np.isfinite(sub[r, c]) else None
        rv = float(raw[r, c]) if raw is not None and np.isfinite(raw[r, c]) else None
        rows.append({
            "layer": k, "description": LAYER_INFO[k], "raw_value": rv, "subscore": sv,
            "weight": weights.get(k, 0.0),
            "contribution": (weights.get(k, 0) * sv / den) if (sv is not None and den) else None,
            "state": "used" if sv is not None else ("not ingested" if raw is None and sub is None else "no data at cell"),
        })
    s = res.score[r, c]
    return {
        "score": float(s) if np.isfinite(s) else None,
        "class": PRIORITY_LEGEND[int(res.classes[r, c])]["label"],
        "evidence_coverage": float(res.coverage[r, c]),
        "sensitivity_std": float(res.sensitivity_std[r, c]) if np.isfinite(res.sensitivity_std[r, c]) else None,
        "evidence": rows,
        "caveat": "Relative priority only. Not a detection or measurement of water ice.",
    }
