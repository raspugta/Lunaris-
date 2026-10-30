"""Run the depth model on a dataset's image layer and compare with its elevation."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.config import resolve
from core.geospatial import terrain
from core.reconstruction.compare import compare_depth_to_elevation
from core.reconstruction.depth_adapter import DepthAnythingV2Adapter, DepthModelUnavailable

_ADAPTER: DepthAnythingV2Adapter | None = None


def get_adapter(cfg: dict) -> DepthAnythingV2Adapter:
    global _ADAPTER
    rc = cfg["reconstruction"]
    if _ADAPTER is None or _ADAPTER.model_id != rc["model_id"]:
        _ADAPTER = DepthAnythingV2Adapter(rc["model_id"], rc.get("device", "auto"),
                                          resolve(cfg["paths"]["processed"]) / "depth_cache", rc.get("max_input_side", 1024))
    return _ADAPTER


def run_depth(an, cfg: dict, progress: Callable[[str, float], None] | None = None,
              cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Returns {status, semantics, info, comparison} and adds ``depth_relative`` to the analysis."""
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    ds = an.dataset
    if "image" not in ds.layers:
        return {"status": "unavailable", "reason": "Dataset has no image layer; monocular depth needs an image."}
    img_layer = ds.layers["image"]
    img = img_layer.data
    valid = np.isfinite(img)
    rows, cols = np.nonzero(valid)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = img[r0:r1, c0:c1]
    progress("loading model", 0.1)
    if cancelled():
        return {"status": "cancelled"}
    try:
        adapter = get_adapter(cfg)
        progress("running Depth Anything V2", 0.3)
        res = adapter.predict(sub, img_layer.manifest["sha256"] + f":{r0},{r1},{c0},{c1}", np.isfinite(sub))
    except DepthModelUnavailable as exc:
        return {"status": "model_unavailable", "reason": str(exc), "environment": DepthAnythingV2Adapter.environment()}
    if cancelled():
        return {"status": "cancelled"}
    progress("comparing with LOLA", 0.8)
    full_rel = np.full(img.shape, np.nan, np.float32)
    full_rel[r0:r1, c0:c1] = res.relative
    factor = an.dataset.grid.width // an.grid.width if an.grid.width else 1
    an.arrays["depth_relative"] = terrain.block_reduce_mean(full_rel, factor, 0.5) if factor > 1 else full_rel
    out: dict[str, Any] = {"status": "ok", "semantics": res.info["semantics"],
                           "info": {k: v for k, v in res.info.items() if k != "semantics"},
                           "label": "relative-depth visualisation (uncalibrated)",
                           "crop_px": [int(r0), int(r1), int(c0), int(c1)]}
    if "elevation" in ds.layers and ds.grid.calibrated:
        eff = ds.layers.get("effective_resolution")
        out["comparison"] = compare_depth_to_elevation(
            full_rel, ds.layers["elevation"].data, img, ds.grid.res_m, cfg["reconstruction"]["min_signal_fraction"],
            float(np.nanmedian(eff.data)) if eff is not None else None)
    else:
        out["comparison"] = None
        out["comparison_reason"] = "No co-registered calibrated elevation; only qualitative inspection is possible."
    progress("done", 1.0)
    return out
