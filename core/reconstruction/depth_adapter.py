"""Depth Anything V2 behind a narrow adapter.

Output semantics
----------------
Depth Anything V2 (Yang et al. 2024) is a *monocular relative* depth model
trained on terrestrial imagery. Its raw output is an affine-invariant
inverse-depth-like quantity ("disparity"): larger = closer to the camera.
For a nadir orbital image "closer to the camera" loosely corresponds to
"higher terrain", but the mapping is unknown, non-metric, and strongly
confounded by illumination (shadows, albedo). We therefore:

* keep the raw output unmodified (``raw``),
* expose a min-max normalised copy labelled *relative depth, unitless*,
* never convert it to metres unless an explicit calibration is applied and
  recorded (see :mod:`core.reconstruction.compare`).

The model is loaded lazily, once, on GPU when available (falling back to CPU),
and results are cached on disk keyed by (image checksum, model id, input size).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import stable_hash

log = logging.getLogger(__name__)


@dataclass
class DepthResult:
    raw: np.ndarray            # model output resized to the input grid, float32
    relative: np.ndarray       # min-max normalised [0, 1] over valid pixels
    info: dict[str, Any]


class DepthModelUnavailable(RuntimeError):
    pass


class DepthAnythingV2Adapter:
    def __init__(self, model_id: str, device: str = "auto", cache_dir: Path | None = None, max_side: int = 1024):
        self.model_id = model_id
        self.requested_device = device
        self.cache_dir = cache_dir
        self.max_side = max_side
        self._model = None
        self._processor = None
        self.device = None

    # -- environment -----------------------------------------------------
    @staticmethod
    def environment() -> dict[str, Any]:
        env: dict[str, Any] = {"torch": None, "transformers": None, "cuda": False, "vram_mb": None}
        try:
            import torch
            env["torch"] = torch.__version__
            env["cuda"] = bool(torch.cuda.is_available())
            if env["cuda"]:
                env["vram_mb"] = int(torch.cuda.get_device_properties(0).total_memory // 2**20)
        except Exception as exc:  # pragma: no cover - environment dependent
            env["error"] = f"torch import failed: {exc}"
        try:
            import transformers
            env["transformers"] = transformers.__version__
        except Exception as exc:  # pragma: no cover
            env["error"] = f"transformers import failed: {exc}"
        return env

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise DepthModelUnavailable(
                f"PyTorch/transformers not installed ({exc}). Install with: pip install -r requirements-ml.txt") from exc
        dev = self.requested_device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # Small model needs ~0.5 GB; refuse GPU if less than 1.5 GB total VRAM.
            if dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 1.5 * 2**30:
                dev = "cpu"
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(dev).eval()
        except Exception as exc:
            raise DepthModelUnavailable(
                f"Could not load '{self.model_id}' ({exc}). First run needs internet to download weights "
                "(~100 MB) into the Hugging Face cache; afterwards it works offline.") from exc
        self.device = dev
        log.info("Loaded %s on %s", self.model_id, dev)

    # -- inference -------------------------------------------------------
    def predict(self, image01: np.ndarray, image_sha256: str, valid: np.ndarray | None = None) -> DepthResult:
        """Run inference on a single-band image scaled to [0, 1] (NaN = no data)."""
        key = stable_hash({"img": image_sha256, "model": self.model_id, "max_side": self.max_side, "v": 2})
        cache = self.cache_dir / f"depth_{key}.npz" if self.cache_dir else None
        if cache and cache.exists():
            z = np.load(cache, allow_pickle=False)
            info = {k: z["info_" + k].item() for k in ("model_id", "device", "runtime_s", "input_size")}
            info["cached"] = True
            return DepthResult(z["raw"], z["relative"], info | {"semantics": SEMANTICS})
        self._load()
        import torch
        from PIL import Image

        h, w = image01.shape
        valid = np.isfinite(image01) if valid is None else valid
        img = np.where(valid, image01, 0.0)
        hi = np.percentile(img[valid], 99.5) if valid.any() else 1.0
        img8 = (np.clip(img / max(hi, 1e-9), 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img8).convert("RGB")
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            pil = pil.resize((max(14, int(w * scale)), max(14, int(h * scale))), Image.BILINEAR)
        t0 = time.time()
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self._model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)
        raw = pred.squeeze().float().cpu().numpy().astype(np.float32)
        runtime = time.time() - t0
        raw[~valid] = np.nan
        rel = normalise(raw)
        info = {"model_id": self.model_id, "device": self.device, "runtime_s": round(runtime, 2),
                "input_size": f"{pil.width}x{pil.height}", "cached": False}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, raw=raw, relative=rel, **{"info_" + k: np.array(v) for k, v in info.items() if k != "cached"})
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return DepthResult(raw, rel, info | {"semantics": SEMANTICS})


SEMANTICS = ("Relative (affine-invariant) monocular depth from Depth Anything V2; unitless; larger = "
             "predicted closer to camera. Not elevation, not metres. Strongly affected by illumination.")


def normalise(a: np.ndarray) -> np.ndarray:
    fin = np.isfinite(a)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    if fin.any():
        lo, hi = np.percentile(a[fin], [1, 99])
        out[fin] = np.clip((a[fin] - lo) / max(hi - lo, 1e-9), 0, 1)
    return out
