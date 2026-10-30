"""FastAPI backend: thin HTTP layer over the core modules + static frontend.

Run:  python -m uvicorn backend.app:app --port 8000
"""
from __future__ import annotations

import base64
import sys
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import ROOT, load_config, resolve
from core.ice_priority.evidence import compute_priority, explain_cell
from core.ingestion.loader import IngestError, load_raster, sha256_file
from core.navigation.astar import PlanningError
from core.navigation.planner import plan_route, suggest_points
from core.pipeline import PIPELINE_VERSION, analyse, build_dataset, load_dataset, registry_status, save_layers
from core.reconstruction.depth_adapter import DepthAnythingV2Adapter
from core.reconstruction.run import run_depth
from core.reporting.figures import colorize, legend_spec, png_bytes
from core.reporting.report import build_pdf, build_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lunar-psr")
CFG = load_config()
app = FastAPI(title="LUNARIS — Lunar PSR Intelligence Platform", version=PIPELINE_VERSION)


@dataclass
class Workspace:
    analysis: Any
    route: dict | None = None
    depth: dict | None = None
    annotations: list[dict] = field(default_factory=list)
    sim: dict | None = None
    sim_frames: dict | None = None
    alvik: dict | None = None
    png_cache: dict[str, bytes] = field(default_factory=dict)


WS: dict[str, Workspace] = {}
UPLOAD_SPECS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _json(o: Any) -> Any:
    from core.reporting.report import _clean
    return _clean(o)


def _ws(ds_id: str) -> Workspace:
    if ds_id not in WS:
        raise HTTPException(404, f"Dataset '{ds_id}' is not loaded. POST /api/datasets/{ds_id}/load first.")
    return WS[ds_id]


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    return {"status": "ok", "version": PIPELINE_VERSION, "environment": DepthAnythingV2Adapter.environment(),
            "model_id": CFG["reconstruction"]["model_id"]}


@app.get("/api/config")
def get_config():
    return CFG


@app.get("/api/datasets")
def datasets():
    items = registry_status(CFG)
    for u in UPLOAD_SPECS.values():
        items.append({"id": u["id"], "title": u["title"], "kind": "upload", "available": True,
                      "layers_present": {k: True for k in u["layers"]}})
    for it in items:
        it["loaded"] = it["id"] in WS
    return items


def _summary(ws: Workspace) -> dict:
    an = ws.analysis
    ds = an.dataset
    return _json({
        "dataset": ds.manifest(), "grid": an.grid.to_dict(), "config_hash": an.config_hash,
        "layers": {k: m.to_dict() | {"legend": legend_spec(m) if m.available else None} for k, m in an.meta.items()}
                  | ({"depth_relative": _depth_meta(an)} if "depth_relative" in an.arrays else {}),
        "stats": an.stats, "processing_log": an.processing_log, "features": an.features[:200],
        "route": ws.route, "depth": ws.depth, "annotations": ws.annotations,
    })


def _depth_meta(an) -> dict:
    from core.provenance import LayerMeta, Status
    m = LayerMeta("depth_relative", "Relative depth (Depth Anything V2)", Status.MODEL_ESTIMATE, "unitless (relative)",
                  "Monocular relative depth from the image layer; larger = predicted closer to camera.",
                  "Depth Anything V2 Small on LROC NAC", "core/reconstruction/depth_adapter.py",
                  ["Relative-depth visualisation, NOT elevation or metres.", "Shading/albedo dominate the prediction in polar light."],
                  an.grid.res_m, "relative", (0.0, 1.0))
    an.meta["depth_relative"] = m
    return m.to_dict() | {"legend": legend_spec(m)}


@app.post("/api/datasets/{ds_id}/load")
def load(ds_id: str):
    t0 = time.time()
    try:
        if ds_id in UPLOAD_SPECS:
            spec = UPLOAD_SPECS[ds_id]
            ds = build_dataset(spec, CFG)
        else:
            ds = load_dataset(ds_id, CFG)
        an = analyse(ds, CFG)
    except IngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    save_layers(an, resolve(CFG["paths"]["processed"]) / "analyses")
    with LOCK:
        WS[ds_id] = Workspace(an)
    log.info("loaded %s in %.1fs (config %s)", ds_id, time.time() - t0, an.config_hash)
    return _summary(WS[ds_id])


@app.get("/api/datasets/{ds_id}/summary")
def summary(ds_id: str):
    return _summary(_ws(ds_id))


@app.get("/api/datasets/{ds_id}/layers/{key}.png")
def layer_png(ds_id: str, key: str):
    ws = _ws(ds_id)
    an = ws.analysis
    if key not in an.arrays:
        raise HTTPException(404, f"Layer '{key}' not available for this dataset.")
    if key == "depth_relative":
        _depth_meta(an)
    ck = f"{key}:{id(an.arrays[key])}"
    if ck not in ws.png_cache:
        ws.png_cache[ck] = png_bytes(colorize(an.arrays[key], an.meta[key]))
    return Response(ws.png_cache[ck], media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/datasets/{ds_id}/mesh")
def mesh(ds_id: str):
    """Downsampled elevation (or relative depth) heightfield for the 3D viewport."""
    an = _ws(ds_id).analysis
    key = "elevation" if "elevation" in an.arrays else ("depth_relative" if "depth_relative" in an.arrays else None)
    if key is None:
        raise HTTPException(404, "No elevation or relative-depth layer to build a 3D surface from.")
    z = an.arrays[key]
    side = CFG["viewer"]["mesh_max_side"]
    step = max(1, int(np.ceil(max(z.shape) / side)))
    zs = z[::step, ::step].astype(np.float32)
    fin = np.isfinite(zs)
    lo, hi = (float(np.nanmin(zs)), float(np.nanmax(zs))) if fin.any() else (0.0, 1.0)
    zs = np.where(fin, zs, lo).astype("<f4")
    return {"key": key, "width": zs.shape[1], "height": zs.shape[0], "step": step, "min": lo, "max": hi,
            "cell_m": (an.grid.res_m * step) if an.grid.res_m and key == "elevation" else None,
            "calibrated": key == "elevation" and an.grid.res_m is not None,
            "label": "LOLA elevation (m)" if key == "elevation" else "Relative-depth visualisation (unitless, not elevation)",
            "nodata_mask": base64.b64encode(np.packbits(~fin).tobytes()).decode(),
            "z": base64.b64encode(zs.tobytes()).decode()}


@app.get("/api/datasets/{ds_id}/inspect")
def inspect(ds_id: str, row: int, col: int):
    an = _ws(ds_id).analysis
    if not (0 <= row < an.grid.height and 0 <= col < an.grid.width):
        raise HTTPException(400, "cell outside grid")
    vals = {}
    for k, a in an.arrays.items():
        v = float(a[row, col])
        m = an.meta.get(k)
        entry = {"value": v if np.isfinite(v) else None, "units": m.units if m else "", "title": m.title if m else k,
                 "status": m.status.value if m else ""}
        if m and m.categorical and np.isfinite(v):
            entry["label"] = m.categorical[int(v)]["label"]
        vals[k] = entry
    x, y = an.grid.pixel_to_world(row, col)
    return _json({"row": row, "col": col, "x_m": float(x) if an.grid.calibrated else None,
                  "y_m": float(y) if an.grid.calibrated else None, "lat_lon": an.lat_lon(row, col), "values": vals,
                  "priority": explain_cell(an.priority, an.priority_inputs, row, col, CFG["ice_priority"])})


class RouteReq(BaseModel):
    start: list[int]
    goal: list[int]
    weights: dict[str, float] | None = None


@app.post("/api/datasets/{ds_id}/route")
def route(ds_id: str, req: RouteReq):
    ws = _ws(ds_id)
    ws.sim = None  # a new route invalidates any previous simulation
    try:
        r = plan_route(ws.analysis, tuple(req.start), tuple(req.goal), CFG,
                       {"weights": req.weights} if req.weights else None)
    except PlanningError as exc:
        raise HTTPException(400, str(exc)) from exc
    ws.route = _json(r)
    return ws.route


@app.get("/api/datasets/{ds_id}/suggest_points")
def suggest(ds_id: str):
    return suggest_points(_ws(ds_id).analysis.arrays["cost"])


class PriorityReq(BaseModel):
    weights: dict[str, float]


@app.post("/api/datasets/{ds_id}/priority")
def priority(ds_id: str, req: PriorityReq):
    ws = _ws(ds_id)
    an = ws.analysis
    if any(v < 0 for v in req.weights.values()):
        raise HTTPException(400, "weights must be non-negative")
    pr = compute_priority(an.priority_inputs, CFG["ice_priority"], req.weights, shape=(an.grid.height, an.grid.width))
    an.priority = pr
    an.arrays["priority_score"] = pr.score
    an.arrays["priority_class"] = pr.classes.astype(np.float32)
    an.arrays["evidence_coverage"] = pr.coverage
    an.arrays["priority_sensitivity"] = pr.sensitivity_std
    an.stats["priority"] = pr.meta
    ws.png_cache.clear()
    return _json(pr.meta)


# ------------------------------------------------------------------ crater library
THUMBS = resolve(CFG["paths"]["processed"]) / "thumbs"
_PSR_FRAC: dict = {}


@app.get("/api/craters")
def craters():
    """Crater library cards: gazetteer facts + availability + thumbnail URL."""
    from core.pipeline import load_registry
    out = []
    for d in load_registry(resolve(CFG["paths"]["datasets_registry"])):
        c = d.get("crater")
        if not c:
            continue
        elev = d["layers"].get("elevation", {}).get("file")
        avail = bool(elev) and resolve(elev).exists()
        psr_frac = None
        pp = d["layers"].get("psr_mask", {}).get("file")
        if avail and pp and resolve(pp).exists():
            key = (pp, resolve(pp).stat().st_mtime)
            if key not in _PSR_FRAC:
                import rasterio
                with rasterio.open(resolve(pp)) as src:
                    a = src.read(1, out_shape=(max(1, src.height // 4), max(1, src.width // 4))).astype(np.float32)
                _PSR_FRAC[key] = float(np.nanmean(np.clip(a, 0, 1)))
            psr_frac = _PSR_FRAC[key]
        out.append({"id": d["id"], "title": d["title"], **c, "available": avail, "fetch": d.get("fetch"), "psr_fraction": psr_frac,
                    "region_note": d.get("region_note"), "covers_features": d.get("covers_features", []),
                    "layers_present": {r: resolve(v["file"]).exists() for r, v in d["layers"].items()},
                    "thumb": f"/api/datasets/{d['id']}/thumb.png" if avail else None})
    return {"source": "IAU/USGS Gazetteer of Planetary Nomenclature", "craters": out}


@app.get("/api/datasets/{ds_id}/thumb.png")
def thumb(ds_id: str):
    """Shaded-relief thumbnail from the dataset's LOLA DEM (illustrative lighting), PSR tinted blue."""
    from core.geospatial.terrain import hillshade
    from core.pipeline import load_registry
    spec = next((d for d in load_registry(resolve(CFG["paths"]["datasets_registry"])) if d["id"] == ds_id), None)
    if spec is None or "elevation" not in spec["layers"]:
        raise HTTPException(404, "unknown dataset or no elevation layer")
    ep = resolve(spec["layers"]["elevation"]["file"])
    if not ep.exists():
        raise HTTPException(404, f"not downloaded: {spec.get('fetch')}")
    THUMBS.mkdir(parents=True, exist_ok=True)
    out = THUMBS / f"{ds_id}.png"
    if not out.exists() or out.stat().st_mtime < ep.stat().st_mtime:
        import rasterio
        from PIL import Image
        with rasterio.open(ep) as src:
            f = max(1, max(src.width, src.height) // 360)
            z = src.read(1, out_shape=(src.height // f, src.width // f)).astype(np.float32)
            res = abs(src.transform.a) * f
        z[z < -1e30] = np.nan
        hs = hillshade(z, res, azimuth_deg=315, altitude_deg=25)
        hs = np.nan_to_num(hs, nan=0.0)
        rgb = np.dstack([hs * 225 + 18, hs * 222 + 18, hs * 214 + 20])
        pp = spec["layers"].get("psr_mask")
        if pp and resolve(pp["file"]).exists():
            with rasterio.open(resolve(pp["file"])) as src:
                psr = np.nan_to_num(src.read(1, out_shape=z.shape).astype(np.float32))
            m = np.clip(psr, 0, 1)[..., None]
            rgb = rgb * (1 - 0.55 * m) + np.array([40, 95, 210]) * 0.55 * m
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out)
    return FileResponse(out, media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/api/datasets/{ds_id}/named_features")
def named_features(ds_id: str):
    """IAU-named features whose centres fall inside the loaded window, in analysis-grid pixels."""
    from core.craters import features_in_bounds
    an = _ws(ds_id).analysis
    if not an.grid.polar_stereo_south:
        return {"features": []}
    feats = features_in_bounds(an.grid.bounds())
    res = an.grid.res_m
    out = []
    for f in feats:
        r, c = an.grid.world_to_pixel(f["x"], f["y"])
        out.append({"name": f["name"], "type": f["type"], "diameter_km": f["diameter_km"], "row": float(r), "col": float(c),
                    "radius_px": f["diameter_km"] * 500.0 / res, "coverage": f["coverage"], "link": f["link"]})
    return {"source": "IAU/USGS Gazetteer of Planetary Nomenclature", "features": out}


# ------------------------------------------------------------------ Arduino Alvik bridge
ALVIK_DIR = resolve(CFG["paths"]["processed"]) / "alvik"


class AlvikReq(BaseModel):
    arena_mm: float = 1219.2
    margin_mm: float = 120.0
    tolerance_mm: float = 6.0
    min_segment_mm: float = 15.0
    max_segments: int = 80
    stop_at_failure: bool = True


@app.post("/api/datasets/{ds_id}/alvik/program")
def alvik_program(ds_id: str, req: AlvikReq):
    """Scale the current route into the arena and generate the Alvik main.py."""
    from core.hardware.alvik import build_program, micropython_source
    ws = _ws(ds_id)
    if not ws.route or not ws.route.get("found"):
        raise HTTPException(400, "Plan a route first (Navigate step).")
    an = ws.analysis
    cells = np.array(ws.route["cells"], float)
    x, y = an.grid.pixel_to_world(cells[:, 0], cells[:, 1])
    frame_s = tags = None
    stop_at = None
    sim_desc = "not simulated"
    if ws.sim and getattr(ws, "sim_frames", None):
        F = ws.sim_frames
        p = ws.sim["rover"]["params"]
        slip_ops = p["max_slip_ops"]["value"]
        frame_s = np.asarray(F["s"], float)
        tags = ["psr" if F["in_psr"][i] else "slip" if F["slip"][i] > slip_ops else "go" for i in range(len(frame_s))]
        if ws.sim["summary"].get("failure"):
            if req.stop_at_failure:
                stop_at = float(ws.sim["summary"]["distance_m"])
            tags[-1] = "fail"
        sim_desc = f"{ws.sim['rover']['name']} on {ws.sim['soil']['name']}: {ws.sim['summary']['verdict']}"
    try:
        prog = build_program(np.c_[x, y], arena_mm=req.arena_mm, margin_mm=req.margin_mm, tolerance_mm=req.tolerance_mm,
                             min_segment_mm=req.min_segment_mm, max_segments=req.max_segments,
                             frame_s=frame_s, frame_tags=tags, stop_at_s=stop_at)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    src = micropython_source(prog, {"dataset_title": an.dataset.title, "sim": sim_desc})
    ALVIK_DIR.mkdir(parents=True, exist_ok=True)
    path = ALVIK_DIR / f"{ds_id}_main.py"
    path.write_text(src)
    ws.alvik = {"program": prog, "path": str(path)}
    return _json({"program": prog, "source": src, "sim": sim_desc, "download": f"/api/datasets/{ds_id}/alvik/main.py"})


@app.get("/api/datasets/{ds_id}/alvik/main.py")
def alvik_download(ds_id: str):
    ws = _ws(ds_id)
    if not getattr(ws, "alvik", None):
        raise HTTPException(404, "Generate the Alvik program first.")
    return FileResponse(ws.alvik["path"], media_type="text/x-python", filename="main.py")


@app.get("/api/alvik/ports")
def alvik_ports(all: bool = False):
    """USB serial ports (the Alvik's Nano ESP32 enumerates as a USB CDC device, Arduino VID 0x2341)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return {"ports": [], "error": "pyserial/mpremote not installed: pip install mpremote"}
    ports = []
    for p in list_ports.comports():
        if p.vid is None and not all:
            continue
        ports.append({"device": p.device, "description": p.description, "vid": p.vid, "pid": p.pid,
                      "likely_alvik": p.vid == 0x2341})
    ports.sort(key=lambda p: not p["likely_alvik"])
    return {"ports": ports}


class UploadReq(BaseModel):
    port: str
    backup: bool = True
    run_now: bool = True


@app.post("/api/datasets/{ds_id}/alvik/upload")
def alvik_upload(ds_id: str, req: UploadReq):
    """Copy the generated program to the Alvik as :main.py with mpremote (backing up the existing main.py)."""
    import shutil
    import subprocess
    ws = _ws(ds_id)
    if not getattr(ws, "alvik", None):
        raise HTTPException(400, "Generate the Alvik program first.")
    exe = shutil.which("mpremote") or str(Path(sys.executable).with_name("mpremote"))
    if not Path(exe).exists():
        raise HTTPException(500, "mpremote not found. Install with: pip install mpremote")
    if not req.port.startswith("/dev/") and not req.port.upper().startswith("COM"):
        raise HTTPException(400, "Port must look like /dev/ttyACM0 or COM3.")
    log = []

    def run(args, timeout=60):
        r = subprocess.run([exe, "connect", req.port, *args], capture_output=True, text=True, timeout=timeout)
        log.append({"cmd": "mpremote connect %s %s" % (req.port, " ".join(args)), "rc": r.returncode,
                    "out": (r.stdout + r.stderr)[-2000:]})
        return r.returncode
    try:
        backup_path = None
        if req.backup:
            ALVIK_DIR.mkdir(parents=True, exist_ok=True)
            backup_path = ALVIK_DIR / f"backup_main_{time.strftime('%Y%m%d_%H%M%S')}.py"
            if run(["fs", "cp", ":main.py", str(backup_path)]) != 0:
                backup_path = None  # no main.py on the board yet (or read failed); continue
        if run(["fs", "cp", ws.alvik["path"], ":main.py"]) != 0:
            raise HTTPException(502, {"message": "Upload failed — is the Alvik switched ON and connected by USB, "
                                      "and is no other program (Arduino Lab, serial monitor) holding the port?", "log": log})
        if req.run_now:
            run(["reset"], timeout=20)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, {"message": "mpremote timed out talking to the board.", "log": log})
    return {"ok": True, "port": req.port, "backup": str(backup_path) if backup_path else None, "log": log,
            "next": "Place the Alvik at the marked start pose, then press the OK (check) button on the robot."}


# ------------------------------------------------------------------ rover simulation
@app.get("/api/rovers")
def rovers():
    from core.rover.sim import load_rover_db
    db = load_rover_db()
    return {"rovers": {k: {"name": v["name"], "agency": v["agency"], "params": v["params"], "references": v["references"],
                           "fidelity_warning": v.get("fidelity_warning"), "designed_for_psr": v.get("designed_for_psr")}
                       for k, v in db["rovers"].items()},
            "soils": db["soils"], "environment": db["environment"]}


class SimReq(BaseModel):
    rover: str = "viper"
    soil: str = "apollo_lrv_nominal"
    speed_mode: str = "nominal"
    sunlit_load_w: float | None = None


@app.post("/api/datasets/{ds_id}/simulate")
def simulate(ds_id: str, req: SimReq):
    from core.rover.run import simulate_route
    ws = _ws(ds_id)
    if req.speed_mode not in ("nominal", "max"):
        raise HTTPException(400, "speed_mode must be 'nominal' or 'max'")
    try:
        res = simulate_route(ws.analysis, ws.route, req.rover, req.soil, req.speed_mode, req.sunlit_load_w)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ws.sim = _json({k: v for k, v in res.items() if k != "frames"})
    ws.sim_frames = {k: (np.asarray(v).tolist() if v is not None else None) for k, v in res["frames"].items()
                     if k in ("s", "in_psr", "slip")}
    return _json(res)


@app.get("/api/datasets/{ds_id}/terrain_patch")
def terrain_patch(ds_id: str, x: float, y: float, size: float = 600.0, spacing: float = 4.0, interp: str = "linear"):
    """Bilinearly upsampled native-DEM patch around (x, y) for the close-up simulation view.

    Heights between the 20 m DEM posts are interpolated, not measured; the response says so.
    """
    from core.rover.sim import Sampler
    an = _ws(ds_id).analysis
    lyr = an.dataset.layers.get("elevation")
    if lyr is None or not an.dataset.grid.calibrated:
        raise HTTPException(404, "no calibrated elevation")
    size = float(min(max(size, 50.0), 2000.0))
    spacing = float(min(max(spacing, 1.0), 20.0))
    n = int(size / spacing) + 1
    xs = x - size / 2 + np.arange(n) * spacing
    ys = y + size / 2 - np.arange(n) * spacing
    X, Y = np.meshgrid(xs, ys)
    if interp not in ("linear", "cubic"):
        raise HTTPException(400, "interp must be 'linear' or 'cubic'")
    if interp == "cubic":
        # Visual-only: cubic spline through the same LOLA posts (C1-smooth, no 20 m shading creases).
        from scipy import ndimage
        a_, _, c_, _, e_, f_ = an.dataset.grid.transform
        col = (X.ravel() - c_) / a_ - 0.5
        row = (Y.ravel() - f_) / e_ - 0.5
        z = ndimage.map_coordinates(np.nan_to_num(lyr.data.astype(np.float64), nan=float(np.nanmean(lyr.data))), [row, col], order=3, mode="nearest").reshape(n, n)
    else:
        z = Sampler(lyr.data.astype(np.float64), an.dataset.grid.transform)(X.ravel(), Y.ravel()).reshape(n, n)
    psr = an.dataset.layers.get("psr_mask")
    pz = Sampler(np.nan_to_num(psr.data).astype(np.float64), an.dataset.grid.transform)(X.ravel(), Y.ravel()).reshape(n, n) if psr else np.zeros_like(z)
    fin = np.isfinite(z)
    zmin = float(np.nanmin(z)) if fin.any() else 0.0
    z = np.where(fin, z, zmin).astype("<f4")
    return {"n": n, "spacing": spacing, "x0": float(xs[0]), "y0": float(ys[0]), "zmin": zmin,
            "z": base64.b64encode(z.tobytes()).decode(),
            "psr": base64.b64encode((pz >= 0.5).astype(np.uint8).tobytes()).decode(),
            "interp": interp,
            "note": f"{'Cubic-spline' if interp == 'cubic' else 'Bilinear'} interpolation of the {an.dataset.grid.res_m:.0f} m LOLA DEM; sub-{an.dataset.grid.res_m:.0f} m relief is not measured."}


# ------------------------------------------------------------------ jobs (depth inference)
def _job_runner(job_id: str, ds_id: str):
    job = JOBS[job_id]
    ws = WS[ds_id]
    def prog(stage, frac):
        job.update(stage=stage, progress=frac)
    try:
        r = run_depth(ws.analysis, CFG, prog, lambda: job.get("cancel", False))
        if r.get("status") == "ok":
            ws.depth = _json(r)
            ws.png_cache.clear()
        job.update(state="cancelled" if r.get("status") == "cancelled" else "done", result=_json(r), progress=1.0)
    except Exception as exc:  # report, never crash the server
        log.exception("depth job failed")
        job.update(state="failed", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/datasets/{ds_id}/depth")
def depth(ds_id: str):
    _ws(ds_id)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "state": "running", "stage": "queued", "progress": 0.0, "dataset": ds_id}
    threading.Thread(target=_job_runner, args=(job_id, ds_id), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    JOBS[job_id]["cancel"] = True
    JOBS[job_id]["stage"] = "cancelling (takes effect between inference stages)"
    return JOBS[job_id]


# ------------------------------------------------------------------ annotations
class Annotation(BaseModel):
    row: int
    col: int
    label: str
    note: str = ""


@app.get("/api/datasets/{ds_id}/annotations")
def get_annotations(ds_id: str):
    return _ws(ds_id).annotations


@app.post("/api/datasets/{ds_id}/annotations")
def add_annotation(ds_id: str, a: Annotation):
    ws = _ws(ds_id)
    item = a.model_dump() | {"id": uuid.uuid4().hex[:8], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "source": "manual annotation (user)", "lat_lon": ws.analysis.lat_lon(a.row, a.col)}
    ws.annotations.append(item)
    return item


@app.delete("/api/datasets/{ds_id}/annotations/{ann_id}")
def del_annotation(ds_id: str, ann_id: str):
    ws = _ws(ds_id)
    ws.annotations = [x for x in ws.annotations if x["id"] != ann_id]
    return {"ok": True}


# ------------------------------------------------------------------ reports
def _report(ds_id: str) -> tuple[dict, Workspace]:
    ws = _ws(ds_id)
    s = build_summary(ws.analysis, ws.route, ws.depth, CFG, ws.annotations)
    s["rover_simulation"] = ws.sim
    return s, ws


@app.get("/api/datasets/{ds_id}/report.json")
def report_json(ds_id: str):
    s, _ = _report(ds_id)
    return JSONResponse(s, headers={"Content-Disposition": f'attachment; filename="{ds_id}_analysis.json"'})


@app.get("/api/datasets/{ds_id}/report.pdf")
def report_pdf(ds_id: str):
    s, ws = _report(ds_id)
    r = ws.route or {}
    pdf = build_pdf(s, ws.analysis, r.get("cells"), r.get("alternative_paths"))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{ds_id}_mission_brief.pdf"'})


# ------------------------------------------------------------------ upload
ROLE_CHOICES = {"elevation", "image"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), role: str = Form("elevation"), title: str = Form(""),
                 source_url: str = Form(""), instrument: str = Form(""), product_id: str = Form("")):
    if role not in ROLE_CHOICES:
        raise HTTPException(400, f"role must be one of {sorted(ROLE_CHOICES)}")
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(415, "Upload a GeoTIFF (.tif) elevation/image raster or a PNG/JPEG image.")
    data = await file.read()
    if len(data) > 300 * 2**20:
        raise HTTPException(413, "File larger than 300 MB; crop to an area of interest first.")
    tmp = resolve(CFG["paths"]["raw"]) / "uploads" / "_incoming"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / name
    p.write_bytes(data)
    digest = sha256_file(p)
    final = resolve(CFG["paths"]["raw"]) / "uploads" / digest[:16] / name
    final.parent.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        p.replace(final)
    else:
        p.unlink()
    meta = {"source_name": title or name, "source_url": source_url or None, "instrument": instrument or None,
            "product_id": product_id or None, "status": "measured" if role == "elevation" else "measured"}
    try:
        lyr = load_raster(final, meta)
    except IngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    if role == "elevation" and not lyr.grid.calibrated:
        role_note = "Elevation upload has no georeferencing; it will be treated as an image (relative depth only)."
        role = "image"
    else:
        role_note = None
    ds_id = f"upload_{digest[:10]}"
    UPLOAD_SPECS[ds_id] = {
        "id": ds_id, "title": f"Upload: {title or name}" + ("" if lyr.manifest["source_verified"] else " (source unverified)"),
        "kind": "upload", "region_note": "User-supplied file. Coverage and provenance as declared by the uploader.",
        "covers_features": [], "citation": source_url or "unverified",
        "layers": {role: {"file": str(final.relative_to(ROOT)), **{k: v for k, v in meta.items() if v}}},
    }
    return _json({"dataset_id": ds_id, "role": role, "manifest": lyr.manifest, "warnings": lyr.warnings + ([role_note] if role_note else [])})


# ------------------------------------------------------------------ static frontend
@app.get("/")
def landing():
    return FileResponse(ROOT / "frontend" / "landing.html")


@app.get("/app")
def console_app():
    return FileResponse(ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
