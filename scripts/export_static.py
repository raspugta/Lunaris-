"""Build the static web demo (Vercel) from the real pipeline.

Runs the FastAPI app in-process, loads every dataset, plans the suggested route, simulates each rover
profile on nominal (Apollo LRV) soil, and writes all responses + assets to ``site/``:

  site/index.html            landing page
  site/app/index.html        console (static mode: window.LUNARIS_STATIC = "/data")
  site/static/...            frontend code + vendored libraries + NASA visual assets
  site/data/...              precomputed JSON, WebP layers, PDF/JSON reports, 16-bit terrain grids

Nothing is invented: every number in the demo is a stored output of the same code the local app runs.

Usage:  python scripts/export_static.py [--datasets id1,id2,...]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SITE = ROOT / "site"
DATA = SITE / "data"
LOSSLESS = {"hazard", "priority_class", "uncertain", "psr"}
MAX_FRAMES = 1500
MAX_TERRAIN_PX = 600  # close-range patches are interpolated from this grid in the browser


def rnd(o, sig=5):
    """Round floats to `sig` significant digits to keep JSON small (display precision only)."""
    if isinstance(o, float):
        return float(f"{o:.{sig}g}") if np.isfinite(o) else None
    if isinstance(o, list):
        return [rnd(v, sig) for v in o]
    if isinstance(o, dict):
        return {k: rnd(v, sig) for k, v in o.items()}
    return o


def decimate_frames(sim: dict) -> dict:
    F = sim["frames"]
    n = len(F["t"])
    if n <= MAX_FRAMES:
        return sim
    keep = np.unique(np.r_[np.linspace(0, n - 1, MAX_FRAMES).round().astype(int), n - 1])
    sim["frames"] = {k: ([v[i] for i in keep] if isinstance(v, list) else v) for k, v in F.items()}
    return sim


def dump(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.write_text(txt)
    return len(txt)


def to_webp(png: bytes, out: Path, lossless: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(io.BytesIO(png))
    im.save(out, "WEBP", lossless=lossless, quality=86 if not lossless else 100, method=6)


def compress_pdf(path: Path) -> None:
    """Shrink a report with Ghostscript (/ebook: 150 dpi images, fonts kept embedded) when gs is available."""
    gs = shutil.which("gs")
    if not gs:
        return
    tmp = path.with_suffix(".small.pdf")
    r = subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/ebook", "-dCompatibilityLevel=1.6",
                        "-dEmbedAllFonts=true", "-dSubsetFonts=true", f"-sOutputFile={tmp}", str(path)], capture_output=True)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size < path.stat().st_size:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    a = ap.parse_args()

    from fastapi.testclient import TestClient

    import backend.app as appmod
    c = TestClient(appmod.app)

    if SITE.exists():
        shutil.rmtree(SITE)
    # ---- frontend
    shutil.copytree(ROOT / "frontend", SITE / "static")
    (SITE / "index.html").write_text((ROOT / "frontend/landing.html").read_text())
    console = (ROOT / "frontend/index.html").read_text().replace(
        '<script type="importmap">', '<script>window.LUNARIS_STATIC = "/data";</script>\n<script type="importmap">', 1)
    (SITE / "app").mkdir(parents=True)
    (SITE / "app/index.html").write_text(console)
    # ---- global data
    health = c.get("/api/health").json()
    health["environment"] = {"torch": "n/a (web demo)", "cuda": False, "vram_mb": None, "transformers": None}
    dump(DATA / "health.json", health)
    dump(DATA / "config.json", c.get("/api/config").json())
    dump(DATA / "rovers.json", c.get("/api/rovers").json())
    craters = c.get("/api/craters").json()
    dump(DATA / "craters.json", craters)
    all_ds = c.get("/api/datasets").json()
    wanted = [d["id"] for d in all_ds if d["available"] and d["id"] != "lola_pds_875s"]
    if a.datasets:
        wanted = [w for w in wanted if w in a.datasets.split(",")]
    dump(DATA / "datasets.json", [dict(d, loaded=False) for d in all_ds if d["id"] in wanted])
    cfg = c.get("/api/config").json()
    for ds in wanted:
        print(f"[{ds}]", flush=True)
        out = DATA / "ds" / ds
        r = c.post(f"/api/datasets/{ds}/load")
        r.raise_for_status()
        summary = r.json()
        for key, m in summary["layers"].items():
            if not m.get("available") or key == "depth_relative":
                continue
            png = c.get(f"/api/datasets/{ds}/layers/{key}.png")
            if png.status_code == 200:
                to_webp(png.content, out / "layers" / f"{key}.webp", key in LOSSLESS or bool(m.get("categorical")))
        dump(out / "mesh.json", c.get(f"/api/datasets/{ds}/mesh").json())
        dump(out / "named_features.json", c.get(f"/api/datasets/{ds}/named_features").json())
        pts = c.get(f"/api/datasets/{ds}/suggest_points").json()
        dump(out / "suggest_points.json", pts)
        route = c.post(f"/api/datasets/{ds}/route", json={"start": pts["start"], "goal": pts["goal"],
                                                           "weights": cfg["navigation"]["weights"]}).json()
        dump(out / "route.json", route)
        # 16-bit terrain grid (+ PSR plane) for in-browser close-range patches
        an = appmod.WS[ds].analysis
        stride = max(1, int(np.ceil(max(an.arrays["elevation"].shape) / MAX_TERRAIN_PX)))
        z = an.arrays["elevation"][::stride, ::stride].astype(np.float64)
        zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
        zs = (zmax - zmin) / 65535.0 or 1.0
        zq = np.round((np.nan_to_num(z, nan=zmin) - zmin) / zs).astype("<u2")
        psr = an.arrays.get("psr")
        psr = psr[::stride, ::stride] if psr is not None else None
        pq = (np.clip(np.nan_to_num(psr), 0, 1) * 255).astype(np.uint8) if psr is not None else np.zeros(z.shape, np.uint8)
        (out / "terrain.bin").write_bytes(zq.tobytes() + pq.tobytes())
        ta, tb, tc, td, te, tf = an.grid.transform
        dump(out / "terrain.json", {"width": int(z.shape[1]), "height": int(z.shape[0]),
                                    "transform": [ta * stride, tb * stride, tc, td * stride, te * stride, tf],
                                    "zmin": zmin, "zscale": zs, "res_m": an.grid.res_m * stride,
                                    "note": "uint16 heights (zmin + q*zscale, metres) then uint8 PSR fraction*255; analysis grid"})
        # Simulations: viper last so the stored report includes the VIPER run.
        for rover in ("pragyan", "perseverance", "viper"):
            sr = c.post(f"/api/datasets/{ds}/simulate", json={"rover": rover, "soil": "apollo_lrv_nominal", "speed_mode": "nominal"})
            if sr.status_code == 200:
                dump(out / f"sim_{rover}.json", rnd(decimate_frames(sr.json())))
        rj = c.get(f"/api/datasets/{ds}/report.json")
        (out / "report.json").write_text(rj.text)
        rp = c.get(f"/api/datasets/{ds}/report.pdf")
        (out / "report.pdf").write_bytes(rp.content)
        compress_pdf(out / "report.pdf")
        dump(out / "summary.json", c.get(f"/api/datasets/{ds}/summary").json() | {"route": None, "depth": None, "annotations": []})
        th = c.get(f"/api/datasets/{ds}/thumb.png")
        if th.status_code == 200:
            to_webp(th.content, out / "thumb.webp", False)
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site/ written: {total / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
