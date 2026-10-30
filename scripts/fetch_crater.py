"""Fetch a real-data window for a named crater from the crater library (configs/craters.yaml).

For each crater:
  1. Centre + diameter from the IAU/USGS Gazetteer subset (configs/gazetteer_south80.json).
  2. Square window side = max(window_factor x D, 30 km), centred on the crater, in south polar
     stereographic metres (IAU:30135-style sphere), snapped to the product grid.
  3. PGDA LOLA products (Barker et al. 2023) at 20 / 40 / 80 m depending on window size:
     elevation, error, effective resolution, slope; PSR map (only published at 20 m, so it is
     read through the COG overviews, i.e. averaged to the window resolution -> PSR *fraction*).
  4. Any PGDA-hosted LROC NAC image (co-registered to LOLA) overlapping >= 5 % of the window.
  5. Diviner PRP + Mazarico illumination resampled onto the window (separate scripts, run here).

Usage:
  python scripts/fetch_crater.py --crater crater_haworth
  python scripts/fetch_crater.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.geospatial.grid import polar_stereo_south_forward  # noqa: E402

BASE = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/"
NAC_IDS = ["M1188426614LE", "M1311409938LE", "M1312308500LE", "M139356343LE", "M139729686LE", "M151364517LE",
           "M154940551LE", "M155368056LE", "M184288222LE"]


def gazetteer() -> dict:
    g = json.loads((ROOT / "configs/gazetteer_south80.json").read_text())
    return {f["name"]: f for f in g["features"]}


def plan(crater: dict, factor: float) -> dict:
    gz = gazetteer()[crater["name"]]
    D = gz["diameter_km"] * 1000.0
    side = max(factor * D, 30000.0)
    res = 20 if side <= 40000 else 40 if side <= 80000 else 80
    x, y = polar_stereo_south_forward(gz["lat"], gz["lon"])
    x, y = float(x), float(y)
    half = math.ceil(side / 2 / res) * res
    # snap to product grid (pixel edges at multiples of res)
    x0 = math.floor((x - half) / res) * res
    y0 = math.floor((y - half) / res) * res
    return {"gazetteer": gz, "res_m": res, "bounds": [x0, y0, x0 + 2 * half, y0 + 2 * half], "centre_xy": [x, y]}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_window(url: str, bounds, res: float, out: Path, resampling=Resampling.average) -> tuple[float, str]:
    left, bottom, right, top = bounds
    with rasterio.open("/vsicurl/" + url) as src:
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        w, h = int(round((right - left) / res)), int(round((top - bottom) / res))
        data = src.read(1, window=win, out_shape=(h, w), boundless=True, fill_value=np.nan, resampling=resampling)
        data = data.astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == np.float32(src.nodata)] = np.nan
        data[data < -1e30] = np.nan
        prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=src.crs,
                    transform=rasterio.transform.from_origin(left, top, res, res), nodata=np.nan,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data, 1)
            dst.update_tags(SOURCE_URL=url, RESAMPLING=resampling.name if res != src.res[0] else "none (native)")
        return float(np.isfinite(data).mean()), src.crs.to_wkt()


def fetch(cid: str, cfg: dict) -> dict:
    crater = next(c for c in cfg["craters"] if c["id"] == cid)
    pl = plan(crater, cfg["window_factor"])
    res = pl["res_m"]
    tag = f"{res}MPP"
    out = ROOT / "data/raw/pgda" / cid
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "elevation": f"LDEM_80S_{tag}_ADJ.TIF", "elevation_uncertainty": f"LDEM_80S_{tag}_ADJ_ERR.TIF",
        "effective_resolution": f"LDEM_80S_{tag}_ADJ_EFFRES.TIF", "slope_reference": f"LDSM_80S_{tag}_ADJ.TIF",
        "psr_mask": "LPSR_80S_20MPP_ADJ.TIF",
    }
    rec = {"crater": crater, "plan": pl, "fetched_utc": datetime.now(timezone.utc).isoformat(), "layers": {}}
    for role, name in files.items():
        p = out / name
        if not p.exists():
            print(f"[{cid}] {role}: {name}", flush=True)
            valid, _ = read_window(BASE + name, pl["bounds"], res, p,
                                   Resampling.average if role == "psr_mask" and res != 20 else Resampling.nearest)
        rec["layers"][role] = {"file": str(p.relative_to(ROOT)), "source_url": BASE + name, "sha256": sha256(p),
                               "note": "20 m PSR map averaged to window resolution (fraction)" if role == "psr_mask" and res != 20 else ""}
    # NAC images overlapping the window
    left, bottom, right, top = pl["bounds"]
    best = None
    for nid in NAC_IDS:
        url = f"{BASE}{nid}_20MPP.TIF"
        try:
            with rasterio.open("/vsicurl/" + url) as s:
                b = s.bounds
        except Exception:
            continue
        ix = max(0, min(right, b.right) - max(left, b.left)) * max(0, min(top, b.top) - max(bottom, b.bottom))
        frac = ix / ((right - left) * (top - bottom))
        if frac > 0 and (best is None or frac > best[1]):
            best = (nid, frac, url)
    if best and best[1] >= 0.05:
        nid, frac, url = best
        p = out / f"{nid}_20MPP.TIF"
        if not p.exists():
            print(f"[{cid}] image: {nid} ({100 * frac:.0f}% bbox overlap)", flush=True)
            read_window(url, pl["bounds"], res, p, Resampling.average)
        rec["layers"]["image"] = {"file": str(p.relative_to(ROOT)), "source_url": url, "sha256": sha256(p), "bbox_overlap": frac}
    (out / "fetch_record.json").write_text(json.dumps(rec, indent=2))
    # Derived per-window products (Diviner PRP, illumination) on the same grid
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts/rasterize_diviner_prp.py"), "--aoi", cid], check=False)
    subprocess.run([py, str(ROOT / "scripts/fetch_illumination_aoi.py"), "--aoi", cid], check=False)
    print(f"[{cid}] done", flush=True)
    return rec


def main() -> int:
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    ap = argparse.ArgumentParser()
    ap.add_argument("--crater")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/craters.yaml").read_text())
    ids = [c["id"] for c in cfg["craters"]] if a.all else [a.crater]
    for cid in ids:
        if a.plan_only:
            pl = plan(next(c for c in cfg["craters"] if c["id"] == cid), cfg["window_factor"])
            b = pl["bounds"]
            print(cid, pl["res_m"], f"{(b[2] - b[0]) / 1000:.1f} km", int((b[2] - b[0]) / pl["res_m"]), "px")
        else:
            fetch(cid, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
