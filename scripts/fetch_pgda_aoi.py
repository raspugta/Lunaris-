"""Fetch a windowed subset of the NASA GSFC PGDA LOLA 20 m/px south-polar products.

Source: Barker et al. (2023), PSJ, doi:10.3847/PSJ/acf3e1, PGDA product 90
        https://pgda.gsfc.nasa.gov/products/90
Files are cloud-optimised GeoTIFFs, so only the requested window is downloaded
via HTTP range requests (GDAL /vsicurl/). Outputs go to data/raw/pgda/ and are
treated as immutable raw inputs thereafter. A JSON sidecar records provenance.

Usage:  python scripts/fetch_pgda_aoi.py --aoi shackleton
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/"

# AOIs in IAU:30135 (Moon south polar stereographic, metres): left, bottom, right, top
AOIS = {
    # Shackleton crater + south pole + connecting ridge toward de Gerlache.
    "shackleton": (-12000.0, -16000.0, 20000.0, 16000.0),
}

PRODUCTS = {
    "LDEM_80S_20MPP_ADJ.TIF": ("LOLA", "elevation", "m", "Height relative to 1737.4 km sphere"),
    "LDEM_80S_20MPP_ADJ_ERR.TIF": ("LOLA", "elevation_uncertainty", "m", "Estimated DEM error"),
    "LDEM_80S_20MPP_ADJ_EFFRES.TIF": ("LOLA", "effective_resolution", "m", "Effective DEM resolution"),
    "LDSM_80S_20MPP_ADJ.TIF": ("LOLA", "slope_reference", "deg", "PGDA reference slope map"),
    "LPSR_80S_20MPP_ADJ.TIF": ("LOLA", "psr_mask", "1", "Permanently shadowed region map (LOLA illumination modelling)"),
    "M139729686LE_20MPP.TIF": ("LROC NAC", "nac_image", "I/F (approx.)", "LROC NAC image resampled to 20 m/px on the DEM grid"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="shackleton", choices=sorted(AOIS))
    args = ap.parse_args()
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    left, bottom, right, top = AOIS[args.aoi]
    out_dir = ROOT / "data" / "raw" / "pgda" / args.aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (instrument, kind, unit, desc) in PRODUCTS.items():
        url = BASE + name
        out = out_dir / name
        print(f"[fetch] {name}")
        with rasterio.open("/vsicurl/" + url) as src:
            win = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
            data = src.read(1, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else float("nan"))
            prof = src.profile.copy()
            prof.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(win),
                        driver="GTiff", compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            for k in ("photometric", "interleave"):
                prof.pop(k, None)
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(data, 1)
                dst.update_tags(SOURCE_URL=url, SOURCE_WINDOW=str(win))
            crs_wkt = src.crs.to_wkt()
        records.append({
            "file": name, "source_url": url, "instrument": instrument, "layer_kind": kind,
            "units": unit, "description": desc, "sha256": sha256(out),
        })
    meta = {
        "aoi": args.aoi, "bounds_m": [left, bottom, right, top], "crs_wkt": crs_wkt,
        "fetched_utc": datetime.now(timezone.utc).isoformat(), "citation":
        "Barker, M. K. et al. (2023), Improved LOLA elevation maps for south pole landing sites, PSJ 4, 183, doi:10.3847/PSJ/acf3e1",
        "product_page": "https://pgda.gsfc.nasa.gov/products/90", "files": records,
    }
    (out_dir / "fetch_record.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {len(records)} rasters to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
