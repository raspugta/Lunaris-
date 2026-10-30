"""Average sun / Earth visibility for an AOI (Mazarico et al. 2011, Icarus 211:1066).

Sources (PGDA product 69):
  sun  : https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF (+ .LBL from imbrium.mit.edu)
  earth: http://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.IMG (+ .LBL)
Both are int16 DN on a 60 m south polar stereographic grid. Value = DN x SCALING_FACTOR
(0.00004, from the labels) = fraction of time the Sun / Earth disc is visible, averaged over
18.6 years. The Earth IMG is fetched with HTTP range requests for the AOI rows only.
Output: data/processed/illumination/<aoi>/{sun,earth}_avg_visibility.tif (float32 fraction, NaN no-data).

Usage:
  curl -O https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF   # into data/raw/illumination/
  curl -O http://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608.LBL
  curl -O http://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.LBL
  python scripts/fetch_illumination_aoi.py --aoi shackleton
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.ingestion import pds3  # noqa: E402

AOIS = {"shackleton": (-12000.0, -16000.0, 20000.0, 16000.0)}
RAW = ROOT / "data/raw/illumination"
EARTH_URL = "http://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.IMG"
CRS = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"


def label_geom(lbl_path: Path):
    lbl = pds3.read_label(lbl_path)
    img = lbl.get("UNCOMPRESSED_FILE", lbl).get("IMAGE") or lbl["IMAGE"]
    proj = lbl["IMAGE_MAP_PROJECTION"]
    scale = float(pds3.value(proj["MAP_SCALE"]))
    l0 = float(pds3.value(proj["LINE_PROJECTION_OFFSET"]))
    s0 = float(pds3.value(proj["SAMPLE_PROJECTION_OFFSET"]))
    return (int(img["LINES"]), int(img["LINE_SAMPLES"]), float(pds3.value(img["SCALING_FACTOR"])), scale,
            (-0.5 - s0) * scale, (l0 + 0.5) * scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="shackleton")
    a = ap.parse_args()
    if a.aoi in AOIS:
        left, bottom, right, top = AOIS[a.aoi]
    else:
        cands = sorted((ROOT / f"data/raw/pgda/{a.aoi}").glob("LDEM_80S_*MPP_ADJ.TIF"))
        if not cands:
            print(f"unknown AOI '{a.aoi}'", file=sys.stderr)
            return 2
        with rasterio.open(cands[0]) as d:
            left, bottom, right, top = d.bounds
    out = ROOT / f"data/processed/illumination/{a.aoi}"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"citation": "Mazarico et al. (2011) Icarus 211:1066, doi:10.1016/j.icarus.2010.10.030", "files": {}}
    for kind in ("sun", "earth"):
        lbl = RAW / ("AVGVISIB_85S_060M_201608.LBL" if kind == "sun" else "AVGVISIB_85S_060M_201608_EARTH.LBL")
        lines, samples, sf, res, x0, y0 = label_geom(lbl)
        r0, r1 = int(np.floor((y0 - top) / res)), int(np.ceil((y0 - bottom) / res))
        c0, c1 = int(np.floor((left - x0) / res)), int(np.ceil((right - x0) / res))
        r0, r1, c0, c1 = max(r0, 0), min(r1, lines), max(c0, 0), min(c1, samples)
        if r1 <= r0 or c1 <= c0:
            print(f"{kind}: window lies outside the 85S illumination product; skipped", file=sys.stderr)
            continue
        if kind == "sun":
            with rasterio.open(RAW / "AVGVISIB_85S_060M_201608.TIF") as d:
                if (d.width, d.height) != (samples, lines) or abs(d.transform.c - x0) > 1e-6:
                    print("TIF geometry does not match its PDS label", file=sys.stderr)
                    return 2
                dn = d.read(1, window=((r0, r1), (c0, c1))).astype(np.int32)
            src = "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF"
        else:
            rb = samples * 2
            req = urllib.request.Request(EARTH_URL, headers={"Range": f"bytes={r0 * rb}-{r1 * rb - 1}"})
            buf = urllib.request.urlopen(req, timeout=600).read()
            if len(buf) != (r1 - r0) * rb:
                print(f"range read returned {len(buf)} bytes, expected {(r1 - r0) * rb}", file=sys.stderr)
                return 2
            dn = np.frombuffer(buf, "<i2").reshape(r1 - r0, samples)[:, c0:c1].astype(np.int32)
            src = EARTH_URL + f" (rows {r0}-{r1 - 1} via HTTP range)"
        frac = dn.astype(np.float32) * np.float32(sf)
        frac[(dn < 0) | (dn > 25000)] = np.nan
        tr = Affine(res, 0, x0 + c0 * res, 0, -res, y0 - r0 * res)
        path = out / f"{kind}_avg_visibility.tif"
        with rasterio.open(path, "w", driver="GTiff", height=frac.shape[0], width=frac.shape[1], count=1, dtype="float32",
                           crs=CRS, transform=tr, nodata=np.nan, compress="deflate") as d:
            d.write(frac, 1)
            d.update_tags(SOURCE=src, SCALING_FACTOR=str(sf), UNITS="fraction of time disc visible (18.6-yr average)")
        rec["files"][kind] = {"source": src, "scaling_factor": sf, "rows": [r0, r1], "cols": [c0, c1],
                              "min": float(np.nanmin(frac)), "max": float(np.nanmax(frac)), "mean": float(np.nanmean(frac))}
    (out / "illumination_record.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
