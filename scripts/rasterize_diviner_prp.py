"""Resample the Diviner Polar Resource Product (south) onto an AOI grid.

Input  : data/raw/diviner/dlre_prp_south.tab (+ .xml label), PDS
         LRO-L-DLRE-5-PRP-V2.0 — fixed-width ASCII, one triangular mesh facet per row.
Output : data/processed/diviner/<aoi>/prp_{temp_max,temp_avg,ice_depth}.tif on
         the same 20 m grid as the PGDA LOLA AOI rasters, plus a JSON record.

Method (documented, deterministic)
----------------------------------
1. Read facet-centre planetocentric lat/lon and the three value fields using
   the byte offsets declared in the PDS4 label (not guessed).
2. Project facet centres to south polar stereographic (R = 1737.4 km).
3. For each target cell centre, take the value of the nearest facet centre
   (nearest neighbour; no smoothing). Cells farther than 1.5 x the local
   median facet spacing from any facet are left NaN (no extrapolation).
4. ``ice_depth = -999`` means "stability depth > 2.8738 m" per the label; it
   is a *valid* value, encoded here as 999.0 so it is not confused with no-data.
5. ``ice_depth = -1`` is NOT defined in the PDS label or catalog (prpds.cat).
   Empirically it occurs only on facets with temp_max <= ~101.3 K (100,319
   facets), while 0 ("stable at the surface") spans ~101.3-130 K. Rather than
   guess its meaning, -1 is treated as no-data (NaN) and counted in the record.

The PRP is a thermal-model fit to the first year of Diviner mapping. Its
native facet spacing is much coarser than 20 m; the JSON record states the
measured spacing so the UI can report the true resolution.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.geospatial.grid import polar_stereo_south_forward  # noqa: E402


def field_slices(xml_text: str) -> dict[str, tuple[int, int]]:
    out = {}
    for m in re.finditer(r"<name>(\w+)</name>\s*<field_number>\d+</field_number>\s*"
                         r"<field_location unit=\"byte\">(\d+)</field_location>\s*<data_type>\w+</data_type>\s*"
                         r"<field_length unit=\"byte\">(\d+)</field_length>", xml_text):
        name, loc, ln = m.group(1), int(m.group(2)), int(m.group(3))
        out[name] = (loc - 1, loc - 1 + ln)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="shackleton")
    ap.add_argument("--tab", default=str(ROOT / "data/raw/diviner/dlre_prp_south.tab"))
    args = ap.parse_args()
    tab = Path(args.tab)
    xml = tab.with_suffix(".xml").read_text()
    fs = field_slices(xml)
    need = ["tri_clon", "tri_clat", "temp_avg", "temp_max", "ice_depth"]
    missing = [n for n in need if n not in fs]
    if missing:
        print(f"label parse failed for fields {missing}", file=sys.stderr)
        return 2
    rec = 210
    size = tab.stat().st_size
    n_rows = size // rec - 1  # first record is the header
    if size % rec:
        print(f"{tab.name} size {size} is not a multiple of {rec}; file truncated? re-download.", file=sys.stderr)
        return 2
    raw = np.memmap(tab, dtype="S1", mode="r", offset=rec, shape=(n_rows, rec))

    cands = sorted(p for p in (ROOT / f"data/raw/pgda/{args.aoi}").glob("LDEM_80S_*MPP_ADJ.TIF"))
    if not cands:
        print(f"no PGDA elevation window for AOI '{args.aoi}'", file=sys.stderr)
        return 2
    ref = cands[0]
    with rasterio.open(ref) as r:
        profile = r.profile.copy()
        t = r.transform
        H, W = r.height, r.width
        left, bottom, right, top = r.bounds
    margin = 3000.0
    from core.geospatial.grid import polar_stereo_south_inverse
    corner_r = max(np.hypot(xx, yy) for xx in (left - margin, right + margin) for yy in (bottom - margin, top + margin))
    lat_cut = float(polar_stereo_south_inverse(corner_r, 0.0)[0]) + 0.05

    cols = {n: [] for n in need}
    chunk = 400_000
    for s in range(0, n_rows, chunk):
        block = np.asarray(raw[s:s + chunk])
        def col(name):
            a, b = fs[name]
            return np.char.strip(block[:, a:b].view(f"S{b - a}").ravel()).astype(np.float64)
        lat = col("tri_clat")
        sel = lat < lat_cut  # coarse prefilter on the window's most equatorward latitude
        if not sel.any():
            continue
        lon = col("tri_clon")[sel]
        x, y = polar_stereo_south_forward(lat[sel], lon)
        inb = (x > left - margin) & (x < right + margin) & (y > bottom - margin) & (y < top + margin)
        idx = np.flatnonzero(sel)[inb]
        cols["tri_clon"].append(x[inb]); cols["tri_clat"].append(y[inb])
        for n in ("temp_avg", "temp_max", "ice_depth"):
            cols[n].append(col(n)[idx])
    X = np.concatenate(cols["tri_clon"]); Y = np.concatenate(cols["tri_clat"])
    if X.size == 0:
        print("no PRP facets inside AOI", file=sys.stderr)
        return 3
    vals = {n: np.concatenate(cols[n]) for n in ("temp_avg", "temp_max", "ice_depth")}
    tree = cKDTree(np.c_[X, Y])
    d_nn, _ = tree.query(np.c_[X, Y], k=2)
    spacing = float(np.median(d_nn[:, 1]))
    cx = t.c + (np.arange(W) + 0.5) * t.a
    cy = t.f + (np.arange(H) + 0.5) * t.e
    gx, gy = np.meshgrid(cx, cy)
    dist, nn = tree.query(np.c_[gx.ravel(), gy.ravel()], k=1)
    far = dist > 1.5 * spacing
    out_dir = ROOT / f"data/processed/diviner/{args.aoi}"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    stats = {}
    for n, fname in [("temp_max", "prp_temp_max.tif"), ("temp_avg", "prp_temp_avg.tif"), ("ice_depth", "prp_ice_depth.tif")]:
        v = vals[n][nn].astype(np.float32)
        if n == "ice_depth":
            undocumented = (v > -1.5) & (v < -0.5)
            v = np.where(v <= -998.0, 999.0, v).astype(np.float32)
            v[undocumented] = np.nan  # undocumented code; see docstring
            stats["ice_depth_undocumented_minus1_fraction"] = float(undocumented[~far].mean())
        else:
            v[v <= 0] = np.nan
        v[far] = np.nan
        v = v.reshape(H, W)
        with rasterio.open(out_dir / fname, "w", **profile) as dst:
            dst.write(v, 1)
            dst.update_tags(SOURCE="DLRE_PRP_SOUTH.TAB", FIELD=n, METHOD="nearest facet centre, max dist 1.5x median spacing",
                            NATIVE_FACET_SPACING_M=f"{spacing:.1f}")
        fin = v[np.isfinite(v)]
        stats[n] = {"min": float(fin.min()), "max": float(fin.max()), "valid_fraction": float(np.isfinite(v).mean())}
    record = {
        "aoi": args.aoi, "n_facets_used": int(X.size), "median_facet_spacing_m": spacing,
        "method": "nearest facet centre; cells > 1.5 x spacing from any facet left NaN",
        "ice_depth_encoding": "999.0 = stability depth deeper than 2.8738 m (label value -999); "
                              "label value -1 is undocumented and set to NaN",
        "stats": stats,
    }
    (out_dir / "prp_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
