"""Crater library: dataset specs generated from configs/craters.yaml + IAU/USGS Gazetteer subset.

Named-feature coverage is *computed*, never asserted: a gazetteer feature is listed as covered only
when its centre lies inside the dataset window ("partial" if its rim circle extends outside).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.config import ROOT
from core.geospatial.grid import polar_stereo_south_forward

GAZ_PATH = ROOT / "configs" / "gazetteer_south80.json"
CRATERS_PATH = ROOT / "configs" / "craters.yaml"
PGDA = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/"
CITE = "Barker et al. (2023) PSJ 4:183 doi:10.3847/PSJ/acf3e1; Paige et al. (2010) Science 330:479; Mazarico et al. (2011) Icarus 211:1066"


@lru_cache(maxsize=1)
def gazetteer() -> dict[str, Any]:
    return json.loads(GAZ_PATH.read_text())


def features_in_bounds(bounds: list[float] | tuple[float, ...]) -> list[dict[str, Any]]:
    left, bottom, right, top = bounds
    out = []
    for f in gazetteer()["features"]:
        x, y = polar_stereo_south_forward(f["lat"], f["lon"])
        x, y = float(x), float(y)
        if left <= x <= right and bottom <= y <= top:
            r = f["diameter_km"] * 500.0
            full = left <= x - r and x + r <= right and bottom <= y - r and y + r <= top
            out.append({**f, "x": x, "y": y, "coverage": "full" if full else "partial (rim extends outside window)"})
    out.sort(key=lambda f: -f["diameter_km"])
    return out


def _layer(role: str, file: str, url: str, product: str, units: str, level: str, status: str, **kw) -> dict[str, Any]:
    return {"file": file, "source_url": url, "product_id": product, "units": units, "processing_level": level,
            "status": status, "license": "NASA data — public domain", **kw}


def crater_specs() -> list[dict[str, Any]]:
    if not CRATERS_PATH.exists():
        return []
    cfg = yaml.safe_load(CRATERS_PATH.read_text())
    gz = {f["name"]: f for f in gazetteer()["features"]}
    specs = []
    for c in cfg["craters"]:
        cid, g = c["id"], gz.get(c["name"])
        if g is None:
            continue
        rec_path = ROOT / "data/raw/pgda" / cid / "fetch_record.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
        res = rec["plan"]["res_m"] if rec else None
        tag = f"{res}MPP" if res else "?"
        d = f"data/raw/pgda/{cid}"
        layers = {
            "elevation": _layer("elevation", f"{d}/LDEM_80S_{tag}_ADJ.TIF", f"{PGDA}LDEM_80S_{tag}_ADJ.TIF", f"LDEM_80S_{tag}_ADJ",
                                "m, height relative to 1737.4 km reference sphere", "Gridded DEM, crossover-adjusted (PGDA)", "measured",
                                source_name=f"NASA GSFC PGDA — LOLA south pole {res} m DEM", instrument="LRO LOLA",
                                acquisition="LRO mission 2009– (multi-year altimetry; see Barker et al. 2023)"),
            "elevation_uncertainty": _layer("elevation_uncertainty", f"{d}/LDEM_80S_{tag}_ADJ_ERR.TIF", f"{PGDA}LDEM_80S_{tag}_ADJ_ERR.TIF",
                                            f"LDEM_80S_{tag}_ADJ_ERR", "m", "Derived uncertainty", "derived",
                                            source_name="PGDA LOLA DEM error estimate", instrument="LRO LOLA"),
            "effective_resolution": _layer("effective_resolution", f"{d}/LDEM_80S_{tag}_ADJ_EFFRES.TIF", f"{PGDA}LDEM_80S_{tag}_ADJ_EFFRES.TIF",
                                           f"LDEM_80S_{tag}_ADJ_EFFRES", "m", "Derived (LOLA shot density)", "derived",
                                           source_name="PGDA LOLA DEM effective resolution", instrument="LRO LOLA"),
            "slope_reference": _layer("slope_reference", f"{d}/LDSM_80S_{tag}_ADJ.TIF", f"{PGDA}LDSM_80S_{tag}_ADJ.TIF",
                                      f"LDSM_80S_{tag}_ADJ", "deg", "Derived slope (PGDA)", "derived",
                                      source_name="PGDA LOLA slope map", instrument="LRO LOLA"),
            "psr_mask": _layer("psr_mask", f"{d}/LPSR_80S_20MPP_ADJ.TIF", f"{PGDA}LPSR_80S_20MPP_ADJ.TIF", "LPSR_80S_20MPP_ADJ",
                               "1 = PSR, 0 = not PSR" + ("" if res == 20 else f" (averaged to {res} m: PSR fraction)"),
                               "Derived from illumination modelling, not image darkness", "derived",
                               source_name="PGDA LOLA permanently shadowed region map", instrument="LRO LOLA (illumination modelling)"),
            "illumination_avg": _layer("illumination_avg", f"data/processed/illumination/{cid}/sun_avg_visibility.tif",
                                       "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_85S_060M_201608.TIF", "AVGVISIB_85S_060M_201608",
                                       "fraction of time the solar disc is visible", "Derived model product (Mazarico et al. 2011)", "derived",
                                       source_name="PGDA average solar visibility, 85S, 60 m/px", instrument="LRO LOLA (illumination modelling)"),
            "earth_visibility_avg": _layer("earth_visibility_avg", f"data/processed/illumination/{cid}/earth_avg_visibility.tif",
                                           "http://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.IMG",
                                           "AVGVISIB_85S_060M_201608_EARTH", "fraction of time the Earth disc is visible",
                                           "Derived model product (Mazarico et al. 2011)", "derived",
                                           source_name="Average Earth visibility, 85S, 60 m/px", instrument="LRO LOLA (visibility modelling)"),
        }
        for role, field, title, units in (("thermal_max", "temp_max", "annual max surface temperature", "K"),
                                          ("thermal_avg", "temp_avg", "annual average temperature, 2 cm depth", "K"),
                                          ("ice_stability_depth", "ice_depth", "modelled water-ice stability depth",
                                           "m (deeper than 2.87 m is encoded as 999)")):
            layers[role] = _layer(role, f"data/processed/diviner/{cid}/prp_{field}.tif",
                                  "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/dlre_prp_south.tab",
                                  f"DLRE_PRP_SOUTH.TAB:{field}", units, "PDS L5 model-fit product, resampled by scripts/rasterize_diviner_prp.py",
                                  "derived", source_name=f"Diviner Polar Resource Product (south), {title}", instrument="LRO Diviner (DLRE)")
        if rec and "image" in rec["layers"]:
            im = rec["layers"]["image"]
            nid = Path(im["file"]).stem.replace("_20MPP", "")
            layers["image"] = _layer("image", im["file"], im["source_url"], f"{nid}_20MPP", "radiance factor (I/F), as distributed",
                                     f"Map-projected NAC image (PGDA), averaged to {res} m", "measured",
                                     source_name=f"LROC NAC {nid}, co-registered to LOLA by PGDA", instrument="LRO LROC NAC")
        bounds = rec["plan"]["bounds"] if rec else None
        feats = features_in_bounds(bounds) if bounds else []
        covers = [f"{f['name']}" + ("" if f["coverage"] == "full" else " (partial)") for f in feats[:8]]
        specs.append({
            "id": cid, "title": f"{c['name']} crater — LOLA {res or '?'} m/px (real data)", "kind": "real",
            "crater": {"name": c["name"], "lat": g["lat"], "lon": g["lon"], "diameter_km": g["diameter_km"], "link": g["link"], "note": c.get("note", "")},
            "region_note": (f"{(bounds[2] - bounds[0]) / 1000:.0f} x {(bounds[3] - bounds[1]) / 1000:.0f} km window centred on the IAU gazetteer "
                            f"centre of {c['name']} ({g['lat']:.2f}°, {g['lon']:.2f}°E, D = {g['diameter_km']:.1f} km).") if bounds else
                           f"Not downloaded yet. Run: python scripts/fetch_crater.py --crater {cid}",
            "covers_features": covers, "fetch": f"python scripts/fetch_crater.py --crater {cid}", "citation": CITE,
            "layers": layers,
        })
    return specs
