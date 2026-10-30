"""Raster/image ingestion with validation and provenance manifests.

Supported inputs
----------------
* GeoTIFF (``.tif/.tiff``) via rasterio — georeferenced if a CRS + transform exist.
* PDS3 IMG + detached LBL (LOLA GDR style polar stereographic products).
* Plain images (``.png/.jpg``) — accepted for relative-depth inference only and
  always marked *uncalibrated, source unverified*.

Raw inputs are never modified. Arrays are returned as float32 with NaN as the
single in-memory no-data representation; the original no-data definition is
recorded in the manifest.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.geospatial.grid import GridSpec, is_polar_stereo_south
from core.ingestion import pds3

MANIFEST_FIELDS = [
    "source_name", "product_id", "source_url", "instrument", "acquisition",
    "crs", "pixel_size_m", "units", "nodata", "processing_level", "license", "sha256",
]
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
GEOTIFF_EXT = {".tif", ".tiff"}
MAX_PIXELS = 64_000_000


class IngestError(ValueError):
    """Raised with an actionable, user-facing message."""


@dataclass
class RasterLayer:
    data: np.ndarray
    grid: GridSpec
    manifest: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finalise_manifest(m: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in MANIFEST_FIELDS if m.get(k) in (None, "", "unknown")]
    m["missing_fields"] = missing
    m["source_verified"] = bool(m.get("source_url")) and bool(m.get("instrument"))
    return m


def load_raster(path: str | Path, meta: dict[str, Any] | None = None, verify_checksum: str | None = None) -> RasterLayer:
    """Load and validate one raster. ``meta`` supplies registry provenance fields."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path.name}. Run the fetch script listed in README → 'Data setup'.")
    ext = path.suffix.lower()
    meta = dict(meta or {})
    checksum = sha256_file(path)
    if verify_checksum and verify_checksum != checksum:
        raise IngestError(f"Checksum mismatch for {path.name}: raw inputs must be immutable. Re-download the file.")
    if ext in GEOTIFF_EXT:
        layer = _load_geotiff(path, meta)
    elif ext == ".img":
        layer = _load_pds3_img(path, meta)
    elif ext in IMAGE_EXT:
        layer = _load_plain_image(path, meta)
    else:
        raise IngestError(f"Unsupported file type '{ext}'. Supported: GeoTIFF, PDS3 IMG+LBL, PNG/JPEG.")
    layer.manifest["sha256"] = checksum
    layer.manifest["file_name"] = path.name
    layer.manifest = _finalise_manifest(layer.manifest)
    valid = np.isfinite(layer.data)
    layer.manifest["valid_fraction"] = float(valid.mean())
    if not valid.any():
        raise IngestError(f"{path.name} contains no valid pixels after no-data masking.")
    if valid.mean() < 0.5:
        layer.warnings.append(f"Only {100 * valid.mean():.1f}% of pixels are valid; coverage gaps will be 'unknown'.")
    return layer


def _check_size(h: int, w: int, name: str) -> None:
    if h < 8 or w < 8:
        raise IngestError(f"{name} is {w}x{h}; at least 8x8 pixels are required.")
    if h * w > MAX_PIXELS:
        raise IngestError(f"{name} is {w}x{h} (> {MAX_PIXELS:,} px). Crop to an area of interest first "
                          "(see scripts/fetch_pgda_aoi.py for windowed extraction).")


def _load_geotiff(path: Path, meta: dict[str, Any]) -> RasterLayer:
    import rasterio

    warnings: list[str] = []
    try:
        src = rasterio.open(path)
    except Exception as exc:  # rasterio raises several types
        raise IngestError(f"Could not open {path.name} as GeoTIFF: {exc}") from exc
    with src:
        _check_size(src.height, src.width, path.name)
        if src.count != 1:
            warnings.append(f"{src.count} bands found; band 1 used.")
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            data[data == np.float32(nodata)] = np.nan
        data[data < -1e30] = np.nan  # float32 -FLT_MAX sentinels used by ISIS products
        crs_wkt = src.crs.to_wkt() if src.crs else None
        t = src.transform
        georef = crs_wkt is not None and not t.is_identity
        if not georef:
            warnings.append("No CRS/geotransform: treated as uncalibrated pixel grid.")
        tags = src.tags()
    polar = is_polar_stereo_south(crs_wkt)
    grid = GridSpec(width=data.shape[1], height=data.shape[0],
                    transform=(t.a, t.b, t.c, t.d, t.e, t.f) if georef else (1, 0, 0, 0, -1, 0),
                    crs_wkt=crs_wkt, polar_stereo_south=polar, calibrated=georef)
    m = {
        **meta,
        "format": "GeoTIFF",
        "crs": "IAU Moon south polar stereographic (R=1737.4 km)" if polar else (crs_wkt[:80] if crs_wkt else None),
        "pixel_size_m": grid.res_m,
        "nodata": "NaN" if nodata is None or not np.isfinite(nodata) else nodata,
        "source_url": meta.get("source_url") or tags.get("SOURCE_URL"),
        "dims": [grid.width, grid.height],
    }
    return RasterLayer(data, grid, m, warnings)


def _load_pds3_img(path: Path, meta: dict[str, Any]) -> RasterLayer:
    lbl_path = next((p for p in (path.with_suffix(".lbl"), path.with_suffix(".LBL")) if p.exists()), None)
    if lbl_path is None:
        raise IngestError(f"{path.name}: PDS3 IMG requires its detached .lbl label alongside it.")
    lbl = pds3.read_label(lbl_path)
    img = lbl.get("UNCOMPRESSED_FILE", lbl).get("IMAGE")
    proj = lbl.get("IMAGE_MAP_PROJECTION")
    if img is None or proj is None:
        raise IngestError(f"{lbl_path.name}: missing IMAGE or IMAGE_MAP_PROJECTION object.")
    lines, samples = int(img["LINES"]), int(img["LINE_SAMPLES"])
    _check_size(lines, samples, path.name)
    bits, stype = int(img["SAMPLE_BITS"]), str(img["SAMPLE_TYPE"])
    if (bits, stype) == (16, "LSB_INTEGER"):
        dtype = "<i2"
    elif (bits, stype) == (32, "PC_REAL"):
        dtype = "<f4"
    else:
        raise IngestError(f"Unsupported PDS3 sample type {stype}/{bits} bits.")
    expected = lines * samples * bits // 8
    if path.stat().st_size < expected:
        raise IngestError(f"{path.name} is truncated ({path.stat().st_size} < {expected} bytes). Re-download.")
    dn = np.memmap(path, dtype=dtype, mode="r", shape=(lines, samples))
    scale = float(pds3.value(img.get("SCALING_FACTOR", 1.0)))
    data = (np.asarray(dn, dtype=np.float32) * np.float32(scale))
    ptype = str(proj["MAP_PROJECTION_TYPE"]).upper()
    clat = float(pds3.value(proj["CENTER_LATITUDE"]))
    mscale_km_or_m = proj["MAP_SCALE"]
    mscale = float(pds3.value(mscale_km_or_m))
    if isinstance(mscale_km_or_m, dict) and "KM" in str(mscale_km_or_m.get("unit", "")).upper():
        mscale *= 1000.0
    s0 = float(pds3.value(proj["SAMPLE_PROJECTION_OFFSET"]))
    l0 = float(pds3.value(proj["LINE_PROJECTION_OFFSET"]))
    # PDS3 polar stereographic (DSMAP_POLAR): x = (S - S0 - 1) * scale,
    # y = (1 + L0 - L) * scale for 1-based sample S / line L at pixel centre.
    # Top-left pixel *edge* in 0-based terms:
    x0 = (-0.5 - s0) * mscale
    y0 = (l0 + 0.5) * mscale
    polar = "POLAR STEREOGRAPHIC" in ptype and clat == -90.0
    if not polar:
        raise IngestError(f"PDS3 projection {ptype} (center lat {clat}) not yet supported; only south polar stereographic.")
    grid = GridSpec(width=samples, height=lines, transform=(mscale, 0.0, x0, 0.0, -mscale, y0),
                    crs_wkt="+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m",
                    polar_stereo_south=True, calibrated=True)
    m = {
        **meta,
        "format": "PDS3 IMG",
        "product_id": meta.get("product_id") or str(lbl.get("UNCOMPRESSED_FILE", {}).get("FILE_NAME", path.name)),
        "instrument": meta.get("instrument") or lbl.get("INSTRUMENT_ID"),
        "acquisition": meta.get("acquisition") or f"{lbl.get('START_TIME')} – {lbl.get('STOP_TIME')}",
        "processing_level": meta.get("processing_level") or lbl.get("DATA_SET_ID"),
        "crs": "Moon south polar stereographic, sphere R=1737.4 km (PDS3 label)",
        "pixel_size_m": mscale,
        "units": "m (height above 1737.4 km sphere)" if img.get("UNIT") == "METER" else img.get("UNIT"),
        "nodata": "none declared",
        "scaling_factor": scale,
        "dims": [samples, lines],
    }
    return RasterLayer(data, grid, m, [])


def _load_plain_image(path: Path, meta: dict[str, Any]) -> RasterLayer:
    from PIL import Image

    try:
        im = Image.open(path)
        im.load()
    except Exception as exc:
        raise IngestError(f"Could not decode image {path.name}: {exc}") from exc
    _check_size(im.height, im.width, path.name)
    arr = np.asarray(im.convert("F") if im.mode in ("I", "I;16", "F") else im.convert("L"), dtype=np.float32)
    mx = float(arr.max()) or 1.0
    arr = arr / mx
    grid = GridSpec(width=arr.shape[1], height=arr.shape[0], transform=(1, 0, 0, 0, -1, 0), calibrated=False)
    m = {
        **meta, "format": f"image/{(im.format or path.suffix[1:]).lower()}", "crs": None,
        "pixel_size_m": None, "units": "normalised DN (uncalibrated)", "nodata": "none",
        "dims": [grid.width, grid.height],
    }
    warn = ["Plain image: no georeferencing or radiometric calibration. Usable only for relative-depth "
            "visualisation; never treated as a scientific raster.",
            "Source unverified unless provenance fields are supplied."]
    return RasterLayer(arr, grid, m, warn)
