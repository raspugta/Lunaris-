"""End-to-end orchestration: dataset -> aligned layers -> analysis products.

Each step is a pure function of (input checksums, configuration), so derived
results are cached under ``data/processed/<hash>/`` and re-used across runs.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.config import ROOT, resolve, stable_hash
from core.detection.features import boulder_detection_feasibility, detect_topographic_anomalies
from core.geospatial import terrain
from core.geospatial.grid import GridSpec, max_scale_error, polar_stereo_south_inverse
from core.ice_priority.evidence import PRIORITY_LEGEND, compute_priority
from core.ingestion.loader import IngestError, RasterLayer, load_raster
from core.navigation.cost import build_cost_surface
from core.provenance import LayerMeta, Status, unavailable

log = logging.getLogger(__name__)
PIPELINE_VERSION = "1.0.0"
CONTINUOUS_ROLES = {"elevation", "elevation_uncertainty", "effective_resolution", "slope_reference",
                    "illumination_avg", "earth_visibility_avg",
                    "image", "thermal_max", "thermal_avg", "ice_stability_depth", "hydrogen"}


@dataclass
class Dataset:
    id: str
    title: str
    kind: str  # real | synthetic | upload
    region_note: str
    covers_features: list[str]
    citation: str
    layers: dict[str, RasterLayer]
    missing_roles: dict[str, str]
    grid: GridSpec
    processing_log: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    crater: dict[str, Any] | None = None

    @property
    def checksum(self) -> str:
        return stable_hash({k: v.manifest.get("sha256") for k, v in sorted(self.layers.items())})

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.id, "title": self.title, "kind": self.kind, "region_note": self.region_note,
            "covers_features": self.covers_features, "citation": self.citation,
            "grid": self.grid.to_dict(), "dataset_checksum": self.checksum,
            "layers": {k: v.manifest for k, v in self.layers.items()},
            "missing_roles": self.missing_roles, "warnings": self.warnings, "crater": self.crater,
        }


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Static registry (configs/datasets.yaml) + crater library generated from configs/craters.yaml."""
    from core.craters import crater_specs
    p = path or ROOT / "configs" / "datasets.yaml"
    return yaml.safe_load(p.read_text())["datasets"] + crater_specs()


def registry_status(cfg: dict) -> list[dict[str, Any]]:
    """Which registered datasets are present on disk (for the dataset selector)."""
    out = []
    for d in load_registry(resolve(cfg["paths"]["datasets_registry"])):
        present = {role: resolve(spec["file"]).exists() for role, spec in d["layers"].items()}
        out.append({"id": d["id"], "title": d["title"], "kind": d["kind"], "crater": d.get("crater"),
                    "available": present.get("elevation", False) or present.get("image", False),
                    "layers_present": present, "fetch": d.get("fetch")})
    return out


# ---------------------------------------------------------------- loading
def _crop(layer: RasterLayer, bounds: list[float]) -> RasterLayer:
    g = layer.grid
    r0, c0 = g.world_to_pixel(bounds[0], bounds[3])
    r1, c1 = g.world_to_pixel(bounds[2], bounds[1])
    r0, c0 = int(round(float(r0) + 0.5)), int(round(float(c0) + 0.5))
    r1, c1 = int(round(float(r1) + 0.5)), int(round(float(c1) + 0.5))
    r0, c0 = max(r0, 0), max(c0, 0)
    data = layer.data[r0:r1, c0:c1].copy()
    a, b, c, d, e, f = g.transform
    t = (a, b, c + c0 * a + r0 * b, d, e, f + c0 * d + r0 * e)
    ng = GridSpec(data.shape[1], data.shape[0], t, g.crs_wkt, g.polar_stereo_south, g.calibrated)
    m = dict(layer.manifest, crop_window_px=[r0, c0, r1, c1], crop_bounds_m=bounds)
    return RasterLayer(data, ng, m, layer.warnings + [f"Cropped to {bounds} m for analysis."])


def _align(layer: RasterLayer, ref: GridSpec, role: str) -> tuple[RasterLayer, str]:
    """Resample ``layer`` onto ``ref`` if their grids differ (rasterio.warp)."""
    if layer.grid.transform == ref.transform and (layer.grid.width, layer.grid.height) == (ref.width, ref.height):
        return layer, "identical grid (no resampling)"
    if not (layer.grid.calibrated and ref.calibrated):
        raise IngestError(f"Layer '{role}' is not georeferenced and cannot be aligned to the elevation grid.")
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    dst = np.full((ref.height, ref.width), np.nan, np.float32)
    rs = Resampling.bilinear if role in CONTINUOUS_ROLES - {"image"} else Resampling.nearest
    crs = ref.crs_wkt or layer.grid.crs_wkt
    reproject(layer.data, dst, src_transform=Affine(*layer.grid.transform), src_crs=layer.grid.crs_wkt or crs,
              dst_transform=Affine(*ref.transform), dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=rs)
    m = dict(layer.manifest, aligned_to_elevation_grid=True, resampling=rs.name)
    return RasterLayer(dst, ref, m, layer.warnings), f"resampled ({rs.name}) onto elevation grid"


def load_dataset(dataset_id: str, cfg: dict) -> Dataset:
    reg = {d["id"]: d for d in load_registry(resolve(cfg["paths"]["datasets_registry"]))}
    if dataset_id not in reg:
        raise IngestError(f"Unknown dataset '{dataset_id}'. Known: {', '.join(reg)}")
    spec = reg[dataset_id]
    return build_dataset(spec, cfg)


def build_dataset(spec: dict[str, Any], cfg: dict, preloaded: dict[str, RasterLayer] | None = None) -> Dataset:
    t0 = time.time()
    layers: dict[str, RasterLayer] = dict(preloaded or {})
    missing: dict[str, str] = {}
    plog: list[dict[str, Any]] = []
    warns: list[str] = []
    for role, lspec in spec.get("layers", {}).items():
        if role in layers:
            continue
        meta = {k: v for k, v in lspec.items() if k not in ("file", "status")}
        meta["status"] = lspec.get("status", "measured")
        meta["role"] = role
        path = resolve(lspec["file"])
        if not path.exists():
            missing[role] = f"file not present ({lspec['file']}); fetch with: {spec.get('fetch', 'see README')}"
            continue
        try:
            lyr = load_raster(path, meta)
        except IngestError as exc:
            missing[role] = str(exc)
            warns.append(f"{role}: {exc}")
            continue
        if spec.get("crop_bounds_m") and lyr.grid.calibrated:
            lyr = _crop(lyr, spec["crop_bounds_m"])
        layers[role] = lyr
        plog.append({"step": "ingest", "role": role, "file": path.name, "sha256": lyr.manifest["sha256"][:16]})
    ref_role = "elevation" if "elevation" in layers else ("image" if "image" in layers else None)
    if ref_role is None:
        raise IngestError(f"Dataset '{spec['id']}' has no elevation or image layer on disk. {spec.get('fetch', '')}")
    ref = layers[ref_role].grid
    for role in list(layers):
        if role == ref_role:
            continue
        try:
            layers[role], how = _align(layers[role], ref, role)
            plog.append({"step": "align", "role": role, "method": how})
        except IngestError as exc:
            missing[role] = str(exc)
            del layers[role]
    for role in ("elevation", "psr_mask", "thermal_max", "ice_stability_depth", "image", "hydrogen"):
        if role not in layers and role not in missing:
            missing[role] = "not provided by this dataset"
    missing.setdefault("hydrogen", "LEND hydrogen not ingested: native footprint (~10+ km) is far coarser than the analysis grid and is not a pixel-level ice label.")
    for role, lyr in layers.items():
        warns.extend(f"{role}: {w}" for w in lyr.warnings)
    plog.append({"step": "load_complete", "seconds": round(time.time() - t0, 2)})
    return Dataset(spec["id"], spec["title"], spec["kind"], spec.get("region_note", ""),
                   spec.get("covers_features", []), spec.get("citation", ""), layers, missing, ref, plog, warns,
                   spec.get("crater"))


# ---------------------------------------------------------------- analysis
@dataclass
class Analysis:
    dataset: Dataset
    grid: GridSpec                      # analysis grid (possibly downsampled)
    arrays: dict[str, np.ndarray]
    meta: dict[str, LayerMeta]
    stats: dict[str, Any]
    config_hash: str
    processing_log: list[dict[str, Any]]
    priority_inputs: dict[str, np.ndarray | None]
    priority: Any
    features: list[dict[str, Any]]

    def lat_lon(self, r: float, c: float) -> tuple[float, float] | None:
        if not self.grid.polar_stereo_south:
            return None
        x, y = self.grid.pixel_to_world(r, c)
        lat, lon = polar_stereo_south_inverse(x, y)
        return float(lat), float(lon)


def _status(ds: Dataset, role: str, default: Status) -> Status:
    if ds.kind == "synthetic":
        return Status.SYNTHETIC
    s = ds.layers[role].manifest.get("status") if role in ds.layers else None
    return Status(s) if s in {x.value for x in Status} else default


def analyse(ds: Dataset, cfg: dict) -> Analysis:
    t0 = time.time()
    acfg, hcfg = cfg["analysis"], cfg["hazard"]
    plog = list(ds.processing_log)
    calibrated = ds.grid.calibrated and "elevation" in ds.layers
    factor = int(acfg["downsample_factor"]) if ds.grid.calibrated else 1
    grid = ds.grid.downsampled(factor)
    res = grid.res_m
    src = f"{ds.title}"
    A: dict[str, np.ndarray] = {}
    M: dict[str, LayerMeta] = {}
    stats: dict[str, Any] = {"downsample_factor": factor, "analysis_res_m": res}

    def red(role: str, mode: str = "mean") -> np.ndarray | None:
        if role not in ds.layers:
            return None
        a = ds.layers[role].data
        return terrain.block_reduce_mode(a, factor) if mode == "frac" else terrain.block_reduce_mean(a, factor)

    elev = red("elevation") if calibrated else None
    if elev is not None:
        A["elevation"] = elev
        fin = elev[np.isfinite(elev)]
        M["elevation"] = LayerMeta("elevation", "Elevation (LOLA)" if ds.kind == "real" else "Elevation",
                                   _status(ds, "elevation", Status.MEASURED), "m",
                                   "Height relative to the 1737.4 km reference sphere.", src,
                                   f"Block-mean {factor}x{factor} of source DEM (cells need >=75% valid sub-pixels).",
                                   ["Gridded from altimetry tracks; interpolated between tracks (see effective resolution).",
                                    "Heights are relative to a sphere, not a geoid."],
                                   res, "terrain", (float(fin.min()), float(fin.max())))
        plog.append({"step": "downsample", "factor": factor, "method": "nan-aware block mean"})
        hs = terrain.hillshade(elev, res)
        A["hillshade"] = hs
        M["hillshade"] = LayerMeta("hillshade", "Hillshade (visual aid)", Status.ILLUSTRATIVE, "unitless",
                                   "Synthetic illumination from azimuth 315°, altitude 30° (grid frame).", "Derived from elevation",
                                   "Lambertian shading of Horn gradients.",
                                   ["Not a simulation of real lunar illumination; polar sun is near the horizon."],
                                   res, "gray", (0.0, 1.0))
        slope = terrain.slope_degrees(elev, res)
        A["slope"] = slope
        M["slope"] = LayerMeta("slope", "Slope", _derived(ds), "degrees",
                               f"Surface slope over a 3x3 ({3 * res:.0f} m) Horn stencil.", "Derived from elevation",
                               "Horn (1981) finite differences on the analysis grid; NaN propagates from no-data.",
                               [f"Baseline {2 * res:.0f} m: slopes at rover-wheel scale can be steeper.",
                                f"Polar stereographic scale error <= {100 * (max_scale_error(grid) or 0):.3f}% ignored."],
                               res, "slope", (0.0, 40.0))
        win = int(acfg["roughness_window"])
        rough = terrain.roughness_plane_rms(elev, win, res)
        A["roughness"] = rough
        M["roughness"] = LayerMeta("roughness", "Roughness (detrended RMS)", _derived(ds), "m",
                                   f"RMS height residual from a best-fit plane in a {win}x{win} window ({win * res:.0f} m baseline).",
                                   "Derived from elevation", "Least-squares plane removal per window (separable correlation).",
                                   ["Scale-dependent; sub-cell roughness (boulders) is not captured."],
                                   res, "magma", (0.0, float(np.nanpercentile(rough, 99)) if np.isfinite(rough).any() else 1.0))
    else:
        for k, t in (("slope", "Slope"), ("roughness", "Roughness")):
            M[k] = unavailable(k, t, "No calibrated elevation raster: slope/roughness would not be physically meaningful.",
                               "Requires a georeferenced elevation product (e.g. LOLA LDEM).")
        slope = rough = None

    unc = red("elevation_uncertainty")
    eff = red("effective_resolution")
    uncertain = None
    if unc is not None or eff is not None:
        uncertain = np.zeros(grid.height * grid.width, bool).reshape(grid.height, grid.width)
        if unc is not None:
            A["elevation_uncertainty"] = unc
            uncertain |= np.isfinite(unc) & (unc > acfg["max_elevation_uncertainty_m"])
            M["elevation_uncertainty"] = LayerMeta("elevation_uncertainty", "DEM uncertainty (PGDA)", _status(ds, "elevation_uncertainty", Status.DERIVED),
                                                   "m", "Producer-estimated elevation error.", src, "Block mean.",
                                                   ["Producer model of error; not ground truth."], res, "inferno",
                                                   (0.0, float(np.nanpercentile(unc, 99))))
        if eff is not None:
            A["effective_resolution"] = eff
            uncertain |= np.isfinite(eff) & (eff > acfg["max_effective_resolution_m"])
            M["effective_resolution"] = LayerMeta("effective_resolution", "DEM effective resolution", _status(ds, "effective_resolution", Status.DERIVED),
                                                  "m", "Effective horizontal resolution of the DEM given LOLA shot density.", src,
                                                  "Block mean.", ["Large values mean heights are interpolated between sparse tracks."],
                                                  res, "inferno", (0.0, float(np.nanpercentile(eff, 99))))
        A["uncertain"] = uncertain.astype(np.float32)
        M["uncertain"] = LayerMeta("uncertain", "Elevation flagged uncertain", _derived(ds), "boolean",
                                   f"error > {acfg['max_elevation_uncertainty_m']} m or effective resolution > {acfg['max_effective_resolution_m']} m",
                                   src, "Threshold on producer uncertainty layers (prototype thresholds).", [],
                                   res, "flag", (0.0, 1.0))
        stats["uncertain_fraction"] = float(uncertain.mean())
    if slope is not None and "slope_reference" in ds.layers:
        sref = red("slope_reference")
        A["slope_reference"] = sref
        M["slope_reference"] = LayerMeta("slope_reference", "Slope — PGDA reference", _status(ds, "slope_reference", Status.DERIVED),
                                         "degrees", "Producer slope map, block-averaged to the analysis grid.", src, "Block mean of 20 m slope.",
                                         ["Different baseline than our slope; differences are expected."], res, "slope", (0.0, 40.0))
        ok = np.isfinite(sref) & np.isfinite(slope)
        diff = slope[ok] - sref[ok]
        stats["slope_vs_reference"] = {
            "pearson_r": float(np.corrcoef(slope[ok], sref[ok])[0, 1]), "mean_diff_deg": float(diff.mean()),
            "mad_deg": float(np.median(np.abs(diff))),
            "note": "Consistency check against the producer slope (same source DEM; not independent validation). "
                    "Our slope uses a coarser baseline, so it is expected to be lower on average.",
        }

    psr = red("psr_mask", "frac")
    if psr is not None:
        A["psr"] = psr
        M["psr"] = LayerMeta("psr", "Permanently shadowed regions", _status(ds, "psr_mask", Status.DERIVED), "fraction of cell",
                             "Fraction of each analysis cell inside a modelled PSR.", src,
                             "Block mean of the producer's 0/1 PSR map (illumination modelling on LOLA topography).",
                             ["Model of illumination over the modelled epoch; small PSRs below grid scale are smoothed."],
                             res, "psr", (0.0, 1.0))
        stats["psr_fraction"] = float(np.nanmean(psr >= 0.5))
    else:
        M["psr"] = unavailable("psr", "Permanently shadowed regions", ds.missing_roles.get("psr_mask", "not provided"),
                               "Requires a PSR map, e.g. PGDA LPSR_80S_20MPP_ADJ.")
    for role, key, title, units, cmap, rng_ in (
            ("thermal_max", "thermal_max", "Annual max temperature (Diviner PRP)", "K", "thermal", (30.0, 330.0)),
            ("thermal_avg", "thermal_avg", "Annual mean temperature, 2 cm depth (Diviner PRP)", "K", "thermal", (30.0, 250.0)),
            ("ice_stability_depth", "ice_stability_depth", "Ice stability depth (Diviner PRP)", "m", "depth", (0.0, 3.0))):
        arr = red(role)
        if arr is not None:
            A[key] = arr
            note = ds.layers[role].manifest
            M[key] = LayerMeta(key, title if ds.kind != "synthetic" else title.replace("Diviner PRP", "SYNTHETIC"),
                               _status(ds, role, Status.DERIVED), units,
                               note.get("source_name", ""), src,
                               "Thermal-model fit (PDS PRP) resampled nearest-facet then block-averaged.",
                               ["Model product fitted to the first year of Diviner data; native facet spacing is coarser than this grid.",
                                "A single temperature does not describe sub-facet conditions (micro cold traps)."]
                               + (["999 encodes 'deeper than 2.87 m'."] if key == "ice_stability_depth" else []),
                               res, cmap, rng_)
        else:
            M[key] = unavailable(key, title, ds.missing_roles.get(role, "not provided"),
                                 "Requires Diviner PRP: run scripts/rasterize_diviner_prp.py after downloading DLRE_PRP_SOUTH.TAB.")
    for role, title, desc in (
            ("illumination_avg", "Average sun visibility (18.6-yr)", "Fraction of time the solar disc is visible (Mazarico et al. 2011)."),
            ("earth_visibility_avg", "Average Earth visibility (18.6-yr)", "Fraction of time the Earth disc is visible: direct-to-Earth comms opportunity.")):
        arr = red(role)
        if arr is not None:
            A[role] = arr
            M[role] = LayerMeta(role, title, _status(ds, role, Status.DERIVED), "fraction of time", desc, src,
                                "LOLA horizon modelling, hourly over 18.6 yr; 60 m product resampled bilinearly, then block mean.",
                                ["Long-term average, not the conditions on any specific date.",
                                 "60 m native resolution; horizons from LOLA topography."], res, "gray" if role == "illumination_avg" else "viridis", (0.0, 1.0))
    M["hydrogen"] = unavailable("hydrogen", "LEND hydrogen", ds.missing_roles.get("hydrogen", ""),
                                "LEND data are kilometre-scale; they could constrain a regional prior only.")

    # Hazard + cost
    if slope is not None:
        hz = terrain.classify_hazard(slope, rough, hcfg, uncertain)
        A["hazard"] = hz.astype(np.float32)
        M["hazard"] = LayerMeta("hazard", "Mobility hazard class", _derived(ds), "class",
                                "Worst of slope class and roughness class; uncertain elevation raises one class.", "Derived",
                                f"Slope edges {hcfg['slope_classes_deg']} deg, blocked >= {hcfg['slope_blocked_deg']} deg; "
                                f"roughness high >= {hcfg['roughness_high_m']} m, blocked >= {hcfg['roughness_blocked_m']} m.",
                                ["PROTOTYPE thresholds, not a certified rover specification.",
                                 "Unknown cells (no data) are never classed as safe."], res, "categorical",
                                (0.0, 4.0), terrain.HAZARD_LEGEND)
        stats["hazard_fractions"] = {terrain.HAZARD_LEGEND[k]["label"]: float((hz == k).mean()) for k in terrain.HAZARD_LEGEND}
    cost, cost_meta = build_cost_surface(slope, rough, uncertain, psr, hcfg, cfg["navigation"], shape=(grid.height, grid.width))
    A["cost"] = cost
    M["cost"] = LayerMeta("cost", "Traversal cost multiplier", _derived(ds), "cost per metre (>= 1)",
                          cost_meta.get("formula", cost_meta.get("reason", "")), "Derived",
                          "See core/navigation/cost.py", ["Relative cost; not energy or time.", "Weights are prototype assumptions."],
                          res, "cost", (1.0, float(np.nanpercentile(cost[np.isfinite(cost)], 99)) if np.isfinite(cost).any() else 2.0))
    stats["cost"] = cost_meta

    # Ice priority
    pin = {"psr": psr, "thermal_max": A.get("thermal_max"), "ice_stability_depth": A.get("ice_stability_depth"), "hydrogen": None}
    pr = compute_priority(pin, cfg["ice_priority"], shape=(grid.height, grid.width))
    A["priority_score"] = pr.score
    A["priority_class"] = pr.classes.astype(np.float32)
    A["evidence_coverage"] = pr.coverage
    A["priority_sensitivity"] = pr.sensitivity_std
    pst = Status.SYNTHETIC if ds.kind == "synthetic" else Status.MODEL_ESTIMATE
    lim = ["Investigation priority, NOT detection of water ice.", "Weights are explicit prototype choices; see sensitivity.",
           f"Layers missing: {', '.join(pr.meta.get('layers_missing', [])) or 'none'}."]
    M["priority_score"] = LayerMeta("priority_score", "Ice investigation priority (score)", pst, "0–1 relative",
                                    "Weighted mean of available evidence sub-scores.", "Derived from PSR + Diviner layers",
                                    "core/ice_priority/evidence.py", lim, res, "priority", (0.0, 1.0))
    M["priority_class"] = LayerMeta("priority_class", "Ice investigation priority (class)", pst, "class",
                                    "Score terciles; 'not assessed' where evidence is insufficient.", "Derived",
                                    "Thresholds at 1/3 and 2/3.", lim, res, "categorical", (0.0, 3.0), PRIORITY_LEGEND)
    M["evidence_coverage"] = LayerMeta("evidence_coverage", "Evidence coverage", _derived(ds), "fraction of weight",
                                       "Share of configured evidence weight with data at each cell.", "Derived", "-", [], res, "viridis", (0.0, 1.0))
    M["priority_sensitivity"] = LayerMeta("priority_sensitivity", "Priority sensitivity (std)", _derived(ds), "score std",
                                          "Std of the score under ±50% and drop-one weight perturbations.", "Derived", "One-at-a-time perturbation.",
                                          ["High values: ranking depends on weight choices."], res, "magma", (0.0, 0.3))
    stats["priority"] = pr.meta

    # Image + features
    if "image" in ds.layers:
        img = red("image")
        A["image"] = img
        M["image"] = LayerMeta("image", "Image (LROC NAC)" if ds.kind == "real" else "Image",
                               _status(ds, "image", Status.MEASURED), ds.layers["image"].manifest.get("units") or "DN",
                               ds.layers["image"].manifest.get("source_name", "uploaded image"), src, "Block mean for display.",
                               ["Single observation; illumination geometry controls what is visible.",
                                "PSR interiors receive no direct light and appear near-black."], res, "gray",
                               (0.0, float(np.nanpercentile(img, 99.5)) if np.isfinite(img).any() else 1.0))
        stats["image_coverage_fraction"] = float(np.isfinite(img).mean())
    feats = []
    if elev is not None and res:
        feats = detect_topographic_anomalies(elev, res)
    stats["boulder_detection"] = boulder_detection_feasibility(res)
    cfg_hash = stable_hash({"v": PIPELINE_VERSION, "ds": ds.checksum, "a": cfg["analysis"], "h": hcfg,
                            "n": cfg["navigation"], "i": cfg["ice_priority"]})
    plog.append({"step": "analyse", "seconds": round(time.time() - t0, 2), "config_hash": cfg_hash})
    return Analysis(ds, grid, A, M, stats, cfg_hash, plog, pin, pr, feats)


def _derived(ds: Dataset) -> Status:
    return Status.SYNTHETIC if ds.kind == "synthetic" else Status.DERIVED


def save_layers(an: Analysis, out_dir: Path) -> Path:
    """Persist derived arrays + metadata (reproducible identifier = config hash)."""
    d = out_dir / an.config_hash
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "layers.npz", **an.arrays)
    (d / "layers_meta.json").write_text(json.dumps({k: v.to_dict() for k, v in an.meta.items()}, indent=2, default=str))
    (d / "manifest.json").write_text(json.dumps(an.dataset.manifest(), indent=2, default=str))
    return d
