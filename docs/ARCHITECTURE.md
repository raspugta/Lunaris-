# Architecture and Phase-0 audit

## Phase 0 — repository reconnaissance (2026-10-30)

| Brief item | Finding |
|---|---|
| Repository inventory | The folder contained only the project build brief (a PDF). There was no git repository, prior code, Depth Anything integration, UI, tests or data. |
| Current entry point / what works | None existed. The brief describes an "existing 2D-to-3D prototype", but none was supplied, so nothing could be preserved or broken. |
| Input data / source verification | No input data was supplied. Data was sourced fresh from verified NASA archives (see README → Data setup). |
| Depth Anything V2 | Not present. Implemented from scratch behind an adapter (Hugging Face `depth-anything/Depth-Anything-V2-Small-hf`). |
| Environment | Linux, Python 3.14, NVIDIA RTX 3050 Laptop (4 GB VRAM), 14 GB RAM, Node 20+ available. No system GDAL; the rasterio wheels bundle it. |
| Compatibility risks | Python 3.14 is new, but all pinned wheels install. A 4 GB GPU only fits small depth models; the adapter falls back to CPU, and the Small model runs on CPU in ~2–3 s for a 1024 px input. |

Because the "existing prototype" was absent, the minimal-change principle applied to the brief's
suggested structure. It was adopted as-is with only the modules needed for a full vertical slice.

## Component diagram

```
             ┌───────────────────────── frontend (static, offline) ─────────────────────────┐
             │ dataset picker · provenance · layer list · 2D map · 3D (three.js) · inspector │
             └───────────────────────────────▲──────────────────────────────────────────────┘
                                             │ JSON / PNG
                              backend/app.py (FastAPI, thin)
                                             │
 ┌───────────── core/pipeline.py ────────────┼───────────────────────────────────────────────┐
 │ load_dataset → build_dataset (ingest, crop, align) → analyse (layers + stats + priority)  │
 └──┬──────────────┬───────────────┬──────────────┬────────────────┬─────────────┬───────────┘
    ingestion/     geospatial/      reconstruction/ ice_priority/    navigation/   reporting/
    loader, pds3   grid, terrain    adapter,compare evidence         cost, A*,     figures,
                                    run                              planner       report
```

* Core modules have no web dependencies and are independently testable.
* Each derived layer carries a `LayerMeta` (status, units, resolution, source, method, limitations).
* Analyses are persisted to `data/processed/analyses/<config_hash>/` (layers.npz + metadata + manifest).
  Depth inference is cached by (image checksum, crop, model, input size).
* Long-running depth inference runs as a background job with progress and cooperative cancellation.

## Phase status

| Phase | Status |
|---|---|
| 0 Audit | done (this document) |
| 1 Stabilise | n/a (no prior app); reproducible setup with pinned requirements |
| 2 Data foundation | done: GeoTIFF/PDS3/PNG ingestion, manifests, checksums, synthetic fixture |
| 3 Terrain core | done: slope, roughness, hazard, uncertainty flags, DA-V2 + LOLA comparison |
| 4 Navigation | done: cost surface, A*, validation, alternatives, 2D/3D display |
| 5 Evidence analysis | done: priority model, sensitivity, inspector, baseline anomaly detector, annotations |
| 6 Polish | done: dashboard, PDF/JSON export, error handling, tests, docs, demo script |
