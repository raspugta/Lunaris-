# Validation results

All numbers come from `python scripts/validate_real_data.py`, which writes
`docs/validation_results.json`. They were measured on the real Shackleton 32 km window
(PGDA LOLA 20 m/px, analysed at 40 m/px). Nothing here is hand-entered.

## 1. Georeferencing: PDS3 IMG vs independent PGDA GeoTIFF

The PDS LOLA GDR `LDEM_875S_20M` (2017 product, georeferenced by our own PDS3 label parser) was compared
with PGDA `LDEM_80S_20MPP_ADJ` (2023 product, GeoTIFF georeferencing) over the identical window.

| Metric | Value |
|---|---|
| Best integer pixel shift (40 m) | **(0, 0)** |
| Median height difference | 0.07 m |
| MAD of difference | 0.74 m |
| 95th percentile \|Δh\| | 3.97 m |
| High-pass correlation | 0.948 |

This confirms the PDS3 coordinate transform (`x = (S − S0 − 1)·scale`) to sub-pixel level. The metre-level
residuals are the expected product-version differences (crossover adjustment in the 2023 release).

Analytic control points for the polar stereographic projection (pole → origin, 1° from the pole ≈ 30.32 km,
forward/inverse round-trip < 1e-9°) are unit-tested in `tests/test_geospatial.py`.

## 2. Terrain derivatives

| Check | Result | Note |
|---|---|---|
| Our 40 m Horn slope vs PGDA slope map (block-averaged) | r = 0.998, median \|Δ\| = 0.36° | Consistency only: same source DEM, not independent validation |
| Slope of an analytic 10 % plane | 5.7106° (exact) | unit test |
| Roughness on white noise σ = 2 m, 7×7 window | σ·√(46/49) within 5 % | unit test |

Hazard composition of the window: 31.8 % low, 16.6 % moderate, 25.3 % high, 25.3 % blocked, 1.0 % unknown.
PSR (cell ≥ 50 % in a modelled PSR): 25.7 %. Elevation flagged uncertain: 0.17 %.

## 3. Route planning

* **A* optimality:** matches brute-force Dijkstra (rel. 1e-6) on 15 random obstacle maps (unit test).
* **Demo route** (suggested points in the largest traversable component):

| | A* candidate | Distance-only A* | Direct line |
|---|---|---|---|
| Length | 39.4 km | 37.0 km | 30.3 km |
| Hazard-weighted cost | 102 420 | 193 096 (+89 %) | ∞ (crosses 282 blocked cells) |
| Max / mean slope | 24.8° / 7.2° | 24.9° / 14.4° | 34.1° / 21.6° |
| Cells in PSR | 1 | 20 | 330 |

Validation flags: all cells traversable ✔, recomputed cost equals planner cost ✔. Runtime 3.1 s on CPU
(800 × 800 grid, ~425 k node expansions).

## 4. Depth Anything V2 (Small) on LROC NAC M139729686LE vs LOLA

| Metric | Value |
|---|---|
| Image footprint in window / co-valid after low-signal mask | 10.2 % / 8.4 % |
| Pearson r (full field) | **0.84** |
| Spearman ρ | 0.72 |
| Pearson r, local relief (high-pass σ ≈ 120 m) | 0.73 |
| Best integer shift | (−1, −1) px, i.e. ~20 m (shading-induced displacement or residual registration) |
| Out-of-sample RMSE after affine fit to LOLA on the other half | 598 m vs test σ 265 m → **skill −1.25** |

**Interpretation.** The model recovers relative structure that correlates strongly with true topography.
However, a single affine mapping fitted on one half of the strip does *not* transfer to the other half. The
output is therefore not usable as metric elevation, and the platform keeps it labelled *relative depth,
uncalibrated*. This is the expected failure mode of a terrestrially trained monocular model under
grazing polar illumination, and reporting it is the point.

## 5. Water-ice investigation priority

Without Diviner thermal data the model returns **not assessed** for 100 % of cells. PSR membership alone
is never allowed to produce a priority, and this is checked by `tests/test_ice_priority.py`.

With the Diviner Polar Resource Product ingested (median facet spacing 239 m, consistent with Diviner's
240 m polar products):

| | Value |
|---|---|
| Layers used | PSR (LOLA), Diviner T<sub>max</sub>, Diviner ice-stability depth |
| Assessed fraction | 100 % |
| Candidate for further investigation / moderate / lower | 23.7 % / 2.9 % / 73.4 % |
| Cells changing class under ≥ 1 of 9 weight scenarios | **31.5 %** |
| Median per-cell score σ across scenarios | 0.016 |

The 31.5 % figure is the honest headline. Most class changes come from the drop-one-layer scenarios, so
the *ranking* depends materially on which evidence is trusted. Read the map as a hypothesis generator.

**Data-quality finding.** The PRP `ice_depth` column contains the value **−1** on 100,319 facets. The PDS
label and `prpds.cat` do not define it. It occurs only where T<sub>max</sub> ≤ ~101.3 K, while the documented
0 ("stable at the surface") spans ~101.3–130 K. We do not guess its meaning: −1 is set to no-data (20.5 %
of the window). Those cells are still assessed from PSR + thermal evidence (coverage 0.75 ≥ 0.6).

## What has *not* been validated

* Rover traversability thresholds against any real vehicle model.
* The ice-priority weights against ground truth. None exists; the outputs are rankings, not detections.
* Rover-scale hazards (boulders), which this resolution cannot resolve.
