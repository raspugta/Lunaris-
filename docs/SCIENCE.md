# Scientific semantics and method notes

## Epistemic status of every layer

| Status | Meaning | Examples here |
|---|---|---|
| measured | Instrument observation, minimally transformed | LOLA DEM (gridded altimetry), LROC NAC radiance |
| derived | Deterministic, documented computation, or a producer's published model product | slope, roughness, hazard, PSR map, Diviner PRP |
| model_estimate | Learned model or heuristic score | Depth Anything V2 output, ice investigation priority |
| illustrative | Visual aid only | hillshade |
| synthetic | Generated demo data | everything in `demo_synthetic` |

## Coordinates

South polar stereographic on a sphere (R = 1737.4 km), true scale at the pole, central meridian 0°:
`rho = 2R tan(pi/4 + lat/2)`, `x = rho sin(lon)`, `y = rho cos(lon)`. Raster "up" (+y) is the 0°E
meridian direction. At the pole every direction is north, so figures label the meridian rather than
drawing a single north arrow. Scale error `k − 1 = 2/(1+sin|lat|) − 1` is below 3.1 × 10⁻⁴ within 2° of
the pole and is ignored, with the bound computed and reported. PDS3 IMG georeferencing follows the
LOLA `DSMAP_POLAR` convention `x = (S − S0 − 1)·scale`. It is validated against the independently
produced PGDA GeoTIFF; see VALIDATION.md.

## Terrain derivatives

* **Downsampling:** a NaN-aware block mean. A coarse cell is valid only if at least 75 % of its sub-cells are valid.
* **Slope:** Horn (1981) 3×3 operator on the analysis grid, so the baseline is 2 cells (80 m at 40 m/px). It is only
  computed from calibrated elevation. Cells touching no-data, *including the no-data cell itself* (Horn
  ignores the centre sample, which is a classic bug this code tests against), are NaN.
* **Roughness:** RMS residual of a least-squares plane in a 5×5 window (200 m). Plane removal separates roughness
  from regional slope. The expected value on white noise, σ·√((n−3)/n), is unit-tested.
* **Hazard class:** the worse of the slope class and the roughness class. Unknown is a separate class. Producer
  uncertainty (DEM error > 0.75 m or effective resolution > 80 m) raises the class by one.

## Traversal cost and A*

Per-metre multiplier `c = 1 + 8(s/25°)² + 3(r/6 m) + 4·uncertain + 1.5·PSR`. A cell is blocked (∞) where the
slope is ≥ 25°, the roughness is ≥ 6 m, or there is no data. Edge cost = step × mean(c). The octile
heuristic times min(c) = 1 is admissible and consistent, so A* is optimal on this graph. This is verified
against brute-force Dijkstra on random maps in `tests/test_navigation.py`. Diagonal moves between two
blocked orthogonal neighbours are forbidden.

The weights are prototype choices. The UI exposes them and every report records them.

## Water-ice investigation priority

The score is a weighted mean of the sub-scores available at each cell:

| Evidence | Sub-score | Rationale |
|---|---|---|
| PSR fraction (LOLA illumination model) | linear 0–1 | no direct sunlight over the modelled epoch. This comes from horizon modelling, *not* image darkness |
| Diviner PRP annual max surface T | 1 at ≤ 100 K → 0 at ≥ 130 K | surface water ice is stable against sublimation over geologic time below ~110 K (Vasavada et al. 1999; Paige et al. 2010) |
| Diviner PRP ice-stability depth | 1 at 0 m → 0 at ≥ 1 m; "> 2.87 m" → 0 | depth where the sublimation rate is < 1 kg m⁻² Gyr⁻¹ (PDS label definition) |
| LEND hydrogen | not used | footprints of ~10+ km. It is not a pixel-level ice label |

A cell is **not assessed** unless the thermal layer is present and at least 60 % of the configured weight
is available. PSR membership alone therefore never yields a priority. Class labels are "lower relative
priority", "moderate relative priority" and "candidate for further investigation", never "ice detected".
Sensitivity: each weight is set to ×0.5, ×1.5 and ×0 (9 scenarios). The UI reports the per-cell standard
deviation and the fraction of cells that change class.

## Monocular depth (Depth Anything V2)

The output is affine-invariant relative depth from a model trained on terrestrial images. We report:
Pearson r and Spearman ρ against LOLA (no units needed), a local-relief correlation after Gaussian
high-pass filtering, the best integer pixel shift (alignment check), coverage, and the excluded
low-signal fraction. A metric RMSE is reported *only* out-of-sample, after an affine fit to LOLA on the
other spatial half. That fit is labelled as calibration against the reference, not an independent
accuracy.

## Feature detection

Rover-scale boulders are sub-pixel at 20–40 m/px. The platform states this and does not claim a
detector. An optional Laplacian-of-Gaussian baseline marks compact topographic pits and mounds as
"unvalidated anomalies". Manual annotations are stored separately.
