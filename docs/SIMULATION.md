# Rover traverse simulation

Open it with the **Rover Simulation** button in the top bar or the **6 · Simulate** step, after you plan a route in step 4.

## What it does

It drives a published rover model along the planned A* route over the native 20 m LOLA DEM, every 0.5 m:

| Quantity | Method | Source of truth |
|---|---|---|
| Pitch, roll, body height | least-squares plane through the wheel-contact heights (bilinear DEM) | LOLA DEM (PGDA) |
| Wheel loads | quasi-static linear load sharing with centre-of-mass height | CG height **assumed** |
| Sinkage, compaction resistance | Bekker pressure–sinkage, rigid wheel | Apollo LRV soil values (Costes et al. 1972, via Carrier/LGI) |
| Thrust vs slip | Janosi–Hanamoto shear; slip solved so thrust = resistance + grade | same |
| Immobilisation | thrust at 100 % slip is still insufficient | — |
| Downhill / lateral slide | braking or lateral demand exceeds soil shear strength | — |
| Static tip-over | any wheel load ≤ 0 | CG **assumed** |
| Ground speed and time | commanded wheel speed × (1 − slip) | VIPER 10 / 20 cm/s (published) |
| PSR exposure | time inside the LOLA PSR map vs the published 9 h shadow-ops limit | PGDA LPSR; NASA VIPER briefing |
| Shadow energy | non-mobility load **derived** from 5420 Wh / 9 h, plus computed mobility power | NASA VIPER briefing |
| Sun / Earth visibility | long-term averages along the path | Mazarico et al. 2011 (PGDA product 69) |
| Expected energy (optional) | average sun × 450 W − **assumed** sunlit load | indicative only |

**Verdicts:**

* **PASS:** completed, no warnings.
* **MARGINAL:** completed, but beyond the 15° design slope, over 40 % slip, or under 10° stability margin.
* **FAIL:** immobilised, tipped over, slid, exceeded the PSR limit, or hit the battery reserve in shadow.
* **INDICATIVE ONLY:** the rover profile relies on unpublished geometry (Pragyan).

Every run also repeats on a *loose-soil* sensitivity case.

## Rover profiles

| Profile | Use | Fidelity |
|---|---|---|
| NASA VIPER | the PSR rover; physics default for lunar operations | specs published by NASA; CG height and drive efficiency assumed |
| Perseverance-class (hypothetical) | NASA's real 3D model; Mars 2020 specs driven on lunar soil | mass, wheels, speed, 45° tilt and 40 cm obstacles published; wheel layout measured from NASA's to-scale model; CG derived from the 45° rating; battery unpublished, so no energy limit |
| ISRO Pragyan | small-rover comparison | INDICATIVE ONLY (geometry unpublished) |

## Rover model

The 3D VIPER is built procedurally from NASA-published dimensions (four 0.5 m spoked wheels with 2.5 cm
grousers, 1.3 m track, ~1 m wheelbase, 1.7 × 1.7 × 2.5 m envelope, three side solar arrays, mast with
cameras, lights and high-gain antenna). NASA has not released a free official VIPER CAD/3D file. The pose
comes only from the simulation frames. Wheel spin = commanded speed × time, so slip shows as wheels turning
faster than the ground moves. Per-wheel suspension offsets are the residuals of the wheel contacts from the
body plane. The close-up terrain is the LOLA DEM bilinearly upsampled to 4 m vertices, at true vertical scale.

## Results on the Shackleton demo route (39.4 km)

| | Apollo LRV soil | Loose-soil sensitivity |
|---|---|---|
| VIPER verdict | **MARGINAL**: completes in 128.8 h | **FAIL**: immobilised at 9.7 km on a 16° slope |
| Warnings | 1.6 km above the 15° design slope; slip up to 74 %; stability margin 7.7° | — |
| PSR time | 0.23 h (well under 9 h) | — |
| Average sun / Earth visibility on route | 30 % / 50 % | — |
| Expected energy (sunlit load 250 W, assumed) | deficit ≈ 44 kWh → ≈ 220 h of parked full-sun charging | — |

Interpretation: VIPER can physically drive this route on nominal soil, but several hundred metres are outside its
design envelope, and the result depends on soil strength. Under the energy estimate, the traverse is a
multi-week campaign with charging stops, consistent in order of magnitude with NASA's VIPER mission planning.

## Physics tests (`tests/test_rover_sim.py`)

* Flat-ground sinkage equals the closed-form Bekker value; pitch and roll are 0.
* On an 8° plane, pitch = 8° driving uphill and roll = −8° driving across it.
* Ground speed = 0.1 × (1 − slip), and drive time matches.
* The slip solution satisfies the thrust/resistance force balance to 1e-6.
* A 30° slope immobilises (traction limit ≈ 26°), and a 5° slope completes.
* 17° raises the 15° design-slope warning. A raised CG tips the rover on a 15° cross-slope.
* A 3.8 km drive inside a PSR breaches the published 9 h limit. The derived load reproduces 5420 Wh over 9 h exactly.

## Cannot be validated with this data

* Obstacles smaller than the DEM (VIPER's 15 cm limit). Rocks and small craters are invisible at 20 m.
* Real PSR regolith strength (never measured in situ).
* Time-specific illumination (only 18.6-year averages are used), thermal survival and comms scheduling.
* Dynamic effects, slip-sinkage and turn-in-place manoeuvres.

Getting closer to a flight-grade answer would need 1 m DEMs (e.g. LROC NAC / ShadowCam stereo), rock-abundance
maps, a time-resolved illumination model, and VIPER's internal CG and power tables.
