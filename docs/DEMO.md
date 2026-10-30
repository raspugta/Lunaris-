# Five-minute demo script

**Narrative:** Inspect → Understand → Prioritise → Navigate → Report.

Before the demo: run the data setup from the README, start the server with
`.venv/bin/python -m uvicorn backend.app:app --port 8000`, and open <http://localhost:8000>.

1. **Inspect (45 s).** Select **[REAL] Shackleton crater & south pole — LOLA 20 m/px** and press **Load**.
   * Point to the green **REAL DATA** badge, and to each layer's product ID, PGDA/PDS URL, instrument, CRS,
     pixel size, SHA-256 and *missing metadata* list.
   * "Features actually covered" lists Shackleton only because this 32 km window contains it.
   * Optional contrast: load **[SYNTHETIC]** to show the striped synthetic banner, then switch back.

2. **Understand (60 s).** Open the *Understand* tab.
   * Status chips: LOLA DEM is *measured*, slope/hazard are *derived*, hillshade is *illustrative*.
   * The slope consistency check against the producer slope map gives r ≈ 0.998.
   * The boulder card says boulders are sub-pixel at 40 m, so no detector is claimed.
   * Press **Run depth model**. Depth Anything V2 runs on the LROC NAC strip. Show the *relative depth* label,
     r ≈ 0.84 against LOLA, and the negative out-of-sample skill: "it sees structure, but it is not topography".

3. **Prioritise (60 s).** Open the *Prioritise* tab.
   * The PSR layer comes from LOLA illumination modelling, not image darkness.
   * Without Diviner thermal data every cell is **Not assessed**, by design. With the Diviner PRP ingested,
     show the classes, then click a cell in a PSR to open the evidence table (raw value → sub-score →
     weight → contribution).
   * Drag the *psr* weight to 0 and recompute. Then show the sensitivity figure (share of cells that change class).

4. **Navigate (75 s).** Open the *Navigate* tab, press **Suggest demo points**, then **Plan route**.
   * The cyan A* route skirts Shackleton's rim. The red dashed direct line is **∞ impassable** (it crosses
     blocked cells). The white distance-only route is ~6 % shorter, but its hazard-weighted cost is ~89 % higher.
   * Validation ticks: every cell is traversable, and the recomputed cost matches.
   * Rotate the 3D view to show the route draped on LOLA heights.
   * Set the slope weight to 20 and replan to show the trade-off.

5. **Report (30 s).** Open the *Report* tab and press **Export PDF brief**. Show the disclaimer, provenance table,
   figure with legend, scale bar and meridian arrow, route comparison table, and limitations. The JSON export
   carries the same content in machine-readable form.

Closing line: *"Every number here traces back to a NASA product, a documented method, or an explicitly
labelled assumption. Where the data can't support a claim, the platform says 'unknown' or 'not assessed'."*
