"""Glue between an Analysis + planned route and the rover simulator."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.rover.sim import SIM_VERSION, Sampler, load_rover_db, pv, simulate

FIDELITY = [
    "Terrain is the LOLA DEM at its native 20 m grid (bilinear). Rocks, boulders and craters smaller than ~20-40 m "
    "are invisible to this simulation; VIPER's published 15 cm obstacle limit cannot be checked from this data.",
    "Wheel-soil forces use the Bekker-Wong model with Apollo LRV soil parameters (Costes et al. 1972). Real PSR regolith "
    "properties are unmeasured and may differ; see the loose-soil sensitivity run.",
    "Quasi-static model: no dynamics, wheel-level suspension articulation, slip-sinkage or turn-in-place time.",
    "Solar and Earth visibility are long-term AVERAGES (Mazarico et al. 2011), not a time-specific illumination simulation.",
    "Parameters tagged 'assumed' are not published by the agency and are shown in the parameter table.",
]


def simulate_route(an, route: dict, rover_id: str, soil_id: str, speed_mode: str = "nominal",
                   sunlit_load_w: float | None = None, ds: float = 0.5) -> dict[str, Any]:
    db = load_rover_db()
    if rover_id not in db["rovers"]:
        raise ValueError(f"unknown rover '{rover_id}'")
    if soil_id not in db["soils"]:
        raise ValueError(f"unknown soil '{soil_id}'")
    if not route or not route.get("found"):
        raise ValueError("plan a route first (Navigate step)")
    ds_layers = an.dataset.layers
    if "elevation" not in ds_layers or not an.dataset.grid.calibrated:
        raise ValueError("simulation needs a calibrated elevation raster")
    rover = db["rovers"][rover_id]
    g = db["environment"]["g_moon"]
    grid = an.dataset.grid
    elev = Sampler(ds_layers["elevation"].data.astype(np.float64), grid.transform)
    psr = Sampler(np.nan_to_num(ds_layers["psr_mask"].data).astype(np.float64), grid.transform) if "psr_mask" in ds_layers else None

    def opt(role):
        if role not in ds_layers:
            return None
        d = ds_layers[role].data.astype(np.float64)
        if np.nanmax(d) > 1.0 + 1e-6:
            raise ValueError(f"{role} must be a 0-1 fraction (run scripts/fetch_illumination_aoi.py)")
        return Sampler(d, grid.transform)  # NaN = outside the product (unknown), never "no sun"

    sun, earth = opt("illumination_avg"), opt("earth_visibility_avg")
    cells = np.array(route["cells"], float)
    x, y = an.grid.pixel_to_world(cells[:, 0], cells[:, 1])
    xy = np.c_[x, y]
    t0 = time.time()
    res = simulate(xy, elev, rover, db["soils"][soil_id], g, psr=psr, sun=sun, earth=earth, speed_mode=speed_mode,
                   ds=ds, sunlit_load_w=sunlit_load_w)
    res["runtime_s"] = round(time.time() - t0, 2)
    # Sensitivity: same route, loose-soil case (if the nominal soil was selected).
    alt_id = "loose_psr_hypothetical" if soil_id != "loose_psr_hypothetical" else "apollo_lrv_nominal"
    alt = simulate(xy, elev, rover, db["soils"][alt_id], g, psr=psr, speed_mode=speed_mode, ds=ds, max_frames=10)
    res["soil_sensitivity"] = {"soil": alt_id, "soil_name": db["soils"][alt_id]["name"],
                               **{k: alt["summary"][k] for k in ("verdict", "failure", "distance_m", "max_slip", "mean_slip",
                                                                  "drive_time_h", "max_sinkage_mm")}}
    res["rover"] = {"id": rover_id, "name": rover["name"], "agency": rover["agency"], "references": rover["references"],
                    "designed_for_psr": rover.get("designed_for_psr"), "fidelity_warning": rover.get("fidelity_warning"),
                    "params": rover["params"]}
    res["soil"] = {"id": soil_id, **db["soils"][soil_id]}
    res["fidelity"] = FIDELITY + ([rover["fidelity_warning"]] if rover.get("fidelity_warning") else [])
    low = any(v.get("tag") == "assumed" for k, v in rover["params"].items()
              if k in ("wheel_diameter_m", "wheel_width_m", "mass_kg", "speed_nominal_mps"))
    res["low_fidelity"] = low
    if low:
        res["summary"]["verdict_raw"] = res["summary"]["verdict"]
        res["summary"]["verdict"] = "INDICATIVE ONLY"
        res["soil_sensitivity"]["verdict"] = "INDICATIVE ONLY"
    res["sim_version"] = SIM_VERSION
    res["speed_mode"] = speed_mode
    res["dimensions"] = {k: pv(rover["params"], k) for k in ("wheel_diameter_m", "wheel_width_m", "grouser_height_m",
                                                               "track_width_m", "wheelbase_m", "body_length_m", "body_width_m",
                                                               "height_m", "ground_clearance_m", "n_wheels", "steer_range_deg")}
    return res
