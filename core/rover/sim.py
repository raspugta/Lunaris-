"""Quasi-static rover traverse simulator over a LOLA DEM.

What is modelled (and how)
--------------------------
* **Kinematics / attitude.** The route polyline is resampled every ``ds`` metres.
  At each pose the wheel contact heights are bilinearly interpolated from the
  *native* (20 m) DEM. Pitch and roll come from a least-squares plane through the
  wheel contacts. Wheel-scale relief below the DEM resolution is NOT represented.
* **Static load distribution.** Linear load sharing with centre-of-mass height
  ``h``: F_j = W cos(pitch) cos(roll) [1/N - h tan(pitch) x_j / sum(x^2)
  - h tan(roll) y_j / sum(y^2)]. A non-positive wheel load means static tip-over.
* **Wheel-soil interaction (Bekker-Wong, rigid wheel).**
  sinkage  z0 = [3 W / ((3-n)(k_c + b k_phi) sqrt(D))]^(2/(2n+1))
  compaction resistance  R_c = (k_c + b k_phi) z0^(n+1) / (n+1)
  contact length  l = sqrt(D z0)
  thrust  H(i) = (b l c + W tan(phi)) [1 - K/(i l) (1 - exp(-i l / K))]   (Janosi-Hanamoto)
  Slip i solves sum H_j(i) = sum R_c,j + m g sin(pitch). If even i = 1 cannot
  supply it, the rover is immobilised. Downhill, the required braking force is
  checked against the shear limit sum(b l c + W tan(phi)). Grousers are
  conservatively ignored (they would add thrust).
* **Speed and time.** Ground speed = commanded wheel speed x (1 - i).
* **Power.** Mobility electrical power = sum(H_j) v_wheel / eta. The non-mobility
  ("hotel") load in shadow is *derived* so that a flat drive at nominal speed uses
  the full battery in the published shadow-operations limit (VIPER: 5420 Wh / 9 h).
* **Shadow budget.** Time inside PSR cells (LOLA PSR map) drains the battery at
  the shadow load. The published PSR-operations limit is enforced as a hard check.
* **Solar / comms context.** Long-term *average* sun and Earth visibility
  (Mazarico et al. 2011) are reported along the path. They are averages over
  18.6 years, not a time-specific illumination simulation, so they do not drive
  pass/fail. An expected-energy estimate using an *assumed* sunlit load is
  reported separately and is labelled as such.

Not modelled: rocks and craters smaller than the DEM resolution, dynamic effects,
wheel-level suspension articulation, steering/turn-in-place time, slip-sinkage,
thermal survival, communications scheduling, and terrain-relative localisation errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml
from scipy import ndimage

from core.config import ROOT

SIM_VERSION = "1.0.0"


def load_rover_db(path=None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "configs" / "rovers.yaml").read_text())


def pv(params: dict, key: str):
    v = params[key]
    return v["value"] if isinstance(v, dict) else v


@dataclass
class Sampler:
    """Bilinear sampler of a north-up raster in world coordinates."""
    data: np.ndarray
    transform: tuple

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, _, c, _, e, f = self.transform
        col = (np.asarray(x) - c) / a - 0.5
        row = (np.asarray(y) - f) / e - 0.5
        return ndimage.map_coordinates(self.data, [row, col], order=1, mode="nearest", cval=np.nan)


def wheel_layout(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Wheel positions (x forward, y left) relative to the body centre."""
    if "wheel_positions_m" in p:
        wp = np.array(pv(p, "wheel_positions_m"), float)
        return wp[:, 0] - wp[:, 0].mean(), wp[:, 1] - wp[:, 1].mean()
    n = int(pv(p, "n_wheels"))
    L, T = pv(p, "wheelbase_m"), pv(p, "track_width_m")
    if n == 4:
        xs = np.array([L / 2, L / 2, -L / 2, -L / 2])
        ys = np.array([T / 2, -T / 2, T / 2, -T / 2])
    elif n == 6:
        xs = np.array([L / 2, L / 2, 0, 0, -L / 2, -L / 2])
        ys = np.array([T / 2, -T / 2] * 3)
    else:
        raise ValueError("only 4- and 6-wheel layouts are supported")
    return xs, ys


def bekker_wheel(W: np.ndarray, soil: dict, b: float, D: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-wheel sinkage z0 (m), compaction resistance R_c (N), contact length l (m), shear limit (N)."""
    n, kc, kphi = soil["n"], soil["k_c"], soil["k_phi"]
    W = np.clip(W, 0, None)
    z0 = (3 * W / ((3 - n) * (kc + b * kphi) * math.sqrt(D))) ** (2 / (2 * n + 1))
    Rc = (kc + b * kphi) * z0 ** (n + 1) / (n + 1)
    l = np.sqrt(D * z0)
    hmax = b * l * soil["cohesion_pa"] + W * math.tan(math.radians(soil["phi_deg"]))
    return z0, Rc, l, hmax


def thrust(i: np.ndarray, hmax: np.ndarray, l: np.ndarray, K: float) -> np.ndarray:
    """Janosi-Hanamoto thrust for slip i (broadcast over wheels)."""
    il = np.clip(i, 1e-6, 1)[..., None] * l
    return hmax * (1 - (K / np.maximum(il, 1e-12)) * (1 - np.exp(-il / K)))


def solve_slip(required: np.ndarray, hmax: np.ndarray, l: np.ndarray, K: float, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised bisection for slip so that sum_j H_j(i) = required. Returns (slip, feasible)."""
    lo = np.zeros(required.shape)
    hi = np.ones(required.shape)
    feasible = thrust(hi, hmax, l, K).sum(-1) >= required
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = thrust(mid, hmax, l, K).sum(-1) >= required
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    slip = np.where(required <= 0, 0.0, hi)
    return np.where(feasible, slip, 1.0), feasible


def resample_path(xy: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.diff(xy, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    keep = seglen > 1e-9
    xy = np.vstack([xy[:1], xy[1:][keep]])
    seglen = seglen[keep]
    s = np.concatenate([[0], np.cumsum(seglen)])
    st = np.arange(0, s[-1], ds)
    st = np.append(st, s[-1]) if st[-1] < s[-1] else st
    x = np.interp(st, s, xy[:, 0])
    y = np.interp(st, s, xy[:, 1])
    # heading from a centred difference over ~one wheelbase, so the 45-deg staircase
    # of an 8-connected grid path does not produce unphysical instantaneous yaw.
    k = max(1, int(round(2.0 / ds)))
    xa, ya = np.interp(np.clip(st + k * ds, 0, s[-1]), s, xy[:, 0]), np.interp(np.clip(st + k * ds, 0, s[-1]), s, xy[:, 1])
    xb, yb = np.interp(np.clip(st - k * ds, 0, s[-1]), s, xy[:, 0]), np.interp(np.clip(st - k * ds, 0, s[-1]), s, xy[:, 1])
    heading = np.arctan2(xa - xb, ya - yb)  # radians clockwise from +y (grid up)
    return st, np.c_[x, y], heading


def simulate(xy_path: np.ndarray, elev: Sampler, rover: dict, soil: dict, g: float, *,
             psr: Sampler | None = None, sun: Sampler | None = None, earth: Sampler | None = None,
             speed_mode: str = "nominal", ds: float = 0.5, start_soc: float = 1.0,
             sunlit_load_w: float | None = None, max_frames: int = 4000) -> dict[str, Any]:
    p = rover["params"]
    m, D, b = pv(p, "mass_kg"), pv(p, "wheel_diameter_m"), pv(p, "wheel_width_m")
    h, eta = pv(p, "cg_height_m"), pv(p, "drive_efficiency")
    v_cmd = pv(p, "speed_nominal_mps") if speed_mode == "nominal" else pv(p, "speed_max_mps")
    xs, ys = wheel_layout(p)
    N = len(xs)
    s, xy, hd = resample_path(xy_path, ds)
    n = len(s)
    ch, sh = np.cos(hd)[:, None], np.sin(hd)[:, None]
    # heading measured clockwise from +y: forward = (sin, cos), left = (-cos, sin)
    wx = xy[:, :1] + xs[None] * sh - ys[None] * ch
    wy = xy[:, 1:] + xs[None] * ch + ys[None] * sh
    wz = elev(wx.ravel(), wy.ravel()).reshape(n, N)
    valid = np.isfinite(wz).all(1)
    # Least-squares plane z = z0 + a x_body + c y_body through the wheel contacts.
    A = np.c_[np.ones(N), xs, ys]
    coef = np.linalg.lstsq(A, np.nan_to_num(wz).T, rcond=None)[0]  # (3, n)
    body_z, dzdx, dzdy = coef
    pitch = np.arctan(dzdx)            # + nose up
    roll = np.arctan(dzdy)             # + left side up
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    Wn = m * g * np.cos(pitch) * np.cos(roll)
    frac = (1 / N - (h * np.tan(pitch))[:, None] * xs[None] / (xs ** 2).sum()
            - (h * np.tan(roll))[:, None] * ys[None] / (ys ** 2).sum())
    Wj = Wn[:, None] * frac
    tipped = (frac <= 0).any(1)
    z0, Rc, l, hmax = bekker_wheel(Wj, soil, b, D)
    grade = m * g * np.sin(pitch)
    required = Rc.sum(1) + grade
    slip, climb_ok = solve_slip(required, hmax, l, soil["K_m"])
    shear_cap = hmax.sum(1)
    sliding = (-required > shear_cap)  # downhill: braking demand beyond shear limit
    lateral_cap = shear_cap
    lateral_slide = (m * g * np.abs(np.sin(roll)) > lateral_cap)
    v = v_cmd * (1 - slip)
    stuck = ~climb_ok | tipped | sliding | lateral_slide | ~valid
    dt = np.where(v > 1e-6, ds / np.maximum(v, 1e-6), np.inf)
    dt[-1] = 0.0
    p_mob = np.clip(required, 0, None) * v_cmd / eta  # electrical W
    # Shadow (hotel) load derived from the published shadow-ops endurance.
    batt = pv(p, "battery_wh")
    psr_lim = pv(p, "psr_ops_limit_h")
    has_batt = batt is not None
    W_flat = m * g / N
    _, Rc_flat, _, _ = bekker_wheel(np.full(N, W_flat), soil, b, D)
    p_mob_flat = Rc_flat.sum() * pv(p, "speed_nominal_mps") / eta
    if p.get("hotel_power_w", {}).get("value") is not None:
        hotel = pv(p, "hotel_power_w")
        hotel_src = "assumed (see rovers.yaml)"
    elif psr_lim and has_batt:
        hotel = batt / psr_lim - p_mob_flat
        hotel_src = f"derived: {batt} Wh / {psr_lim} h - flat-drive mobility {p_mob_flat:.1f} W"
    else:
        hotel, hotel_src = 0.0, "unknown"
    in_psr = (psr(xy[:, 0], xy[:, 1]) >= 0.5) if psr is not None else np.zeros(n, bool)
    sun_f = np.clip(sun(xy[:, 0], xy[:, 1]), 0, 1) if sun is not None else None
    earth_f = np.clip(earth(xy[:, 0], xy[:, 1]), 0, 1) if earth is not None else None
    if sun_f is not None and not np.isfinite(sun_f).any():
        sun_f = None
    if earth_f is not None and not np.isfinite(earth_f).any():
        earth_f = None
    # Stop integration at the first stuck pose.
    first_stuck = int(np.argmax(stuck)) if stuck.any() else None
    end = first_stuck if first_stuck is not None else n - 1
    dt_eff = dt.copy()
    dt_eff[end:] = 0.0
    t = np.concatenate([[0], np.cumsum(dt_eff[:-1])])
    # Shadow battery budget (published-constraint based).
    e_shadow = np.where(in_psr, (hotel + p_mob) * dt_eff / 3600.0, 0.0)
    usable = batt * (start_soc - pv(p, "battery_reserve_frac")) if has_batt else None
    shadow_used = np.cumsum(e_shadow)
    # continuous PSR exposure
    run, max_run = 0.0, 0.0
    psr_run = np.zeros(n)
    for k in range(end + 1):
        run = run + dt_eff[k] if in_psr[k] else 0.0
        psr_run[k] = run
        max_run = max(max_run, run)
    soc_shadow = np.clip(start_soc - shadow_used / batt, 0, 1) if has_batt else np.full(n, np.nan)
    # Expected energy estimate (assumed sunlit load; average illumination).
    exp = None
    if sun_f is not None and sunlit_load_w is not None and has_batt and pv(p, "solar_peak_w") and np.isfinite(sun_f).all():
        solar = pv(p, "solar_peak_w")
        f = np.where(in_psr, 0.0, sun_f)
        p_net = f * (solar - sunlit_load_w) - (1 - f) * hotel - p_mob
        e = np.cumsum(p_net * dt_eff / 3600.0)
        soc_exp = np.clip(start_soc + e / batt, 0.0, 1.0)  # displayed state of charge (bounded)
        usable_e = batt * (start_soc - pv(p, "battery_reserve_frac"))
        deficit = max(0.0, -float(e[min(len(e) - 1, len(e) - 1)]) - usable_e)
        charge_rate = solar - sunlit_load_w
        exp = {"soc": soc_exp, "label": f"EXPECTED-VALUE estimate: average illumination x {solar} W solar; "
               f"sunlit non-mobility load {sunlit_load_w} W is ASSUMED; shadow load {hotel:.0f} W derived. "
               "Not a time-specific power simulation.",
               "min_soc": float(np.min(soc_exp[:end + 1])), "end_soc": float(soc_exp[end]),
               "net_energy_wh": float(e[end]), "energy_deficit_wh": deficit,
               "parked_full_sun_charging_h": (deficit / charge_rate) if charge_rate > 0 else None,
               "note": "Continuous driving is energy-infeasible under this estimate; the rover would have to park at "
                       "sunlit sites to recharge for about the stated time." if deficit > 0 else
                       "Energy-feasible without parking under this estimate."}

    # ----- verdict
    events: list[dict[str, Any]] = []

    def ev(kind, sev, k, msg):
        events.append({"type": kind, "severity": sev, "s_m": float(s[k]), "t_h": float(t[k] / 3600), "index": int(k), "message": msg})

    fail_reason = None
    if first_stuck is not None:
        k = first_stuck
        if not valid[k]:
            fail_reason = "left DEM coverage / no elevation data"
        elif tipped[k]:
            fail_reason = f"static tip-over (pitch {np.degrees(pitch[k]):.1f}°, roll {np.degrees(roll[k]):.1f}°)"
        elif not climb_ok[k]:
            fail_reason = f"immobilised: slope {slope[k]:.1f}° needs more thrust than the soil can provide at 100 % slip"
        elif sliding[k]:
            fail_reason = f"uncontrolled downhill slide on {slope[k]:.1f}° (braking demand exceeds soil shear strength)"
        else:
            fail_reason = f"lateral slide on {np.degrees(roll[k]):.1f}° cross-slope"
        ev("failure", "fail", k, fail_reason)
    seg = slice(0, end + 1)
    max_slip_ops = pv(p, "max_slip_ops")
    req_slope = pv(p, "max_slope_req_deg")
    over_slip = np.flatnonzero(slip[seg] > max_slip_ops)
    if over_slip.size:
        ev("high_slip", "warn", int(over_slip[0]), f"slip {100 * slip[over_slip[0]]:.0f}% > {100 * max_slip_ops:.0f}% operational threshold "
           f"({over_slip.size * ds:.0f} m of route)")
    if req_slope:
        over = np.flatnonzero(slope[seg] > req_slope)
        if over.size:
            ev("design_slope", "warn", int(over[0]), f"slope {slope[over[0]]:.1f}° exceeds the {req_slope}° design requirement "
               f"({over.size * ds:.0f} m of route)")
    # Static tip angles about the support-polygon edges (outermost wheel rows/sides).
    ends = (xs == xs.max()) | (xs == xs.min())  # side tipping axis runs front wheel -> rear wheel
    tip_lat = math.degrees(math.atan(np.abs(ys[ends]).min() / h))
    tip_lon = math.degrees(math.atan(min(xs.max(), -xs.min()) / h))
    margin = np.minimum(tip_lon - np.abs(np.degrees(pitch)), tip_lat - np.abs(np.degrees(roll)))
    low = np.flatnonzero(margin[seg] < 10)
    if low.size and first_stuck is None:
        ev("stability", "warn", int(low[0]), f"static stability margin {margin[low[0]]:.1f}° < 10°")
    if psr_lim is not None and has_batt and max_run / 3600 > psr_lim:
        k = int(np.argmax(psr_run > psr_lim * 3600))
        ev("psr_limit", "fail", k, f"continuous PSR exposure exceeds the {psr_lim} h limit")
        fail_reason = fail_reason or f"PSR exposure {max_run / 3600:.1f} h > {psr_lim} h limit"
    if has_batt and shadow_used[end] > usable:
        k = int(np.argmax(shadow_used > usable))
        ev("battery", "fail", k, "battery reserve reached while in shadow")
        fail_reason = fail_reason or "battery reserve reached in shadow"
    if exp and exp["min_soc"] < pv(p, "battery_reserve_frac"):
        k = int(np.argmax(exp["soc"][:end + 1] < pv(p, "battery_reserve_frac")))
        ev("energy_expected", "info", k, "EXPECTED-value energy estimate falls below reserve (uses an assumed sunlit load; indicative only)")
    events.sort(key=lambda e: e["s_m"])
    warns = [e for e in events if e["severity"] == "warn"]
    verdict = "FAIL" if fail_reason else ("MARGINAL" if warns else "PASS")

    # ----- frames for playback (decimated) + summary
    ev_idx = np.array([e["index"] for e in events if e["index"] <= end], dtype=int)
    idx = np.unique(np.concatenate([np.linspace(0, end, min(max_frames, end + 1)).round().astype(int), ev_idx]))
    frames = {
        "s": s[idx], "t": t[idx], "x": xy[idx, 0], "y": xy[idx, 1], "z": body_z[idx], "heading": np.degrees(hd[idx]),
        "pitch": np.degrees(pitch[idx]), "roll": np.degrees(roll[idx]), "slope": slope[idx], "slip": slip[idx],
        "speed": v[idx], "sinkage_mm": 1000 * z0[idx].mean(1), "power_w": p_mob[idx] + np.where(in_psr[idx], hotel, 0),
        "p_mob_w": p_mob[idx], "soc_shadow": soc_shadow[idx], "in_psr": in_psr[idx].astype(int),
        "margin_deg": margin[idx], "wheel_z": wz[idx], "wheel_load_n": Wj[idx],
        "sun_avg": sun_f[idx] if sun_f is not None else None,
        "earth_avg": earth_f[idx] if earth_f is not None else None,
        "soc_expected": exp["soc"][idx] if exp else None,
    }
    distance = float(s[end])
    summary = {
        "verdict": verdict, "failure": fail_reason,
        "distance_m": distance, "route_length_m": float(s[-1]), "completed_fraction": distance / max(float(s[-1]), 1e-9),
        "drive_time_h": float(t[end] / 3600), "mean_speed_mps": distance / max(float(t[end]), 1e-9),
        "max_slope_deg": float(np.nanmax(slope[seg])), "max_pitch_deg": float(np.nanmax(np.abs(np.degrees(pitch[seg])))),
        "max_roll_deg": float(np.nanmax(np.abs(np.degrees(roll[seg])))), "max_slip": float(slip[seg].max()),
        "mean_slip": float(slip[seg].mean()), "max_sinkage_mm": float(1000 * z0[seg].max()),
        "min_stability_margin_deg": float(np.nanmin(margin[seg])), "static_tip_angles_deg": {"longitudinal": tip_lon, "lateral": tip_lat},
        "psr_time_h": float(dt_eff[in_psr].sum() / 3600), "max_continuous_psr_h": max_run / 3600,
        "shadow_energy_wh": float(shadow_used[end]) if has_batt else None, "usable_battery_wh": usable,
        "energy_model": "battery shadow budget" if has_batt else "not modelled (battery capacity unpublished)",
        "mobility_energy_wh": float((p_mob * dt_eff).sum() / 3600),
        "hotel_power_shadow_w": hotel, "hotel_power_source": hotel_src,
        "mean_sun_visibility": float(np.nanmean(sun_f[seg])) if sun_f is not None and np.isfinite(sun_f[seg]).any() else None,
        "mean_earth_visibility": float(np.nanmean(earth_f[seg])) if earth_f is not None and np.isfinite(earth_f[seg]).any() else None,
        "illumination_coverage_fraction": float(np.isfinite(sun_f[seg]).mean()) if sun_f is not None else 0.0,
        "expected_energy": {k: v for k, v in exp.items() if k != "soc"} if exp else None,
    }
    return {"summary": summary, "events": events, "frames": frames, "n_steps": n, "ds_m": ds,
            "end_index": end, "speed_cmd_mps": v_cmd}
