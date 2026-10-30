// Browser port of core/hardware/alvik.py (used by the static web demo). Same algorithm and output format:
// fit the route into the arena, RDP-simplify, emit rotate()/move() steps (+ = left turn), render main.py.
const ARENA_4FT = 1219.2;
const ALVIK = { mm_s: 100, deg_s: 100, overhead_s: 0.2 };

function rdpIdx(P, eps) {
  const keep = new Uint8Array(P.length); keep[0] = keep[P.length - 1] = 1;
  const stack = [[0, P.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop(); if (b <= a + 1) continue;
    const [px, py] = P[a], [qx, qy] = P[b], sx = qx - px, sy = qy - py, L = Math.hypot(sx, sy);
    let best = -1, bi = -1;
    for (let i = a + 1; i < b; i++) {
      const d = L === 0 ? Math.hypot(P[i][0] - px, P[i][1] - py) : Math.abs(sx * (P[i][1] - py) - sy * (P[i][0] - px)) / L;
      if (d > best) { best = d; bi = i; }
    }
    if (best > eps) { keep[bi] = 1; stack.push([a, bi], [bi, b]); }
  }
  const out = []; keep.forEach((k, i) => k && out.push(i)); return out;
}

function maxDev(full, pts) {
  let worst = 0;
  for (const f of full) {
    let best = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[i + 1], abx = bx - ax, aby = by - ay, L2 = abx * abx + aby * aby || 1e-12;
      const t = Math.max(0, Math.min(1, ((f[0] - ax) * abx + (f[1] - ay) * aby) / L2));
      best = Math.min(best, Math.hypot(f[0] - (ax + t * abx), f[1] - (ay + t * aby)));
    }
    worst = Math.max(worst, best);
  }
  return worst;
}

export function buildProgram(summary, route, sim, opt, render) {
  const arena = opt.arena_mm ?? ARENA_4FT, margin = opt.margin_mm ?? 120, minSeg = opt.min_segment_mm ?? 15, maxSeg = opt.max_segments ?? 80;
  const [a, b, c, d, e, f] = summary.grid.transform;
  let xy = route.cells.map(([r, q]) => [c + (q + 0.5) * a + (r + 0.5) * b, f + (q + 0.5) * d + (r + 0.5) * e]);
  const lo = [Math.min(...xy.map((p) => p[0])), Math.min(...xy.map((p) => p[1]))], hi = [Math.max(...xy.map((p) => p[0])), Math.max(...xy.map((p) => p[1]))];
  let s = [0]; for (let i = 1; i < xy.length; i++) s.push(s[i - 1] + Math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]));
  // optional truncation at the simulated failure (scale stays that of the full route)
  let tags = null, stopAt = null, simDesc = "not simulated";
  if (sim) {
    const F = sim.frames, slipOps = sim.rover.params.max_slip_ops.value;
    tags = F.s.map((_, i) => (F.in_psr[i] ? "psr" : F.slip[i] > slipOps ? "slip" : "go"));
    if (sim.summary.failure) { tags[tags.length - 1] = "fail"; if (opt.stop_at_failure !== false) stopAt = sim.summary.distance_m; }
    simDesc = `${sim.rover.name} on ${sim.soil.name}: ${sim.summary.verdict}`;
  }
  if (stopAt != null && stopAt < s[s.length - 1]) {
    const k = s.findIndex((v) => v >= stopAt), t = (stopAt - s[k - 1]) / Math.max(s[k] - s[k - 1], 1e-9);
    const end = [xy[k - 1][0] + t * (xy[k][0] - xy[k - 1][0]), xy[k - 1][1] + t * (xy[k][1] - xy[k - 1][1])];
    xy = [...xy.slice(0, k), end]; s = [...s.slice(0, k), stopAt];
  }
  const usable = arena - 2 * margin, span = Math.max(hi[0] - lo[0], hi[1] - lo[1]), k = usable / span;
  const off = [(usable - (hi[0] - lo[0]) * k) / 2, (usable - (hi[1] - lo[1]) * k) / 2];
  const A = xy.map(([x, y]) => [(x - lo[0]) * k + margin + off[0], (y - lo[1]) * k + margin + off[1]]);
  let eps = opt.tolerance_mm ?? 6, idx;
  for (let it = 0; it < 12; it++) {
    idx = rdpIdx(A, eps);
    const merged = [idx[0]];
    for (let i = 1; i < idx.length; i++) {
      const pm = A[merged[merged.length - 1]], pi = A[idx[i]];
      if (Math.hypot(pi[0] - pm[0], pi[1] - pm[1]) >= minSeg || i === idx.length - 1) merged.push(idx[i]);
    }
    idx = merged; if (idx.length - 1 <= maxSeg) break; eps *= 1.5;
  }
  const pts = idx.map((i) => A[i]), sk = idx.map((i) => s[i]);
  const h0 = (Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) * 180) / Math.PI;
  const steps = []; let cur = h0;
  for (let i = 0; i < pts.length - 1; i++) {
    const dx = pts[i + 1][0] - pts[i][0], dy = pts[i + 1][1] - pts[i][1], h = (Math.atan2(dy, dx) * 180) / Math.PI;
    const turn = i ? Math.round(((((h - cur + 180) % 360) + 360) % 360 - 180) * 10) / 10 : 0;
    let led = "go";
    if (tags) {
      const set = new Set(); sim.frames.s.forEach((v, j) => { if (v >= sk[i] && v <= sk[i + 1]) set.add(tags[j]); });
      led = set.has("fail") ? "fail" : set.has("slip") ? "slip" : set.has("psr") ? "psr" : "go";
    }
    steps.push({ turn_deg: turn, move_mm: Math.round(Math.hypot(dx, dy) * 10) / 10, led, world_from_m: sk[i], world_to_m: sk[i + 1] });
    cur = i ? cur + turn : h;
  }
  const nTurns = steps.filter((x) => Math.abs(x.turn_deg) >= 0.5).length;
  const len = steps.reduce((t, x) => t + x.move_mm, 0);
  const prog = {
    arena_mm: arena, margin_mm: margin, scale_mm_per_m: k, scale_ratio: `1 : ${Math.round(1000 / k).toLocaleString("en-US")}`,
    world_length_m: s[s.length - 1], arena_length_mm: len, start_mm: pts[0], start_heading_deg_from_up_cw: Math.round((((90 - h0) % 360) + 360) % 360 * 10) / 10,
    goal_mm: pts[pts.length - 1], waypoints_mm: pts, full_path_mm: A.filter((_, i) => i % Math.max(1, Math.floor(A.length / 600)) === 0),
    steps, n_steps: steps.length, max_deviation_mm: maxDev(A, pts), tolerance_mm: eps,
    inside_safe_area: pts.every(([x, y]) => x >= margin - 1e-6 && y >= margin - 1e-6 && x <= arena - margin + 1e-6 && y <= arena - margin + 1e-6),
    estimated_run_s: Math.round((len / ALVIK.mm_s + steps.reduce((t, x) => t + Math.abs(x.turn_deg), 0) / ALVIK.deg_s + ALVIK.overhead_s * (steps.length + nTurns)) * 10) / 10,
    truncated_at_failure_m: stopAt,
  };
  const source = render(prog, { dataset_title: summary.dataset.title, sim: simDesc });
  const download = URL.createObjectURL(new Blob([source], { type: "text/x-python" }));
  return { program: prog, source, sim: simDesc, download };
}

export function micropythonSource(p, meta) {
  const steps = p.steps.map((s) => `    (${s.turn_deg.toFixed(1)}, ${s.move_mm.toFixed(1)}, '${s.led}')`).join(",\n");
  const now = new Date().toISOString().slice(0, 16).replace("T", " ");
  return `# LUNARIS -> Arduino Alvik mission program (auto-generated, do not edit by hand)
# Generated : ${now} UTC
# Dataset   : ${meta.dataset_title}
# Route     : ${(p.world_length_m / 1000).toFixed(2)} km on the Moon -> ${(p.arena_length_mm / 10).toFixed(1)} cm in the arena (scale ${p.scale_ratio})
# Rover sim : ${meta.sim}
# Arena     : ${p.arena_mm.toFixed(0)} mm square (4 ft x 4 ft), safety margin ${p.margin_mm.toFixed(0)} mm
#
# PLACE THE ALVIK:
#   centre of the robot at x = ${(p.start_mm[0] / 10).toFixed(1)} cm from the LEFT edge, y = ${(p.start_mm[1] / 10).toFixed(1)} cm from the NEAR edge,
#   facing ${p.start_heading_deg_from_up_cw.toFixed(0)} deg clockwise from straight 'up' (away from you).
# Press OK (check) to start. Press CANCEL (x) at any time to stop; press OK again to restart.
# LEDs: green = normal, blue = inside a permanently shadowed region, yellow = high wheel slip in the
#       simulation, red = segment where the simulated rover failed. This is a geometric replay only.

from arduino_alvik import ArduinoAlvik
from time import sleep_ms

# (turn_deg [+ = left/CCW], move_mm, led) per step
STEPS = [
${steps}
]

COLORS = {'go': (0, 1, 0), 'psr': (0, 0, 1), 'slip': (1, 1, 0), 'fail': (1, 0, 0)}


def leds(alvik, rgb):
    alvik.left_led.set_color(*rgb)
    alvik.right_led.set_color(*rgb)


def wait_target(alvik):
    # Non-blocking wait so CANCEL can abort mid-motion (pattern from the official examples).
    while not alvik.is_target_reached():
        if alvik.get_touch_cancel():
            alvik.brake()
            return False
        sleep_ms(20)
    return True


def run(alvik):
    alvik.reset_pose(0, 0, 0)
    for i in range(len(STEPS)):
        turn, dist, tag = STEPS[i]
        leds(alvik, COLORS.get(tag, (0, 1, 0)))
        print('step %d/%d  turn %.1f deg  move %.1f mm  [%s]' % (i + 1, len(STEPS), turn, dist, tag))
        if abs(turn) >= 0.5:
            alvik.rotate(turn, 'deg', blocking=False)
            if not wait_target(alvik):
                return False
        alvik.move(dist, 'mm', blocking=False)
        if not wait_target(alvik):
            return False
    x, y, theta = alvik.get_pose()
    print('finished: odometry pose x=%s cm y=%s cm theta=%s deg' % (x, y, theta))
    return True


alvik = ArduinoAlvik()
alvik.begin()
try:
    while True:
        # Ready: blink green until OK is pressed.
        on = True
        while not alvik.get_touch_ok():
            leds(alvik, (0, 1, 0) if on else (0, 0, 0))
            on = not on
            sleep_ms(300)
        while alvik.get_touch_ok():
            sleep_ms(50)
        sleep_ms(700)  # hands off
        ok = run(alvik)
        alvik.brake()
        for _ in range(6):  # finished = white flashes, aborted = red flashes
            leds(alvik, (1, 1, 1) if ok else (1, 0, 0))
            sleep_ms(200)
            leds(alvik, (0, 0, 0))
            sleep_ms(200)
        while alvik.get_touch_cancel():
            sleep_ms(50)
except KeyboardInterrupt:
    pass
finally:
    alvik.stop()
`;
}
