// Thin fetch wrapper with actionable error messages.
// Static mode (window.LUNARIS_STATIC = "/data"): used by the Vercel demo build. Requests are served from
// files precomputed by scripts/export_static.py; anything that needs the Python backend fails with a
// clear "run locally" message instead of pretending to work.
const STATIC = typeof window !== "undefined" ? window.LUNARIS_STATIC : null;
export const isStatic = !!STATIC;
const NEEDS_LOCAL = (what) =>
  new Error(`${what} needs the full LUNARIS backend. This web demo serves precomputed results; run LUNARIS locally for it (see README → Quick start).`);

async function req(method, url, body, isForm = false) {
  if (STATIC) return staticReq(method, url, body);
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opt.body = body;
    else { opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  }
  let r;
  try { r = await fetch(url, opt); }
  catch (e) { throw new Error(`Backend unreachable (${e.message}). Is the server running?`); }
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = j.detail ? (typeof j.detail === "string" ? j.detail : (j.detail.message || JSON.stringify(j.detail))) : JSON.stringify(j); } catch {}
    throw new Error(`${r.status}: ${detail}`);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.blob();
}

// ------------------------------------------------------------------ static demo routing
const cache = new Map();
async function sj(path) {
  if (!cache.has(path)) cache.set(path, fetch(`${STATIC}/${path}`).then((r) => { if (!r.ok) throw new Error(`missing ${path}`); return r.json(); }));
  return cache.get(path);
}
const sameCell = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];

async function staticReq(method, url, body) {
  const u = new URL(url, location.origin), p = u.pathname, m = p.match(/^\/api\/datasets\/([^/]+)\/?(.*)$/);
  if (p === "/api/health") return sj("health.json");
  if (p === "/api/config") return sj("config.json");
  if (p === "/api/datasets") return sj("datasets.json");
  if (p === "/api/craters") return sj("craters.json");
  if (p === "/api/rovers") return sj("rovers.json");
  if (p === "/api/alvik/ports") return { ports: [], error: "USB upload needs the local server. Use “Download main.py” and copy it with Arduino Lab for MicroPython." };
  if (p === "/api/upload") throw NEEDS_LOCAL("Uploading your own raster");
  if (!m) throw NEEDS_LOCAL(p);
  const [, id, rest] = m;
  if (rest === "load" || rest === "summary") return sj(`ds/${id}/summary.json`);
  if (rest === "mesh") return sj(`ds/${id}/mesh.json`);
  if (rest === "named_features") return sj(`ds/${id}/named_features.json`);
  if (rest === "suggest_points") return sj(`ds/${id}/suggest_points.json`);
  if (rest === "annotations" && method === "GET") return [];
  if (rest === "route") {
    const pts = await sj(`ds/${id}/suggest_points.json`), cfg = await sj("config.json");
    const w0 = cfg.navigation.weights, w = body?.weights || w0;
    const sameW = Object.keys(w0).every((k) => Math.abs((w[k] ?? w0[k]) - w0[k]) < 1e-9);
    if (sameCell(body?.start, pts.start) && sameCell(body?.goal, pts.goal) && sameW) return sj(`ds/${id}/route.json`);
    throw NEEDS_LOCAL("Planning a custom route (other start/goal or weights)");
  }
  if (rest === "simulate") {
    const b = body || {};
    if ((b.soil || "apollo_lrv_nominal") === "apollo_lrv_nominal" && (b.speed_mode || "nominal") === "nominal" && b.sunlit_load_w == null
        && ["viper", "perseverance", "pragyan"].includes(b.rover || "viper")) {
      try { return await sj(`ds/${id}/sim_${b.rover || "viper"}.json`); } catch { /* fall through */ }
    }
    throw NEEDS_LOCAL("This simulation configuration (precomputed: each rover on Apollo LRV soil, nominal speed)");
  }
  if (rest === "terrain_patch") return terrainPatch(id, u.searchParams);
  if (rest === "inspect") throw NEEDS_LOCAL("Per-cell inspection");
  if (rest === "priority") throw NEEDS_LOCAL("Re-weighting the ice-priority model");
  if (rest === "depth") throw NEEDS_LOCAL("Running Depth Anything V2");
  if (rest.startsWith("annotations")) throw NEEDS_LOCAL("Saving annotations");
  if (rest === "alvik/program") {
    const { buildProgram, micropythonSource } = await import("/static/alvik_gen.js");
    const sum = await sj(`ds/${id}/summary.json`);
    const route = window.__lunarisRoute, sim = window.__lunarisSim;
    if (!route?.found) throw new Error("Plan a route first (Navigate step).");
    return buildProgram(sum, route, sim, body || {}, micropythonSource);
  }
  if (rest === "alvik/upload") throw NEEDS_LOCAL("Uploading over USB");
  throw NEEDS_LOCAL(rest);
}

// Close-range terrain for the 3D simulation, interpolated in the browser from the exported 16-bit elevation grid
// (Catmull-Rom cubic spline through the same LOLA-derived posts, as the backend's visual 'cubic' mode).
const terrains = new Map();
async function terrainGrid(id) {
  if (!terrains.has(id)) terrains.set(id, (async () => {
    const meta = await sj(`ds/${id}/terrain.json`);
    const buf = await (await fetch(`${STATIC}/ds/${id}/terrain.bin`)).arrayBuffer();
    const n = meta.width * meta.height;
    return { meta, z: new Uint16Array(buf, 0, n), psr: new Uint8Array(buf, n * 2, n) };
  })());
  return terrains.get(id);
}
async function terrainPatch(id, q) {
  const { meta, z, psr } = await terrainGrid(id), W = meta.width, H = meta.height;
  const [a, , c, , e, f] = meta.transform;
  const x = +q.get("x"), y = +q.get("y"), size = Math.min(Math.max(+q.get("size") || 600, 50), 2000), sp = Math.min(Math.max(+q.get("spacing") || 4, 1), 20);
  const n = Math.floor(size / sp) + 1, out = new Float32Array(n * n), po = new Uint8Array(n * n);
  const Z = (r, cc) => { r = Math.max(0, Math.min(H - 1, r)); cc = Math.max(0, Math.min(W - 1, cc)); return meta.zmin + z[r * W + cc] * meta.zscale; };
  const cr = (p0, p1, p2, p3, t) => p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
  let zmin = Infinity;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const wx = x - size / 2 + j * sp, wy = y + size / 2 - i * sp;
    const fc = (wx - c) / a - 0.5, fr = (wy - f) / e - 0.5, c0 = Math.floor(fc), r0 = Math.floor(fr), tx = fc - c0, ty = fr - r0;
    const rows = [-1, 0, 1, 2].map((dr) => cr(Z(r0 + dr, c0 - 1), Z(r0 + dr, c0), Z(r0 + dr, c0 + 1), Z(r0 + dr, c0 + 2), tx));
    const v = cr(rows[0], rows[1], rows[2], rows[3], ty);
    out[i * n + j] = v; zmin = Math.min(zmin, v);
    const rr = Math.max(0, Math.min(H - 1, Math.round(fr))), qq = Math.max(0, Math.min(W - 1, Math.round(fc)));
    po[i * n + j] = psr[rr * W + qq] >= 128 ? 1 : 0;
  }
  const b64 = (u8) => { let s = ""; for (let i = 0; i < u8.length; i += 32768) s += String.fromCharCode.apply(null, u8.subarray(i, i + 32768)); return btoa(s); };
  return { n, spacing: sp, x0: x - size / 2, y0: y + size / 2, zmin, z: b64(new Uint8Array(out.buffer)), psr: b64(po), interp: "cubic",
           note: `Web demo: cubic spline through the ${meta.res_m} m analysis grid (finer detail needs the local install).` };
}

// Map a backend URL used directly by <img>/<a> elements to its static equivalent.
export function assetUrl(url) {
  if (!STATIC) return url;
  const u = new URL(url, location.origin), m = u.pathname.match(/^\/api\/datasets\/([^/]+)\/(.*)$/);
  if (!m) return url;
  const [, id, rest] = m;
  const lm = rest.match(/^layers\/(.+)\.png$/);
  if (lm) return `${STATIC}/ds/${id}/layers/${lm[1]}.webp`;
  if (rest === "thumb.png") return `${STATIC}/ds/${id}/thumb.webp`;
  if (rest === "report.pdf" || rest === "report.json") return `${STATIC}/ds/${id}/${rest}`;
  return url;
}

export const api = {
  get: (u) => req("GET", u),
  post: (u, b) => req("POST", u, b ?? {}),
  del: (u) => req("DELETE", u),
  form: (u, fd) => req("POST", u, fd, true),
};
