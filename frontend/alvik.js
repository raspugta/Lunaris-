// Arduino Alvik mission uploader: arena preview, program generation, USB upload via the backend (mpremote).
import { api } from "/static/api.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const TAGC = { go: "#5ef0a8", psr: "#7fb8ff", slip: "#ffd67a", fail: "#ff6b6b" };
const ALVIK = { mm_s: 100, deg_s: 100, overhead_s: 0.2, body_mm: 100 }; // official library constants
let st = { ds: null, prog: null, anim: null };

export function initAlvik(getDs) {
  const sync = (id, unit) => { $(`#${id}V`).textContent = `${$(`#${id}`).value} ${unit}`; };
  [["alMargin", "mm"], ["alTol", "mm"], ["alMin", "mm"]].forEach(([id, u]) => $(`#${id}`).addEventListener("input", () => sync(id, u)));
  $("#alvikBtn").addEventListener("click", () => open(getDs()));
  $("#alvikClose").addEventListener("click", close);
  $("#alvikModal").addEventListener("click", (e) => { if (e.target.id === "alvikModal") close(); });
  $("#alGen").addEventListener("click", generate);
  $("#alRefresh").addEventListener("click", ports);
  $("#alAll").addEventListener("change", ports);
  $("#alUpload").addEventListener("click", upload);
  $("#arenaPlay").addEventListener("click", preview);
}

function close() { $("#alvikModal").classList.add("hidden"); cancelAnimationFrame(st.anim?.raf); st.anim = null; }
async function open(ds) {
  st.ds = ds; $("#alvikModal").classList.remove("hidden");
  drawArena(null);
  await Promise.all([generate(), ports()]);
}

async function generate() {
  $("#alMsg").textContent = ""; $("#alMsg").className = "msg";
  try {
    const r = await api.post(`/api/datasets/${st.ds}/alvik/program`, {
      margin_mm: +$("#alMargin").value, tolerance_mm: +$("#alTol").value, min_segment_mm: +$("#alMin").value,
      stop_at_failure: $("#alStopFail").checked,
    });
    st.prog = r.program; st.src = r.source;
    $("#alCode").textContent = r.source;
    $("#alDownload").href = r.download; $("#alDlBtn").disabled = false;
    $("#alUpload").disabled = !$("#alPort").value;
    stats(r);
    drawArena(st.prog);
  } catch (e) { $("#alStats").innerHTML = `<div class="card warn">${esc(e.message)}</div>`; }
}

function stats(r) {
  const p = r.program, sx = p.start_mm[0] / 10, sy = p.start_mm[1] / 10;
  $("#alStats").innerHTML = `
    <div class="card"><b>Program</b> <span class="okpill ${p.inside_safe_area ? "good" : "bad"}">${p.inside_safe_area ? "inside safe area" : "OUTSIDE SAFE AREA"}</span>
      <table class="kv">
        <tr><td>Scale</td><td class="mono">${esc(p.scale_ratio)}</td></tr>
        <tr><td>Lunar route → arena</td><td class="mono">${(p.world_length_m / 1000).toFixed(2)} km → ${(p.arena_length_mm / 10).toFixed(1)} cm${p.truncated_at_failure_m ? " (stops at sim failure)" : ""}</td></tr>
        <tr><td>Steps</td><td class="mono">${p.n_steps} (rotate + move)</td></tr>
        <tr><td>Max deviation from scaled route</td><td class="mono">${p.max_deviation_mm.toFixed(1)} mm</td></tr>
        <tr><td>Estimated run time</td><td class="mono">${p.estimated_run_s.toFixed(0)} s</td></tr>
        <tr><td>Simulation</td><td>${esc(r.sim)}</td></tr>
      </table></div>
    <div class="card note"><b>Place the Alvik</b>
      <div class="place"><span>From LEFT edge</span><b>${sx.toFixed(1)} cm</b><span>From NEAR edge</span><b>${sy.toFixed(1)} cm</b>
      <span>Facing</span><b>${p.start_heading_deg_from_up_cw.toFixed(0)}° clockwise from "up"</b></div>
      <p class="ld">"Up" = straight away from you across the grid. Then press <b>OK (✓)</b> on the Alvik. <b>✗</b> stops it at any time.
      Expect a few cm of drift by the end: the Alvik navigates by wheel odometry only.</p></div>`;
}

function drawArena(p, robot = null) {
  const cv = $("#arena"), g = cv.getContext("2d"), W = cv.width, A = p ? p.arena_mm : 1219.2, s = W / A;
  const X = (x) => x * s, Y = (y) => W - y * s;
  g.clearRect(0, 0, W, W);
  g.fillStyle = "#0b0d11"; g.fillRect(0, 0, W, W);
  // 1-ft grid (304.8 mm) + 6-in minor lines
  for (let mm = 0; mm <= A + 1; mm += 152.4) {
    const major = Math.abs((mm / 304.8) - Math.round(mm / 304.8)) < 1e-6;
    g.strokeStyle = major ? "rgba(236,230,218,.28)" : "rgba(236,230,218,.08)"; g.lineWidth = major ? 1.5 : 1;
    g.beginPath(); g.moveTo(X(mm), 0); g.lineTo(X(mm), W); g.stroke(); g.beginPath(); g.moveTo(0, Y(mm)); g.lineTo(W, Y(mm)); g.stroke();
  }
  g.fillStyle = "#8d8a83"; g.font = "11px 'JetBrains Mono'";
  for (let f = 1; f < 4; f++) { g.fillText(`${f} ft`, X(f * 304.8) + 4, W - 6); g.fillText(`${f} ft`, 4, Y(f * 304.8) - 4); }
  g.fillText("NEAR EDGE (you) ↓", W / 2 - 60, W - 6); g.save(); g.translate(14, W / 2 + 40); g.rotate(-Math.PI / 2); g.fillText("LEFT EDGE", 0, 0); g.restore();
  if (!p) { g.fillStyle = "#8d8a83"; g.font = "14px 'Space Grotesk'"; g.fillText("Generating…", W / 2 - 40, W / 2); return; }
  // safety margin
  g.strokeStyle = "rgba(255,90,31,.5)"; g.setLineDash([6, 6]);
  g.strokeRect(X(p.margin_mm), Y(A - p.margin_mm), (A - 2 * p.margin_mm) * s, (A - 2 * p.margin_mm) * s); g.setLineDash([]);
  // full-resolution scaled route
  g.strokeStyle = "rgba(236,230,218,.35)"; g.lineWidth = 1.2; g.beginPath();
  p.full_path_mm.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.stroke();
  // program steps coloured by simulated condition
  const w = p.waypoints_mm;
  for (let i = 0; i < p.steps.length; i++) {
    g.strokeStyle = TAGC[p.steps[i].led] || TAGC.go; g.lineWidth = 4; g.lineCap = "round";
    g.beginPath(); g.moveTo(X(w[i][0]), Y(w[i][1])); g.lineTo(X(w[i + 1][0]), Y(w[i + 1][1])); g.stroke();
  }
  w.forEach(([x, y], i) => { if (i && i < w.length - 1) { g.fillStyle = "#06070a"; g.strokeStyle = "#ece6da"; g.lineWidth = 1.5; g.beginPath(); g.arc(X(x), Y(y), 3.5, 0, 6.283); g.fill(); g.stroke(); } });
  // goal
  const [gx, gy] = p.goal_mm; g.fillStyle = "#ffd400"; g.font = "bold 18px sans-serif"; g.fillText("★", X(gx) - 7, Y(gy) + 6);
  // robot (start or animated pose)
  const pose = robot || { x: p.start_mm[0], y: p.start_mm[1], h: (90 - p.start_heading_deg_from_up_cw) };
  const rx = X(pose.x), ry = Y(pose.y), r = (ALVIK.body_mm / 2) * s, a = (-pose.h * Math.PI) / 180;
  g.save(); g.translate(rx, ry); g.rotate(a);
  g.fillStyle = "rgba(0,151,157,.85)"; g.strokeStyle = "#fff"; g.lineWidth = 1.5;
  g.beginPath(); g.roundRect(-r, -r, 2 * r, 2 * r, r * 0.35); g.fill(); g.stroke();
  g.fillStyle = "#fff"; g.beginPath(); g.moveTo(r + 6, 0); g.lineTo(r - 4, -6); g.lineTo(r - 4, 6); g.closePath(); g.fill();
  g.restore();
  if (!robot) { g.fillStyle = "#e9ffef"; g.font = "600 12px 'Space Grotesk'"; g.fillText("START", rx + r + 6, ry - r - 2); }
}

function preview() {
  if (!st.prog) return;
  cancelAnimationFrame(st.anim?.raf);
  const p = st.prog, w = p.waypoints_mm;
  // timeline of primitive actions at the official Alvik speeds
  const acts = []; let h = 90 - p.start_heading_deg_from_up_cw;
  p.steps.forEach((s, i) => {
    if (Math.abs(s.turn_deg) >= 0.5) acts.push({ kind: "turn", from: h, to: h + s.turn_deg, at: w[i], dur: ALVIK.overhead_s + Math.abs(s.turn_deg) / ALVIK.deg_s });
    h += s.turn_deg;
    acts.push({ kind: "move", from: w[i], to: w[i + 1], h, dur: ALVIK.overhead_s + s.move_mm / ALVIK.mm_s });
  });
  const total = acts.reduce((a, b) => a + b.dur, 0), speed = 4; // 4x preview
  const t0 = performance.now();
  const step = (now) => {
    let t = ((now - t0) / 1000) * speed, k = 0;
    if (t >= total) t = total;
    let acc = 0; while (k < acts.length - 1 && acc + acts[k].dur < t) { acc += acts[k].dur; k++; }
    const a = acts[k], u = Math.min(1, Math.max(0, (t - acc - ALVIK.overhead_s) / Math.max(a.dur - ALVIK.overhead_s, 1e-6)));
    const pose = a.kind === "turn" ? { x: a.at[0], y: a.at[1], h: a.from + (a.to - a.from) * u }
      : { x: a.from[0] + (a.to[0] - a.from[0]) * u, y: a.from[1] + (a.to[1] - a.from[1]) * u, h: a.h };
    drawArena(p, pose);
    $("#arenaClock").textContent = `${t.toFixed(1)} / ${total.toFixed(1)} s (real time, shown ×${speed})`;
    if (t < total) st.anim = { raf: requestAnimationFrame(step) };
  };
  st.anim = { raf: requestAnimationFrame(step) };
}

async function ports() {
  try {
    const r = await api.get(`/api/alvik/ports${$("#alAll").checked ? "?all=true" : ""}`);
    const sel = $("#alPort");
    sel.innerHTML = r.ports.length ? r.ports.map((p) => `<option value="${esc(p.device)}">${esc(p.device)} — ${esc(p.description)}${p.likely_alvik ? " (Arduino)" : ""}</option>`).join("")
      : `<option value="">— no USB device found: connect the Alvik and switch it ON —</option>`;
    $("#alUpload").disabled = !sel.value || !st.prog;
    if (r.error) { $("#alMsg").textContent = r.error; $("#alMsg").className = "msg err"; }
  } catch (e) { $("#alMsg").textContent = e.message; $("#alMsg").className = "msg err"; }
}

async function upload() {
  const port = $("#alPort").value; if (!port) return;
  const b = $("#alUpload"); b.disabled = true; $("#alMsg").className = "msg"; $("#alMsg").textContent = `Uploading to ${port}…`;
  try {
    const r = await api.post(`/api/datasets/${st.ds}/alvik/upload`, { port, backup: true, run_now: true });
    $("#alMsg").textContent = `✓ Uploaded as main.py${r.backup ? ` (previous main.py backed up to ${r.backup})` : ""}.\n${r.next}`;
  } catch (e) { $("#alMsg").className = "msg err"; $("#alMsg").textContent = e.message; }
  finally { b.disabled = false; }
}
