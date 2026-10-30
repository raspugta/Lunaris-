// Lunar PSR Intelligence — dashboard controller.
import { api, assetUrl, isStatic } from "/static/api.js";
import { Map2D } from "/static/map2d.js";
import { Viewer3D } from "/static/viewer3d.js";
import { SimView, drawChart } from "/static/sim.js";
import { initAlvik } from "/static/alvik.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : typeof v === "number" ? (Math.abs(v) >= 1e4 ? v.toFixed(0) : v.toFixed(d)) : esc(v));
const pct = (v) => (v === null || v === undefined ? "—" : `${(100 * v).toFixed(1)}%`);
const STATUS_TXT = { measured: "measured", derived: "derived", model_estimate: "model estimate", illustrative: "illustrative", synthetic: "synthetic" };
const chip = (s) => `<span class="chip ${s}">${STATUS_TXT[s] || s}</span>`;

const S = { ds: null, sum: null, layer: null, mode: "inspect", start: null, goal: null, route: null, cfg: null, job: null };
const map = new Map2D($("#map"), { onHover, onClick });
const v3 = new Viewer3D($("#view3d"));

// ---------------------------------------------------------------- boot
async function boot() {
  if (isStatic) banner("info", "Web demo: results were precomputed by the LUNARIS pipeline on real NASA data. Custom routes, re-weighting, the depth model and USB upload need a local install (see README).");
  try {
    const [h, cfg] = await Promise.all([api.get("/api/health"), api.get("/api/config")]);
    S.cfg = cfg;
    const e = h.environment;
    $("#envInfo").textContent = `v${h.version} · torch ${e.torch || "n/a"} · ${e.cuda ? `CUDA ${e.vram_mb} MB` : "CPU"} · ${h.model_id.split("/").pop()}`;
    $("#exag").value = cfg.viewer.default_vertical_exaggeration; $("#exagVal").textContent = cfg.viewer.default_vertical_exaggeration;
    v3.exag = cfg.viewer.default_vertical_exaggeration;
  } catch (err) { banner("error", err.message); }
  await refreshDatasets();
  loadCraters();
}
// ---------------------------------------------------------------- crater library
let CRATERS = [];
async function loadCraters() {
  try { CRATERS = (await api.get("/api/craters")).craters; } catch { CRATERS = []; }
  const card = (c) => {
    const lp = c.layers_present || {};
    const badges = [`${c.diameter_km.toFixed(0)} km`, c.psr_fraction != null ? `PSR ${(100 * c.psr_fraction).toFixed(0)}% of window` : null, lp.thermal_max ? "Diviner" : null, lp.image ? "LROC NAC" : null, lp.illumination_avg ? "Illumination" : null].filter(Boolean);
    return `<button class="ccard ${c.available ? "" : "off"} ${S.ds === c.id ? "cur" : ""}" data-id="${esc(c.id)}" ${c.available ? "" : `title="Not downloaded. Run: ${esc(c.fetch)}"`}>
      ${c.thumb ? `<img class="img" loading="lazy" src="${assetUrl(c.thumb)}" alt="LOLA shaded relief of ${esc(c.name)}">` : `<div class="img"></div>`}
      <div class="shade"></div>
      <div class="badges">${badges.map((b, i) => `<span class="b ${i === 0 ? "hot" : ""}">${esc(b)}</span>`).join("")}</div>
      <div class="meta"><div class="nm">${esc(c.name)}</div>
        <div class="sub">${Math.abs(c.lat).toFixed(2)}°S · ${c.lon.toFixed(1)}°E${c.available ? "" : " · not downloaded"}</div>
        <div class="note">${esc(c.note || "")}</div></div></button>`;
  };
  const html = `<div class="cgrid">${CRATERS.map(card).join("")}</div>`;
  $("#craterGrid").innerHTML = html;
  $("#craterInline").innerHTML = CRATERS.length ? `<h4>Crater library</h4>${html}` : "";
  document.querySelectorAll(".ccard").forEach((el) => el.addEventListener("click", () => {
    const c = CRATERS.find((x) => x.id === el.dataset.id);
    if (!c?.available) return;
    $("#craterModal").classList.add("hidden");
    $("#datasetSelect").value = c.id; loadDataset(c.id);
  }));
}
$("#craterBtn").addEventListener("click", () => { $("#craterModal").classList.remove("hidden"); loadCraters(); });
$("#craterClose").addEventListener("click", () => $("#craterModal").classList.add("hidden"));
$("#craterModal").addEventListener("click", (e) => { if (e.target.id === "craterModal") $("#craterModal").classList.add("hidden"); });
window.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#craterModal").classList.add("hidden"); });

async function refreshDatasets(selectId) {
  const list = await api.get("/api/datasets");
  const sel = $("#datasetSelect");
  sel.innerHTML = list.map((d) => `<option value="${esc(d.id)}" ${d.available ? "" : "disabled"}>${d.kind === "synthetic" ? "[SYNTHETIC] " : d.kind === "upload" ? "[UPLOAD] " : "[REAL] "}${esc(d.title)}${d.available ? "" : " — not downloaded"}</option>`).join("");
  const firstReal = list.find((d) => d.available && d.kind === "real") || list.find((d) => d.available);
  sel.value = selectId || (firstReal && firstReal.id) || "";
  S.registry = list;
}

// ---------------------------------------------------------------- UI helpers
function banner(kind, html) { const b = $("#banner"); b.className = `banner ${kind}`; b.innerHTML = html; if (!html) b.classList.add("hidden"); }
function loading(on, text = "Working…") { $("#loading").classList.toggle("hidden", !on); $("#loadingText").textContent = text; }
document.querySelectorAll(".steps button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(t) {
  document.querySelectorAll(".steps button").forEach((x) => x.classList.toggle("active", x.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.id === `tab-${t}`));
}
document.querySelectorAll("#modeSeg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
function setMode(m) { S.mode = m; document.querySelectorAll("#modeSeg button").forEach((x) => x.classList.toggle("active", x.dataset.mode === m)); }
$("#opacity").addEventListener("input", (e) => { map.opacity = e.target.value / 100; map.draw(); update3DTexture(); });
$("#showFeatures").addEventListener("change", (e) => { map.showFeatures = e.target.checked; map.draw(); });
$("#showNames").addEventListener("change", (e) => { map.showNames = e.target.checked; map.draw(); });
$("#fitBtn").addEventListener("click", () => { map._userMoved = false; map.fit(); });
$("#exag").addEventListener("input", (e) => { $("#exagVal").textContent = e.target.value; v3.setExaggeration(+e.target.value); });
$("#reset3d").addEventListener("click", () => v3.resetView());
$("#layerSelect").addEventListener("change", (e) => selectLayer(e.target.value));
$("#loadBtn").addEventListener("click", () => loadDataset($("#datasetSelect").value));

// ---------------------------------------------------------------- dataset load
async function loadDataset(id) {
  if (!id) return;
  loading(true, "Ingesting, validating and analysing…");
  $("#loadBtn").disabled = true;
  try {
    const sum = await api.post(`/api/datasets/${id}/load`);
    S.ds = id; S.sum = sum; S.start = S.goal = null; S.route = null;
    const kind = sum.dataset.kind;
    const badge = $("#kindBadge"); badge.className = `badge ${kind}`; badge.textContent = kind === "real" ? "REAL DATA" : kind === "synthetic" ? "SYNTHETIC DEMO DATA" : "UPLOAD · " + (Object.values(sum.dataset.layers)[0]?.source_verified ? "declared source" : "SOURCE UNVERIFIED");
    if (kind === "synthetic") banner("synthetic", "SYNTHETIC DEMO DATA — procedurally generated terrain. Nothing shown here describes the Moon.");
    else if (kind === "upload") banner("info", "User upload — provenance is as declared by the uploader and has not been verified.");
    else banner("", "");
    map.setGrid(sum.grid.width, sum.grid.height);
    map.features = sum.features || []; map.annotations = sum.annotations || [];
    map.named = []; api.get(`/api/datasets/${id}/named_features`).then((r) => { map.named = r.features; map.draw(); }).catch(() => {});
    document.querySelectorAll(".ccard").forEach((c) => c.classList.toggle("cur", c.dataset.id === id));
    map.route = null; map.alts = {}; map.start = map.goal = null;
    const layers = Object.entries(sum.layers).filter(([, m]) => m.available);
    $("#layerSelect").innerHTML = layers.map(([k, m]) => `<option value="${k}">${esc(m.title)}</option>`).join("");
    await map.setBase(sum.layers.hillshade?.available ? layerUrl("hillshade") : null);
    const def = ["hazard", "elevation", "image", "depth_relative"].find((k) => sum.layers[k]?.available) || layers[0][0];
    $("#layerSelect").value = def;
    try {
      const mesh = await api.get(`/api/datasets/${id}/mesh`);
      v3.setTerrain(mesh, sum.grid.width, sum.grid.height);
      $("#label3d").textContent = mesh.label + (mesh.calibrated ? ` · ${mesh.cell_m.toFixed(0)} m mesh spacing` : " · display-scaled");
      $("#title3d").textContent = mesh.calibrated ? "3D terrain — LOLA-calibrated heights" : "3D — relative-depth visualisation";
    } catch (err) { $("#label3d").textContent = `3D unavailable: ${err.message}`; }
    await selectLayer(def);
    S.sim = null; exitSim();
    renderInspect(); renderLayers(); renderPriority(); renderNav(); renderReport(); renderSimPanel();
    $("#emptyState").classList.add("hidden"); $("#inspectBody").classList.remove("hidden");
  } catch (err) { banner("error", `Load failed — ${esc(err.message)}`); }
  finally { loading(false); $("#loadBtn").disabled = false; }
}
const layerUrl = (k) => assetUrl(`/api/datasets/${S.ds}/layers/${k}.png`) + (isStatic ? "" : `?t=${Date.now()}`);

async function selectLayer(k) {
  S.layer = k; $("#layerSelect").value = k;
  const m = S.sum.layers[k];
  await map.setLayer(k === "hillshade" ? null : layerUrl(k));
  renderLegend(m);
  $("#layerStatus").innerHTML = `${chip(m.status)} <b>${esc(m.title)}</b> · ${esc(m.units)}${m.resolution_m ? ` · ${m.resolution_m.toFixed(0)} m/px` : " · uncalibrated grid"}`;
  document.querySelectorAll(".layer-item").forEach((el) => el.classList.toggle("sel", el.dataset.k === k));
  update3DTexture();
}
function update3DTexture() {
  if (!S.sum) return;
  const urls = [S.sum.layers.hillshade?.available ? assetUrl(`/api/datasets/${S.ds}/layers/hillshade.png`) : null,
    S.layer !== "hillshade" ? assetUrl(`/api/datasets/${S.ds}/layers/${S.layer}.png`) + (isStatic ? "" : `?t=${S.layerVersion || 0}`) : null];
  v3.setTexture(urls, map.opacity).catch(() => {});
}
function renderLegend(m) {
  const L = m.legend; let h = `<div><b>${esc(m.title)}</b></div>`;
  if (!L) { $("#legend").innerHTML = h; return; }
  if (L.type === "categorical") h += L.items.map((i) => `<div class="cat"><span class="sw" style="background:${i.color}"></span>${esc(i.label)}</div>`).join("");
  else h += `<div class="ramp" style="background:linear-gradient(90deg,${L.stops.join(",")})"></div><div class="ends"><span>${fmt(L.min, 1)}</span><span>${esc(L.units)}</span><span>${fmt(L.max, 1)}</span></div>`;
  h += `<div class="ends"><span><span class="sw" style="display:inline-block;background:#282a30"></span> no data / unknown</span></div>`;
  $("#legend").innerHTML = h;
}

// ---------------------------------------------------------------- map interaction
let hoverT = 0;
function onHover(c) {
  if (!c || !S.sum) { $("#readout").textContent = "—"; return; }
  const now = performance.now(); if (now - hoverT < 60) return; hoverT = now;
  const g = S.sum.grid; let txt = `row ${c.row} col ${c.col}`;
  if (g.polar_stereo_south) {
    const [a, b, cc, d, e, f] = g.transform, x = cc + (c.col + 0.5) * a, y = f + (c.row + 0.5) * e;
    const rho = Math.hypot(x, y), lat = (2 * Math.atan(rho / (2 * 1737400)) - Math.PI / 2) * 180 / Math.PI;
    const lon = ((Math.atan2(x, y) * 180 / Math.PI) + 360) % 360;
    txt += ` · ${lat.toFixed(4)}°, ${lon.toFixed(2)}°E · x ${(x / 1000).toFixed(2)} y ${(y / 1000).toFixed(2)} km`;
  }
  $("#readout").textContent = txt;
}
async function onClick(c) {
  if (!S.sum) return;
  if (S.mode === "start") { S.start = [c.row, c.col]; map.start = S.start; map.draw(); renderNav(); setMode("goal"); return; }
  if (S.mode === "goal") { S.goal = [c.row, c.col]; map.goal = S.goal; map.draw(); renderNav(); return; }
  if (S.mode === "annotate") {
    const label = prompt("Annotation label (e.g. 'boulder field?', 'check in NAC'):"); if (!label) return;
    const note = prompt("Optional note:") || "";
    const a = await api.post(`/api/datasets/${S.ds}/annotations`, { row: c.row, col: c.col, label, note });
    map.annotations.push(a); map.draw(); renderReport(); return;
  }
  map.pick = [c.row, c.col]; map.draw();
  try { showInspector(await api.get(`/api/datasets/${S.ds}/inspect?row=${c.row}&col=${c.col}`)); }
  catch (err) { banner("error", esc(err.message)); }
}
function showInspector(d) {
  const p = d.priority;
  const rows = Object.entries(d.values).map(([k, v]) => `<tr><td>${esc(v.title)} ${v.status ? chip(v.status) : ""}</td><td class="mono">${v.label ? esc(v.label) : fmt(v.value, 3)} ${v.value !== null && !v.label ? esc(v.units) : ""}</td></tr>`).join("");
  const ev = p.evidence.map((e) => `<tr><td>${esc(e.description)}</td><td>${fmt(e.raw_value, 2)}</td><td>${fmt(e.subscore, 2)}</td><td>${fmt(e.weight, 2)}</td><td>${e.contribution === null ? `<span class="miss">${esc(e.state)}</span>` : fmt(e.contribution, 3)}</td></tr>`).join("");
  $("#inspector").innerHTML = `<button class="ghost close" id="closeInsp">✕</button>
    <h3>Cell ${d.row}, ${d.col}</h3>
    <div class="mono">${d.lat_lon ? `${d.lat_lon[0].toFixed(5)}°, ${d.lat_lon[1].toFixed(3)}°E` : "no geographic coordinates (uncalibrated grid)"}${d.x_m !== null ? ` · x ${fmt(d.x_m, 0)} m, y ${fmt(d.y_m, 0)} m` : ""}</div>
    <h4>Investigation-priority evidence</h4>
    <div class="card note"><b>${esc(p.class)}</b> · score ${fmt(p.score, 3)} · coverage ${pct(p.evidence_coverage)} · sensitivity σ ${fmt(p.sensitivity_std, 3)}<br><small>${esc(p.caveat)}</small></div>
    <table class="grid"><tr><th>Evidence</th><th>raw</th><th>sub</th><th>w</th><th>contrib.</th></tr>${ev}</table>
    <h4>All layers at this cell</h4><table class="kv">${rows}</table>`;
  $("#inspector").classList.remove("hidden");
  $("#closeInsp").onclick = () => { $("#inspector").classList.add("hidden"); map.pick = null; map.draw(); };
}

// ---------------------------------------------------------------- 1. Inspect (provenance)
function renderInspect() {
  const d = S.sum.dataset, g = S.sum.grid;
  const layerCards = Object.entries(d.layers).map(([role, m]) => `
    <div class="card"><div class="lt"><b>${esc(role)}</b> ${chip(m.status || "measured")} ${m.source_verified ? '<span class="okc">source documented</span>' : '<span class="miss">source unverified</span>'}</div>
    <table class="kv">
      <tr><td>Product</td><td class="mono">${esc(m.product_id || m.file_name)}</td></tr>
      <tr><td>Source</td><td>${esc(m.source_name || "—")}</td></tr>
      <tr><td>URL</td><td class="mono">${m.source_url ? `<a href="${esc(m.source_url)}" target="_blank" rel="noopener" style="color:#7fd8ff">${esc(m.source_url)}</a>` : '<span class="miss">missing</span>'}</td></tr>
      <tr><td>Instrument</td><td>${esc(m.instrument || "—")}</td></tr>
      <tr><td>Acquisition</td><td>${esc(m.acquisition || "—")}</td></tr>
      <tr><td>Processing level</td><td>${esc(m.processing_level || "—")}</td></tr>
      <tr><td>CRS</td><td>${esc(m.crs || "—")}</td></tr>
      <tr><td>Pixel size</td><td>${m.pixel_size_m ? fmt(m.pixel_size_m, 1) + " m" : '<span class="miss">uncalibrated</span>'}</td></tr>
      <tr><td>Units</td><td>${esc(m.units || "—")}</td></tr>
      <tr><td>No-data</td><td>${esc(m.nodata)}</td></tr>
      <tr><td>Valid pixels</td><td>${pct(m.valid_fraction)}</td></tr>
      <tr><td>License / terms</td><td>${esc(m.license || "—")}</td></tr>
      <tr><td>SHA-256</td><td class="mono">${esc((m.sha256 || "").slice(0, 32))}…</td></tr>
      <tr><td>Missing metadata</td><td class="${m.missing_fields?.length ? "miss" : "okc"}">${m.missing_fields?.length ? esc(m.missing_fields.join(", ")) : "none"}</td></tr>
    </table></div>`).join("");
  const missing = Object.entries(d.missing_roles || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  $("#inspectBody").innerHTML = `
    <h3>${esc(d.title)}</h3>
    <p>${esc(d.region_note)}</p>
    <table class="kv">
      <tr><td>Dataset ID</td><td class="mono">${esc(d.dataset_id)}</td></tr>
      <tr><td>Kind</td><td>${esc(d.kind)}</td></tr>
      <tr><td>Features actually covered</td><td>${d.covers_features?.length ? esc(d.covers_features.join(", ")) : "none claimed"}</td></tr>
      <tr><td>Analysis grid</td><td>${g.width} × ${g.height} cells${g.res_m ? ` @ ${g.res_m.toFixed(0)} m` : " (uncalibrated)"}</td></tr>
      <tr><td>Projection</td><td>${g.polar_stereo_south ? "South polar stereographic, R = 1737.4 km (grid-up = 0°E meridian)" : "none / pixel grid"}</td></tr>
      <tr><td>Dataset checksum</td><td class="mono">${esc(d.dataset_checksum)}</td></tr>
      <tr><td>Analysis config hash</td><td class="mono">${esc(S.sum.config_hash)}</td></tr>
      <tr><td>Citation</td><td>${esc(d.citation)}</td></tr>
    </table>
    ${d.warnings?.length ? `<div class="card warn">${d.warnings.map(esc).join("<br>")}</div>` : ""}
    <h4>Input layers & provenance</h4>${layerCards}
    <h4>Layers not ingested</h4><table class="kv">${missing || "<tr><td>none</td></tr>"}</table>
    <h4>Processing log</h4><div class="mono" style="font-size:10.5px">${S.sum.processing_log.map((s) => esc(JSON.stringify(s))).join("<br>")}</div>`;
}

// ---------------------------------------------------------------- 2. Understand (layers)
function renderLayers() {
  const st = S.sum.stats;
  const items = Object.entries(S.sum.layers).map(([k, m]) => `
    <div class="card layer-item ${m.available ? "" : "off"} ${k === S.layer ? "sel" : ""}" data-k="${k}">
      <div class="lt"><span>${esc(m.title)}</span>${m.available ? chip(m.status) : chip("unavailable")}</div>
      <div class="ld">${esc(m.description)}</div>
      <div class="ld"><b>Units:</b> ${esc(m.units)} · <b>Res:</b> ${m.resolution_m ? m.resolution_m.toFixed(0) + " m" : "—"} · <b>Source:</b> ${esc(m.source)}</div>
      <div class="ld"><b>Method:</b> ${esc(m.method)}</div>
      ${m.limitations?.length ? `<ul>${m.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : ""}
      ${!m.available ? `<div class="ld miss">Unavailable: ${esc(m.unavailable_reason)}</div>` : ""}
    </div>`).join("");
  const hz = st.hazard_fractions ? Object.entries(st.hazard_fractions) : [];
  const colors = S.sum.layers.hazard?.categorical || {};
  const bar = hz.length ? `<div class="bar">${hz.map(([l, f], i) => `<span title="${esc(l)} ${pct(f)}" style="width:${100 * f}%;background:${Object.values(colors)[i]?.color}"></span>`).join("")}</div>
    <table class="grid">${hz.map(([l, f]) => `<tr><td>${esc(l)}</td><td>${pct(f)}</td></tr>`).join("")}</table>` : "<p>No hazard layer (needs calibrated elevation).</p>";
  const sv = st.slope_vs_reference;
  $("#layerList").innerHTML = `
    <h3>Terrain & evidence layers</h3>
    <p class="ld">Every layer states what it is, its units, resolution, source, method and limitations. Click a layer to display it.</p>
    <div class="card note"><b>Legend of status</b><br>${chip("measured")} instrument observation · ${chip("derived")} documented computation / producer model product · ${chip("model_estimate")} learned model or heuristic score · ${chip("illustrative")} visual aid only · ${chip("synthetic")} demo data</div>
    <h4>Hazard summary</h4>${bar}
    ${sv ? `<div class="card"><b>Slope consistency check</b><br>r = ${fmt(sv.pearson_r, 4)} · median |Δ| = ${fmt(sv.mad_deg, 2)}° vs producer slope<br><small>${esc(sv.note)}</small></div>` : ""}
    ${st.boulder_detection ? `<div class="card warn"><b>Boulder detection:</b> ${esc(st.boulder_detection.reason)}<br><small>The optional "anomalies" overlay shows ${S.sum.features.length} unvalidated LoG topographic anomalies (pits/mounds), not boulders.</small></div>` : ""}
    <h4>Layers</h4>${items}
    <h4>Monocular depth (Depth Anything V2)</h4>${depthPanel()}`;
  document.querySelectorAll(".layer-item").forEach((el) => el.addEventListener("click", () => { if (S.sum.layers[el.dataset.k]?.available) selectLayer(el.dataset.k); }));
  bindDepth();
}
function depthPanel() {
  const d = S.sum.depth;
  const hasImg = !!S.sum.dataset.layers.image;
  if (!hasImg) return `<div class="card">This dataset has no image layer, so monocular depth cannot run.</div>`;
  let h = `<div class="card"><p>Runs <span class="mono">${esc(S.cfg.reconstruction.model_id)}</span> on the image layer. Output is <b>relative depth (unitless)</b> unless calibrated.</p>
    <div class="btnrow"><button class="primary" id="runDepth">Run depth model</button><button id="cancelDepth" class="hidden danger">Cancel</button></div>
    <div class="progress hidden" id="depthProg"><span style="width:0%"></span></div><div class="msg" id="depthMsg"></div></div>`;
  if (d && d.status === "ok") {
    const c = d.comparison;
    h += `<div class="card"><b>${esc(d.label)}</b> ${chip("model_estimate")}<br><small>${esc(d.semantics)}</small><br><span class="mono">${esc(d.info.device)} · ${fmt(d.info.runtime_s, 1)} s · input ${esc(d.info.input_size)}${d.info.cached ? " · cached" : ""}</span></div>`;
    if (c && c.status === "ok") {
      const o = c.affine_calibration_oos;
      h += `<div class="card"><b>Comparison with LOLA elevation</b>
        <table class="grid">
          <tr><td>Co-valid coverage of AOI</td><td>${pct(c.coverage_fraction)}</td></tr>
          <tr><td>Image footprint / low-signal excluded</td><td>${pct(c.image_footprint_fraction)} / ${pct(c.low_signal_excluded_fraction)}</td></tr>
          <tr><td>Pearson r (full field)</td><td>${fmt(c.pearson_r_full, 3)}</td></tr>
          <tr><td>Spearman ρ (full field)</td><td>${fmt(c.spearman_rho_full, 3)}</td></tr>
          <tr><td>Pearson r (local relief, ${esc(c.local_relief_scale)})</td><td>${fmt(c.pearson_r_local_relief, 3)}</td></tr>
          <tr><td>Best integer shift (row, col)</td><td>${esc(JSON.stringify(c.alignment.best_integer_shift_px))}</td></tr>
          <tr><td>Resolution: image / DEM effective (median)</td><td>${fmt(c.resolution.image_grid_m, 0)} m / ${fmt(c.resolution.dem_effective_resolution_median_m, 0)} m</td></tr>
          ${o ? `<tr><td>Out-of-sample RMSE after affine fit</td><td>${fmt(o.rmse_m, 0)} m (test σ ${fmt(o.test_elev_std_m, 0)} m, skill ${fmt(o.skill_vs_mean, 2)})</td></tr>` : ""}
        </table><small>${esc(c.alignment.interpretation)}. ${o ? esc(o.label) + (o.skill_vs_mean < 0 ? " Negative skill: an affine calibration does not transfer across the scene — the prediction is not usable as metric topography." : "") : ""}</small>
        <div class="msg">Calibration status: <b>${esc(c.calibration_status)}</b></div></div>`;
    } else if (c) h += `<div class="card warn">${esc(c.reason || c.status)}</div>`;
  } else if (d) h += `<div class="card warn">${esc(d.reason || d.status)}</div>`;
  return h;
}
function bindDepth() {
  const b = $("#runDepth"); if (!b) return;
  b.onclick = async () => {
    b.disabled = true; $("#depthProg").classList.remove("hidden"); $("#cancelDepth").classList.remove("hidden");
    try {
      let job = await api.post(`/api/datasets/${S.ds}/depth`); S.job = job.id;
      $("#cancelDepth").onclick = () => api.post(`/api/jobs/${job.id}/cancel`);
      while (job.state === "running") {
        await new Promise((r) => setTimeout(r, 700));
        job = await api.get(`/api/jobs/${job.id}`);
        $("#depthProg span").style.width = `${100 * job.progress}%`; $("#depthMsg").textContent = job.stage;
      }
      if (job.state === "failed") throw new Error(job.error);
      if (job.result?.status !== "ok") { $("#depthMsg").textContent = job.result?.reason || job.result?.status; $("#depthMsg").className = "msg err"; return; }
      S.sum = await api.get(`/api/datasets/${S.ds}/summary`);
      const opts = Object.entries(S.sum.layers).filter(([, m]) => m.available);
      $("#layerSelect").innerHTML = opts.map(([k, m]) => `<option value="${k}">${esc(m.title)}</option>`).join("");
      renderLayers(); await selectLayer("depth_relative"); renderReport();
    } catch (err) { $("#depthMsg").textContent = err.message; $("#depthMsg").className = "msg err"; }
    finally { b.disabled = false; $("#cancelDepth")?.classList.add("hidden"); }
  };
}

// ---------------------------------------------------------------- 3. Prioritise
function renderPriority() {
  const p = S.sum.stats.priority || {}, w = p.weights || S.cfg.ice_priority.weights;
  const used = new Set(p.layers_used || []);
  const sliders = Object.entries(S.cfg.ice_priority.weights).map(([k, v]) => `
    <div class="slider-row"><span>${esc(k)} ${used.has(k) ? "" : '<span class="miss">(no data)</span>'}</span>
      <input type="range" min="0" max="1" step="0.05" value="${w[k] ?? v}" data-w="${k}"><output>${(w[k] ?? v).toFixed(2)}</output></div>`).join("");
  const cf = p.class_fractions ? Object.entries(p.class_fractions).map(([l, f]) => `<tr><td>${esc(l)}</td><td>${pct(f)}</td></tr>`).join("") : "";
  const s = p.sensitivity;
  $("#priorityBody").innerHTML = `
    <h3>Water-ice investigation priority</h3>
    <div class="card warn"><b>Not an ice detector.</b> ${esc(p.wording || "")} Darkness alone is never used; PSR membership comes from LOLA illumination modelling.</div>
    <table class="kv">
      <tr><td>Layers used</td><td>${esc((p.layers_used || []).join(", ") || "none")}</td></tr>
      <tr><td>Layers missing</td><td class="miss">${esc((p.layers_missing || []).join(", ") || "none")}</td></tr>
      <tr><td>Assessed fraction</td><td>${pct(p.assessed_fraction)}</td></tr>
      <tr><td>Rule</td><td>assessed only if evidence coverage ≥ ${S.cfg.ice_priority.min_evidence_fraction} and ${esc(S.cfg.ice_priority.require_layers.join(", "))} present</td></tr>
    </table>
    <h4>Weights (explicit, editable)</h4>${sliders}
    <div class="btnrow"><button class="primary" id="applyW">Recompute</button><button id="resetW">Reset to config</button></div>
    <h4>Class areas</h4><table class="grid">${cf}</table>
    ${s ? `<h4>Sensitivity</h4><div class="card">${s.scenarios} scenarios (each weight ×(1±${s.perturbation}) and drop-one). <b>${pct(s.fraction_of_assessed_cells_changing_class)}</b> of assessed cells change class in at least one scenario; median score σ ${fmt(s.median_score_std, 3)}. View the <a href="#" id="showSens" style="color:#7fd8ff">sensitivity layer</a>.</div>` : ""}
    <h4>Transfer functions</h4>
    <div class="card note ld">psr: fraction of cell in modelled PSR → [0,1]<br>thermal_max: 1 at ≤ ${S.cfg.ice_priority.thermal_full_score_K} K → 0 at ≥ ${S.cfg.ice_priority.thermal_zero_score_K} K (~110 K sublimation stability)<br>ice_stability_depth: 1 at 0 m → 0 at ≥ ${S.cfg.ice_priority.stability_depth_zero_score_m} m<br>hydrogen (LEND): not used per-pixel (km-scale footprint)</div>
    <div class="btnrow"><button id="showPrio">Show priority class</button><button id="showScore">Show score</button><button id="showCov">Show evidence coverage</button></div>
    <p class="ld">Click any cell in Inspect mode to see each layer's raw value, sub-score, weight and contribution.</p>`;
  document.querySelectorAll("[data-w]").forEach((el) => el.addEventListener("input", () => { el.nextElementSibling.textContent = (+el.value).toFixed(2); }));
  $("#applyW").onclick = () => applyWeights(Object.fromEntries([...document.querySelectorAll("[data-w]")].map((el) => [el.dataset.w, +el.value])));
  $("#resetW").onclick = () => applyWeights(S.cfg.ice_priority.weights);
  $("#showPrio").onclick = () => selectLayer("priority_class");
  $("#showScore").onclick = () => selectLayer("priority_score");
  $("#showCov").onclick = () => selectLayer("evidence_coverage");
  const sl = $("#showSens"); if (sl) sl.onclick = (e) => { e.preventDefault(); selectLayer("priority_sensitivity"); };
}
async function applyWeights(w) {
  loading(true, "Recomputing priority…");
  try {
    S.sum.stats.priority = await api.post(`/api/datasets/${S.ds}/priority`, { weights: w });
    S.layerVersion = Date.now();
    renderPriority();
    if (S.layer.startsWith("priority") || S.layer === "evidence_coverage") await selectLayer(S.layer);
  } catch (err) { banner("error", esc(err.message)); } finally { loading(false); }
}

// ---------------------------------------------------------------- 4. Navigate
function renderNav() {
  const w = S.cfg.navigation.weights, h = S.cfg.hazard;
  const r = S.route;
  const cal = S.sum.grid.res_m !== null;
  let res = "";
  if (r && r.found) {
    const a = r.alternatives || {}, s = r.summary;
    const row = (lab, k, d = 0, unit = "") => `<tr><td>${lab}</td><td>${fmt(s[k], d)}${unit}</td><td>${fmt(a.distance_only?.[k], d)}${unit}</td><td>${fmt(a.direct_line?.[k], d)}${unit}</td></tr>`;
    res = `<div class="card warn"><b>${esc(r.label)}</b></div>
      <table class="grid"><tr><th></th><th style="color:#00e5ff">A* candidate</th><th>distance-only</th><th style="color:#ff5470">direct line</th></tr>
      ${cal ? row("Length", "length_m", 0, " m") : row("Length (cells)", "length_cells", 1)}
      <tr><td>Path cost</td><td>${fmt(s.cost, 0)}</td><td>${fmt(a.distance_only?.cost, 0)}</td><td>${a.direct_line?.cost === null ? '<span class="miss">∞ impassable</span>' : fmt(a.direct_line?.cost, 0)}</td></tr>${row("Max slope", "max_slope_deg", 1, "°")}${row("Mean slope", "mean_slope_deg", 1, "°")}
      ${row("Blocked cells crossed", "blocked_cells_crossed")}${row("Cells in PSR", "cells_in_psr")}${row("Total ascent", "total_ascent_m", 0, " m")}
      <tr><td>Why rejected</td><td>—</td><td>${esc(a.distance_only?.why_rejected)}</td><td>${esc(a.direct_line?.why_rejected)}</td></tr></table>
      <h4>Hazard exposure along candidate</h4><table class="grid">${Object.entries(s.hazard_exposure || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v} cells</td></tr>`).join("")}</table>
      <div class="card note ld"><b>Cost semantics.</b> ${esc(r.cost_semantics)}<br><b>Validation:</b> all cells traversable: ${r.validation.all_cells_traversable ? "✔" : "✘"} · recomputed cost matches: ${r.validation.recomputed_cost_matches ? "✔" : "✘"} · nodes expanded ${r.expanded_nodes.toLocaleString()}${cal ? "" : "<br><span class='miss'>Distance in cells only: grid not calibrated.</span>"}</div>
      ${r.start_latlon ? `<div class="mono">start ${r.start_latlon[0].toFixed(4)}°, ${r.start_latlon[1].toFixed(2)}°E → goal ${r.goal_latlon[0].toFixed(4)}°, ${r.goal_latlon[1].toFixed(2)}°E</div>` : ""}`;
  } else if (r) res = `<div class="card warn">No route: ${esc(r.reason)}</div>`;
  $("#navBody").innerHTML = `
    <h3>Rover traversability & A* route</h3>
    <p class="ld">8-neighbour A* with octile heuristic (admissible: cost multiplier ≥ 1). No-data and unknown cells are blocked; uncertain elevation is penalised. Diagonal corner-cutting past blocked cells is disallowed.</p>
    <table class="kv"><tr><td>Start</td><td class="mono">${S.start ? S.start.join(", ") : '<span class="miss">click "Set start" then the map</span>'}</td></tr>
    <tr><td>Goal</td><td class="mono">${S.goal ? S.goal.join(", ") : '<span class="miss">click "Set goal" then the map</span>'}</td></tr></table>
    <h4>Cost weights</h4>
    ${Object.entries(w).map(([k, v]) => `<div class="slider-row"><span>${esc(k)}</span><input type="range" min="0" max="20" step="0.5" value="${(r?.weights || w)[k]}" data-nw="${k}"><output>${(+(r?.weights || w)[k]).toFixed(1)}</output></div>`).join("")}
    <div class="ld">Blocked: slope ≥ ${h.slope_blocked_deg}°, roughness ≥ ${h.roughness_blocked_m} m. Thresholds are prototype assumptions, not a rover spec.</div>
    <div class="btnrow"><button class="primary" id="planBtn" ${S.start && S.goal ? "" : "disabled"}>Plan route</button><button id="demoPts">Suggest demo points</button><button id="clearRoute">Clear</button><button id="showCost">Show cost map</button></div>
    <div class="msg" id="navMsg"></div>${res}`;
  document.querySelectorAll("[data-nw]").forEach((el) => el.addEventListener("input", () => { el.nextElementSibling.textContent = (+el.value).toFixed(1); }));
  $("#planBtn").onclick = plan;
  $("#clearRoute").onclick = () => { S.route = null; S.start = S.goal = null; Object.assign(map, { route: null, alts: {}, start: null, goal: null }); map.draw(); v3.setRoute(null); renderNav(); };
  $("#showCost").onclick = () => selectLayer("cost");
  $("#demoPts").onclick = async () => {
    try {
      const p = await api.get(`/api/datasets/${S.ds}/suggest_points`);
      if (!p.ok) { $("#navMsg").textContent = p.reason; return; }
      S.start = p.start; S.goal = p.goal; map.start = S.start; map.goal = S.goal; map.draw(); renderNav();
      $("#navMsg").textContent = p.note;
    } catch (err) { $("#navMsg").textContent = err.message; }
  };
  const snap = (name) => { const v = S.route?.[`nearest_traversable_${name}`]; return v ? `<button data-snap="${name}">Snap ${name} to nearest traversable cell (${v.join(", ")})</button>` : ""; };
  const snaps = snap("start") + snap("goal");
  if (snaps) { $("#navMsg").insertAdjacentHTML("afterend", `<div class="btnrow">${snaps}</div>`);
    document.querySelectorAll("[data-snap]").forEach((b) => b.onclick = () => { const n = b.dataset.snap; S[n] = S.route[`nearest_traversable_${n}`]; map[n] = S[n]; map.draw(); plan(); }); }
};
async function plan() {
  const weights = Object.fromEntries([...document.querySelectorAll("[data-nw]")].map((el) => [el.dataset.nw, +el.value]));
  loading(true, "Planning A* route…");
  try {
    const r = await api.post(`/api/datasets/${S.ds}/route`, { start: S.start, goal: S.goal, weights });
    S.route = r; window.__lunarisRoute = r; window.__lunarisSim = null;
    map.route = r.found ? r.cells : null; map.alts = r.alternative_paths || {}; map.draw();
    v3.setRoute(r.found ? r.cells : null);
    renderNav(); renderReport(); S.sim = null; renderSimPanel();
  } catch (err) { $("#navMsg").textContent = err.message; $("#navMsg").className = "msg err"; }
  finally { loading(false); }
}

// ---------------------------------------------------------------- 5. Report
function renderReport() {
  const anns = map.annotations || [];
  $("#reportBody").innerHTML = `
    <h3>Mission brief export</h3>
    <p class="ld">Includes region, provenance, processing steps, layer availability, terrain findings, route, investigation-priority evidence, depth comparison, assumptions, uncertainty and limitations.</p>
    <div class="card warn"><b>Disclaimer:</b> Not a mission certification. Not proof of water ice.</div>
    <table class="kv"><tr><td>Route included</td><td>${S.route?.found ? "yes" : '<span class="miss">no — plan one in step 4</span>'}</td></tr>
    <tr><td>Depth comparison included</td><td>${S.sum.depth ? "yes" : "no"}</td></tr>
    <tr><td>Manual annotations</td><td>${anns.length}</td></tr></table>
    <div class="btnrow"><a href="${assetUrl(`/api/datasets/${S.ds}/report.pdf`)}" target="_blank"><button class="primary">Export PDF brief</button></a>
    <a href="${assetUrl(`/api/datasets/${S.ds}/report.json`)}" target="_blank"><button>Export JSON summary</button></a></div>
    <h4>Manual annotations</h4>
    ${anns.length ? anns.map((a) => `<div class="card"><b>${esc(a.label)}</b> <span class="mono">(${a.row}, ${a.col})</span> <button class="ghost" data-del="${a.id}">delete</button><br><small>${esc(a.note)}</small></div>`).join("") : '<p class="ld">Use "Annotate" mode on the map to add manual observations (kept separate from detector output).</p>'}`;
  document.querySelectorAll("[data-del]").forEach((b) => b.onclick = async () => {
    await api.del(`/api/datasets/${S.ds}/annotations/${b.dataset.del}`);
    map.annotations = map.annotations.filter((a) => a.id !== b.dataset.del); map.draw(); renderReport();
  });
}

// ---------------------------------------------------------------- upload
$("#uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault(); const msg = $("#uploadMsg"); msg.className = "msg"; msg.textContent = "Uploading and validating…";
  try {
    const r = await api.form("/api/upload", new FormData(e.target));
    msg.textContent = `Accepted as ${r.role}. Missing metadata: ${r.manifest.missing_fields.join(", ") || "none"}.\n${r.warnings.join("\n")}`;
    await refreshDatasets(r.dataset_id); await loadDataset(r.dataset_id);
  } catch (err) { msg.className = "msg err"; msg.textContent = err.message; }
});

// ---------------------------------------------------------------- 6. Rover simulation
let simView = null, ROVERS = null;
const TAGC = (t) => `<span class="tagchip ${t}">${t}</span>`;
const hms = (sec) => { sec = Math.max(0, Math.round(sec)); const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), x = sec % 60; return `T+${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}`; };
$("#simToggle").addEventListener("click", () => { if ($("#simOverlay").classList.contains("hidden")) { showTab("simulate"); if (S.sim) enterSim(); else runSim(); } else exitSim(); });
$("#simClose").addEventListener("click", () => exitSim());

async function renderSimPanel() {
  const ready = S.sum && S.route?.found;
  $("#alvikBtn").disabled = !ready;
  $("#alvikBtn").title = ready ? "Shrink this route to a 4 ft grid and upload it to an Arduino Alvik" : "Plan a route first";
  $("#simToggle").disabled = !ready;
  $("#simToggle").title = ready ? "Drive the rover along the planned route" : "Load a dataset and plan a route first";
  if (!S.sum) return;
  if (!ROVERS) { try { ROVERS = await api.get("/api/rovers"); } catch (e) { $("#simBody").innerHTML = `<div class="card warn">${esc(e.message)}</div>`; return; } }
  const r = S.simReq || { rover: "perseverance", soil: "apollo_lrv_nominal", speed_mode: "nominal", sunlit_load_w: "" };
  const rv = ROVERS.rovers[r.rover];
  const params = Object.entries(rv.params).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${v.value ?? "—"}</td><td>${TAGC(v.tag)}</td><td class="ld">${esc(v.src)}</td></tr>`).join("");
  const sim = S.sim;
  let res = "";
  if (sim) {
    const m = sim.summary, ss = sim.soil_sensitivity, vcls = m.verdict.split(" ")[0];
    const row = (l, v) => `<tr><td>${l}</td><td class="mono">${v}</td></tr>`;
    res = `<h4>Result</h4><div class="verdict ${vcls}" style="display:inline-block">${esc(m.verdict)}</div>
      ${m.failure ? `<div class="card warn"><b>Failure:</b> ${esc(m.failure)}</div>` : ""}
      ${sim.low_fidelity ? `<div class="card warn"><b>Low-fidelity profile.</b> Raw model verdict: ${esc(m.verdict_raw)}. ${esc(sim.rover.fidelity_warning || "")}</div>` : ""}
      <table class="kv">
        ${row("Distance driven", `${(m.distance_m / 1000).toFixed(2)} / ${(m.route_length_m / 1000).toFixed(2)} km (${(100 * m.completed_fraction).toFixed(0)}%)`)}
        ${row("Drive time (continuous)", `${m.drive_time_h.toFixed(1)} h`)}${row("Mean ground speed", `${(100 * m.mean_speed_mps).toFixed(1)} cm/s (cmd ${(100 * sim.speed_cmd_mps).toFixed(0)} cm/s)`)}
        ${row("Max slope / pitch / roll", `${m.max_slope_deg.toFixed(1)}° / ${m.max_pitch_deg.toFixed(1)}° / ${m.max_roll_deg.toFixed(1)}°`)}
        ${row("Slip max / mean", `${(100 * m.max_slip).toFixed(0)}% / ${(100 * m.mean_slip).toFixed(0)}%`)}
        ${row("Max sinkage", `${m.max_sinkage_mm.toFixed(1)} mm`)}
        ${row("Static tip angles (lon / lat)", `${m.static_tip_angles_deg.longitudinal.toFixed(1)}° / ${m.static_tip_angles_deg.lateral.toFixed(1)}° (CG height assumed)`)}
        ${row("Min stability margin", `${m.min_stability_margin_deg.toFixed(1)}°`)}
        ${row("Time in PSR (max continuous)", `${m.psr_time_h.toFixed(2)} h (${m.max_continuous_psr_h.toFixed(2)} h)`)}
        ${row("Shadow energy used / usable", m.shadow_energy_wh == null ? esc(m.energy_model) : `${m.shadow_energy_wh.toFixed(0)} / ${m.usable_battery_wh.toFixed(0)} Wh`)}
        ${row("Mobility energy (whole drive)", `${m.mobility_energy_wh.toFixed(0)} Wh`)}
        ${row("Shadow non-mobility load", m.hotel_power_shadow_w == null ? "—" : `${m.hotel_power_shadow_w.toFixed(0)} W <span class="ld">(${esc(m.hotel_power_source)})</span>`)}
        ${row("Mean avg-sun / avg-Earth visibility", m.mean_sun_visibility == null ? "not ingested" : `${(100 * m.mean_sun_visibility).toFixed(0)}% / ${m.mean_earth_visibility == null ? "—" : (100 * m.mean_earth_visibility).toFixed(0) + "%"}`)}
      </table>
      ${m.expected_energy ? `<div class="card note ld"><b>Expected energy (indicative):</b> net ${m.expected_energy.net_energy_wh.toFixed(0)} Wh over the drive; deficit beyond usable battery ${m.expected_energy.energy_deficit_wh.toFixed(0)} Wh${m.expected_energy.parked_full_sun_charging_h != null && m.expected_energy.energy_deficit_wh > 0 ? ` → ≈${m.expected_energy.parked_full_sun_charging_h.toFixed(0)} h parked in full sun` : ""}. ${esc(m.expected_energy.note)}<br>${esc(m.expected_energy.label)}</div>` : ""}
      <div class="card"><b>Soil sensitivity — ${esc(ss.soil_name)}:</b> ${esc(ss.verdict)}${ss.failure ? ` — ${esc(ss.failure)}` : ""}; drove ${(ss.distance_m / 1000).toFixed(2)} km, max slip ${(100 * ss.max_slip).toFixed(0)}%.</div>
      <h4>Events (click to jump)</h4>
      <table class="grid">${sim.events.length ? sim.events.map((e, k) => `<tr class="evrow" data-ev="${k}"><td>${e.severity === "fail" ? "⛔" : e.severity === "warn" ? "⚠" : "ℹ"} ${esc(e.message)}</td><td>${(e.s_m / 1000).toFixed(2)} km</td></tr>`).join("") : "<tr><td>No events</td></tr>"}</table>
      <div class="btnrow"><button class="primary" id="simOpen">Open 3D simulation</button></div>
      <h4>Model fidelity — read before trusting</h4><ul class="ld">${sim.fidelity.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`;
  }
  $("#simBody").innerHTML = `
    <h3>Rover traverse simulation</h3>
    <p class="ld">Drives a published rover model along the planned A* route over the native ${S.sum.dataset.grid.res_m ? (S.sum.dataset.grid.res_m).toFixed(0) : ""} m LOLA DEM: attitude from wheel contacts, Bekker–Wong wheel–soil forces, slip, sinkage, static stability, PSR time and shadow energy budget.</p>
    ${S.route?.found ? "" : '<div class="card warn">Plan a route in step 4 first.</div>'}
    <div class="slider-row" style="grid-template-columns:1fr 2fr"><span>Rover</span><select id="simRover">${Object.entries(ROVERS.rovers).map(([k, v]) => `<option value="${k}" ${k === r.rover ? "selected" : ""}>${esc(v.name)}</option>`).join("")}</select></div>
    <div class="slider-row" style="grid-template-columns:1fr 2fr"><span>Regolith</span><select id="simSoil">${Object.entries(ROVERS.soils).map(([k, v]) => `<option value="${k}" ${k === r.soil ? "selected" : ""}>${esc(v.name)}</option>`).join("")}</select></div>
    <div class="slider-row" style="grid-template-columns:1fr 2fr"><span>Drive speed</span><select id="simSpeed"><option value="nominal" ${r.speed_mode === "nominal" ? "selected" : ""}>Nominal (prospecting)</option><option value="max" ${r.speed_mode === "max" ? "selected" : ""}>Maximum</option></select></div>
    <div class="slider-row" style="grid-template-columns:1fr 2fr"><span>Sunlit load (W, optional, ASSUMED)</span><input id="simSunLoad" type="number" min="0" step="10" placeholder="leave blank = no energy estimate" value="${r.sunlit_load_w ?? ""}"></div>
    <div class="btnrow"><button class="primary" id="simRun" ${S.route?.found ? "" : "disabled"}>Run simulation</button></div>
    <div class="msg" id="simMsg"></div>
    ${rv.fidelity_warning ? `<div class="card warn">${esc(rv.fidelity_warning)}</div>` : ""}
    ${res}
    <h4>Rover parameters & provenance</h4>
    <table class="grid">${params}</table>
    <div class="ld">${rv.references.map(esc).join("<br>")}</div>
    <h4>Regolith model</h4><div class="card ld">${esc(ROVERS.soils[r.soil].name)}: n=${ROVERS.soils[r.soil].n}, k<sub>c</sub>=${ROVERS.soils[r.soil].k_c} N/m², k<sub>φ</sub>=${ROVERS.soils[r.soil].k_phi} N/m³, c=${ROVERS.soils[r.soil].cohesion_pa} Pa, φ=${ROVERS.soils[r.soil].phi_deg}°, K=${ROVERS.soils[r.soil].K_m} m<br>${esc(ROVERS.soils[r.soil].src)}</div>`;
  const read = () => ({ rover: $("#simRover").value, soil: $("#simSoil").value, speed_mode: $("#simSpeed").value,
    sunlit_load_w: $("#simSunLoad").value === "" ? null : +$("#simSunLoad").value });
  ["#simRover", "#simSoil", "#simSpeed"].forEach((q) => $(q).addEventListener("change", () => { S.simReq = read(); renderSimPanel(); }));
  $("#simRun").onclick = () => { S.simReq = read(); runSim(); };
  const o = $("#simOpen"); if (o) o.onclick = () => enterSim();
  document.querySelectorAll("[data-ev]").forEach((el) => el.onclick = () => { enterSim().then(() => { simView.playing = false; simView.setTime(S.sim.events[+el.dataset.ev].t_h * 3600); syncPlay(); }); });
}

async function runSim() {
  S.simReq = S.simReq || { rover: "perseverance", soil: "apollo_lrv_nominal", speed_mode: "nominal", sunlit_load_w: null };
  loading(true, "Simulating rover traverse (terramechanics at 0.5 m steps)…");
  try {
    S.sim = await api.post(`/api/datasets/${S.ds}/simulate`, S.simReq); window.__lunarisSim = S.sim;
    await renderSimPanel(); renderReport(); await enterSim();
  } catch (err) { const m = $("#simMsg"); if (m) { m.textContent = err.message; m.className = "msg err"; } else banner("error", esc(err.message)); }
  finally { loading(false); }
}

async function enterSim() {
  if (!S.sim) return;
  $("#simOverlay").classList.remove("hidden"); $("#simToggle").classList.add("on"); $("#simToggle").lastChild.textContent = " Exit Simulation";
  if (!simView) {
    simView = new SimView($("#simView"), api);
    simView.onFrame = (f) => updateHud(f);
    simView.onEnd = () => syncPlay();
    $("#simPlay").onclick = () => { if (simView.simT >= simView.tEnd) simView.setTime(0); simView.playing = !simView.playing; syncPlay(); };
    $("#simRestart").onclick = () => { simView.setTime(0); simView.playing = false; syncPlay(); };
    $("#simRate").onchange = (e) => (simView.rate = +e.target.value);
    $("#simScrub").oninput = (e) => { simView.playing = false; syncPlay(); simView.setTime((e.target.value / 1000) * simView.tEnd); };
    document.querySelectorAll("#camSeg button").forEach((b) => b.onclick = () => { document.querySelectorAll("#camSeg button").forEach((x) => x.classList.toggle("active", x === b)); simView.setCamMode(b.dataset.cam); });
    $("#sunAz").oninput = (e) => { $("#sunAzV").textContent = `${e.target.value}°`; simView.setSunAz(+e.target.value); };
    $("#detailTog").onchange = (e) => simView.setDetail(e.target.checked);
    $("#expo").oninput = (e) => { $("#expoV").textContent = (+e.target.value).toFixed(2); simView.setExposure(+e.target.value); };
    simView.setExposure(+$("#expo").value);
    simView.onSunAz = (v) => { $("#sunAz").value = Math.round(v); $("#sunAzV").textContent = `${Math.round(v)}°`; };
    window.lunaris = { simView };
    window.addEventListener("keydown", (e) => {
      if ($("#simOverlay").classList.contains("hidden") || e.target.matches("input, select, textarea")) return;
      if (e.code === "Space") { e.preventDefault(); $("#simPlay").click(); }
      if (e.key === "Escape") exitSim();
    });
  }
  if (simView.sim !== S.sim) {
    loading(true, "Building close-range terrain and rover model…");
    try { await simView.load(S.ds, S.sim); } finally { loading(false); }
    simView.rate = +$("#simRate").value;
    S.miniBase = await new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = assetUrl(`/api/datasets/${S.ds}/layers/hillshade.png`); });
  }
  const m = S.sim.summary, v = m.verdict.split(" ")[0];
  $("#simVerdict").className = `verdict ${v}`; $("#simVerdict").textContent = `${S.sim.rover.name.split("(")[0].trim()} · ${m.verdict}${m.failure ? " — " + m.failure : ""}`;
  $("#simLabel").textContent = `${S.sim.rover.name} · LOLA terrain at true scale · ${S.sim.soil.name} · sun azimuth & regolith texture illustrative · not a mission certification`;
  const T = S.sim.frames.t, tEnd = T[T.length - 1] || 1;
  $("#evMarks").innerHTML = S.sim.events.map((e) => `<i title="${esc(e.message)}" style="left:${(100 * Math.min(1, e.t_h * 3600 / tEnd)).toFixed(2)}%;background:${e.severity === "fail" ? "#ff6b6b" : e.severity === "warn" ? "#ffb347" : "#9be7ff"}"></i>`).join("");
  simView.resize(); syncPlay(); simView.setTime(simView.simT);
}
function exitSim() {
  $("#simOverlay").classList.add("hidden"); $("#simToggle").classList.remove("on"); $("#simToggle").lastChild.textContent = " Rover Simulation";
  if (simView) { simView.playing = false; }
}
function syncPlay() { $("#simPlay").textContent = simView.playing ? "❚❚" : "▶"; }

function updateHud(f) {
  const sim = S.sim, p = sim.rover.params, slipOps = p.max_slip_ops.value, reqSlope = p.max_slope_req_deg.value;
  const cls = (bad, warn) => (bad ? "badv" : warn ? "warnv" : "");
  const cell = (k, v, unit, c = "", bar = null, tag = "") => `<div class="cell"><div class="k">${k}${tag ? ` <span class="tag">${tag}</span>` : ""}</div><div class="v ${c}">${v}<small>${unit}</small></div>${bar == null ? "" : `<span class="bar"><i style="width:${Math.max(0, Math.min(100, bar * 100)).toFixed(0)}%"></i></span>`}</div>`;
  $("#simHud").innerHTML =
    `<div class="big"><div><div class="k">Mission elapsed</div><div class="v">${hms(f.t).slice(2)}</div></div><div style="text-align:right"><div class="k">Odometer</div><div class="v">${(f.s / 1000).toFixed(2)}<small>km</small></div></div></div>` +
    cell("Ground speed", (100 * f.speed).toFixed(1), "cm/s", "", f.speed / (sim.speed_cmd_mps || 1)) +
    cell("Wheel slip", (100 * f.slip).toFixed(0), "%", cls(f.slip >= 0.999, f.slip > slipOps), f.slip) +
    cell("Slope", f.slope.toFixed(1), "°", cls(false, reqSlope && f.slope > reqSlope), reqSlope ? f.slope / (reqSlope * 2) : null) +
    cell("Sinkage", f.sinkage_mm.toFixed(1), "mm") +
    cell("Pitch", f.pitch.toFixed(1), "°") + cell("Roll", f.roll.toFixed(1), "°") +
    cell("Tip margin", f.margin_deg.toFixed(1), "°", cls(f.margin_deg < 0, f.margin_deg < 10), null, p.cg_height_m.tag === "assumed" ? "CG assumed" : "") +
    cell("Drive power", f.power_w.toFixed(0), "W") +
    (sim.summary.shadow_energy_wh != null ? cell("Shadow SoC", (100 * f.soc_shadow).toFixed(0), "%", "", f.soc_shadow) : cell("Energy", "n/a", "", "", null, "unpublished battery")) +
    (f.soc_exp != null ? cell("Expected SoC", (100 * f.soc_exp).toFixed(0), "%", cls(false, f.soc_exp < 0.1), f.soc_exp, "indicative") : cell("Heading", ((f.heading + 360) % 360).toFixed(0), "°")) +
    (f.sun != null ? cell("Avg sun", (100 * f.sun).toFixed(0), "%", "", f.sun, "18.6 yr") : "") +
    (f.earth != null ? cell("Avg Earth", (100 * f.earth).toFixed(0), "%", "", f.earth, "DTE comms") : "") +
    (f.in_psr ? `<div class="flag psr">◐ INSIDE PERMANENT SHADOW · HEADLIGHTS ON</div>` : `<div class="flag sun">○ OUTSIDE PSR</div>`);
  $("#simClock").textContent = `${hms(f.t)} / ${hms(simView.tEnd)}`;
  if (!simView.playing || performance.now() - (S._lastScrub || 0) > 200) { $("#simScrub").value = Math.round(1000 * f.t / Math.max(simView.tEnd, 1e-9)); S._lastScrub = performance.now(); }
  const F = sim.frames;
  if (performance.now() - (S._lastChart || 0) > 150) {
    S._lastChart = performance.now();
    drawChart($("#chSlope"), F, "slope", f.s, { label: "slope ° · distance", color: "#ffb347", min: 0, lines: reqSlope ? [[reqSlope, "#ff6b6b", `${reqSlope}° design`]] : [] });
    drawChart($("#chSlip"), F, "slip", f.s, { label: "wheel slip · distance", color: "#9be7ff", min: 0, max: 1, lines: [[slipOps, "#ff6b6b", `${100 * slipOps}% ops`]] });
    drawChart($("#chSoc"), F, F.soc_expected ? "soc_expected" : (sim.summary.shadow_energy_wh != null ? "soc_shadow" : "__none"), f.s, { label: F.soc_expected ? "expected SoC (indicative)" : "shadow-budget SoC", color: "#5ef0a8", min: 0, max: 1 });
    drawMini(f);
  }
}
function drawMini(f) {
  const cv = $("#simMini"), g = cv.getContext("2d"), W = cv.width, H = cv.height, gr = S.sum.grid;
  g.clearRect(0, 0, W, H);
  if (S.miniBase) g.drawImage(S.miniBase, 0, 0, W, H);
  const [a, , c, , e, ff] = gr.transform; const toPx = (x, y) => [((x - c) / a) / gr.width * W, ((y - ff) / e) / gr.height * H];
  const F = S.sim.frames; g.strokeStyle = "#00e5ff"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < F.x.length; i += 4) { const [px, py] = toPx(F.x[i], F.y[i]); i ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke();
  for (const ev of S.sim.events) { const i = F.s.findIndex((s) => s >= ev.s_m); if (i < 0) continue; const [px, py] = toPx(F.x[i], F.y[i]); g.fillStyle = ev.severity === "fail" ? "#ff4d5e" : "#ffd400"; g.beginPath(); g.arc(px, py, 3, 0, 6.283); g.fill(); }
  const [px, py] = toPx(f.x, f.y); g.fillStyle = "#fff"; g.strokeStyle = "#000"; g.beginPath(); g.arc(px, py, 5, 0, 6.283); g.fill(); g.stroke();
  g.fillStyle = "#c9d2de"; g.font = "10px system-ui"; g.fillText("route overview (not to scale)", 6, H - 6);
}

initAlvik(() => S.ds);
boot();
