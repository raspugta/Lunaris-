// Rover traverse simulation viewer.
// Geometry rules: every rover pose comes from a backend simulation frame (linear interpolation between
// frames). Terrain heights are LOLA (near: bilinear 4 m upsample of the 20 m DEM; far: 125 m mesh of the
// whole 32 km window). The fine regolith texture, sun azimuth and Earth placement are ILLUSTRATIVE and are
// labelled as such in the UI; they never feed the physics.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { buildRover, buildPerseverance } from "/static/rovers3d.js";

const b64f32 = (s) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const lerp = (a, b, t) => a + (b - a) * t;
const lerpAngle = (a, b, t) => { const d = ((b - a + 540) % 360) - 180; return a + d * t; };
const SUN_EL_DEG = 1.5; // polar sun stays within ~±1.6° of the horizon; illustrative fixed elevation

// Tileable regolith micro-relief (visual only): small craters + multi-scale noise -> normal & albedo maps.
function makeRegolithTextures(size = 512, seed = 7) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const h = new Float32Array(size * size);
  const blobs = (n, rmin, rmax, depth) => {
    for (let k = 0; k < n; k++) {
      const cx = rnd() * size, cy = rnd() * size, r = rmin + rnd() * (rmax - rmin), d = depth * (0.5 + rnd());
      const R = Math.ceil(r * 1.4);
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        const q = Math.hypot(dx, dy) / r; if (q > 1.4) continue;
        const x = ((Math.floor(cx + dx) % size) + size) % size, y = ((Math.floor(cy + dy) % size) + size) % size;
        h[y * size + x] += q < 1 ? -d * (1 - q * q) : d * 0.35 * Math.exp(-(((q - 1) / 0.18) ** 2));
      }
    }
  };
  blobs(6, 22, 60, 4); blobs(45, 5, 14, 1.6); blobs(420, 1, 4, 0.6);
  for (let i = 0; i < h.length; i++) h[i] += (rnd() - 0.5) * 0.9;
  const nc = document.createElement("canvas"); nc.width = nc.height = size;
  const ac = document.createElement("canvas"); ac.width = ac.height = size;
  const ng = nc.getContext("2d"), ag = ac.getContext("2d");
  const nim = ng.createImageData(size, size), aim = ag.createImageData(size, size);
  const H = (x, y) => h[(((y % size) + size) % size) * size + (((x % size) + size) % size)];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const gx = (H(x + 1, y) - H(x - 1, y)) * 0.5, gy = (H(x, y - 1) - H(x, y + 1)) * 0.5;
    let nx = -gx * 0.28, ny = -gy * 0.28, nz = 1; const l = Math.hypot(nx, ny, nz); nx /= l; ny /= l; nz /= l;
    const i = (y * size + x) * 4;
    nim.data.set([(nx * 0.5 + 0.5) * 255, (ny * 0.5 + 0.5) * 255, (nz * 0.5 + 0.5) * 255, 255], i);
    const a = 150 + Math.max(-30, Math.min(30, H(x, y) * 2.5)) + (rnd() - 0.5) * 18;
    aim.data.set([a, a * 0.985, a * 0.96, 255], i);
  }
  ng.putImageData(nim, 0, 0); ag.putImageData(aim, 0, 0);
  const n = new THREE.CanvasTexture(nc), a = new THREE.CanvasTexture(ac);
  for (const t of [n, a]) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; }
  a.colorSpace = THREE.SRGBColorSpace;
  return { normal: n, albedo: a };
}

export class SimView {
  constructor(host, api) {
    this.host = host; this.api = api;
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true }));
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace; r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.05;
    r.setClearColor(0x000000);
    host.appendChild(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 80000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true; this.controls.maxPolarAngle = Math.PI * 0.497;
    // Light: no atmosphere -> hard sun, almost no fill. Earthshine is a faint blue hemisphere term.
    this.scene.add((this.fill = new THREE.HemisphereLight(0x9fb0cc, 0x2a2724, 0.35))); // earthshine + light scattered off sunlit terrain (visibility aid)
    this.sun = new THREE.DirectionalLight(0xfff6ea, 5.5); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096); this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.02;
    Object.assign(this.sun.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 1, far: 2500 });
    this.scene.add(this.sun, this.sun.target);
    this.headlight = new THREE.SpotLight(0xfff3d6, 0, 80, 0.55, 0.45, 1.2); this.headlight.castShadow = true;
    this.scene.add(this.headlight, this.headlight.target);
    this.sunAz = 135; // degrees clockwise from grid-up (0°E meridian); illustrative, user-adjustable
    this.regolith = makeRegolithTextures();
    this.detail = true;
    this.buildSky();
    this.camMode = "chase"; this.playing = false; this.rate = 100; this.simT = 0; this.patch = null;
    new ResizeObserver(() => this.resize()).observe(host);
    let last = performance.now();
    const loop = (now) => { requestAnimationFrame(loop); const dt = Math.min(0.1, (now - last) / 1000); last = now; this.tick(dt); this.controls.update(); this.renderer.render(this.scene, this.camera); };
    requestAnimationFrame(loop);
  }
  buildSky() {
    // Stars: magnitude-like size/brightness distribution, rendered as a sphere at "infinity".
    const n = 6000, p = new Float32Array(n * 3), c = new Float32Array(n * 3), sz = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * 6.283, s = Math.sqrt(1 - u * u);
      p.set([s * Math.cos(a), u, s * Math.sin(a)], i * 3);
      const m = Math.pow(Math.random(), 3); sz[i] = 0.6 + m * 2.6;
      const tint = Math.random(); c.set(tint < 0.15 ? [1, 0.85, 0.7] : tint > 0.85 ? [0.75, 0.85, 1] : [1, 1, 1], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3)); g.setAttribute("sz", new THREE.BufferAttribute(sz, 1));
    this.stars = new THREE.Points(g, new THREE.ShaderMaterial({
      vertexColors: true, depthWrite: false, transparent: true,
      vertexShader: `attribute float sz; varying vec3 vC; void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position * 40000., 1.); gl_Position = projectionMatrix * mv; gl_PointSize = sz; }`,
      fragmentShader: `varying vec3 vC; void main(){ float d = length(gl_PointCoord - .5); if (d > .5) discard; gl_FragColor = vec4(vC, smoothstep(.5, 0., d)); }`,
    }));
    this.stars.frustumCulled = false; this.scene.add(this.stars);
    // Sun disc + glare sprite
    const cv = document.createElement("canvas"); cv.width = cv.height = 256; const gg = cv.getContext("2d");
    const grd = gg.createRadialGradient(128, 128, 0, 128, 128, 128);
    grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(0.08, "rgba(255,252,240,1)"); grd.addColorStop(0.12, "rgba(255,240,210,.55)");
    grd.addColorStop(0.35, "rgba(255,220,170,.10)"); grd.addColorStop(1, "rgba(255,200,150,0)");
    gg.fillStyle = grd; gg.fillRect(0, 0, 256, 256);
    this.sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
    this.sunSprite.scale.setScalar(9000); this.scene.add(this.sunSprite);
    // Earth: NASA Blue Marble, lit by the same sun so its phase is physically consistent with the sun direction.
    const tex = new THREE.TextureLoader().load("/static/assets/earth/blue_marble_2048.jpg"); tex.colorSpace = THREE.SRGBColorSpace;
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0 }));
    this.scene.add(this.earth);
  }
  resize() { const r = this.host.getBoundingClientRect(); if (!r.width) return; this.renderer.setSize(r.width, r.height); this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix(); }

  async load(dsId, sim) {
    this.dsId = dsId; this.sim = sim; this.F = sim.frames; this.n = this.F.t.length; this.tEnd = this.F.t[this.n - 1];
    if (this.rover) this.scene.remove(this.rover.root);
    this.rover = sim.rover.id === "perseverance" ? await buildPerseverance({ ...sim.dimensions, model_glb: sim.rover.params.model_glb }) : buildRover(sim.rover.id, sim.dimensions);
    this.rover.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.scene.add(this.rover.root);
    this.origin = { x: this.F.x[0], y: this.F.y[0], z: this.F.z[0] };
    this.sunAz = (this.F.heading[0] + 215) % 360; // default: sun behind-left of the chase camera (illustrative)
    this.onSunAz && this.onSunAz(this.sunAz);
    if (this.patch) { this.scene.remove(this.patch); this.patch.geometry.dispose(); } this.patch = null; this.patchCenter = null;
    await this.buildFar();
    this.simT = 0; this.playing = false; this._fetching = false;
    await this.ensurePatch(this.F.x[0], this.F.y[0], true);
    this.buildTracks();
    this.setTime(0); this.resetCamera();
  }
  w2s(x, y, z) { return new THREE.Vector3(x - this.origin.x, z - this.origin.z, -(y - this.origin.y)); }
  regolithMaterial(extra = {}, normal = true) {
    const m = new THREE.MeshStandardMaterial({ color: 0x9a958d, roughness: 1, metalness: 0, ...extra });
    if (this.detail) { m.map = this.regolith.albedo; if (normal) { m.normalMap = this.regolith.normal; m.normalScale = new THREE.Vector2(0.9, 0.9); } }
    // Anti-tiling: blend each texture with a rotated, rescaled copy of itself.
    m.onBeforeCompile = (sh) => {
      const mix2 = (tex, uv) => `mix(texture2D(${tex}, ${uv}), texture2D(${tex}, mat2(0.8,-0.6,0.6,0.8) * ${uv} * 0.37 + 0.21), 0.5)`;
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <map_fragment>", `#ifdef USE_MAP\n vec4 sampledDiffuseColor = ${mix2("map", "vMapUv")};\n diffuseColor *= sampledDiffuseColor;\n#endif`)
        .replace("#include <normal_fragment_maps>", THREE.ShaderChunk.normal_fragment_maps.replace(
          "vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;", `vec3 mapN = ${mix2("normalMap", "vNormalMapUv")}.xyz * 2.0 - 1.0;`));
      if (m.userData.extraCompile) m.userData.extraCompile(sh);
    };
    return m;
  }
  async buildFar() {
    // Whole 32 km window at true scale; fragments under the near patch are discarded to avoid z-fighting.
    const m = await this.api.get(`/api/datasets/${this.dsId}/mesh`);
    const z = b64f32(m.z), W = m.width, H = m.height, cs = m.cell_m;
    const g = this.gridT = (await this.api.get(`/api/datasets/${this.dsId}/summary`)).grid;
    const [a, , c, , e, f] = g.transform;
    const pos = new Float32Array(W * H * 3);
    for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) {
      const x = c + (q * m.step + 0.5 * m.step) * a, y = f + (r * m.step + 0.5 * m.step) * e;
      const v = this.w2s(x, y, z[r * W + q] - 0.8); pos.set([v.x, v.y, v.z], (r * W + q) * 3);
    }
    const idx = []; for (let r = 0; r < H - 1; r++) for (let q = 0; q < W - 1; q++) { const i = r * W + q; idx.push(i, i + W, i + 1, i + 1, i + W, i + W + 1); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
    const uv = new Float32Array(W * H * 2); for (let i = 0; i < W * H; i++) uv.set([pos[i * 3] / 400, pos[i * 3 + 2] / 400], i * 2);
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    const mat = this.regolithMaterial({}, false);
    this.patchUniform = { value: new THREE.Vector4(0, 0, 0, 0) };
    mat.userData.extraCompile = (sh) => {
      sh.uniforms.uPatch = this.patchUniform;
      sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 vWp;").replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\nvWp = (modelMatrix * vec4(transformed,1.)).xyz;");
      sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\nvarying vec3 vWp; uniform vec4 uPatch;").replace("void main() {", "void main() {\n if (uPatch.w > 0. && abs(vWp.x - uPatch.x) < uPatch.w && abs(vWp.z - uPatch.z) < uPatch.w) discard;");
    };
    if (this.far) { this.scene.remove(this.far); this.far.geometry.dispose(); }
    this.far = new THREE.Mesh(geo, mat); this.far.receiveShadow = true; this.scene.add(this.far);
  }
  async ensurePatch(x, y, force = false) {
    if (this._fetching) return;
    if (!force && this.patchCenter && Math.hypot(x - this.patchCenter[0], y - this.patchCenter[1]) < 160) return;
    this._fetching = true;
    try {
      const p = await this.api.get(`/api/datasets/${this.dsId}/terrain_patch?x=${x}&y=${y}&size=600&spacing=3&interp=cubic`);
      const z = b64f32(p.z), psr = b64u8(p.psr), n = p.n;
      const geo = new THREE.BufferGeometry(); const pos = new Float32Array(n * n * 3), col = new Float32Array(n * n * 3), uv = new Float32Array(n * n * 2);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const i = r * n + c, wx = p.x0 + c * p.spacing, wy = p.y0 - r * p.spacing, v = this.w2s(wx, wy, z[i]);
        pos.set([v.x, v.y, v.z], i * 3); uv.set([v.x / 9, v.z / 9], i * 2);
        col.set(psr[i] ? [0.82, 0.84, 0.9] : [1, 1, 1], i * 3);
      }
      const idx = []; for (let r = 0; r < n - 1; r++) for (let c = 0; c < n - 1; c++) { const a = r * n + c; idx.push(a, a + n, a + 1, a + 1, a + n, a + n + 1); }
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("color", new THREE.BufferAttribute(col, 3)); geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
      geo.setIndex(idx); geo.computeVertexNormals();
      const mesh = new THREE.Mesh(geo, this.regolithMaterial({ vertexColors: true })); mesh.receiveShadow = true; mesh.castShadow = true;
      if (this.patch) { this.scene.remove(this.patch); this.patch.geometry.dispose(); }
      this.patch = mesh; this.scene.add(mesh); this.patchCenter = [x, y]; this.patchNote = p.note;
      const cc = this.w2s(x, y, 0); this.patchUniform.value.set(cc.x, 0, cc.z, 300 - 6);
    } finally { this._fetching = false; }
  }
  buildTracks() {
    // Wheel ruts: two dark ribbons (left/right wheel lines) along the simulated path, revealed up to the current time.
    if (this.tracks) this.tracks.forEach((t) => { this.scene.remove(t); t.geometry.dispose(); });
    const d = this.sim.dimensions, half = (d.track_width_m || 1.3) / 2, w = Math.max(0.12, d.wheel_width_m || 0.2);
    const F = this.F, n = this.n;
    this.tracks = [-1, 1].map((side) => {
      const pos = new Float32Array(n * 2 * 3), idx = [];
      for (let i = 0; i < n; i++) {
        const h = (F.heading[i] * Math.PI) / 180, lx = -Math.cos(h), ly = Math.sin(h); // left unit vector (grid)
        for (let k = 0; k < 2; k++) {
          const off = side * half + (k ? w / 2 : -w / 2);
          const v = this.w2s(F.x[i] + lx * off, F.y[i] + ly * off, F.z[i] + off * Math.tan((F.roll[i] * Math.PI) / 180) + 0.03);
          pos.set([v.x, v.y, v.z], (i * 2 + k) * 3);
        }
        if (i < n - 1) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
      }
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(idx);
      const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x3d3a36, roughness: 1, transparent: true, opacity: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }));
      m.receiveShadow = true; this.scene.add(m); return m;
    });
  }
  frameAt(t) {
    const T = this.F.t; let lo = 0, hi = this.n - 1;
    if (t <= T[0]) return [0, 0, 0]; if (t >= T[hi]) return [hi, hi, 0];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; T[m] <= t ? (lo = m) : (hi = m); }
    return [lo, hi, (t - T[lo]) / Math.max(T[hi] - T[lo], 1e-9)];
  }
  sample(t) {
    const [a, b, u] = this.frameAt(t), F = this.F, out = { i: u < 0.5 ? a : b, a };
    for (const k of ["s", "x", "y", "z", "pitch", "roll", "slope", "slip", "speed", "sinkage_mm", "power_w", "soc_shadow", "margin_deg"]) out[k] = lerp(F[k][a], F[k][b], u);
    out.heading = lerpAngle(F.heading[a], F.heading[b], u);
    out.in_psr = F.in_psr[out.i]; out.t = t;
    out.sun = F.sun_avg ? lerp(F.sun_avg[a], F.sun_avg[b], u) : null; out.earth = F.earth_avg ? lerp(F.earth_avg[a], F.earth_avg[b], u) : null;
    out.soc_exp = F.soc_expected ? lerp(F.soc_expected[a], F.soc_expected[b], u) : null;
    out.wz = F.wheel_z[a].map((v, j) => lerp(v, F.wheel_z[b][j], u));
    return out;
  }
  setExposure(v) { this.renderer.toneMappingExposure = v; }
  setSunAz(deg) { this.sunAz = deg; if (this.cur) this.setTime(this.simT); }
  setDetail(on) { this.detail = on; for (const m of [this.patch, this.far]) if (m) { const mat = m.material; mat.map = on ? this.regolith.albedo : null; if (m === this.patch) mat.normalMap = on ? this.regolith.normal : null; mat.needsUpdate = true; } }
  placeSky(p) {
    // Directions in scene space: grid +y (0°E meridian) is scene -Z; azimuth measured clockwise from it.
    const dir = (azDeg, elDeg) => { const a = (azDeg * Math.PI) / 180, e = (elDeg * Math.PI) / 180; return new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e)); };
    const sd = dir(this.sunAz, SUN_EL_DEG);
    this.sun.position.copy(p).addScaledVector(sd, 1200); this.sun.target.position.copy(p);
    this.sunSprite.position.copy(this.camera.position).addScaledVector(sd, 30000);
    // Earth: from the south pole it sits near the horizon toward the 0°E meridian (elevation varies ±~7° with libration).
    const ed = dir(0, 4);
    const D = 20000; this.earth.position.copy(this.camera.position).addScaledVector(ed, D);
    this.earth.scale.setScalar(D * Math.tan((1.9 * Math.PI) / 180 / 2)); // ~1.9° apparent diameter
    this.stars.position.copy(this.camera.position);
  }
  setTime(t) {
    this.simT = Math.max(0, Math.min(t, this.tEnd));
    const s = this.sample(this.simT); this.cur = s;
    const R = this.rover.root, p = this.w2s(s.x, s.y, s.z);
    R.position.copy(p);
    R.rotation.set(0, 0, 0); R.rotation.order = "YZX";
    R.rotation.y = Math.PI / 2 - (s.heading * Math.PI) / 180;
    R.rotation.z = (s.pitch * Math.PI) / 180; R.rotation.x = (s.roll * Math.PI) / 180;
    // Wheel spin = commanded wheel speed x time (slip shows as wheels turning faster than the ground moves).
    const ang = (this.sim.speed_cmd_mps * this.simT) / (this.rover.D / 2);
    const tp = Math.tan((s.pitch * Math.PI) / 180), tr = Math.tan((s.roll * Math.PI) / 180), lim = this.rover.maxOffset ?? 0.2;
    this.rover.wheels.forEach((w, j) => {
      if (w.axis === "x") w.mesh.rotation.x = ang; else w.mesh.rotation.z = -ang;
      const plane = s.z + w.x * tp - w.z * tr;
      const off = Math.max(-lim, Math.min(lim, (s.wz[j] ?? plane) - plane));
      w.mesh.position.y = (w.baseY ?? this.rover.D / 2) + off;
    });
    this.headlight.intensity = s.in_psr ? 260 : 0;
    const fwd = new THREE.Vector3(1, 0, 0).applyEuler(R.rotation);
    this.headlight.position.copy(p).add(new THREE.Vector3(0, 2.0, 0)).addScaledVector(fwd, 0.8);
    this.headlight.target.position.copy(p).addScaledVector(fwd, 14);
    this.placeSky(p);
    // reveal tracks behind the rover
    if (this.tracks) { const k = (this.cur.a + 1) * 6; this.tracks.forEach((m) => m.geometry.setDrawRange(0, k)); }
    this.ensurePatch(s.x, s.y);
    this.onFrame && this.onFrame(s);
  }
  tick(dt) {
    if (!this.sim) return;
    if (this.playing) { this.setTime(this.simT + dt * this.rate); if (this.simT >= this.tEnd) { this.playing = false; this.onEnd && this.onEnd(); } }
    const R = this.rover.root;
    if (this.camMode === "chase" && this.cur) {
      const back = new THREE.Vector3(-8.5, 3.0, 3.2).applyAxisAngle(new THREE.Vector3(0, 1, 0), R.rotation.y);
      this.camera.position.lerp(R.position.clone().add(back), 1 - Math.exp(-dt * 2.5));
      this.controls.target.lerp(R.position.clone().add(new THREE.Vector3(0, 1.0, 0)), 1 - Math.exp(-dt * 5));
    } else if ((this.camMode === "orbit" || this.camMode === "cinematic") && this.cur) {
      const d = R.position.clone().add(new THREE.Vector3(0, 1.0, 0)), delta = d.clone().sub(this.controls.target);
      this.controls.target.add(delta); this.camera.position.add(delta);
      if (this.camMode === "cinematic") { const off = this.camera.position.clone().sub(this.controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), dt * 0.12); this.camera.position.copy(this.controls.target).add(off); }
    }
    if (this.cur) this.placeSky(R.position);
  }
  resetCamera() { const P = this.rover.root.position; this.camera.position.set(P.x - 8, P.y + 3, P.z + 5); this.controls.target.set(P.x, P.y + 1, P.z); }
  setCamMode(m) {
    this.camMode = m;
    const P = this.rover.root.position;
    if (m === "overview") { this.controls.target.copy(P); this.camera.position.set(P.x - 900, P.y + 1400, P.z + 900); }
    if (m === "ground") { this.controls.target.set(P.x, P.y + 1.2, P.z); this.camera.position.set(P.x + 6, P.y + 0.6, P.z + 6); }
    if (m === "cinematic") { this.controls.target.set(P.x, P.y + 1, P.z); this.camera.position.set(P.x - 5, P.y + 1.6, P.z + 9); }
  }
}

// Line chart of a frame series against distance, with threshold lines and a cursor.
export function drawChart(cv, F, key, cursorS, opts) {
  const g = cv.getContext("2d"), W = cv.clientWidth, H = cv.clientHeight, d = devicePixelRatio || 1;
  if (cv.width !== W * d) { cv.width = W * d; cv.height = H * d; }
  g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
  const xs = F.s, ys = F[key]; if (!ys || ys.every((v) => v == null || Number.isNaN(v))) { g.fillStyle = "#6b6862"; g.font = "10px 'JetBrains Mono'"; g.fillText(`${opts.label}: not modelled`, 10, H / 2); return; }
  const sMax = xs[xs.length - 1] || 1; let lo = opts.min ?? Math.min(...ys), hi = opts.max ?? Math.max(...ys); if (hi - lo < 1e-9) hi = lo + 1;
  const X = (s) => 8 + (W - 16) * (s / sMax), Y = (v) => H - 16 - (H - 30) * ((v - lo) / (hi - lo));
  const grad = g.createLinearGradient(0, 14, 0, H - 16); grad.addColorStop(0, opts.color + "55"); grad.addColorStop(1, opts.color + "00");
  g.beginPath(); ys.forEach((v, i) => { const x = X(xs[i]), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.lineTo(X(sMax), H - 16); g.lineTo(X(0), H - 16); g.closePath(); g.fillStyle = grad; g.fill();
  g.strokeStyle = opts.color; g.lineWidth = 1.4; g.beginPath(); ys.forEach((v, i) => { const x = X(xs[i]), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  (opts.lines || []).forEach(([v, c, lab]) => { g.strokeStyle = c; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(8, Y(v)); g.lineTo(W - 8, Y(v)); g.stroke(); g.setLineDash([]); g.fillStyle = c; g.font = "9px 'JetBrains Mono'"; g.fillText(lab, W - 8 - g.measureText(lab).width, Y(v) - 3); });
  g.fillStyle = "#a5a19a"; g.font = "600 9px 'JetBrains Mono'"; g.fillText(opts.label.toUpperCase(), 10, 12);
  if (cursorS != null) { g.strokeStyle = "#fff"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(cursorS), 16); g.lineTo(X(cursorS), H - 16); g.stroke(); }
}
