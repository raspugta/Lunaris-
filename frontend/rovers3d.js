// Procedural rover models built from published dimensions (configs/rovers.yaml).
// Local frame: +x forward, +y up, +z right. Origin = centre of the wheel-contact plane.
// These are geometry-accurate *representations* (wheel size, track, wheelbase, overall
// envelope), not agency CAD models; fine details are stylised.
import * as THREE from "three";

const M = {
  foil: new THREE.MeshStandardMaterial({ color: 0xc9a227, metalness: 0.85, roughness: 0.35 }),
  foilDark: new THREE.MeshStandardMaterial({ color: 0x8a6a12, metalness: 0.8, roughness: 0.45 }),
  white: new THREE.MeshStandardMaterial({ color: 0xe8ebee, metalness: 0.1, roughness: 0.6 }),
  cell: new THREE.MeshStandardMaterial({ color: 0x0d1d3a, metalness: 0.5, roughness: 0.25, emissive: 0x020611 }),
  cellGrid: new THREE.MeshStandardMaterial({ color: 0x9aa6b8, metalness: 0.7, roughness: 0.4 }),
  alu: new THREE.MeshStandardMaterial({ color: 0xb8bcc2, metalness: 0.9, roughness: 0.3 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x1b1d22, metalness: 0.4, roughness: 0.6 }),
  lens: new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 1, roughness: 0.05 }),
  lamp: new THREE.MeshStandardMaterial({ color: 0xfff6d0, emissive: 0xfff2b0, emissiveIntensity: 1.5 }),
};
const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
const cyl = (r1, r2, h, mat, seg = 24) => new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg), mat);

// Spoked rigid wheel with grousers. Axis along local z.
function makeWheel(D, b, grouser, spokes = 8, nGrousers = 24) {
  const g = new THREE.Group();
  const r = D / 2;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(r - grouser, r - grouser, b, 40, 1, true), M.alu);
  rim.rotation.x = Math.PI / 2; rim.material.side = THREE.DoubleSide; g.add(rim);
  for (const s of [-1, 1]) { const ring = new THREE.Mesh(new THREE.TorusGeometry(r - grouser, 0.008, 6, 40), M.alu); ring.position.z = s * b / 2; g.add(ring); }
  const hub = cyl(r * 0.18, r * 0.18, b * 0.9, M.dark); hub.rotation.x = Math.PI / 2; g.add(hub);
  for (let k = 0; k < spokes; k++) {
    const sp = box(r - grouser - r * 0.15, 0.012, 0.012, M.alu); const a = (k / spokes) * Math.PI * 2;
    sp.position.set(Math.cos(a) * (r - grouser) / 2, Math.sin(a) * (r - grouser) / 2, 0); sp.rotation.z = a; g.add(sp);
  }
  for (let k = 0; k < nGrousers; k++) {
    const gr = box(grouser, 0.006, b, M.alu); const a = (k / nGrousers) * Math.PI * 2;
    gr.position.set(Math.cos(a) * (r - grouser / 2), Math.sin(a) * (r - grouser / 2), 0); gr.rotation.z = a; g.add(gr);
  }
  return g;
}

function solarPanel(w, h) {
  const g = new THREE.Group();
  g.add(box(w, h, 0.03, M.cellGrid));
  const cells = box(w * 0.94, h * 0.92, 0.035, M.cell); g.add(cells);
  return g;
}

export function buildRover(id, d) {
  const root = new THREE.Group();
  const body = new THREE.Group(); root.add(body);
  const wheels = [];
  const D = d.wheel_diameter_m, b = d.wheel_width_m, L = d.wheelbase_m, T = d.track_width_m, cl = d.ground_clearance_m;
  if (id === "viper") {
    // Chassis
    const chW = Math.min(d.body_width_m - 0.3, T - b - 0.05), chL = d.body_length_m - 0.3, chH = 0.5;
    const chassis = box(chL, chH, chW, M.foil); chassis.position.y = D / 2 + cl + chH / 2 - 0.2; body.add(chassis);
    const deck = box(chL * 0.96, 0.04, chW * 0.96, M.white); deck.position.y = chassis.position.y + chH / 2 + 0.02; body.add(deck); // radiator
    const top = deck.position.y;
    // Three vertical solar arrays (left, right, rear) — VIPER's sun is near the horizon at the pole.
    const pH = Math.max(0.6, d.height_m - top - 0.75);
    for (const [x, z, ry, w] of [[0, -chW / 2 - 0.03, 0, chL], [0, chW / 2 + 0.03, 0, chL], [-chL / 2 - 0.03, 0, Math.PI / 2, chW]]) {
      const p = solarPanel(w, pH); p.position.set(x, top + pH / 2, z); p.rotation.y = ry; body.add(p);
    }
    // Mast with camera/light head and high-gain antenna, total height = published height.
    const mastH = d.height_m - top - 0.2;
    const mast = cyl(0.035, 0.045, mastH, M.alu); mast.position.set(chL / 2 - 0.18, top + mastH / 2, 0.0); body.add(mast);
    const head = box(0.18, 0.12, 0.5, M.white); head.position.set(chL / 2 - 0.15, top + mastH + 0.02, 0); body.add(head);
    for (const z of [-0.18, 0.18]) { const cam = cyl(0.035, 0.035, 0.08, M.lens, 16); cam.rotation.z = Math.PI / 2; cam.position.set(chL / 2 - 0.04, top + mastH + 0.02, z); body.add(cam); }
    for (const z of [-0.08, 0.08]) { const lamp = box(0.02, 0.05, 0.06, M.lamp); lamp.position.set(chL / 2 - 0.05, top + mastH - 0.06, z); body.add(lamp); }
    const dish = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 12, 0, Math.PI * 2, 0, Math.PI / 3.2), M.white);
    dish.position.set(chL / 2 - 0.35, top + mastH - 0.25, 0); dish.rotation.z = -Math.PI / 2.4; body.add(dish);
    // TRIDENT drill housing (aft-centre) and NSS/NIRVSS boxes (stylised)
    const drill = box(0.16, 0.9, 0.16, M.foilDark); drill.position.set(-chL / 2 + 0.25, chassis.position.y + 0.1, 0); body.add(drill);
    const hazcam = box(0.06, 0.06, 0.4, M.dark); hazcam.position.set(chL / 2 + 0.02, chassis.position.y, 0); body.add(hazcam);
    // Wheel modules: suspension leg from chassis corner to wheel hub.
    for (const [sx, sz] of [[1, -1], [1, 1], [-1, -1], [-1, 1]]) {
      const w = makeWheel(D, b, d.grouser_height_m); w.position.set(sx * L / 2, D / 2, sz * T / 2); root.add(w);
      const leg = box(0.08, 0.08, Math.abs(sz * T / 2) - chW / 2 + 0.05, M.alu);
      leg.position.set(sx * L / 2, D / 2 + 0.12, sz * (chW / 2 + (T / 2 - chW / 2) / 2)); root.add(leg);
      const strut = box(0.06, cl + 0.15, 0.06, M.alu); strut.position.set(sx * L / 2, D / 2 + cl / 2 + 0.05, sz * (T / 2 - b / 2 - 0.05)); root.add(strut);
      wheels.push({ mesh: w, leg, strut, x: sx * L / 2, z: sz * T / 2 });
    }
  } else {
    // Pragyan-style six-wheel rocker-bogie rover (dimensions largely estimated; see rovers.yaml).
    const chL = d.body_length_m * 0.7, chW = d.body_width_m * 0.6, chH = 0.16;
    const chassis = box(chL, chH, chW, M.foil); chassis.position.y = D / 2 + cl + chH / 2; body.add(chassis);
    const panel = solarPanel(chL * 1.05, chW * 1.1); panel.rotation.x = -Math.PI / 2; panel.rotation.y = 0.0;
    panel.position.set(0, chassis.position.y + chH / 2 + 0.03, 0); panel.rotateX(0.35); body.add(panel);
    const mast = cyl(0.015, 0.015, 0.18, M.alu); mast.position.set(chL / 2 - 0.05, chassis.position.y + 0.17, 0); body.add(mast);
    for (const z of [-0.05, 0.05]) { const cam = box(0.04, 0.04, 0.04, M.dark); cam.position.set(chL / 2 - 0.03, chassis.position.y + 0.27, z); body.add(cam); }
    for (const sx of [1, 0, -1]) for (const sz of [-1, 1]) {
      const w = makeWheel(D, b, d.grouser_height_m, 6, 16); w.position.set(sx * L / 2, D / 2, sz * T / 2); root.add(w);
      wheels.push({ mesh: w, x: sx * L / 2, z: sz * T / 2 });
    }
    for (const sz of [-1, 1]) { const rocker = box(L * 1.05, 0.03, 0.03, M.alu); rocker.position.set(0, D / 2 + 0.06, sz * (T / 2 - b / 2 - 0.02)); root.add(rocker); }
  }
  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return { root, body, wheels, D };
}

// ---------------------------------------------------------------------------------------------
// NASA Perseverance (Mars 2020) — official NASA 3D Resources model, to scale in metres.
// Model frame: +z forward, +y up, +x left. Its six wheels are merged into three meshes; they are
// split per wheel here (triangles assigned by centroid) so each wheel can spin and follow terrain.
async function loadGLTF(url) {
  // Fresh parse each time (the browser caches the file). Cloning would break the model's skinned mesh.
  const { GLTFLoader } = await import("/static/vendor/loaders/GLTFLoader.js");
  const { DRACOLoader } = await import("/static/vendor/loaders/DRACOLoader.js");
  const draco = new DRACOLoader(); draco.setDecoderPath("/static/vendor/draco/");
  const loader = new GLTFLoader(); loader.setDRACOLoader(draco);
  const g = await loader.loadAsync(url);
  return g.scene;
}

export async function buildPerseverance(d) {
  const model = await loadGLTF(d.model_glb || "/static/assets/models/perseverance.glb");
  model.updateMatrixWorld(true);
  const root = new THREE.Group(), body = new THREE.Group();
  root.add(body);
  // Wheel centres in model coords (measured; see configs/rovers.yaml wheel_positions_m).
  const centres = [];
  let wheelsGroup = null;
  model.traverse((o) => { if (o.name === "Wheels_objs") wheelsGroup = o; });
  const wheelMeshes = [];
  if (wheelsGroup) {
    const pieces = new Map();
    wheelsGroup.traverse((m) => {
      if (!m.isMesh) return;
      const geo = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
      geo.applyMatrix4(m.matrixWorld);
      const pos = geo.attributes.position, nrm = geo.attributes.normal, uv = geo.attributes.uv;
      const zmin = -1.4, zmax = 1.4;
      for (let t = 0; t < pos.count; t += 3) {
        const cx = (pos.getX(t) + pos.getX(t + 1) + pos.getX(t + 2)) / 3, cz = (pos.getZ(t) + pos.getZ(t + 1) + pos.getZ(t + 2)) / 3;
        const key = `${cx < 0 ? "R" : "L"}${cz > 0.5 ? "F" : cz < -0.6 ? "B" : "M"}|${m.material.uuid}`;
        if (!pieces.has(key)) pieces.set(key, { mat: m.material, p: [], n: [], u: [] });
        const pc = pieces.get(key);
        for (let k = 0; k < 3; k++) {
          pc.p.push(pos.getX(t + k), pos.getY(t + k), pos.getZ(t + k));
          if (nrm) pc.n.push(nrm.getX(t + k), nrm.getY(t + k), nrm.getZ(t + k));
          if (uv) pc.u.push(uv.getX(t + k), uv.getY(t + k));
        }
      }
    });
    const byWheel = new Map();
    for (const [key, pc] of pieces) {
      const w = key.split("|")[0];
      if (!byWheel.has(w)) byWheel.set(w, []);
      byWheel.get(w).push(pc);
    }
    for (const [w, list] of byWheel) {
      const box = new THREE.Box3();
      list.forEach((pc) => { for (let i = 0; i < pc.p.length; i += 3) box.expandByPoint(new THREE.Vector3(pc.p[i], pc.p[i + 1], pc.p[i + 2])); });
      const c = box.getCenter(new THREE.Vector3());
      const hub = new THREE.Group();
      list.forEach((pc) => {
        const g = new THREE.BufferGeometry();
        const P = new Float32Array(pc.p); for (let i = 0; i < P.length; i += 3) { P[i] -= c.x; P[i + 1] -= c.y; P[i + 2] -= c.z; }
        g.setAttribute("position", new THREE.BufferAttribute(P, 3));
        if (pc.n.length) g.setAttribute("normal", new THREE.Float32BufferAttribute(pc.n, 3)); else g.computeVertexNormals();
        if (pc.u.length) g.setAttribute("uv", new THREE.Float32BufferAttribute(pc.u, 2));
        const mesh = new THREE.Mesh(g, pc.mat); mesh.castShadow = true; hub.add(mesh);
      });
      hub.position.copy(c);
      wheelMeshes.push({ name: w, hub, c });
    }
    wheelsGroup.visible = false;
  }
  model.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  // Re-frame: model +z forward -> local +x forward; centre on the wheel centroid.
  const holder = new THREE.Group();
  holder.add(model);
  wheelMeshes.forEach((w) => holder.add(w.hub));
  const cz = wheelMeshes.length ? wheelMeshes.reduce((a, w) => a + w.c.z, 0) / wheelMeshes.length : 0;
  holder.position.z = -cz;
  const frame = new THREE.Group(); frame.rotation.y = Math.PI / 2; frame.add(holder);
  body.add(frame);
  // Order wheels like the simulator: FL, FR, ML, MR, RL, RR  (left = model +x).
  const order = ["LF", "RF", "LM", "RM", "LB", "RB"];
  const wheels = order.map((k) => {
    const w = wheelMeshes.find((m) => m.name === k);
    if (!w) return null;
    return { mesh: w.hub, x: w.c.z - cz, z: -w.c.x, baseY: w.c.y, axis: "x" };
  }).filter(Boolean);
  return { root, body, wheels, D: d.wheel_diameter_m, maxOffset: 0.12 };
}
