// 3D heightfield viewer (three.js r160, vendored). Drapes the active 2D layer and the route.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

export class Viewer3D {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x05070a);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 1, 1e7);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-1, 0.35, 0.6); this.scene.add(sun);
    this.exag = 2; this.mesh = null; this.routeLine = null; this.data = null;
    new ResizeObserver(() => this.resize()).observe(el);
    const loop = () => { requestAnimationFrame(loop); this.controls.update(); this.renderer.render(this.scene, this.camera); };
    loop();
  }
  resize() {
    const r = this.el.getBoundingClientRect(); if (!r.width) return;
    this.renderer.setSize(r.width, r.height); this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix();
  }
  setTerrain(m, gridW, gridH) {
    const bin = Uint8Array.from(atob(m.z), (c) => c.charCodeAt(0));
    const z = new Float32Array(bin.buffer);
    this.data = { ...m, z, gridW, gridH };
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    const cell = m.calibrated ? m.cell_m : 1;
    const Wm = (m.width - 1) * cell, Hm = (m.height - 1) * cell;
    this.sizeM = Math.max(Wm, Hm);
    const geo = new THREE.PlaneGeometry(Wm, Hm, m.width - 1, m.height - 1);
    this.geo = geo; this._applyHeights();
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(geo, mat); this.mesh.rotation.x = -Math.PI / 2; this.scene.add(this.mesh);
    this.resetView();
  }
  _zScale() {
    const d = this.data; if (!d) return 1;
    if (d.calibrated) return this.exag;                          // metres, true vertical scale x exaggeration
    return (this.sizeM * 0.15 / Math.max(d.max - d.min, 1e-9)) * this.exag; // relative depth: display-only scaling
  }
  _applyHeights() {
    const d = this.data, pos = this.geo.attributes.position, s = this._zScale();
    for (let i = 0; i < pos.count; i++) pos.setZ(i, (d.z[i] - d.min) * s);
    pos.needsUpdate = true; this.geo.computeVertexNormals();
  }
  setExaggeration(v) { this.exag = v; if (this.data) { this._applyHeights(); this.setRoute(this._route); } }
  async setTexture(urls, opacity) {
    if (!this.mesh) return;
    const imgs = await Promise.all(urls.filter(Boolean).map((u) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = u; })));
    if (!imgs.length) return;
    const cv = document.createElement("canvas"); cv.width = imgs[0].width; cv.height = imgs[0].height;
    const g = cv.getContext("2d"); g.imageSmoothingEnabled = false;
    imgs.forEach((im, k) => { g.globalAlpha = k === 0 || imgs.length === 1 ? 1 : opacity; g.drawImage(im, 0, 0, cv.width, cv.height); });
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.NearestFilter;
    if (this.mesh.material.map) this.mesh.material.map.dispose();
    this.mesh.material.map = tex; this.mesh.material.needsUpdate = true;
  }
  _h(row, col) {
    const d = this.data; const r = Math.min(d.height - 1, Math.round(row / d.step)), c = Math.min(d.width - 1, Math.round(col / d.step));
    return (d.z[r * d.width + c] - d.min) * this._zScale();
  }
  setRoute(cells) {
    this._route = cells;
    if (this.routeLine) { this.scene.remove(this.routeLine); this.routeLine.geometry.dispose(); this.routeLine = null; }
    if (!cells || !cells.length || !this.data) return;
    const d = this.data, cell = d.calibrated ? d.cell_m / d.step : 1 / d.step;
    const Wm = (d.width - 1) * (d.calibrated ? d.cell_m : 1), Hm = (d.height - 1) * (d.calibrated ? d.cell_m : 1);
    const lift = this.sizeM * 0.004;
    const pts = cells.map(([r, c]) => new THREE.Vector3(c * cell - Wm / 2, this._h(r, c) + lift, r * cell - Hm / 2));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    this.routeLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x00e5ff, depthTest: false }));
    this.routeLine.renderOrder = 2; this.scene.add(this.routeLine);
  }
  resetView() {
    const s = this.sizeM || 1000;
    this.camera.near = s / 1000; this.camera.far = s * 20; this.camera.updateProjectionMatrix();
    this.camera.position.set(-s * 0.15, s * 0.75, s * 0.85); this.controls.target.set(0, 0, 0); this.controls.update();
  }
}
