// LUNARIS landing: NASA SVS CGI Moon Kit (LROC colour + LOLA-derived normal map) rendered with three.js,
// choreographed by GSAP ScrollTrigger. The canvas sits above the hero title so the Moon occludes it.
import * as THREE from "three";

const { gsap, ScrollTrigger } = window;
gsap.registerPlugin(ScrollTrigger);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ renderer / scene
const canvas = document.getElementById("moon");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
camera.position.set(0, 0, 10);

const small = innerWidth < 900;
const tl = new THREE.TextureLoader();
const color = tl.load(`/static/assets/moon/moon_color_${small ? "2k" : "4k"}.jpg`);
color.colorSpace = THREE.SRGBColorSpace;
color.anisotropy = renderer.capabilities.getMaxAnisotropy();
const normal = tl.load(`/static/assets/moon/moon_normal_${small ? "2k" : "4k"}.jpg`);
normal.anisotropy = color.anisotropy;

const moon = new THREE.Mesh(
  new THREE.SphereGeometry(1, 192, 128),
  new THREE.MeshStandardMaterial({ map: color, normalMap: normal, normalScale: new THREE.Vector2(1.4, 1.4), roughness: 1, metalness: 0 })
);
const pivot = new THREE.Group(); // scroll-driven tilt/position
pivot.add(moon);
scene.add(pivot);
moon.rotation.y = -Math.PI / 2; // put the nearside (0° longitude) towards the camera

const sun = new THREE.DirectionalLight(0xfff4e6, 3.2);
sun.position.set(-6, 1.5, 4);
scene.add(sun);
scene.add(new THREE.AmbientLight(0x6f86b0, 0.06)); // faint earthshine

// Starfield (separate scene layer behind the moon; drifts with scroll)
const stars = (() => {
  const n = 2600, p = new Float32Array(n * 3), s = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    p[i * 3] = (Math.random() - 0.5) * 90; p[i * 3 + 1] = (Math.random() - 0.5) * 60; p[i * 3 + 2] = -20 - Math.random() * 40;
    s[i] = Math.random();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  g.setAttribute("seed", new THREE.BufferAttribute(s, 1));
  const m = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, uniforms: { uTime: { value: 0 }, uPx: { value: renderer.getPixelRatio() } },
    vertexShader: `attribute float seed; uniform float uTime; uniform float uPx; varying float vA;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.); gl_Position = projectionMatrix * mv;
        gl_PointSize = (0.6 + seed * 1.8) * uPx; vA = 0.35 + 0.65 * (0.5 + 0.5 * sin(uTime * (0.6 + seed * 2.0) + seed * 40.)); }`,
    fragmentShader: `varying float vA; void main(){ float d = length(gl_PointCoord - .5); if (d > .5) discard; gl_FragColor = vec4(vec3(1.), vA * smoothstep(.5, .0, d)); }`,
  });
  const pts = new THREE.Points(g, m);
  scene.add(pts);
  return pts;
})();

// ------------------------------------------------------------------ layout helpers
// Moon "pose" in screen-relative units: x/y in viewport half-widths, size in fraction of viewport height.
const state = { x: 0, y: -0.04, size: 0.62, tilt: 0, spin: 0, sunAz: -0.9, sunEl: 0.25, dim: 1 };
function applyPose() {
  const h = 2 * Math.tan((camera.fov * Math.PI) / 360) * camera.position.z; // visible height at z=0
  const w = h * camera.aspect;
  pivot.position.set((state.x * w) / 2, (state.y * h) / 2, 0);
  pivot.scale.setScalar((state.size * h) / 2);
  pivot.rotation.x = state.tilt;
  const r = 7;
  sun.position.set(Math.sin(state.sunAz) * r * Math.cos(state.sunEl), Math.sin(state.sunEl) * r, Math.cos(state.sunAz) * r * Math.cos(state.sunEl));
  sun.intensity = 3.2 * state.dim;
}
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// ------------------------------------------------------------------ pointer parallax
const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
addEventListener("pointermove", (e) => { mouse.tx = e.clientX / innerWidth - 0.5; mouse.ty = e.clientY / innerHeight - 0.5; });

// ------------------------------------------------------------------ scroll choreography
const isNarrow = () => innerWidth < 900;
const key = (sel, pose, extra = {}) => gsap.to(state, {
  ...pose, ease: "none",
  scrollTrigger: { trigger: sel, start: "top bottom", end: "top top", scrub: 1.2, invalidateOnRefresh: true, ...extra },
});
if (!reduce) {
  // Hero -> mission: Moon swings right and tilts its south pole toward us; sun drops to grazing incidence.
  key("#mission", { x: () => (isNarrow() ? 0.35 : 0.55), y: 0.05, size: () => (isNarrow() ? 0.7 : 0.95), tilt: -1.15, spin: 1.2, sunAz: -1.5, sunEl: 0.05 });
  key("#approach", { x: () => (isNarrow() ? -0.35 : -0.55), y: -0.05, size: 0.85, tilt: -0.55, spin: 2.6, sunAz: -0.4, sunEl: 0.35 });
  key("#capabilities", { x: 0, y: 0.1, size: 1.35, tilt: -0.2, spin: 3.6, sunAz: 0.6, sunEl: 0.2, dim: 0.55 });
  key("#numbers", { x: () => (isNarrow() ? 0.4 : 0.62), y: 0.55, size: 0.38, tilt: 0.2, spin: 4.6, sunAz: -0.8, sunEl: 0.4, dim: 1 });
  key("#data", { x: -0.62, y: -0.5, size: 0.3, tilt: 0.1, spin: 5.4 });
  key("#launch", { x: 0, y: -1.25, size: 2.2, tilt: -1.4, spin: 6.2, sunAz: -1.55, sunEl: 0.02, dim: 1 });

  // After the hero, drop the canvas behind content so cards and copy read clearly.
  ScrollTrigger.create({
    trigger: "#mission", start: "top 55%",
    onEnter: () => (canvas.style.zIndex = 1), onLeaveBack: () => (canvas.style.zIndex = 10),
  });

  // Title letters fly apart with depth-varying speed; full form fades.
  gsap.utils.toArray(".title span").forEach((el, i, all) => {
    const c = i - (all.length - 1) / 2;
    gsap.to(el, { yPercent: -60 - Math.abs(c) * 25, xPercent: c * 22, opacity: 0, rotate: c * 4, ease: "none",
      scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true } });
  });
  gsap.to([".fullform", ".hero-foot", ".hero-meta", ".scroll-hint"], { opacity: 0, yPercent: -60, ease: "none",
    scrollTrigger: { trigger: ".hero", start: "top top", end: "45% top", scrub: true } });

  // Intro: letters rise in; Moon fades up.
  // (letter/caption intro is a CSS animation so it never competes with the scroll-scrubbed tweens)
  gsap.from(state, { size: 0.2, dim: 0, duration: 2.4, ease: "expo.out" });

  // Text reveals
  gsap.utils.toArray(".reveal").forEach((el) => gsap.to(el, { opacity: 1, y: 0, duration: 1.1, ease: "power3.out",
    scrollTrigger: { trigger: el, start: "top 85%" } }));
  gsap.from(".card", { opacity: 0, y: 60, duration: 1, ease: "power3.out", stagger: 0.08, scrollTrigger: { trigger: ".grid", start: "top 80%" } });

  // Count-up stats
  gsap.utils.toArray(".stat .v").forEach((el) => {
    const to = +el.dataset.to, dec = +el.dataset.dec, o = { v: 0 };
    gsap.to(o, { v: to, duration: 2, ease: "power2.out", scrollTrigger: { trigger: el, start: "top 85%" },
      onUpdate: () => (el.textContent = o.v.toFixed(dec)) });
  });
} else {
  document.querySelectorAll(".stat .v").forEach((el) => (el.textContent = (+el.dataset.to).toFixed(+el.dataset.dec)));
}

// ------------------------------------------------------------------ render loop
const clock = new THREE.Clock();
function frame() {
  const t = clock.getElapsedTime();
  mouse.x += (mouse.tx - mouse.x) * 0.05; mouse.y += (mouse.ty - mouse.y) * 0.05;
  applyPose();
  moon.rotation.y = -Math.PI / 2 + state.spin + (reduce ? 0 : t * 0.035) + mouse.x * 0.25;
  pivot.rotation.z = mouse.x * -0.05;
  pivot.rotation.x = state.tilt + mouse.y * 0.12;
  stars.material.uniforms.uTime.value = t;
  stars.position.y = (scrollY / innerHeight) * 1.6;
  stars.rotation.z = scrollY * 0.00004;
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
frame();
