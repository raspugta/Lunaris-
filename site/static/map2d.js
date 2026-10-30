// 2D map: pan/zoom canvas rendering of a layer over hillshade with route/markers.
export class Map2D {
  constructor(canvas, { onHover, onClick }) {
    this.cv = canvas; this.ctx = canvas.getContext("2d");
    this.onHover = onHover; this.onClick = onClick;
    this.w = 0; this.h = 0; this.scale = 1; this.ox = 0; this.oy = 0;
    this.base = null; this.layer = null; this.opacity = 0.85;
    this.route = null; this.alts = {}; this.start = null; this.goal = null;
    this.annotations = []; this.features = []; this.showFeatures = false; this.pick = null;
    this.named = []; this.showNames = true;
    this._drag = null;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("mousedown", (e) => { this._drag = { x: e.clientX, y: e.clientY, ox: this.ox, oy: this.oy, moved: false }; });
    window.addEventListener("mouseup", (e) => {
      if (this._drag && !this._drag.moved && e.target === this.cv) { const c = this.cellAt(e); if (c) this.onClick(c); }
      this._drag = null;
    });
    canvas.addEventListener("mousemove", (e) => this._move(e));
    canvas.addEventListener("mouseleave", () => this.onHover(null));
  }
  resize() {
    const r = this.cv.parentElement.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    this.cv.width = Math.max(1, r.width * d); this.cv.height = Math.max(1, r.height * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0); this.vw = r.width; this.vh = r.height;
    if (!this._userMoved && this.w) this.fit(); this.draw();
  }
  setGrid(w, h) { this.w = w; this.h = h; this._userMoved = false; this.fit(); }
  fit() {
    if (!this.w || !this.vw) return;
    this.scale = Math.min(this.vw / this.w, this.vh / this.h) * 0.96;
    this.ox = (this.vw - this.w * this.scale) / 2; this.oy = (this.vh - this.h * this.scale) / 2;
    this.draw();
  }
  async _img(url) { const im = new Image(); im.src = url; await im.decode(); return im; }
  async setBase(url) { this.base = url ? await this._img(url) : null; this.draw(); }
  async setLayer(url) { this.layer = url ? await this._img(url) : null; this.draw(); }
  cellAt(e) {
    const r = this.cv.getBoundingClientRect();
    const col = Math.floor((e.clientX - r.left - this.ox) / this.scale), row = Math.floor((e.clientY - r.top - this.oy) / this.scale);
    return row >= 0 && col >= 0 && row < this.h && col < this.w ? { row, col } : null;
  }
  _wheel(e) {
    e.preventDefault();
    const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    this._userMoved = true;
    const f = Math.exp(-e.deltaY * 0.0015), ns = Math.min(Math.max(this.scale * f, 0.2), 60);
    this.ox = mx - (mx - this.ox) * (ns / this.scale); this.oy = my - (my - this.oy) * (ns / this.scale); this.scale = ns; this.draw();
  }
  _move(e) {
    if (this._drag) {
      const dx = e.clientX - this._drag.x, dy = e.clientY - this._drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) this._drag.moved = true;
      if (this._drag.moved) { this._userMoved = true; this.ox = this._drag.ox + dx; this.oy = this._drag.oy + dy; this.draw(); }
    }
    this.onHover(this.cellAt(e));
  }
  _pt(r, c) { return [this.ox + (c + 0.5) * this.scale, this.oy + (r + 0.5) * this.scale]; }
  _path(cells, color, width, dash) {
    if (!cells || cells.length < 2) return;
    const g = this.ctx; g.save(); g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.lineJoin = "round";
    g.beginPath(); cells.forEach(([r, c], i) => { const [x, y] = this._pt(r, c); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.restore();
  }
  draw() {
    const g = this.ctx; if (!g) return;
    g.clearRect(0, 0, this.vw, this.vh); g.imageSmoothingEnabled = false;
    const W = this.w * this.scale, H = this.h * this.scale;
    if (this.base) g.drawImage(this.base, this.ox, this.oy, W, H);
    if (this.layer) { g.globalAlpha = this.base ? this.opacity : 1; g.drawImage(this.layer, this.ox, this.oy, W, H); g.globalAlpha = 1; }
    if (this.w) { g.strokeStyle = "#2a3445"; g.strokeRect(this.ox, this.oy, W, H); }
    this._path(this.alts.direct_line, "#ff5470", 1.2, [5, 4]);
    this._path(this.alts.distance_only, "#ffffff", 1.2, [2, 3]);
    if (this.route) { this._path(this.route, "#001820", 4.5); this._path(this.route, "#00e5ff", 2.4); }
    if (this.showFeatures) for (const f of this.features) {
      const [x, y] = this._pt(f.row, f.col); g.strokeStyle = f.kind === "pit" ? "#ff9f1c" : "#c38bff"; g.lineWidth = 1;
      g.beginPath(); g.arc(x, y, Math.max(3, 1.5 * this.scale), 0, 6.283); g.stroke();
    }
    if (this.showNames) for (const f of this.named) {
      const [x, y] = this._pt(f.row, f.col), r = f.radius_px * this.scale;
      if (r < 6) continue;
      g.save(); g.strokeStyle = "rgba(236,230,218,.55)"; g.setLineDash([3, 4]); g.lineWidth = 1;
      g.beginPath(); g.arc(x, y, r, 0, 6.283); g.stroke(); g.setLineDash([]);
      g.font = `600 ${Math.max(10, Math.min(15, r / 6))}px "Space Grotesk", sans-serif`; g.textAlign = "center";
      g.lineWidth = 3; g.strokeStyle = "rgba(0,0,0,.75)"; g.strokeText(f.name, x, y - r - 5); g.fillStyle = "#f4efe6"; g.fillText(f.name, x, y - r - 5);
      g.restore();
    }
    for (const a of this.annotations) {
      const [x, y] = this._pt(a.row, a.col); g.fillStyle = "#ff4fd8"; g.beginPath(); g.moveTo(x, y - 9); g.lineTo(x + 5, y); g.lineTo(x - 5, y); g.fill();
      g.font = "11px system-ui"; g.fillStyle = "#ffd6f5"; g.fillText(a.label, x + 6, y - 2);
    }
    const mark = (p, fill, label) => {
      if (!p) return; const [x, y] = this._pt(p[0], p[1]);
      g.fillStyle = fill; g.strokeStyle = "#000"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 6, 0, 6.283); g.fill(); g.stroke();
      g.font = "bold 11px system-ui"; g.fillStyle = "#fff"; g.fillText(label, x + 8, y + 4);
    };
    mark(this.start, "#00e5ff", "START"); mark(this.goal, "#ffd400", "GOAL");
    if (this.pick) { const [x, y] = this._pt(this.pick[0], this.pick[1]); g.strokeStyle = "#fff"; g.lineWidth = 1; g.strokeRect(x - 6, y - 6, 12, 12); }
  }
}
