"""LUNARIS technical report (PDF) — layout modelled on NASA Technical Memorandum conventions.

Structure: cover with document control, executive summary with key results, numbered sections,
numbered table/figure captions, running header/footer with "Page X of Y", appendices.
Typography and palette follow the LUNARIS frontend (Unbounded / Space Grotesk / JetBrains Mono;
obsidian, bone, ember, ice), adapted for print on white paper.
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.reporting.figures import map_figure

FONT_DIR = Path(__file__).with_name("fonts")
INK, MUTED, RULE, TINT = "#15171c", "#6b675f", "#d9d4ca", "#f6f4ef"
OBSIDIAN, BONE, EMBER, SOLAR, ICE = "#07080b", "#ece6da", "#ff5a1f", "#ffb347", "#1f7fa6"
PROJECT = "LUNARIS"
FULL_NAME = "Lunar Uncharted Navigation, Autonomy & Resource Intelligence System"
_SUBS = {"ρ": "rho", "σ": "sigma", "Δ": "delta ", "≥": ">=", "≤": "<=", "→": "->", "✓": "yes", "★": "*", "✘": "no", "✔": "yes"}


def _t(s: Any) -> str:
    """Escape for ReportLab paragraphs and replace glyphs missing from the embedded Latin subsets."""
    s = "" if s is None else str(s)
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fonts() -> dict[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    names = {"display": ("Unbounded-Black", "unbounded-latin-900-normal"), "display7": ("Unbounded-Bold", "unbounded-latin-700-normal"),
             "sans": ("SpaceGrotesk", "space-grotesk-latin-400-normal"), "sansM": ("SpaceGrotesk-Medium", "space-grotesk-latin-500-normal"),
             "sansB": ("SpaceGrotesk-Bold", "space-grotesk-latin-700-normal"), "mono": ("JetBrainsMono", "jetbrains-mono-latin-400-normal"),
             "monoB": ("JetBrainsMono-SemiBold", "jetbrains-mono-latin-600-normal")}
    out = {}
    try:
        for key, (name, file) in names.items():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f"{file}.ttf")))
            out[key] = name
        from reportlab.lib.fonts import addMapping
        addMapping("SpaceGrotesk", 0, 0, "SpaceGrotesk"); addMapping("SpaceGrotesk", 1, 0, "SpaceGrotesk-Bold")
        addMapping("SpaceGrotesk", 0, 1, "SpaceGrotesk"); addMapping("SpaceGrotesk", 1, 1, "SpaceGrotesk-Bold")
    except Exception:  # fonts missing -> standard PDF fonts
        out = {"display": "Helvetica-Bold", "display7": "Helvetica-Bold", "sans": "Helvetica", "sansM": "Helvetica",
               "sansB": "Helvetica-Bold", "mono": "Courier", "monoB": "Courier-Bold"}
    return out


def _cover_image(an, route_cells) -> bytes | None:
    """Shaded relief + PSR tint + candidate route; no axes (cover art built from the analysed data)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = an.arrays.get("hillshade")
    if hs is None:
        return None
    rgb = np.dstack([np.nan_to_num(hs, nan=0.0)] * 3) * np.array([0.93, 0.91, 0.87]) + 0.04
    psr = an.arrays.get("psr")
    if psr is not None:
        m = np.clip(np.nan_to_num(psr), 0, 1)[..., None] * 0.6
        rgb = rgb * (1 - m) + np.array([0.12, 0.42, 0.78]) * m
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(rgb, 0, 1), interpolation="bilinear")
    if route_cells:
        c = np.array(route_cells)
        ax.plot(c[:, 1], c[:, 0], color="#ff5a1f", lw=2.6, solid_capstyle="round")
        ax.plot(c[0, 1], c[0, 0], "o", ms=9, mfc="#ffffff", mec="#ff5a1f", mew=2)
        ax.plot(c[-1, 1], c[-1, 0], "o", ms=9, mfc="#ff5a1f", mec="#ffffff", mew=2)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def build_pdf(summary: dict[str, Any], an, route_cells: list | None, alt_routes: dict | None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                    PageTemplate, Paragraph, Spacer, Table, TableStyle)

    F = _fonts()
    C = {k: colors.HexColor(v) for k, v in dict(ink=INK, muted=MUTED, rule=RULE, tint=TINT, obs=OBSIDIAN, bone=BONE,
                                                  ember=EMBER, solar=SOLAR, ice=ICE).items()}
    W, H = A4
    ds = summary["dataset"]
    crater = ds.get("crater") or {}
    region = crater.get("name") or ds["title"].split("—")[0].strip()
    when = datetime.fromisoformat(summary["generated_utc"])
    rep_id = f"LUN-TR-{when:%Y%m%d}-{hashlib.sha1((summary['config_hash'] + ds['dataset_checksum']).encode()).hexdigest()[:6].upper()}"

    st = {
        "body": ParagraphStyle("body", fontName=F["sans"], fontSize=9.4, leading=13.6, textColor=C["ink"], spaceAfter=5, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName=F["sans"], fontSize=7.6, leading=10.2, textColor=C["muted"]),
        "cell": ParagraphStyle("cell", fontName=F["sans"], fontSize=7.8, leading=10, textColor=C["ink"]),
        "cellm": ParagraphStyle("cellm", fontName=F["mono"], fontSize=7.2, leading=9.6, textColor=C["ink"]),
        "h1": ParagraphStyle("h1", fontName=F["sansB"], fontSize=15, leading=19, textColor=C["ink"], spaceBefore=4, spaceAfter=7, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName=F["sansB"], fontSize=11, leading=14, textColor=C["ink"], spaceBefore=8, spaceAfter=4),
        "kick": ParagraphStyle("kick", fontName=F["monoB"], fontSize=7.4, leading=9, textColor=C["ember"], spaceBefore=12, spaceAfter=2, keepWithNext=1),
        "cap": ParagraphStyle("cap", fontName=F["sans"], fontSize=8, leading=11, textColor=C["muted"], spaceBefore=3, spaceAfter=9),
        "capk": ParagraphStyle("capk", fontName=F["sans"], fontSize=8, leading=11, textColor=C["muted"], spaceBefore=3, spaceAfter=5, keepWithNext=1),
        "bullet": ParagraphStyle("bullet", fontName=F["sans"], fontSize=9.2, leading=13, textColor=C["ink"], leftIndent=11, bulletIndent=0, spaceAfter=2),
    }
    P = lambda s, k="cell": Paragraph(_t(s), st[k])  # noqa: E731
    counters = {"sec": 0, "tab": 0, "fig": 0}

    def section(title, need_mm=70):
        counters["sec"] += 1
        return [CondPageBreak(need_mm * mm), Paragraph(f"SECTION {counters['sec']:02d}", st["kick"]), Paragraph(_t(title), st["h1"])]

    def table(rows, widths, caption=None, mono_cols=(), zebra=True):
        data = [[P(c, "cellm" if (j in mono_cols and i) else "cell") if not isinstance(c, Paragraph) else c
                 for j, c in enumerate(r)] for i, r in enumerate(rows)]
        hdr_style = ParagraphStyle("hdr", parent=st["cell"], fontName=F["sansB"], textColor=C["bone"])
        data[0] = [Paragraph(_t(c), hdr_style) for c in rows[0]]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        ts = [("BACKGROUND", (0, 0), (-1, 0), C["obs"]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LINEBELOW", (0, 0), (-1, 0), 1.4, C["ember"]), ("LINEBELOW", (0, 1), (-1, -1), 0.35, C["rule"]),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
        if zebra:
            ts += [("BACKGROUND", (0, i), (-1, i), C["tint"]) for i in range(2, len(data), 2)]
        t.setStyle(TableStyle(ts))
        out = [t, Spacer(1, 8)]
        if caption:
            counters["tab"] += 1
            out.insert(0, Paragraph(f"<font name='{F['sansB']}' color='{INK}'>Table {counters['tab']}.</font> {_t(caption)}", st["capk"]))
        # short tables stay in one piece; long ones split across pages (header row repeats), caption stays attached
        return KeepTogether(out) if len(rows) <= 8 else out

    def figure(png, width_mm, caption):
        from PIL import Image as PILImage
        w, h = PILImage.open(io.BytesIO(png)).size
        counters["fig"] += 1
        img = Image(io.BytesIO(png), width=width_mm * mm, height=width_mm * mm * h / w)
        return KeepTogether([img, Paragraph(f"<font name='{F['sansB']}' color='{INK}'>Figure {counters['fig']}.</font> {_t(caption)}", st["cap"])])

    def fmt(v, f="{:.1f}", none="—"):
        return none if v is None else f.format(v)

    tf = summary.get("terrain_findings") or {}
    route = summary.get("route")
    pri = summary.get("ice_investigation_priority") or {}
    sim = summary.get("rover_simulation")
    dep = summary.get("depth_reconstruction")
    hz = tf.get("hazard_fractions") or {}
    rs = (route or {}).get("summary") or {}

    # ------------------------------------------------------------------ page decorations
    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                if self._pageNumber > 1:
                    self._chrome(n)
                super().showPage()
            super().save()

        def _chrome(self, n):
            self.saveState()
            self.setFillColor(C["ember"]); self.rect(18 * mm, H - 13 * mm, 6, 6, stroke=0, fill=1)
            self.setFont(F["monoB"], 7); self.setFillColor(C["ink"])
            self.drawString(18 * mm + 10, H - 12.6 * mm, f"{PROJECT}  ·  TECHNICAL REPORT  ·  {region.upper()}")
            self.setFont(F["mono"], 7); self.setFillColor(C["muted"])
            self.drawRightString(W - 18 * mm, H - 12.6 * mm, rep_id)
            self.setStrokeColor(C["rule"]); self.setLineWidth(0.6); self.line(18 * mm, H - 15.5 * mm, W - 18 * mm, H - 15.5 * mm)
            self.line(18 * mm, 14 * mm, W - 18 * mm, 14 * mm)
            self.setFont(F["sans"], 7); self.drawString(18 * mm, 9.5 * mm, FULL_NAME)
            self.setFont(F["monoB"], 7); self.setFillColor(C["ink"])
            self.drawRightString(W - 18 * mm, 9.5 * mm, f"Page {self._pageNumber} of {n}")
            self.restoreState()

    def cover(c, _doc):
        c.saveState()
        band = H * 0.47
        c.setFillColor(C["obs"]); c.rect(0, H - band, W, band, stroke=0, fill=1)
        # subtle glow arcs
        for i, a in enumerate((0.10, 0.06, 0.035)):
            c.setFillColor(colors.Color(1, 0.35, 0.12, alpha=a)); c.circle(W * 0.86, H - band * 0.22, 60 + 45 * i, stroke=0, fill=1)
        c.setFillColor(C["ember"]); c.rect(0, H - band - 3, W, 3, stroke=0, fill=1)
        x0 = 20 * mm
        c.setFont(F["monoB"], 8); c.setFillColor(C["ember"]); c.drawString(x0, H - 22 * mm, "TECHNICAL REPORT")
        c.setFont(F["mono"], 8); c.setFillColor(colors.HexColor("#a9a398")); c.drawRightString(W - 20 * mm, H - 22 * mm, rep_id)
        c.setFont(F["display"], 54); c.setFillColor(C["bone"]); c.drawString(x0 - 2, H - 52 * mm, PROJECT)
        c.setFont(F["sans"], 8.6); c.setFillColor(colors.HexColor("#b9b2a5")); c.drawString(x0, H - 60 * mm, FULL_NAME.upper())
        c.setFont(F["sansB"], 24); c.setFillColor(colors.white)
        c.drawString(x0, H - 84 * mm, "Traverse and Volatile-Prospect")
        c.drawString(x0, H - 94 * mm, "Assessment")
        c.setFont(F["sansM"], 13); c.setFillColor(C["solar"])
        c.drawString(x0, H - 106 * mm, _t(f"{region} · Lunar South Polar Region").replace("&amp;", "&"))
        c.setFont(F["sans"], 9); c.setFillColor(colors.HexColor("#b9b2a5"))
        c.drawString(x0, H - 114 * mm, f"Issued {when:%d %B %Y}  ·  Prepared with the {PROJECT} analysis pipeline")
        # cover art
        art = _cover_image(an, route_cells)
        if art:
            from reportlab.lib.utils import ImageReader
            size = 78 * mm
            ix, iy = W - 20 * mm - size, H - band - 16 * mm - size
            c.drawImage(ImageReader(io.BytesIO(art)), ix, iy, size, size)
            c.setStrokeColor(C["ink"]); c.setLineWidth(0.6); c.rect(ix, iy, size, size)
            c.setFont(F["sans"], 6.6); c.setFillColor(C["muted"])
            c.drawString(ix, iy - 9, "LOLA shaded relief, PSR (blue) and candidate traverse (orange)")
        # document control
        rows = [("Report number", rep_id), ("Date of issue", f"{when:%Y-%m-%d %H:%M} UTC"),
                ("Study region", region), ("Dataset", ds["title"]),
                ("Region centre", f"{abs(crater['lat']):.2f} S, {crater['lon']:.2f} E · D = {crater['diameter_km']:.1f} km" if crater else "—"),
                ("Analysis grid", f"{summary['analysis_grid']['width']} x {summary['analysis_grid']['height']} cells @ {fmt(summary['analysis_grid'].get('res_m'), '{:.0f}')} m"),
                ("Configuration", summary["config_hash"]), ("Dataset checksum", ds["dataset_checksum"]),
                ("Primary sources", "NASA LRO: LOLA (PGDA), Diviner (PDS), LROC NAC; LOLA illumination models"),
                ("Distribution", "Unrestricted — derived from NASA open data")]
        from reportlab.pdfbase.pdfmetrics import stringWidth
        y = H - band - 22 * mm
        col_w = 62 * mm
        c.setFont(F["monoB"], 7.4); c.setFillColor(C["ember"]); c.drawString(x0, y, "DOCUMENT CONTROL"); y -= 5 * mm
        for k, v in rows:
            words, lines, cur = str(v).split(" "), [], ""
            for wd in words:
                trial = (cur + " " + wd).strip()
                if stringWidth(trial, F["mono"], 7) <= col_w:
                    cur = trial
                else:
                    lines.append(cur); cur = wd
            lines.append(cur)
            c.setFont(F["sans"], 7.4); c.setFillColor(C["muted"]); c.drawString(x0, y, k)
            c.setFont(F["mono"], 7); c.setFillColor(C["ink"])
            for i, ln in enumerate(lines[:3]):
                c.drawString(x0 + 28 * mm, y - i * 3.4 * mm, ln)
            y -= (len(lines[:3]) - 1) * 3.4 * mm
            c.setStrokeColor(C["rule"]); c.setLineWidth(0.4); c.line(x0, y - 2.2 * mm, x0 + 90 * mm, y - 2.2 * mm)
            y -= 6.4 * mm
        c.setFont(F["sans"], 7); c.setFillColor(C["muted"])
        c.drawString(x0, 14 * mm, f"{PROJECT} · {FULL_NAME}")
        c.drawRightString(W - 20 * mm, 14 * mm, "Lunar surface data: NASA/GSFC/ASU/UCLA/MIT")
        c.restoreState()

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, title=f"{PROJECT} Technical Report — {region}", author=f"{PROJECT} analysis pipeline",
                          subject=FULL_NAME, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=20 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[Frame(0, 0, W, H, id="c")], onPage=cover),
                          PageTemplate(id="body", frames=[frame])])
    el: list = [NextPageTemplate("body"), PageBreak()]

    def add(x):
        el.extend(x) if isinstance(x, list) else el.append(x)

    # ------------------------------------------------------------------ executive summary
    el += [Paragraph("EXECUTIVE SUMMARY", st["kick"]), Paragraph("Summary of findings", st["h1"])]
    parts = [f"This report documents an orbital-data assessment of the {_t(region)} study area using the {PROJECT} pipeline. "]
    if hz:
        low = hz.get("Low", 0) * 100; blk = hz.get("Blocked (impassable)", 0) * 100
        parts.append(f"Terrain derived from LOLA laser altimetry classifies {low:.0f}% of the analysis area as low mobility hazard "
                     f"and {blk:.0f}% as impassable at the adopted slope and roughness thresholds. ")
    if tf.get("psr_fraction") is not None:
        parts.append(f"Permanently shadowed regions occupy {100 * tf['psr_fraction']:.1f}% of the window. ")
    if route and route.get("found"):
        parts.append(f"A hazard-weighted candidate traverse of {rs.get('length_m', 0) / 1000:.1f} km was identified, with a maximum "
                     f"slope of {fmt(rs.get('max_slope_deg'))}° and no impassable cells crossed. ")
    if pri.get("status") == "ok":
        cand = (pri.get("class_fractions") or {}).get("Candidate for further investigation", 0) * 100
        parts.append(f"The volatile investigation-priority model flags {cand:.1f}% of the area as candidates for further investigation. ")
    if sim:
        m = sim["summary"]
        parts.append(f"A traverse simulation of the {_t(sim['rover']['name'])} returned <b>{_t(m['verdict'])}</b>"
                     + (f" ({_t(m['failure'])})" if m.get("failure") else "") + ". ")
    el.append(Paragraph("".join(parts), st["body"]))

    tiles = [("ROUTE LENGTH", f"{rs.get('length_m', 0) / 1000:.1f} km" if rs else "—"),
             ("MAX SLOPE ON ROUTE", f"{fmt(rs.get('max_slope_deg'))}°" if rs else "—"),
             ("PERMANENT SHADOW", f"{100 * tf['psr_fraction']:.1f}%" if tf.get("psr_fraction") is not None else "—"),
             ("ROVER SIMULATION", sim["summary"]["verdict"] if sim else "not run")]
    kst = ParagraphStyle("k", fontName=F["monoB"], fontSize=6.6, leading=8, textColor=colors.HexColor("#a9a398"))
    vst = ParagraphStyle("v", fontName=F["display7"], fontSize=15, leading=19, textColor=C["bone"])
    vst_s = ParagraphStyle("vs", parent=vst, fontSize=11.5, leading=15)
    tt = Table([[[Paragraph(k, kst), Spacer(1, 3), Paragraph(_t(v), vst if len(str(v)) <= 7 else vst_s)] for k, v in tiles]], colWidths=[doc.width / 4] * 4, rowHeights=[22 * mm])
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C["obs"]), ("LINEAFTER", (0, 0), (-2, 0), 0.6, colors.HexColor("#2a2d35")),
                            ("LINEABOVE", (0, 0), (-1, 0), 2, C["ember"]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9)]))
    el += [Spacer(1, 4), tt, Spacer(1, 10)]

    # ------------------------------------------------------------------ 1 introduction
    el += section("Introduction and scope")
    el.append(Paragraph(
        f"{PROJECT} is a decision-support system for permanently shadowed regions (PSRs) near the lunar poles. It combines laser altimetry, "
        "thermal-model products, illumination models and orbital imagery into traceable data layers, plans rover traverses over a documented "
        "cost surface, and evaluates them with a terramechanics-based rover simulation. Every layer in this report is classified as measured, "
        "derived, model estimate or illustrative, and cells without valid data are treated as unknown rather than safe.", st["body"]))
    el.append(Paragraph(f"<b>Study region.</b> {_t(ds.get('region_note', ''))} Named features inside the window (IAU gazetteer): "
                        f"{_t(', '.join(ds.get('covers_features') or []) or 'none')}.", st["body"]))

    # ------------------------------------------------------------------ 2 data
    el += section("Data sources")
    rows = [["Layer", "Product", "Instrument", "Units", "SHA-256"]]
    for role, m in ds["layers"].items():
        rows.append([role.replace("_", " "), m.get("product_id") or m.get("file_name"), m.get("instrument") or "—",
                     m.get("units") or "—", (m.get("sha256") or "")[:12]])
    add(table(rows, [28 * mm, 42 * mm, 34 * mm, 42 * mm, 24 * mm], "Input data products and checksums.", mono_cols=(1, 4)))
    if ds.get("missing_roles"):
        add(table([["Layer", "Status"]] + [[k.replace("_", " "), v] for k, v in ds["missing_roles"].items()],
                        [40 * mm, 130 * mm], "Layers not ingested for this study region."))
    el.append(Paragraph(f"<b>Citation.</b> {_t(ds.get('citation', ''))}", st["small"]))

    # ------------------------------------------------------------------ 3 methods
    el += section("Methods")
    el.append(Paragraph(
        "Rasters are ingested with checksum verification, aligned to the elevation grid and block-averaged to the analysis resolution "
        "(cells require at least 75% valid sub-cells). Slope uses the Horn (1981) operator; roughness is the RMS residual from a local "
        "least-squares plane. Hazard classes take the worse of the slope and roughness classes; producer uncertainty raises a cell by one class. "
        "Traverses use an 8-connected A* search with an admissible octile heuristic over a per-metre cost multiplier.", st["body"]))
    steps = []
    for s in summary.get("processing_log", []):
        k = s.get("step")
        if k == "ingest":
            steps.append(["Ingest", f"{s.get('role', '').replace('_', ' ')} — {s.get('file')}"])
        elif k == "align":
            steps.append(["Align", f"{s.get('role', '').replace('_', ' ')} — {s.get('method')}"])
        elif k == "downsample":
            steps.append(["Resample", f"factor {s.get('factor')} — {s.get('method')}"])
        elif k == "analyse":
            steps.append(["Analyse", f"{s.get('seconds')} s · configuration {s.get('config_hash')}"])
    if steps:
        add(table([["Stage", "Detail"]] + steps, [26 * mm, 144 * mm], "Processing record."))

    # ------------------------------------------------------------------ 4 terrain
    el += section("Terrain assessment", need_mm=175 if "hazard" in an.arrays else 70)
    if "hazard" in an.arrays:
        img = map_figure(an.arrays["hazard"], an.meta["hazard"], an.grid.res_m, an.grid.polar_stereo_south, ds["title"],
                         route_cells, alt_routes, tuple(route["start"]) if route else None, tuple(route["goal"]) if route else None,
                         title=" ")
        el.append(figure(img, 150, "Mobility hazard classes with the candidate traverse (cyan), distance-only alternative and direct line. "
                                   "Up is the 0°E meridian direction of the polar stereographic grid."))
    if hz:
        add(table([["Hazard class", "Area fraction"]] + [[k, f"{100 * v:.1f}%"] for k, v in hz.items()],
                        [90 * mm, 40 * mm], "Hazard class distribution over the analysis area."))
    if tf.get("slope_vs_reference"):
        s = tf["slope_vs_reference"]
        el.append(Paragraph(f"Derived slope agrees with the producer slope map with r = {s['pearson_r']:.3f} and a median absolute "
                            f"difference of {s['mad_deg']:.2f}° (a consistency check on the same source elevation model).", st["body"]))
    if tf.get("boulder_detection"):
        el.append(Paragraph(f"<b>Boulder-scale hazards.</b> {_t(tf['boulder_detection']['reason'])}", st["body"]))

    # ------------------------------------------------------------------ 5 traverse
    el += section("Traverse planning")
    if route and route.get("found"):
        alts = route.get("alternatives", {})

        def g(d, k, f="{:.0f}"):
            v = (d or {}).get(k)
            return "impassable" if (k == "cost" and d is not None and v is None) else ("—" if v is None else f.format(v))
        rows = [["Metric", "Candidate (hazard-weighted)", "Distance-only", "Direct line"]]
        for k, lab, f in [("length_m", "Length (m)", "{:,.0f}"), ("cost", "Path cost", "{:,.0f}"), ("max_slope_deg", "Maximum slope (°)", "{:.1f}"),
                          ("mean_slope_deg", "Mean slope (°)", "{:.1f}"), ("cells_in_psr", "Cells in PSR", "{:.0f}"),
                          ("blocked_cells_crossed", "Blocked cells crossed", "{:.0f}"), ("total_ascent_m", "Total ascent (m)", "{:,.0f}")]:
            rows.append([lab, g(rs, k, f), g(alts.get("distance_only"), k, f), g(alts.get("direct_line"), k, f)])
        add(table(rows, [46 * mm, 44 * mm, 40 * mm, 40 * mm], "Candidate traverse compared with rejected alternatives."))
        el.append(Paragraph(_t(route.get("cost_semantics", "")), st["small"]))
        sl, gl = route.get("start_latlon"), route.get("goal_latlon")
        if sl and gl:
            el.append(Paragraph(f"Start {abs(sl[0]):.4f}° S, {sl[1]:.2f}° E; goal {abs(gl[0]):.4f}° S, {gl[1]:.2f}° E (planetocentric).", st["body"]))
        if rs.get("hazard_exposure"):
            add(table([["Hazard class along route", "Cells"]] + [[k, v] for k, v in rs["hazard_exposure"].items()],
                            [90 * mm, 30 * mm], "Hazard exposure along the candidate traverse."))
    else:
        el.append(Paragraph("No traverse was planned for this report." if not route else f"No traverse found: {_t(route.get('reason'))}", st["body"]))

    # ------------------------------------------------------------------ 6 volatiles
    el += section("Volatile investigation priority")
    el.append(Paragraph(f"The priority model combines {_t(', '.join(pri.get('layers_used', [])) or 'no layers')} with explicit weights; "
                        f"layers not used: {_t(', '.join(pri.get('layers_missing', [])) or 'none')}. A cell is assessed only when thermal "
                        f"evidence is present. Assessed fraction: {100 * (pri.get('assessed_fraction') or 0):.1f}%.", st["body"]))
    if pri.get("class_fractions"):
        add(table([["Priority class", "Area fraction"]] + [[k, f"{100 * v:.1f}%"] for k, v in pri["class_fractions"].items()],
                        [90 * mm, 40 * mm], "Investigation-priority class distribution."))
    if pri.get("sensitivity"):
        s = pri["sensitivity"]
        el.append(Paragraph(f"Under {s['scenarios']} weight-perturbation scenarios, {100 * s['fraction_of_assessed_cells_changing_class']:.1f}% "
                            "of assessed cells change class at least once; the ranking should be read with this sensitivity in mind.", st["body"]))
    if "priority_class" in an.arrays and pri.get("status") == "ok":
        img = map_figure(an.arrays["priority_class"], an.meta["priority_class"], an.grid.res_m, an.grid.polar_stereo_south, ds["title"], title=" ")
        el.append(figure(img, 125, "Relative investigation priority for water ice (ranking of evidence, not a detection)."))

    # ------------------------------------------------------------------ 7 simulation
    el += section("Rover traverse simulation")
    if sim:
        m = sim["summary"]
        el.append(Paragraph(f"The {_t(sim['rover']['name'])} was driven along the candidate traverse over the native-resolution elevation model on "
                            f"{_t(sim['soil']['name'])}, with Bekker–Wong wheel–soil forces, static stability and shadow-time checks. "
                            f"Verdict: <b>{_t(m['verdict'])}</b>{(' — ' + _t(m['failure'])) if m.get('failure') else ''}.", st["body"]))
        rows = [["Quantity", "Value"],
                ["Distance driven / route length", f"{m['distance_m'] / 1000:.2f} / {m['route_length_m'] / 1000:.2f} km"],
                ["Drive time (continuous)", f"{m['drive_time_h']:.1f} h"],
                ["Maximum slope / pitch / roll", f"{m['max_slope_deg']:.1f}° / {m['max_pitch_deg']:.1f}° / {m['max_roll_deg']:.1f}°"],
                ["Wheel slip, maximum / mean", f"{100 * m['max_slip']:.0f}% / {100 * m['mean_slip']:.0f}%"],
                ["Maximum sinkage", f"{m['max_sinkage_mm']:.1f} mm"],
                ["Minimum static stability margin", f"{m['min_stability_margin_deg']:.1f}°"],
                ["Time in PSR (longest continuous)", f"{m['psr_time_h']:.2f} h ({m['max_continuous_psr_h']:.2f} h)"],
                ["Shadow energy / usable battery", f"{m['shadow_energy_wh']:.0f} / {m['usable_battery_wh']:.0f} Wh" if m.get("shadow_energy_wh") is not None else "not modelled"]]
        ss = sim.get("soil_sensitivity") or {}
        rows.append(["Loose-regolith sensitivity case", f"{ss.get('verdict')} — {ss.get('failure') or 'completed'} ({(ss.get('distance_m') or 0) / 1000:.2f} km)"])
        add(table(rows, [70 * mm, 100 * mm], "Rover simulation results."))
        if sim.get("events"):
            add(table([["Event", "Distance (km)"]] + [[e["message"], f"{e['s_m'] / 1000:.2f}"] for e in sim["events"]],
                            [140 * mm, 30 * mm], "Simulation events."))
        assumed = [k.replace("_", " ") for k, v in sim["rover"]["params"].items() if isinstance(v, dict) and v.get("tag") == "assumed"]
        el.append(Paragraph(f"Unpublished rover parameters adopted as assumptions: {_t(', '.join(assumed) or 'none')}. "
                            + _t(" ".join(sim.get("fidelity", [])[:3])), st["small"]))
    else:
        el.append(Paragraph("No rover simulation was performed for this report.", st["body"]))

    # ------------------------------------------------------------------ 8 depth
    if dep and dep.get("comparison"):
        el += section("Monocular depth reconstruction")
        c = dep["comparison"]
        rows = [["Metric", "Value"], ["Co-valid coverage", f"{100 * c.get('coverage_fraction', 0):.1f}%"],
                ["Pearson r (full field)", f"{c.get('pearson_r_full', float('nan')):.3f}"],
                ["Spearman rank correlation", f"{c.get('spearman_rho_full', float('nan')):.3f}"],
                ["Pearson r (local relief)", f"{c.get('pearson_r_local_relief', float('nan')):.3f}"],
                ["Best integer shift (px)", str(c.get("alignment", {}).get("best_integer_shift_px"))]]
        oos = c.get("affine_calibration_oos")
        if oos:
            rows.append(["Out-of-sample RMSE after affine fit", f"{oos['rmse_m']:.0f} m (test sigma {oos['test_elev_std_m']:.0f} m)"])
        el.append(Paragraph(_t(dep.get("semantics", "")), st["body"]))
        add(table(rows, [80 * mm, 70 * mm], "Depth Anything V2 relative depth compared with LOLA elevation."))

    # ------------------------------------------------------------------ assumptions / limitations
    el += section("Assumptions")
    el += [Paragraph(_t(a), st["bullet"], bulletText="–") for a in summary.get("assumptions", [])]
    el += section("Limitations")
    el += [Paragraph(_t(a), st["bullet"], bulletText="–") for a in summary.get("limitations", [])]

    # ------------------------------------------------------------------ appendix
    el += [PageBreak(), Paragraph("APPENDIX A", st["kick"]), Paragraph("Layer catalogue", st["h1"])]
    rows = [["Layer", "Status", "Units", "Method"]]
    for v in summary["layers"].values():
        rows.append([v["title"], v["status"] if v["available"] else "unavailable", v["units"], v["method"] or (v.get("unavailable_reason") or "")])
    add(table(rows, [44 * mm, 24 * mm, 28 * mm, 74 * mm], "Analysis layers, epistemic status and method."))
    el.append(Spacer(1, 10))
    el.append(Paragraph(f"Prepared with {PROJECT} ({FULL_NAME}). Configuration {summary['config_hash']}; dataset checksum "
                        f"{ds['dataset_checksum']}; generated {summary['generated_utc'][:19]} UTC.", st["small"]))

    doc.build(el, canvasmaker=NumberedCanvas)
    return buf.getvalue()
