"""Map rendering shared by the API (PNG tiles) and reports (figures).

Every exported figure carries a legend/colour bar with units, a scale bar
only when the grid is calibrated, an orientation indicator (the 0°E meridian
direction for south polar stereographic grids), and source attribution.
"""
from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from core.provenance import LayerMeta  # noqa: E402

CMAPS = {
    "terrain": "cividis", "slope": LinearSegmentedColormap.from_list("slope", ["#1d3b2a", "#3f8f5a", "#d9c35a", "#e0752d", "#b3203a"]),
    "magma": "magma", "inferno": "inferno", "gray": "gray", "viridis": "viridis",
    "psr": LinearSegmentedColormap.from_list("psr", ["#0b0e14", "#1f3b73", "#4fa3ff"]),
    "thermal": "coolwarm", "depth": "viridis_r",
    "priority": LinearSegmentedColormap.from_list("prio", ["#0f1830", "#2b4a7a", "#3f8fc0", "#9fe3ff"]),
    "cost": LinearSegmentedColormap.from_list("cost", ["#16302a", "#5b8a3c", "#d9b43a", "#c85a2d", "#7a1f2a"]),
    "flag": ListedColormap(["#00000000", "#ff4fd8"]), "relative": "Greys_r",
}
NODATA_RGBA = (40, 42, 48, 255)


def _cmap(meta: LayerMeta):
    if meta.categorical:
        n = max(int(k) for k in meta.categorical) + 1
        cols = ["#000000"] * n
        for k, v in meta.categorical.items():
            cols[int(k)] = v["color"]
        return ListedColormap(cols), 0, n - 1
    lo, hi = meta.value_range or (0.0, 1.0)
    c = CMAPS.get(meta.colormap, meta.colormap)
    return (plt.get_cmap(c) if isinstance(c, str) else c), lo, hi


def colorize(arr: np.ndarray, meta: LayerMeta) -> np.ndarray:
    """Array -> RGBA uint8 using the layer's colour map; NaN -> neutral no-data grey."""
    cmap, lo, hi = _cmap(meta)
    if meta.categorical:
        idx = np.nan_to_num(arr, nan=0).astype(int)
        rgba = (cmap(idx) * 255).astype(np.uint8)
    else:
        norm = np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1)
        rgba = (cmap(np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[~np.isfinite(arr)] = NODATA_RGBA
    if meta.colormap == "flag":
        rgba[np.isfinite(arr) & (arr < 0.5)] = (0, 0, 0, 0)
    return rgba


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=False)
    return buf.getvalue()


def legend_spec(meta: LayerMeta) -> dict[str, Any]:
    if meta.categorical:
        return {"type": "categorical", "items": [{"value": int(k), **v} for k, v in meta.categorical.items()]}
    cmap, lo, hi = _cmap(meta)
    stops = [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, 8)]
    return {"type": "continuous", "min": lo, "max": hi, "units": meta.units, "stops": stops}


def map_figure(arr: np.ndarray, meta: LayerMeta, res_m: float | None, polar: bool, attribution: str,
               route: list[tuple[int, int]] | None = None, alt_routes: dict[str, list] | None = None,
               start: tuple[int, int] | None = None, goal: tuple[int, int] | None = None,
               hillshade: np.ndarray | None = None, title: str | None = None, label: str | None = None) -> bytes:
    """Publication-style PNG of one layer with route overlay, legend, scale and orientation."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    h, w = arr.shape
    if hillshade is not None and not meta.categorical:
        ax.imshow(hillshade, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    rgba = colorize(arr, meta)
    if hillshade is not None:
        rgba = rgba.copy()
        rgba[..., 3] = np.where(np.isfinite(arr), 200, 255)
    ax.imshow(rgba, interpolation="nearest")
    if meta.categorical:
        from matplotlib.patches import Patch
        handles = [Patch(color=v["color"], label=v["label"]) for v in meta.categorical.values()]
        ax.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.85)
    else:
        cmap, lo, hi = _cmap(meta)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"{meta.title} [{meta.units}]", fontsize=8)
    colors = {"distance_only": "#ffffff", "direct_line": "#ff5470"}
    for name, cells in (alt_routes or {}).items():
        if cells:
            ax.plot([c[1] for c in cells], [c[0] for c in cells], "--", color=colors.get(name, "#cccccc"), lw=1.0,
                    label=name.replace("_", " "))
    if route:
        ax.plot([c[1] for c in route], [c[0] for c in route], "-", color="#00e5ff", lw=2.0, label="candidate route (A*)")
    if start:
        ax.plot(start[1], start[0], "o", ms=7, mfc="#00e5ff", mec="k")
    if goal:
        ax.plot(goal[1], goal[0], "*", ms=11, mfc="#ffd400", mec="k")
    if route or alt_routes:
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)
        if meta.categorical:  # keep both legends
            from matplotlib.patches import Patch
            ax.add_artist(ax.legend(handles=[Patch(color=v["color"], label=v["label"]) for v in meta.categorical.values()],
                                    loc="lower left", fontsize=6.5, framealpha=0.85))
            ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)
    if res_m:
        km = _nice_length(w * res_m / 5.0)
        px = km / res_m
        x0, y0 = w * 0.95 - px, h * 0.95
        ax.plot([x0, x0 + px], [y0, y0], color="w", lw=3, solid_capstyle="butt")
        ax.plot([x0, x0 + px], [y0, y0], color="k", lw=1, solid_capstyle="butt")
        ax.text(x0 + px / 2, y0 - h * 0.015, f"{km / 1000:g} km" if km >= 1000 else f"{km:g} m", color="w", ha="center",
                fontsize=7, weight="bold")
    else:
        ax.text(w * 0.95, h * 0.95, "no scale: grid not calibrated", color="w", fontsize=7, ha="right")
    if polar:
        ax.annotate("", xy=(w * 0.93, h * 0.70), xytext=(w * 0.93, h * 0.80),
                    arrowprops=dict(arrowstyle="-|>", color="w", lw=1.5))
        ax.text(w * 0.93, h * 0.825, "0°E\nmeridian", color="w", fontsize=6, ha="center", va="top")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or meta.title, fontsize=10)
    foot = f"Source: {attribution}  |  Status: {meta.status.value}"
    if label:
        foot += f"  |  {label}"
    fig.text(0.01, 0.01, foot, fontsize=5.5, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _nice_length(m: float) -> float:
    exp = 10 ** np.floor(np.log10(max(m, 1e-9)))
    for f in (5, 2, 1):
        if f * exp <= m:
            return float(f * exp)
    return float(exp)
