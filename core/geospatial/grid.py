"""Grid geometry: pixel <-> projected <-> planetocentric coordinates.

Conventions
-----------
* Pixel indices are ``(row, col)`` with ``(0, 0)`` the top-left cell; the
  world coordinate of a cell is its *centre*.
* Projected coordinates follow the affine transform of the raster
  (GDAL convention: ``x = c + col*a + row*b``, ``y = f + col*d + row*e``).
* For lunar south-polar products the CRS is the IAU:30135-style spherical
  polar stereographic projection (R = 1737.4 km, true scale at the pole,
  central meridian 0 deg). For that CRS:

      rho = 2 R tan(pi/4 + lat/2)        (lat negative, radians)
      x = rho sin(lon),  y = rho cos(lon)

  so +y (raster "up") points along the 0 deg E meridian and +x along 90 deg E.
  At the pole every horizontal direction is geographic north, so figures label
  the 0 deg E meridian direction rather than a single "north" arrow.
* The map scale factor is k = 2 / (1 + sin|lat|); within 2 deg of the pole
  k - 1 < 3.1e-4, so grid spacing is used as ground distance with < 0.04 %
  error. This bound is checked in :func:`max_scale_error`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MOON_RADIUS_M = 1_737_400.0


@dataclass(frozen=True)
class GridSpec:
    """Regular north-up grid. ``transform`` is a 6-tuple (a, b, c, d, e, f)."""

    width: int
    height: int
    transform: tuple[float, float, float, float, float, float]
    crs_wkt: str | None = None
    polar_stereo_south: bool = False
    calibrated: bool = False  # True when pixel size is a physical ground distance

    @property
    def pixel_size(self) -> tuple[float, float]:
        a, b, _, d, e, _ = self.transform
        return math.hypot(a, d), math.hypot(b, e)

    @property
    def res_m(self) -> float | None:
        if not self.calibrated:
            return None
        px, py = self.pixel_size
        return 0.5 * (px + py)

    def pixel_to_world(self, row: np.ndarray | float, col: np.ndarray | float):
        a, b, c, d, e, f = self.transform
        r = np.asarray(row, dtype=float) + 0.5
        cc = np.asarray(col, dtype=float) + 0.5
        return c + cc * a + r * b, f + cc * d + r * e

    def world_to_pixel(self, x: np.ndarray | float, y: np.ndarray | float):
        a, b, c, d, e, f = self.transform
        det = a * e - b * d
        if det == 0:
            raise ValueError("singular transform")
        x = np.asarray(x, dtype=float) - c
        y = np.asarray(y, dtype=float) - f
        col = (e * x - b * y) / det - 0.5
        row = (-d * x + a * y) / det - 0.5
        return row, col

    def downsampled(self, factor: int) -> "GridSpec":
        a, b, c, d, e, f = self.transform
        return GridSpec(
            width=self.width // factor, height=self.height // factor,
            transform=(a * factor, b * factor, c, d * factor, e * factor, f),
            crs_wkt=self.crs_wkt, polar_stereo_south=self.polar_stereo_south,
            calibrated=self.calibrated,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform
        x0, x1 = c, c + a * self.width
        y0, y1 = f, f + e * self.height
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height, "transform": list(self.transform),
            "polar_stereo_south": self.polar_stereo_south, "calibrated": self.calibrated,
            "res_m": self.res_m, "bounds": list(self.bounds()),
        }


def polar_stereo_south_forward(lat_deg, lon_deg, radius: float = MOON_RADIUS_M):
    """Planetocentric (lat, east lon) in degrees -> south polar stereographic (x, y) metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    rho = 2.0 * radius * np.tan(np.pi / 4.0 + lat / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def polar_stereo_south_inverse(x, y, radius: float = MOON_RADIUS_M):
    """South polar stereographic (x, y) metres -> planetocentric (lat, east lon) degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    lat = 2.0 * np.arctan(rho / (2.0 * radius)) - np.pi / 2.0
    lon = np.degrees(np.arctan2(x, y)) % 360.0
    return np.degrees(lat), lon


def scale_factor(lat_deg) -> np.ndarray:
    """Point scale factor of the polar stereographic projection (true at pole)."""
    s = np.abs(np.sin(np.radians(np.asarray(lat_deg, dtype=float))))
    return 2.0 / (1.0 + s)


def max_scale_error(grid: GridSpec) -> float | None:
    """Largest |k-1| over the grid corners (None for non-polar grids)."""
    if not grid.polar_stereo_south:
        return None
    l, b, r, t = grid.bounds()
    xs = np.array([l, r, l, r])
    ys = np.array([b, b, t, t])
    lat, _ = polar_stereo_south_inverse(xs, ys)
    return float(np.max(np.abs(scale_factor(lat) - 1.0)))


def is_polar_stereo_south(crs_wkt: str | None) -> bool:
    if not crs_wkt:
        return False
    w = crs_wkt.lower()
    return ("polar_stereographic" in w or "polar stereographic" in w or "stere" in w) and "-90" in w and (
        "moon" in w or "1737400" in w)
