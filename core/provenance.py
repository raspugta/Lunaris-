"""Provenance primitives shared by every layer in the platform.

Every raster surfaced to a user carries a :class:`LayerMeta` that states what it
measures, its units, resolution, source, processing method, limitations, and —
crucially — its *epistemic status*:

* ``measured``      – an instrument observation, minimally transformed.
* ``derived``       – computed from measured data by a documented, deterministic method
                      (includes model products published by instrument teams).
* ``model_estimate``– output of a learned model or heuristic score; not a measurement.
* ``illustrative``  – visualisation aid only; carries no quantitative meaning.
* ``synthetic``     – generated demo data; does not describe the Moon.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    MODEL_ESTIMATE = "model_estimate"
    ILLUSTRATIVE = "illustrative"
    SYNTHETIC = "synthetic"


@dataclass
class LayerMeta:
    key: str
    title: str
    status: Status
    units: str
    description: str
    source: str
    method: str
    limitations: list[str] = field(default_factory=list)
    resolution_m: float | None = None
    colormap: str = "viridis"
    value_range: tuple[float, float] | None = None
    categorical: dict[int, dict[str, str]] | None = None  # value -> {label, color}
    available: bool = True
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        if self.categorical is not None:
            d["categorical"] = {str(k): v for k, v in self.categorical.items()}
        return d


def unavailable(key: str, title: str, reason: str, needs: str) -> LayerMeta:
    """A placeholder for a layer that cannot be produced from the loaded data."""
    return LayerMeta(
        key=key, title=title, status=Status.DERIVED, units="-", description=needs,
        source="not ingested", method="-", limitations=[reason], available=False,
        unavailable_reason=reason,
    )
