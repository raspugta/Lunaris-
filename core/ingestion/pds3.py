"""Minimal PDS3 label (ODL) parser sufficient for LOLA GDR IMG products.

Only the subset of ODL used by LOLA/LRO labels is supported: ``KEY = VALUE``
pairs, nested ``OBJECT``/``END_OBJECT`` blocks, quoted multi-line strings,
``<unit>`` suffixes and ``{a, b}`` sets. Unknown constructs raise ValueError
rather than being guessed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_UNIT_RE = re.compile(r"^(?P<val>.+?)\s*<(?P<unit>[^>]+)>$")


def _coerce(raw: str) -> Any:
    raw = raw.strip()
    m = _UNIT_RE.match(raw)
    unit = None
    if m:
        raw, unit = m.group("val").strip(), m.group("unit").strip()
    if raw.startswith('"') and raw.endswith('"'):
        val: Any = " ".join(raw[1:-1].split())
    elif raw.startswith("'") and raw.endswith("'"):
        val = raw[1:-1]
    elif raw.startswith("{") or raw.startswith("("):
        inner = raw[1:-1]
        val = [_coerce(x) for x in inner.split(",") if x.strip()]
    else:
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
    return {"value": val, "unit": unit} if unit else val


def parse_label(text: str) -> dict[str, Any]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    root: dict[str, Any] = {}
    stack = [root]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "END":
            break
        if "=" not in line:
            raise ValueError(f"unparseable PDS3 line: {line!r}")
        key, val = (s.strip() for s in line.split("=", 1))
        # multi-line quoted strings / sets
        opener = val[:1]
        closer = {'"': '"', "{": "}", "(": ")"}.get(opener)
        if closer and (val.count(closer) < (2 if closer == '"' else 1)):
            while i < len(lines) and val.count(closer) < (2 if closer == '"' else 1):
                val += " " + lines[i].strip()
                i += 1
        if key == "OBJECT":
            child: dict[str, Any] = {}
            stack[-1].setdefault(val, child)
            stack.append(child)
        elif key == "END_OBJECT":
            if len(stack) == 1:
                raise ValueError("unbalanced END_OBJECT")
            stack.pop()
        else:
            stack[-1][key] = _coerce(val)
    return root


def read_label(path: str | Path) -> dict[str, Any]:
    return parse_label(Path(path).read_text(encoding="latin-1"))


def value(v: Any) -> Any:
    return v["value"] if isinstance(v, dict) and "value" in v else v
