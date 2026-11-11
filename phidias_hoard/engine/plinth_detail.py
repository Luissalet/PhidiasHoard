"""Lay out the engraving and figure recess outside the sculpt's footprint."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from . import export, finish
from .sculpt import Sculpted, SculptSettings, plinth


SCRIPT = Path(__file__).with_name("blender_plinth.py")


def decorate(scene: Sculpted, settings: SculptSettings, blender: str, folder: Path):
    lo, hi = scene.P.min(0), scene.P.max(0)
    margin = max(2.0, settings.plinth_margin_mm)
    base = settings.base_mm if settings.ground else 0.0
    depth = settings.plinth_socket_depth_mm
    top = float(lo[2]) - base + 0.01
    cuts = {}
    if settings.plinth_socket_mm:
        radius = settings.plinth_socket_mm / 2
        cy = float((lo[1] + hi[1]) / 2)
        cx = float(hi[0]) + margin + radius
        hi[0] = cx + radius
        lo[1], hi[1] = min(lo[1], cy - radius), max(hi[1], cy + radius)
        cuts["socket"] = {"center": [cx, cy], "radius": radius, "depth": depth}
    if settings.plinth_name:
        font_size = min(5.0, max(2.4, (hi[0] - lo[0]) / 12))
        band = 1.6 * font_size + 2
        cuts["engraving"] = {"name": settings.plinth_name, "font_size": font_size,
                             "center": [float((lo[0] + hi[0]) / 2), float(lo[1] - band / 2)],
                             "max_width": float(hi[0] - lo[0])}
        lo[1] -= band
    P, F = plinth(scene, settings.plinth, margin, base, max(2.0, depth + 1.0), bounds=(lo, hi))
    with tempfile.TemporaryDirectory(prefix="plinth-", dir=folder) as temp:
        root = Path(temp)
        export.write_ply(root / "plate.ply", P, F)
        (root / "cuts.json").write_text(json.dumps({"top": top, **cuts}), encoding="utf-8")
        result = subprocess.run(
            [blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPT), "--", str(root)],
            capture_output=True, text=True, errors="replace", timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = root / "plate.stl"
        if result.returncode or not output.is_file():
            log = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-8:])
            raise RuntimeError(f"Blender no pudo grabar la peana.\n{log}")
        return finish.read_stl(output)
