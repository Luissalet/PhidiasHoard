"""Run the print split and publish the validated parts and their 3MF package."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from . import export, finish, three_mf
from .refine import edge_ids


SCRIPT = Path(__file__).with_name("blender_split.py")


def _manifold(points, faces):
    _, ids = edge_ids(faces, len(points))
    return len(faces) > 0 and np.all(np.bincount(ids.reshape(-1)) == 2)


def create(P, F, spec, folder, blender):
    if not _manifold(P, F):
        raise ValueError("La malla tiene que ser estanca para hacer el corte y los encajes.")
    axis = "xyz".index(spec["axis"])
    extent = float(np.ptp(P[:, axis]))
    position = spec["position_mm"] if spec["position_mm"] is not None else extent / 2
    if not 0.1 < position < extent - 0.1:
        raise ValueError(f"El plano tiene que cortar dentro de la pieza (0–{extent:.2f} mm).")
    (folder / "cut.json").write_text(json.dumps({"axis": axis, "position": position + float(P[:, axis].min()),
                                               "diameter": spec["diameter_mm"], "depth": spec["depth_mm"],
                                               "clearance": spec["clearance_mm"], "pins": spec["pins"]}), encoding="utf-8")
    export.write_ply(folder / "source.ply", P, F)
    run = subprocess.run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPT),
                          "--", str(folder.resolve())], capture_output=True, text=True, errors="replace", timeout=1800,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    files = [folder / "piece-a.stl", folder / "piece-b.stl"]
    if run.returncode or any(not f.is_file() for f in files):
        detail = "\n".join((run.stderr + "\n" + run.stdout).strip().splitlines()[-12:])
        raise ValueError(f"Blender no pudo partir la malla.\n{detail}")
    parts = []
    offset = 0.0
    for label, file in zip(("A", "B"), files):
        points, faces = finish.read_stl(file)
        if not _manifold(points, faces):
            raise ValueError(f"La parte {label} salió abierta o vacía. Prueba otro plano de corte.")
        points -= points.min(0)
        export.write_stl(file, points, faces, f"Phidias {label}")
        layout = points.copy()
        layout[:, 0] += offset
        offset += float(np.ptp(points[:, 0])) + 5
        parts.append((f"Pieza {label}", layout, faces, None))
    three_mf.write(folder / "pieces.3mf", parts, [("Material", (205, 197, 182))])
    return {**spec, "position_mm": position, "triangles": [len(p[2]) for p in parts],
            "size_mm": [[round(float(d), 2) for d in np.ptp(p[1], axis=0)] for p in parts]}
