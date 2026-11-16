"""Turn a high-resolution result into a game asset with Blender: retopology, UVs, baked normals and occlusion, glTF/FBX."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from . import export

SCRIPT = Path(__file__).with_name("blender_asset.py")
FORMATS = ("glb", "fbx")
FILES = {"glb": "asset.glb", "fbx": "asset.fbx"}
TEXTURES = ("asset_normal.png", "asset_ao.png")


def create(P: np.ndarray, F: np.ndarray, spec: dict, folder: Path, blender: str, name: str = "asset") -> dict:
    """Run the pipeline in ``folder``; returns the report and leaves the models, the textures and the low mesh there."""
    formats = [f for f in FORMATS if f in spec["formats"]]
    if not formats:
        raise ValueError("Elige al menos un formato de salida: glb o fbx.")
    export.write_ply(folder / "high.ply", P, F)
    job = {"target_faces": int(spec["target_faces"]), "method": spec["method"], "texture_px": int(spec["texture_px"]),
           "ao_samples": int(spec["ao_samples"]), "size_m": float(spec["size_m"]), "formats": formats,
           "cage_pct": float(spec["cage_pct"]), "uv_angle": float(spec["uv_angle"]),
           "name": "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "asset"}
    (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
    run = subprocess.run([blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPT), "--",
                          str(folder.resolve())], capture_output=True, text=True, errors="replace", timeout=7200,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    wanted = [folder / FILES[f] for f in formats] + [folder / t for t in TEXTURES] + [folder / "report.json", folder / "low_P.npy"]
    if run.returncode or any(not path.is_file() for path in wanted):
        detail = "\n".join((run.stderr + "\n" + run.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"Blender no pudo preparar el asset.\n{detail}")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    report["spec"] = {k: spec[k] for k in ("target_faces", "method", "texture_px", "ao_samples", "size_m", "formats", "cage_pct", "uv_angle")}
    return report
