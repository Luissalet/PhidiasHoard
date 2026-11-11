"""Closing a sculpt into one watertight solid with Blender running in the background."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT = Path(__file__).with_name("blender_close.py")
IMPORT_SCRIPT = Path(__file__).with_name("blender_import.py")


def find_blender(configured: str = "") -> Optional[str]:
    if configured and Path(configured).is_file():
        return configured
    env = os.environ.get("PHIDIAS_BLENDER") or os.environ.get("BLENDER")
    if env and Path(env).is_file():
        return env
    found = shutil.which("blender")
    if found:
        return found
    if sys.platform == "win32":
        roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"]
        best: list[tuple[tuple[int, ...], str]] = []
        for root in roots:
            if not root.is_dir():
                continue
            for d in root.iterdir():
                exe = d / "blender.exe"
                if exe.is_file():
                    nums = tuple(int(x) for x in "".join(c if c.isdigit() else " " for c in d.name).split())
                    best.append((nums, str(exe)))
        if best:
            return sorted(best)[-1][1]
    for cand in ("/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"):
        if Path(cand).is_file():
            return cand
    return None


def import_model(blender: str, source: Path, glb: Path, timeout_s: float = 180) -> None:
    cmd = [blender, "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python",
           str(IMPORT_SCRIPT), "--", str(source.resolve()), str(glb.resolve())]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    run = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s, creationflags=flags)
    if run.returncode or not glb.is_file():
        tail = "\n".join((run.stdout + "\n" + run.stderr).strip().splitlines()[-10:])
        raise RuntimeError(f"Blender did not export the model.\n{tail}")


def close_mesh(blender: str, ply_in: Path, stl_out: Path, voxel_mm: float, shell_mm: float, max_faces: int,
               timeout_s: float = 3600) -> dict:
    cmd = [blender, "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(ply_in), str(stl_out),
           repr(float(voxel_mm)), repr(float(shell_mm)), str(int(max_faces))]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, creationflags=flags)
    marks = [line for line in (run.stdout or "").splitlines() if line.startswith("PHIDIAS ")]
    if run.returncode != 0 or not Path(stl_out).is_file():
        tail = "\n".join(((run.stdout or "") + "\n" + (run.stderr or "")).strip().splitlines()[-12:])
        raise RuntimeError(f"Blender did not write the closed mesh.\n{tail}")
    return {"log": marks}


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A binary STL back as welded points and triangles."""
    from .meshio import weld
    with open(path, "rb") as fh:
        fh.seek(80)
        count = int(np.frombuffer(fh.read(4), "<u4")[0])
        rec = np.fromfile(fh, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")], count=count)
    pts = rec["v"].reshape(-1, 3).astype(np.float64)
    ext = float((pts.max(0) - pts.min(0)).max()) if len(pts) else 1.0
    P, inv = weld(pts, max(ext, 1e-9) * 1e-6)
    return P, inv.reshape(-1, 3).astype(np.int32)
