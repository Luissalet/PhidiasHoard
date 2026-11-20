"""Closing a sculpt into one watertight solid with Blender running in the background."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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


def close_mesh(blender: str, ply_in: Path, output: Path, voxel_mm: float, shell_mm: float, max_faces: int,
               timeout_s: float = 3600, *, bridges: tuple = ()) -> dict:
    cmd = [blender, "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(ply_in), str(output),
           repr(float(voxel_mm)), repr(float(shell_mm)), str(int(max_faces))]
    if bridges:
        recipe = output.with_name("bridges.json")
        recipe.write_text(json.dumps(bridges), encoding="utf-8")
        cmd.append(str(recipe))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, creationflags=flags)
    marks = [line for line in (run.stdout or "").splitlines() if line.startswith("PHIDIAS ")]
    if run.returncode != 0 or not output.is_file():
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


def _components(P: np.ndarray, F: np.ndarray):
    return connected_components(
        coo_matrix((np.ones(2 * len(F), dtype=bool), (F[:, :2].reshape(-1), F[:, 1:].reshape(-1))),
                   shape=(len(P), len(P))), directed=False)


def bridge_spans(points: np.ndarray, faces: np.ndarray, max_gap: float) -> tuple[list, int]:
    """Nearest points of each nearby shell and the main solid, in print millimetres."""
    count, labels = _components(points, faces)
    if count <= 1:
        return [], count
    counts = np.bincount(labels)
    main_label = counts.argmax()
    main = points[labels == main_label]
    tree = cKDTree(main)
    bridges = []
    for label in np.flatnonzero((counts > 1) & (np.arange(count) != main_label)):
        part = points[labels == label]
        distances, matches = tree.query(part, distance_upper_bound=max_gap)
        nearest = distances.argmin()
        if 0 < distances[nearest] <= max_gap:
            bridges.append([main[matches[nearest]].tolist(), part[nearest].tolist()])
    return bridges, count


def fill_cavities(P: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop inward-facing closed shells left by the remesh and any islands they enclose."""
    from .sculpt import open_edges

    count, labels = _components(P, F)
    if count < 2:
        return P, F, 0
    component = labels[F[:, 0]]
    volume = np.zeros(count)
    for start in range(0, len(F), 200_000):
        indices = slice(start, start + 200_000)
        a, b, c = P[F[indices]].transpose(1, 0, 2)
        signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        volume += np.bincount(component[indices], signed, minlength=count)
    eps = 1e-12 * float(np.ptp(P, axis=0).max()) ** 3
    inward = np.flatnonzero(volume < -eps)
    outward = np.flatnonzero(volume > eps)
    if not len(inward) or not len(outward):
        return P, F, 0

    lo = np.full((count, 3), np.inf)
    hi = np.full((count, 3), -np.inf)
    np.minimum.at(lo, labels, P)
    np.maximum.at(hi, labels, P)
    removed = np.zeros(count, dtype=bool)
    cavities = 0
    for cavity in inward:
        if not np.any(np.all((lo[outward] < lo[cavity]) & (hi[cavity] < hi[outward]), axis=1)):
            continue
        shell = F[component == cavity]
        if open_edges(shell, len(P)):
            continue
        removed[cavity] = True
        cavities += 1
        islands = outward[np.all((lo[cavity] < lo[outward]) & (hi[outward] < hi[cavity]), axis=1)]
        for island in islands:
            if not removed[island] and _inside_shell(P[np.flatnonzero(labels == island)[0]], P[shell]):
                removed[island] = True
    if not cavities:
        return P, F, 0
    keep = ~removed[labels]
    remap = np.cumsum(keep, dtype=np.int32) - 1
    return P[keep], remap[F[~removed[component]]], cavities


def _inside_shell(point: np.ndarray, triangles: np.ndarray) -> bool:
    """A closed shell subtends 4π steradians from an interior point and zero from outside."""
    a, b, c = (triangles - point).transpose(1, 0, 2)
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numer = np.einsum("ij,ij->i", a, np.cross(b, c))
    denom = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
             + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb)
    return bool(abs(np.arctan2(numer, denom).sum()) > np.pi)
