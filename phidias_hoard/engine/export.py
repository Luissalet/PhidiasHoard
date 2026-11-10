"""Writing meshes: binary STL (millimetres) for the slicer and compact binary buffers for the viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_stl(path: Path, P: np.ndarray, F: np.ndarray, name: str = "phidias") -> int:
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, length, out=np.zeros_like(n), where=length > 1e-20)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = np.stack([a, b, c], axis=1)
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:79].ljust(80, b" "))
        fh.write(np.uint32(len(F)).tobytes())
        rec.tofile(fh)
    return int(Path(path).stat().st_size)


def write_ply(path: Path, P: np.ndarray, F: np.ndarray) -> None:
    """Binary PLY with shared points (what Blender reads fastest)."""
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
                  f"property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        P.astype("<f4").tofile(fh)
        rec = np.zeros(len(F), dtype=[("k", "u1"), ("v", "<i4", 3)])
        rec["k"] = 3
        rec["v"] = F
        rec.tofile(fh)


def viewer_buffers(P: np.ndarray, F: np.ndarray) -> bytes:
    """``uint32 points, uint32 triangles, float32 xyz..., uint32 indices...`` for the three.js viewer."""
    return (np.array([len(P), len(F)], "<u4").tobytes() + P.astype("<f4").tobytes() + F.astype("<u4").tobytes())
