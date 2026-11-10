"""Relief from a normal map, integrated on the mesh itself.

A normal map says how the surface tilts at every point. Following those tilts along the edges of the refined mesh and
solving for the height at every point that fits them best gives the relief the artist painted, and it works whatever
the UV layout is: tiling textures, mirrored halves or a baked atlas where every triangle has its own island."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from . import heights


def slope_maps(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dh/du and dh/dv (per unit of surface length) from a tangent-space normal map."""
    n = normal.astype(np.float32) / 127.5 - 1.0
    nz = np.clip(n[..., 2], 0.2, None)
    return np.clip(-n[..., 0] / nz, -4, 4).astype(np.float32), np.clip(-n[..., 1] / nz, -4, 4).astype(np.float32)


class SlopeDetail:
    """Where a normal map changes inside a triangle (so the triangle needs more triangles)."""

    def __init__(self, sx: np.ndarray, sy: np.ndarray):
        self.px, self.py = heights.RangePyramid(sx), heights.RangePyramid(sy)

    def variation(self, uv: np.ndarray) -> np.ndarray:
        return np.maximum(self.px.variation(uv), self.py.variation(uv))


def integrate(P: np.ndarray, F: np.ndarray, UV: np.ndarray, sx: np.ndarray, sy: np.ndarray, cutoff: float,
              iterations: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Heights (same units as ``P``) at the points used by ``F`` that best follow the slopes. Undulation wider than
    ``cutoff`` is damped, so the relief stays detail and never bends walls. Returns ``(point indices, heights)``."""
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    uv = UV.astype(np.float64)
    du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
    du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
    det = du1 * dv2 - du2 * dv1
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    T = (e1 * dv2[:, None] - e2 * dv1[:, None]) * inv[:, None]
    B = (e2 * du1[:, None] - e1 * du2[:, None]) * inv[:, None]
    tl, bl = np.linalg.norm(T, axis=1, keepdims=True), np.linalg.norm(B, axis=1, keepdims=True)
    ok &= (tl[:, 0] > 1e-12) & (bl[:, 0] > 1e-12)
    T = np.divide(T, tl, out=np.zeros_like(T), where=tl > 1e-12)
    B = np.divide(B, bl, out=np.zeros_like(B), where=bl > 1e-12)
    flat = UV.reshape(-1, 2)
    gx = heights.sample(sx, flat).reshape(-1, 3).astype(np.float64)
    gy = heights.sample(sy, flat).reshape(-1, 3).astype(np.float64)

    used, compact = np.unique(F, return_inverse=True)
    C = compact.reshape(-1, 3)
    n = len(used)
    rows_i, rows_j, delta = [], [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = P[F[:, b]] - P[F[:, a]]
        d = 0.5 * (gx[:, a] + gx[:, b]) * np.einsum("ij,ij->i", e, T) + 0.5 * (gy[:, a] + gy[:, b]) * np.einsum("ij,ij->i", e, B)
        rows_i.append(C[ok, a]); rows_j.append(C[ok, b]); delta.append(d[ok])
    i, j, d = np.concatenate(rows_i), np.concatenate(rows_j), np.concatenate(delta)
    if not len(i):
        return used, np.zeros(n)
    ones = np.ones(len(i))
    lap = sparse.coo_matrix((np.concatenate([ones, ones, -ones, -ones]),
                             (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))), shape=(n, n)).tocsr()
    rhs = np.bincount(j, weights=d, minlength=n) - np.bincount(i, weights=d, minlength=n)
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    mass = np.zeros(n)
    for k in range(3):
        mass += np.bincount(C[:, k], weights=area / 3.0, minlength=n)
    lam = 1.0 / max(cutoff, 1e-9) ** 2
    A = lap + sparse.diags(mass * lam + 1e-12)
    diag = A.diagonal()
    pre = sparse.diags(1.0 / np.maximum(diag, 1e-12))
    h, _ = cg(A, rhs, M=pre, rtol=1e-4, maxiter=iterations)
    return used, h
