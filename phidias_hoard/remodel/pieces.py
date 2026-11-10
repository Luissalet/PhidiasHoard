"""Taking a game model apart: its connected pieces, each with its position, its profile (the size of its outline at
every height it has corners), the materials on it and how many times their textures repeat across it. Pieces of the
same shape are told to be instances of one type. A first guess says what each piece is."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..engine.meshio import Scene

KINDS = ("roof", "body", "block", "lathe", "stairs", "keep", "drop")


def split(scene: Scene, scale: float, dropped: set[str]) -> list[dict[str, Any]]:
    """Pieces in millimetres (``scale`` mm per model unit), lowest first."""
    n, F = len(scene.P), scene.F
    P = (scene.P - scene.P.min(0)) * scale
    _, label = connected_components(coo_matrix((np.ones(len(F) * 2, np.int8), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])),
                                               shape=(n, n)), directed=False)
    piece_of = label[F[:, 0]]
    names = [m.name for m in scene.materials]
    areas = scene.face_areas() * scale * scale
    out = []
    for k in np.unique(piece_of):
        sel = piece_of == k
        f = F[sel]
        used = np.unique(f)
        pts = P[used]
        lo, hi = pts.min(0), pts.max(0)
        size = hi - lo
        tol = max(float(size[2]) * 0.01, 1e-4)
        zs = np.sort(pts[:, 2])
        levels = [zs[0]]
        for z in zs[1:]:
            if z - levels[-1] > tol:
                levels.append(z)
        profile = []
        for z in levels:
            q = pts[np.abs(pts[:, 2] - z) <= tol]
            profile.append([round(float(z), 4), round(float(np.ptp(q[:, 0])), 4), round(float(np.ptp(q[:, 1])), 4)])
        mats: dict[str, dict[str, Any]] = {}
        for i in np.unique(scene.M[sel]):
            ms = sel & (scene.M == i)
            uv = scene.UV[ms].reshape(-1, 2)
            mats[names[i]] = {"share": round(float(areas[ms].sum() / max(areas[sel].sum(), 1e-12)), 3),
                              "repeat_u": round(float(np.ptp(uv[:, 0])), 2), "repeat_v": round(float(np.ptp(uv[:, 1])), 2)}
        remap = {int(v): j for j, v in enumerate(used)}
        piece = {"id": int(k), "triangles_count": int(sel.sum()), "center": [round(float(v), 4) for v in ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])],
                 "size": [round(float(v), 4) for v in size], "profile": profile, "materials": mats,
                 "points": np.round(pts, 4).tolist(), "triangles": [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in f],
                 "dropped": bool(mats) and all(name in dropped for name in mats)}
        piece["signature"] = f'{piece["triangles_count"]}:{"x".join(f"{v:.1f}" for v in size)}:{"+".join(sorted(mats))}'
        out.append(piece)
    out.sort(key=lambda p: (p["center"][2], p["id"]))
    types: dict[str, int] = {}
    for p in out:
        p["type"] = types.setdefault(p["signature"], len(types) + 1)
    return out


def guess(piece: dict[str, Any], size_mm: float) -> tuple[str, dict[str, Any]]:
    """What a piece is, from its shape alone."""
    if piece["dropped"]:
        return "drop", {}
    prof = piece["profile"]
    sx, sy, sz = piece["size"]
    wide = max(sx, sy, 1e-9)
    if len(prof) < 2 or min(sx, sy) < 1e-6:
        return "keep", {}
    first, last = prof[0], prof[-1]
    radii = [max(r[1], r[2]) for r in prof]
    squarish = all(abs(r[1] - r[2]) <= 0.25 * max(r[1], r[2], 1e-9) for r in prof if max(r[1], r[2]) > 0.05 * wide)
    ups = sum(1 for a, b in zip(radii, radii[1:]) if b > a * 1.05)
    downs = sum(1 for a, b in zip(radii, radii[1:]) if b < a * 0.95)
    if squarish and len(prof) >= 5 and ups >= 2 and downs >= 2:
        return "lathe", {"segments": 24}
    if squarish and sz > 2.2 * wide and len(prof) >= 3:
        return "lathe", {"segments": 16}
    shrink = min(radii[1:]) / max(radii[0], 1e-9)
    if shrink < 0.8 and radii[0] >= max(radii) * 0.999 and sz < 0.9 * wide:
        return "roof", {"tiles": int(np.clip(round(wide / 2.4), 8, 36)), "rows": 6}
    constant = max(radii) - min(radii) < 0.02 * wide
    if constant and sz > 0.18 * wide:
        return "body", {"bays_x": 3, "bays_y": 3, "zones": ["panel"] * (len(prof) - 1)}
    return "block", {"bevel": round(float(np.clip(size_mm / 600.0, 0.1, 0.6)), 2)}
