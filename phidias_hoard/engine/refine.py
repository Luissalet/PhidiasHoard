"""Adaptive refinement: triangles are split only where the relief needs them, and neighbours always agree on which
edges are split, so the mesh never opens (no T-junctions)."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def edge_ids(F: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges (k, 2) and, for each triangle, the ids of its edges v0-v1, v1-v2, v2-v0."""
    a = F[:, [0, 1, 2]].astype(np.int64)
    b = F[:, [1, 2, 0]].astype(np.int64)
    key = np.minimum(a, b) * np.int64(n_points) + np.maximum(a, b)
    uniq, inverse = np.unique(key.reshape(-1), return_inverse=True)
    edges = np.stack([uniq // n_points, uniq % n_points], axis=1)
    return edges, inverse.reshape(-1, 3)


def split_once(P: np.ndarray, F: np.ndarray, UV: np.ndarray, parent: np.ndarray, want: np.ndarray, min_edge: float,
               max_cuts: int = 0):
    """One pass: every edge longer than ``min_edge`` that touches a triangle with ``want`` is cut in half (only the
    ``max_cuts`` longest when there are more)."""
    edges, fe = edge_ids(F, len(P))
    length = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
    touched = np.zeros(len(edges), bool)
    touched[fe[want].reshape(-1)] = True
    cut = touched & (length > min_edge)
    if max_cuts and int(cut.sum()) > max_cuts:
        which = np.nonzero(cut)[0]
        longest = which[np.argpartition(length[which], -max_cuts)[-max_cuts:]]
        cut = np.zeros(len(edges), bool)
        cut[longest] = True
    if not cut.any():
        return P, F, UV, parent, 0
    new_index = np.full(len(edges), -1, np.int64)
    new_index[cut] = len(P) + np.arange(int(cut.sum()))
    P2 = np.concatenate([P, 0.5 * (P[edges[cut, 0]] + P[edges[cut, 1]])])
    flags = cut[fe]                                       # (m, 3)
    count = flags.sum(axis=1)
    outF, outUV, outParent = [], [], []
    keep = count == 0
    outF.append(F[keep]); outUV.append(UV[keep]); outParent.append(parent[keep])

    def rolled(sel: np.ndarray, shift: np.ndarray):
        idx = (np.arange(3)[None, :] + shift[:, None]) % 3
        rows = np.arange(len(sel))[:, None]
        return F[sel][rows, idx], UV[sel][rows, idx], new_index[fe[sel]][rows, idx]

    # one edge cut: bring it to position 0 -> (v0, m0, v2), (m0, v1, v2)
    sel = np.nonzero(count == 1)[0]
    if len(sel):
        f, uv, m = rolled(sel, np.argmax(flags[sel], axis=1))
        m0 = m[:, 0].astype(F.dtype)
        uvm = 0.5 * (uv[:, 0] + uv[:, 1])
        outF += [np.stack([f[:, 0], m0, f[:, 2]], 1), np.stack([m0, f[:, 1], f[:, 2]], 1)]
        outUV += [np.stack([uv[:, 0], uvm, uv[:, 2]], 1), np.stack([uvm, uv[:, 1], uv[:, 2]], 1)]
        outParent += [parent[sel]] * 2
    # two edges cut: bring the uncut one to position 2 -> (m0, v1, m1), (v0, m0, m1), (v0, m1, v2)
    sel = np.nonzero(count == 2)[0]
    if len(sel):
        f, uv, m = rolled(sel, (np.argmin(flags[sel], axis=1) + 1) % 3)
        m0, m1 = m[:, 0].astype(F.dtype), m[:, 1].astype(F.dtype)
        u0, u1 = 0.5 * (uv[:, 0] + uv[:, 1]), 0.5 * (uv[:, 1] + uv[:, 2])
        outF += [np.stack([m0, f[:, 1], m1], 1), np.stack([f[:, 0], m0, m1], 1), np.stack([f[:, 0], m1, f[:, 2]], 1)]
        outUV += [np.stack([u0, uv[:, 1], u1], 1), np.stack([uv[:, 0], u0, u1], 1), np.stack([uv[:, 0], u1, uv[:, 2]], 1)]
        outParent += [parent[sel]] * 3
    # three edges cut: four triangles
    sel = np.nonzero(count == 3)[0]
    if len(sel):
        f, uv = F[sel], UV[sel]
        m = new_index[fe[sel]].astype(F.dtype)
        u0, u1, u2 = 0.5 * (uv[:, 0] + uv[:, 1]), 0.5 * (uv[:, 1] + uv[:, 2]), 0.5 * (uv[:, 2] + uv[:, 0])
        outF += [np.stack([f[:, 0], m[:, 0], m[:, 2]], 1), np.stack([m[:, 0], f[:, 1], m[:, 1]], 1),
                 np.stack([m[:, 2], m[:, 1], f[:, 2]], 1), np.stack([m[:, 0], m[:, 1], m[:, 2]], 1)]
        outUV += [np.stack([uv[:, 0], u0, u2], 1), np.stack([u0, uv[:, 1], u1], 1),
                  np.stack([u2, u1, uv[:, 2]], 1), np.stack([u0, u1, u2], 1)]
        outParent += [parent[sel]] * 4
    return P2, np.concatenate(outF).astype(F.dtype), np.concatenate(outUV).astype(np.float32), np.concatenate(outParent), int(cut.sum())


def refine(P: np.ndarray, F: np.ndarray, UV: np.ndarray, needs: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
           min_edge: float, *, max_faces: int = 6_000_000, max_passes: int = 14,
           progress: Optional[Callable[[int, int], None]] = None):
    """Split until no triangle that ``needs(F, UV, parent)`` detail has an edge longer than ``min_edge``, or the triangle
    budget is reached. Returns ``P, F, UV, parent, info``; ``parent`` maps every triangle to the one it came from."""
    parent = np.arange(len(F), dtype=np.int64)
    info = {"passes": 0, "stopped": "done"}
    for i in range(max_passes):
        want = needs(F, UV, parent)
        if not want.any():
            break
        room = (max_faces - len(F)) // 2                  # a cut edge adds about two triangles
        if room < max(1, max_faces // 400):                # the last crumbs of the budget are not worth a pass
            info["stopped"] = "budget"
            break
        P, F, UV, parent, cuts = split_once(P, F, UV, parent, want, min_edge, max_cuts=room)
        info["passes"] = i + 1
        if progress:
            progress(i + 1, len(F))
        if cuts == 0:
            break
    else:
        info["stopped"] = "passes"
    return P, F, UV, parent, info
