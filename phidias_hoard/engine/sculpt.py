"""The sculpt itself: scale to print size, refine where each material's relief varies, push the surface along its
normals by that relief (in millimetres), and mark the ground so it can be closed into a block."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import heights, patterns, surface
from .meshio import Scene
from .refine import edge_ids, refine


def clean_enrich(e: Any) -> Optional[dict]:
    if not isinstance(e, dict) or e.get("pattern") not in patterns.PATTERNS:
        return None
    return {"pattern": e["pattern"], "size_mm": float(np.clip(float(e.get("size_mm", 2.0) or 2.0), 0.3, 60.0)),
            "depth_mm": float(np.clip(float(e.get("depth_mm", 0.3) or 0.3), 0.0, 10.0))}


@dataclass
class MaterialRecipe:
    recipe: str = "auto"         # auto, normal, lines, relief, flat, drop
    depth_mm: float = 0.6        # how far the deepest relief goes at print size
    invert: bool = False
    smooth: float = 0.0
    cutout: bool = False         # cut away what the texture leaves see-through (fences, leaves, railings)
    steps: int = 0               # recipe steps: step edges per texture tile (0 = counted from the texture)
    enrich: Optional[dict] = None  # detail the texture does not hold: {"pattern", "size_mm", "depth_mm"}

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "MaterialRecipe":
        d = d or {}
        return cls(str(d.get("recipe", "auto")), float(d.get("depth_mm", 0.6)), bool(d.get("invert", False)), float(d.get("smooth", 0.0)),
                   bool(d.get("cutout", False)), int(d.get("steps", 0) or 0), clean_enrich(d.get("enrich")))


@dataclass
class SculptSettings:
    size_mm: float = 150.0       # longest side of the print
    min_edge_mm: float = 0.25    # smallest triangle edge: the detail the printer can show
    tolerance_mm: float = 0.03   # relief below this inside a triangle is not worth more triangles
    max_faces: int = 6_000_000
    ground: bool = True          # close the ground into a solid block down to the base
    base_mm: float = 2.0         # thickness under the lowest point
    plinth: str = "none"         # none, rect, round: a plate under the whole scene
    plinth_margin_mm: float = 3.0
    sheet_mm: float = 0.8        # thickness given to cut-out sheets so they print (0 leaves them as a single surface)
    normal_cutoff_mm: float = 6.0  # relief from normal maps keeps features up to this size; wider undulation is dropped

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "SculptSettings":
        d = d or {}
        s = cls()
        for k in s.__dataclass_fields__:
            if k in d and d[k] is not None:
                setattr(s, k, type(getattr(s, k))(d[k]))
        s.size_mm = float(np.clip(s.size_mm, 10, 2000))
        s.min_edge_mm = float(np.clip(s.min_edge_mm, 0.05, 20))
        s.max_faces = int(np.clip(s.max_faces, 10_000, 60_000_000))
        return s


@dataclass
class Sculpted:
    P: np.ndarray
    F: np.ndarray
    M: np.ndarray
    ground: np.ndarray                    # per triangle: part of the ground surface
    scale: float                          # millimetres per model unit
    info: dict[str, Any] = field(default_factory=dict)


def point_normals(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(P)
    for k in range(3):
        for axis in range(3):
            n[:, axis] += np.bincount(F[:, k], weights=fn[:, axis], minlength=len(P))
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 1e-12)


def find_ground(P: np.ndarray, F: np.ndarray, cell: float, seed: int = 7) -> np.ndarray:
    """Triangles that face up and have nothing underneath them: filling below them down to the base buries nothing.
    Sampled on a grid of ``cell`` millimetres seen from below."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    up = cross[:, 2] / np.maximum(2 * area, 1e-12)
    # samples per triangle in proportion to its footprint, at least 3
    count = np.clip(np.ceil(np.abs(cross[:, 2]) * 0.5 / (cell * cell) * 2.0).astype(np.int64), 3, 4000)
    owner = np.repeat(np.arange(len(F)), count)
    rng = np.random.default_rng(seed)
    r1, r2 = np.sqrt(rng.random(len(owner))), rng.random(len(owner))
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    pts = a[owner] * w0[:, None] + b[owner] * w1[:, None] + c[owner] * w2[:, None]
    lo = P.min(0)
    nx = int(np.ceil((P[:, 0].max() - lo[0]) / cell)) + 1
    ix = np.floor((pts[:, 0] - lo[0]) / cell).astype(np.int64)
    iy = np.floor((pts[:, 1] - lo[1]) / cell).astype(np.int64)
    key = iy * nx + ix
    order = np.lexsort((pts[:, 2], key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    floor = np.empty(len(order))
    floor[order] = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    lowest = pts[order][floor.astype(np.int64), 2]         # per sample: the lowest z of its cell
    tol = max(cell * 1.5, 1e-6)
    is_low = pts[:, 2] <= lowest + tol
    share = np.bincount(owner, weights=is_low, minlength=len(F)) / np.maximum(count, 1)
    return (up > 0.25) & (share > 0.6)


def sculpt(scene: Scene, recipes: dict[str, MaterialRecipe], settings: SculptSettings,
           progress: Optional[Callable[[str, float], None]] = None) -> Sculpted:
    say = progress or (lambda stage, frac: None)
    ext = scene.extents
    scale = settings.size_mm / max(float(ext.max()), 1e-9)
    P = (scene.P - scene.P.min(0)) * scale
    F, UV, M = scene.F.copy(), scene.UV.copy(), scene.M.copy()

    say("materiales", 0.02)
    maps: list[Optional[np.ndarray]] = []
    pyramids: list[Optional[heights.RangePyramid]] = []
    depth = np.zeros(len(scene.materials), np.float32)
    drop = np.zeros(len(scene.materials), bool)
    slopes: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    alphas: dict[int, tuple[np.ndarray, heights.RangePyramid]] = {}
    stairs: dict[int, dict[str, tuple[int, float]]] = {}
    enrich: dict[int, dict] = {}
    for i, mat in enumerate(scene.materials):
        r = recipes.get(mat.name) or MaterialRecipe()
        if r.recipe != "drop" and r.enrich and r.enrich["depth_mm"] > 0:
            enrich[i] = r.enrich
        if r.recipe == "drop":
            drop[i] = True
        elif r.cutout and mat.albedo is not None and mat.albedo.shape[2] == 4 and int(mat.albedo[..., 3].min()) < 128:
            alpha = heights._enlarge((mat.albedo[..., 3] >= 128).astype(np.float32), soften=0.3)
            alphas[i] = (alpha, heights.RangePyramid(alpha))
        if r.recipe == "steps" and mat.albedo is not None:
            pattern = heights.step_pattern(mat.albedo)
            if r.steps > 0:                                # the user's count, on the axis the texture suggests (v by default)
                axis = "u" if pattern["u"][0] > pattern["v"][0] else "v"
                pattern = {a: ((r.steps, pattern[a][1]) if a == axis else (0, 0.0)) for a in ("u", "v")}
            if pattern["u"][0] or pattern["v"][0]:
                stairs[i] = pattern
            maps.append(None); pyramids.append(None)
            continue
        if r.recipe in ("drop", "flat", "steps") or r.depth_mm <= 0:
            maps.append(None); pyramids.append(None)
            continue
        if r.recipe in ("auto", "normal") and heights.has_normal_detail(mat.normal):
            sx, sy = surface.slope_maps(mat.normal)
            slopes[i] = (sx, sy, -1.0 if r.invert else 1.0)
            maps.append(None); pyramids.append(surface.SlopeDetail(sx, sy)); depth[i] = r.depth_mm
            continue
        h = heights.height_map(mat.albedo, mat.normal, r.recipe, invert=r.invert, smooth=r.smooth, atlas=mat.atlas)
        if float(np.abs(h).max()) < 1e-4:
            maps.append(None); pyramids.append(None)
            continue
        maps.append(h); pyramids.append(heights.RangePyramid(h)); depth[i] = r.depth_mm
    if drop.any():
        keep = ~drop[M]
        F, UV, M = F[keep], UV[keep], M[keep]

    say("suelo", 0.06)
    ground0 = find_ground(P, F, max(settings.size_mm / 400.0, 0.2)) if settings.ground and len(F) else np.zeros(len(F), bool)
    M0 = M
    ramp0 = np.zeros(len(F), bool)                         # sloped faces of stair materials: all of them get fine triangles
    if stairs:
        a0, b0, c0 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n0 = np.cross(b0 - a0, c0 - a0)
        nz = np.abs(n0[:, 2]) / np.maximum(np.linalg.norm(n0, axis=1), 1e-12)
        ramp0 = np.isin(M, list(stairs)) & (nz > 0.15) & (nz < 0.985)

    def needs(P_, F_, UV_, parent):
        mats = M0[parent]
        want = np.zeros(len(F_), bool)
        for i, pyr in enumerate(pyramids):
            if pyr is None:
                continue
            sel = np.nonzero(mats == i)[0]
            if len(sel):
                if i in slopes:                           # slope change inside the triangle
                    want[sel] = pyr.variation(UV_[sel]) > 4.0 * settings.tolerance_mm
                else:
                    want[sel] = pyr.variation(UV_[sel]) * depth[i] > settings.tolerance_mm
        want |= ramp0[parent]
        for i, e in enrich.items():                        # enough triangles to draw the pattern's features
            sel = np.nonzero(mats == i)[0]
            if len(sel):
                f = F_[sel]
                longest = np.maximum.reduce([np.linalg.norm(P_[f[:, k]] - P_[f[:, (k + 1) % 3]], axis=1) for k in range(3)])
                want[sel] |= longest > max(e["size_mm"] / 5.0, 1e-6)
        for i, (_alpha, pyr) in alphas.items():            # the outline of a cut-out crosses the triangle
            sel = np.nonzero(mats == i)[0]
            if len(sel):
                want[sel] |= pyr.variation(UV_[sel]) > 0.5
        return want

    say("refinando", 0.1)
    edge = settings.min_edge_mm                           # the budget is shared out inside refine, longest edges first
    P, F, UV, parent, info = refine(P, F, UV, needs, edge, max_faces=settings.max_faces,
                                    progress=lambda i, n: say("refinando", min(0.1 + 0.05 * i, 0.7)))
    if info["stopped"] == "budget" and len(F):               # what the budget really reached: the longest edge left to cut
        want = needs(P, F, UV, parent)
        if want.any():
            fw = F[want]
            lengths = np.concatenate([np.linalg.norm(P[fw[:, k]] - P[fw[:, (k + 1) % 3]], axis=1) for k in range(3)])
            edge = max(edge, float(np.percentile(lengths, 90)))
    M = M0[parent]
    ground = ground0[parent]
    if alphas:
        keep = np.ones(len(F), bool)
        for i, (alpha, _pyr) in alphas.items():
            sel = np.nonzero(M == i)[0]
            if len(sel):
                keep[sel] = heights.sample(alpha, UV[sel].mean(axis=1)) >= 0.5
        F, UV, M, ground = F[keep], UV[keep], M[keep], ground[keep]
    used = np.unique(F)                                   # points of removed triangles would stretch the measures
    if len(used) < len(P):
        remap = np.full(len(P), -1, np.int64)
        remap[used] = np.arange(len(used))
        P, F = P[used], remap[F].astype(np.int32)

    say("esculpiendo", 0.75)
    disp_sum = np.zeros(len(P))
    disp_n = np.zeros(len(P))
    normals = point_normals(P, F)
    for i, h in enumerate(maps):
        sel = np.nonzero(M == i)[0]
        if not len(sel):
            continue
        idx = F[sel].reshape(-1)
        if i in enrich:
            e = enrich[i]
            pts = np.unique(idx)
            extra = np.zeros(len(P))
            name = e["pattern"]
            if name in patterns.ORIENTED:
                height = patterns.oriented(name, P[pts], np.searchsorted(pts, F[sel]), normals[pts], e["size_mm"])
            else:
                height = patterns.triplanar(name, P[pts], normals[pts], e["size_mm"])
            extra[pts] = height * e["depth_mm"]
            disp_sum += np.bincount(idx, weights=extra[idx], minlength=len(P))
        if i in slopes:
            sx, sy, sign = slopes[i]
            used, hp = surface.integrate(P, F[sel], UV[sel], sx, sy, settings.normal_cutoff_mm)
            top = float(np.percentile(np.abs(hp), 99.5)) if len(hp) else 0.0
            full = np.zeros(len(P))
            if top > 1e-9:
                full[used] = np.clip(hp / top, -1.6, 1.6) * depth[i] * sign
            disp_sum += np.bincount(idx, weights=full[idx], minlength=len(P))
            disp_n += np.bincount(idx, minlength=len(P))
            continue
        if h is None:
            disp_n += np.bincount(idx, minlength=len(P))
            continue
        val = heights.sample(h, UV[sel].reshape(-1, 2)) * depth[i]
        disp_sum += np.bincount(idx, weights=val, minlength=len(P))
        disp_n += np.bincount(idx, minlength=len(P))
    disp = disp_sum / np.maximum(disp_n, 1)
    P = P + normals * disp[:, None]
    stair_faces = 0
    if stairs:
        dz_sum, dz_n = np.zeros(len(P)), np.zeros(len(P))
        for i, pattern in stairs.items():
            sel = np.nonzero(M == i)[0]
            if not len(sel):
                continue
            f, uv = F[sel], UV[sel].astype(np.float64)
            z = P[f][..., 2]
            du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
            du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
            det = du1 * dv2 - du2 * dv1
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            dz1, dz2 = z[:, 1] - z[:, 0], z[:, 2] - z[:, 0]
            zu = (dz1 * dv2 - dz2 * dv1) * inv             # rise per unit of u and of v
            zv = (dz2 * du1 - dz1 * du2) * inv
            (ku, ou), (kv, ov) = pattern["u"], pattern["v"]
            use_u = (np.abs(zu) * (ku > 0) >= np.abs(zv) * (kv > 0)) & (ku > 0)
            k = np.where(use_u, ku, kv).astype(np.float64)
            off = np.where(use_u, ou, ov)
            slope = np.where(use_u, zu, zv)
            w = np.where(use_u[:, None], uv[..., 0], uv[..., 1])
            good = ok & (k > 0) & (np.abs(slope) > 1e-9)
            phase = (w * k[:, None] - (off * k)[:, None]) % 1.0
            phase = np.where(slope[:, None] < 0, 1.0 - phase, phase)
            dz = -(np.abs(slope) / np.maximum(k, 1.0))[:, None] * (phase - 0.5)
            dz = np.where(good[:, None], dz, 0.0)
            # a triangle whose corners sit on both sides of a step edge would smear the riser: keep each corner's own tread
            idx = f.reshape(-1)
            dz_sum += np.bincount(idx, weights=dz.reshape(-1), minlength=len(P))
            dz_n += np.bincount(idx, minlength=len(P))
            stair_faces += int(good.sum())
        P[:, 2] += dz_sum / np.maximum(dz_n, 1)
    sheets = 0
    if alphas and settings.sheet_mm > 0:
        sheet = np.isin(M, list(alphas)) & ~ground
        if sheet.any():
            back_pts, back_tris = _offset_solid(P, F[sheet], -normals * settings.sheet_mm)
            P = np.concatenate([P, back_pts])
            F = np.concatenate([F, back_tris])
            M = np.concatenate([M, np.full(len(back_tris), M[sheet][0], M.dtype)])
            ground = np.concatenate([ground, np.zeros(len(back_tris), bool)])
            sheets = int(sheet.sum())
    say("esculpido", 0.85)
    info.update({"edge_mm": round(edge, 3), "edge_asked_mm": settings.min_edge_mm, "triangles": int(len(F)), "points": int(len(P)), "scale_mm_per_unit": scale,
                 "size_mm": [round(float(v), 2) for v in (P.max(0) - P.min(0))] if len(P) else [0, 0, 0],
                 "ground_triangles": int(ground.sum()), "sheet_triangles": sheets, "stair_triangles": stair_faces, "enriched_materials": len(enrich),
                 "max_relief_mm": round(float(np.abs(disp).max()), 3) if len(disp) else 0.0})
    return Sculpted(P, F, M, ground, scale, info)


def _offset_solid(P: np.ndarray, G: np.ndarray, offset: np.ndarray, z_flat: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """The other side of a surface and the walls that join both along its outline: with the surface itself, a closed
    solid. The other side is every point moved by ``offset`` (per point), or dropped to the height ``z_flat``.
    Returns the new points and triangles (indices continue after ``P``)."""
    used = np.unique(G)
    remap = np.full(len(P), -1, np.int64)
    remap[used] = len(P) + np.arange(len(used))
    pts = P[used].copy()
    if z_flat is None:
        pts += offset[used]
    else:
        pts[:, 2] = z_flat
    back = remap[G][:, ::-1]
    # outline: directed edges whose opposite does not belong to the surface
    a = G[:, [0, 1, 2]].reshape(-1).astype(np.int64)
    b = G[:, [1, 2, 0]].reshape(-1).astype(np.int64)
    n = np.int64(len(P))
    lone = ~np.isin(a * n + b, b * n + a)
    a, b = a[lone], b[lone]
    walls = np.concatenate([np.stack([b, a, remap[a]], 1), np.stack([b, remap[a], remap[b]], 1)])
    return pts, np.concatenate([back, walls]).astype(np.int32)


def close_ground(s: Sculpted, base_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Walls from the outline of the ground down to the base and a bottom under it: with the ground itself, a closed
    block. The bottom is a fan from one centre point to the outline (flat, so it costs one triangle per outline edge
    instead of a copy of the whole ground). Returns extra points and triangles (indices continue after ``s.P``)."""
    G = s.F[s.ground]
    if not len(G):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    z_base = float(s.P[:, 2].min()) - base_mm
    a = G[:, [0, 1, 2]].reshape(-1).astype(np.int64)
    b = G[:, [1, 2, 0]].reshape(-1).astype(np.int64)
    n = np.int64(len(s.P))
    lone = ~np.isin(a * n + b, b * n + a)                 # outline: edges whose opposite is not ground
    a, b = a[lone], b[lone]
    if not len(a):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    rim = np.unique(np.concatenate([a, b]))
    remap = np.full(len(s.P), -1, np.int64)
    remap[rim] = len(s.P) + np.arange(len(rim))
    pts = s.P[rim].copy()
    pts[:, 2] = z_base
    centre = np.array([[pts[:, 0].mean(), pts[:, 1].mean(), z_base]])
    c = np.full(len(a), len(s.P) + len(rim), np.int64)
    tris = np.concatenate([np.stack([b, a, remap[a]], 1), np.stack([b, remap[a], remap[b]], 1), np.stack([remap[b], remap[a], c], 1)])
    return np.concatenate([pts, centre]), tris.astype(np.int32)


def plinth(s: Sculpted, kind: str, margin: float, base_mm: float, thickness: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A closed plate under the whole scene (its top at the base level). Points and triangles of its own."""
    if kind not in ("rect", "round") or not len(s.P):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    lo, hi = s.P.min(0), s.P.max(0)
    top = lo[2] - base_mm + 0.01
    bot = top - thickness
    if kind == "rect":
        x0, y0, x1, y1 = lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin
        ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    else:
        c = 0.5 * (lo[:2] + hi[:2])
        r = 0.5 * float(np.linalg.norm(hi[:2] - lo[:2])) + margin
        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ring = c + r * np.stack([np.cos(t), np.sin(t)], 1)
    k = len(ring)
    pts = np.concatenate([np.c_[ring, np.full(k, top)], np.c_[ring, np.full(k, bot)],
                          [[*ring.mean(0), top]], [[*ring.mean(0), bot]]])
    i = np.arange(k)
    j = (i + 1) % k
    tris = np.concatenate([np.stack([np.full(k, 2 * k), i, j], 1), np.stack([np.full(k, 2 * k + 1), j + k, i + k], 1),
                           np.stack([i, i + k, j + k], 1), np.stack([i, j + k, j], 1)])
    return pts, tris.astype(np.int32)


def assemble(s: Sculpted, settings: SculptSettings) -> tuple[np.ndarray, np.ndarray]:
    """The printable mesh: sculpted surface, ground block and plinth, in millimetres with the base at z = 0."""
    pts, tris = [s.P], [s.F]
    n = len(s.P)
    if settings.ground:
        gp, gt = close_ground(s, settings.base_mm)
        pts.append(gp); tris.append(gt); n += len(gp)
    pp, pt = plinth(s, settings.plinth, settings.plinth_margin_mm, settings.base_mm if settings.ground else 0.0)
    if len(pp):
        pts.append(pp); tris.append(pt + n)
    P = np.concatenate(pts)
    F = np.concatenate(tris).astype(np.int32)
    if len(P):
        P = P - np.array([P[:, 0].min(), P[:, 1].min(), P[:, 2].min()])
    return P, F


def open_edges(F: np.ndarray, n_points: int) -> int:
    """Edges used by exactly one triangle (0 on a closed surface)."""
    _, fe = edge_ids(F, n_points)
    return int((np.bincount(fe.reshape(-1)) == 1).sum())
