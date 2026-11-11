"""Taking a game model apart: its connected pieces, each with its position, its profile (the size of its outline at
every height it has corners), the materials on it and how many times their textures repeat across it. Pieces of the
same shape are told to be instances of one type. A first guess says what each piece is."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..engine.meshio import Scene

KINDS = ("roof", "tiled", "body", "block", "lathe", "sweep", "stairs", "organic", "keep", "drop")


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
        world = P[used]
        # the piece's own frame: turned about the vertical so its outline is as tight as it can be
        yaw = _tightest_yaw(world[:, :2])
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        mid = 0.5 * (world[:, :2].min(0) + world[:, :2].max(0))
        local_xy = (world[:, :2] - mid) @ np.array([[c, s_], [-s_, c]])
        shift = 0.5 * (local_xy.min(0) + local_xy.max(0))
        local_xy -= shift
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        centre_xy = mid + shift @ np.array([[cy_, sy_], [-sy_, cy_]])
        pts = np.column_stack([local_xy, world[:, 2]])
        lo, hi = pts.min(0), pts.max(0)
        size = hi - lo
        profile, fits, extra = _shape(pts, f, remap_of(used))
        mats: dict[str, dict[str, Any]] = {}
        for i in np.unique(scene.M[sel]):
            ms = sel & (scene.M == i)
            uv = scene.UV[ms].reshape(-1, 2)
            mats[names[i]] = {"share": round(float(areas[ms].sum() / max(areas[sel].sum(), 1e-12)), 3),
                              "repeat_u": round(float(np.ptp(uv[:, 0])), 2), "repeat_v": round(float(np.ptp(uv[:, 1])), 2)}
        remap = remap_of(used)
        # a slope big enough to carry tiles, and the material that covers most of it
        tri = pts[np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in f])]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(nrm, axis=1)
        steep = np.abs(nrm[:, 2]) / np.maximum(2 * area, 1e-12)
        sloped = (steep > 0.2) & (steep < 0.98)
        by_mat = {names[i]: float(area[sloped & (scene.M[sel] == i)].sum()) for i in np.unique(scene.M[sel])}
        best = max(by_mat, key=by_mat.get) if by_mat else ""
        fits["tiled"] = bool(best and by_mat[best] > 0.25 * float(area.sum()) and max(size[0], size[1]) > 20 and len(f) >= 8)
        if fits["tiled"]:
            extra["tiles_on"] = [best]
            on = sloped & (scene.M[sel] == names.index(best))
            mid = tri[on].mean(1)
            away = (mid[:, 0] * nrm[on, 0] + mid[:, 1] * nrm[on, 1]) * np.sign(nrm[on, 2])
            sides = {(int(abs(n[0]) >= abs(n[1])), bool((n[0] if abs(n[0]) >= abs(n[1]) else n[1]) * n[2] > 0)) for n in nrm[on]}
            extra["sheds"] = bool(float(area[on][away > 0].sum()) > 0.9 * float(area[on].sum()) and len(sides) >= 3)   # slopes falling away from its middle
        extra["triangle_materials"] = [names[i] for i in scene.M[sel]]
        # a figure: an irregular piece with a texture of its own (a statue, an animal, a carved ornament)
        own = [i for i in np.unique(scene.M[sel])]
        whole = len(own) == 1 and not scene.materials[own[0]].atlas and scene.materials[own[0]].albedo is not None \
            and 0.5 <= mats[names[own[0]]]["repeat_u"] <= 1.5 and 0.5 <= mats[names[own[0]]]["repeat_v"] <= 1.5
        extra["figure"] = bool(whole and len(f) >= 40 and len(profile) >= 12 and not (fits["rect"] or fits["round"] or fits["sweep"]))
        extra["uvs"] = np.round(scene.UV[sel].astype(np.float64), 5).tolist()
        piece = {"id": int(k), "triangles_count": int(sel.sum()),
                 "center": [round(float(centre_xy[0]), 4), round(float(centre_xy[1]), 4), round(float(lo[2]), 4)],
                 "yaw": round(float(yaw), 5), "size": [round(float(v), 4) for v in size], "profile": profile, "materials": mats, "fits": fits, **extra,
                 "points": np.round(pts, 4).tolist(), "triangles": [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in f],
                 "dropped": bool(mats) and all(name in dropped for name in mats)}
        piece["signature"] = f'{piece["triangles_count"]}:{"x".join(f"{v:.1f}" for v in size)}:{"+".join(sorted(mats))}'
        out.append(piece)
    out.sort(key=lambda p: (p["center"][2], p["id"]))
    top = max((p["center"][2] + p["size"][2] for p in out), default=0.0)
    for p in out:                                          # a roof stands up on something and its texture tiles; sloping ground is not one
        tiles = p["materials"].get((p.get("tiles_on") or [""])[0], {})
        p["roof_like"] = bool(p.pop("sheds", False) and p["center"][2] > 0.2 * top and max(tiles.get("repeat_u", 0), tiles.get("repeat_v", 0)) >= 2)
    types: dict[str, int] = {}
    for p in out:
        p["type"] = types.setdefault(p["signature"], len(types) + 1)
    return out


def remap_of(used: np.ndarray) -> dict[int, int]:
    return {int(v): j for j, v in enumerate(used)}


def _levels(z: np.ndarray, tol: float) -> list[float]:
    zs = np.sort(z)
    levels = [float(zs[0])]
    for v in zs[1:]:
        if v - levels[-1] > tol:
            levels.append(float(v))
    return levels


def _shape(pts: np.ndarray, faces: np.ndarray, remap: dict[int, int]) -> tuple[list[list[float]], dict[str, bool], dict[str, Any]]:
    """What a piece is, measured: its outline at every height, which builders its shape really fits, and what those
    builders need to know (a polygon's sides, a ring's hole, a beam's sections, a roof's eaves)."""
    size = pts.max(0) - pts.min(0)
    wide = max(float(size[0]), float(size[1]), 1e-9)
    tol = max(float(size[2]) * 0.01, 1e-4)
    levels = _levels(pts[:, 2], tol)
    at = [pts[np.abs(pts[:, 2] - z) <= tol] for z in levels]
    extra: dict[str, Any] = {}

    def outline(square: bool) -> tuple[list[list[float]], float, float]:
        prof, err, off = [], 0.0, 0.0
        for z, q in zip(levels, at):
            if square:                                     # a square piece seen whole: game models leave out the sides nobody sees
                sx = sy = 2 * float(np.abs(q[:, :2]).max())
                c0 = np.zeros(2)
            else:
                sx, sy = float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))
                c0 = 0.5 * (q[:, :2].min(0) + q[:, :2].max(0))
            prof.append([round(float(z), 4), round(sx, 4), round(sy, 4)])
            off = max(off, float(np.abs(c0).max()))
            d = np.minimum(np.abs(np.abs(q[:, 0] - c0[0]) - sx / 2), np.abs(np.abs(q[:, 1] - c0[1]) - sy / 2))
            err = max(err, float(d.max()))
        return prof, err, off

    profile, rect_err, off_centre = outline(False)
    centred = off_centre < 0.04 * wide
    rect = centred and rect_err < 0.03 * wide
    if not rect and abs(float(size[0]) - float(size[1])) < 0.03 * wide and len(pts) >= 12:
        prof2, err2, _ = outline(True)
        if err2 < 0.03 * wide:
            profile, rect, centred, rect_err = prof2, True, True, err2
            extra["completed"] = True                      # rebuilt whole from the sides the model does have
    few = 2 <= len(levels) <= 10
    # the surface between the outlines must be the piece's surface too: relief inside it means it is not a plain solid
    if rect and few and not _lofted(pts, faces, remap, profile, tol, wide):
        rect = False
    shell_err = 0.0
    for q in at:
        gx, gy = float(size[0]) / 2 - np.abs(q[:, 0]), float(size[1]) / 2 - np.abs(q[:, 1])
        shell_err = max(shell_err, float(np.minimum(gx, gy).max()))   # how far inside the piece's outline a point sits
    polar = _polar(at, levels, wide)
    same = all(abs(a - b) <= 0.12 * max(a, b, 1e-9) for _, a, b in profile)
    fits = {"rect": bool(rect and few),
            "round": bool(centred and 2 <= len(levels) <= 40 and (polar is not None or (rect_err < 0.03 * wide and same and not extra.get("completed")))),
            "shell": bool(centred and few and shell_err < 0.13 * wide and float(size[2]) > 0.18 * wide)}
    if polar is not None:
        extra["polar"] = polar
    sweep = None if (fits["rect"] or fits["round"]) else _sweep(pts)
    fits["sweep"] = sweep is not None
    if sweep is not None:
        extra["sections"] = sweep
    if fits["rect"]:
        roof = _roof(pts, profile, levels, at)
        if roof is not None:
            extra["roof"] = roof
    return profile, fits, extra


def _lofted(pts: np.ndarray, faces: np.ndarray, remap: dict[int, int], profile: list[list[float]], tol: float, wide: float) -> bool:
    """Whether every face lies on the solid its outlines describe: on a flat level or on the wall between two levels."""
    if not len(faces):
        return True
    idx = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces])
    mid = pts[idx].mean(1)
    zs = np.array([r[0] for r in profile])
    hx = np.interp(mid[:, 2], zs, [r[1] / 2 for r in profile])
    hy = np.interp(mid[:, 2], zs, [r[2] / 2 for r in profile])
    flat = np.abs(mid[:, 2][:, None] - zs[None, :]).min(1) <= 2 * tol
    wall = np.minimum(np.abs(np.abs(mid[:, 0]) - hx), np.abs(np.abs(mid[:, 1]) - hy)) <= 0.04 * wide
    return bool((flat | wall).mean() >= 0.97)


def _polar(at: list[np.ndarray], levels: list[float], wide: float) -> Optional[dict[str, Any]]:
    """A piece that turns about its axis: how many sides its outline has (0 for a smooth round), where the first
    corner points, the radius at every height and the hole down its middle when it is a ring."""
    tol = 0.035 * wide
    radii, sides, phases, holes = [], [], [], []
    for z, q in zip(levels, at):
        r = np.linalg.norm(q[:, :2], axis=1)
        outer = float(r.max())
        if outer < tol:
            radii.append([round(float(z), 4), 0.0])
            holes.append(0.0)
            continue
        rest = r[r < outer - tol]
        inner = float(rest.max()) if len(rest) and rest.max() > tol else 0.0
        if len(rest) and np.any((rest < inner - tol) & (rest > tol)):
            return None                                    # more than a wall and a hole: not a turned shape
        ang = np.sort(np.degrees(np.arctan2(q[r >= outer - tol, 1], q[r >= outer - tol, 0])) % 360)
        ang = ang[np.r_[True, np.diff(ang) > 3]]
        if len(ang) > 1 and ang[-1] - ang[0] > 357:
            ang = ang[:-1]
        if len(ang) < 3:
            return None
        gaps = np.diff(np.r_[ang, ang[0] + 360])
        n = int(round(360 / max(float(gaps.min()), 1e-6)))  # corners may be missing on the side nobody sees
        if n > 64 or len(ang) < n / 2 + 1 or np.abs(gaps / (360 / n) - np.round(gaps / (360 / n))).max() > 0.2:
            return None
        if inner > 0:                                      # a ring has its inner corners behind the outer ones
            ia = np.degrees(np.arctan2(q[np.abs(r - inner) <= tol, 1], q[np.abs(r - inner) <= tol, 0])) % 360
            off = np.abs((ia[:, None] - ang[None, :] + 180) % 360 - 180).min(1)
            if off.max() > 0.2 * 360 / n:
                return None
        radii.append([round(float(z), 4), round(outer, 4)])
        sides.append(n)
        phases.append(float(ang[0] % (360 / n)))
        holes.append(inner)
    if not sides or len(set(sides)) != 1 or len(radii) < 2:
        return None
    step = 360 / sides[0]
    if max(min(abs(p - phases[0]), step - abs(p - phases[0])) for p in phases) > 0.12 * step:
        return None                                        # the corners turn from one level to the next: a twisted shape
    hole = float(np.median([h for h in holes if h > 0])) if holes[0] > 0 and holes[-1] > 0 else 0.0
    return {"sides": int(sides[0]), "phase": round(phases[0], 2), "radii": radii, "hole": round(hole, 4)}


def _sweep(pts: np.ndarray) -> Optional[dict[str, Any]]:
    """A beam that runs along the piece's long side, straight or curved: its box section at every station
    [x, y0, y1, z0, z1], or None when the piece is not that."""
    size = pts.max(0) - pts.min(0)
    axis = "x" if size[0] >= size[1] else "y"
    if axis == "y":
        pts = pts[:, [1, 0, 2]]
    long_ = float(max(size[0], size[1]))
    if long_ < 1e-6:
        return None
    stations = _levels(pts[:, 0], max(long_ * 0.04, 1e-4))
    if not 2 <= len(stations) <= 24:
        return None
    out = []
    for x in stations:
        q = pts[(pts[:, 0] >= x - 1e-9) & (pts[:, 0] <= x + max(long_ * 0.04, 1e-4))]
        y0, y1, z0, z1 = float(q[:, 1].min()), float(q[:, 1].max()), float(q[:, 2].min()), float(q[:, 2].max())
        out.append([round(float(q[:, 0].mean()), 4), round(y0, 4), round(y1, 4), round(z0, 4), round(z1, 4)])
    width = float(np.median([s[2] - s[1] for s in out]))
    height = float(np.median([s[4] - s[3] for s in out]))
    if long_ < 2.5 * max(width, height) or max(width, height) < 1e-6:
        return None
    if max(s[2] - s[1] for s in out) > 2.5 * max(width, 0.02 * long_) or max(s[4] - s[3] for s in out) > 2.5 * max(height, 0.02 * long_):
        return None                                        # something sticks out of it: more than a beam
    return {"axis": axis, "stations": out}


def _roof(pts: np.ndarray, profile: list[list[float]], levels: list[float], at: list[np.ndarray]) -> Optional[dict[str, Any]]:
    """The eaves of a hipped roof as the model has them: how far the corners rise above the middle of each side,
    how much of the side is level, the fascia's height and the slope's real line (size at each height)."""
    z0 = levels[0]
    eave = max(r[1] for r in profile)
    half = eave / 2
    if half < 1e-6:
        return None
    corner_z = [z for z, q in zip(levels, at) if np.any((np.abs(q[:, 0]) > 0.93 * half) & (np.abs(q[:, 1]) > 0.93 * half))]
    if not corner_z:
        return None
    low = at[0]
    flat = float(np.minimum(np.abs(low[:, 0]), np.abs(low[:, 1])).max()) / half
    lift = float(corner_z[0] - z0)
    wide_levels = [z for (z, sx, sy) in profile if max(sx, sy) > 0.97 * eave]
    mid_top = [z for z in wide_levels if z < corner_z[0] - 1e-6]
    fascia = float((mid_top[-1] if mid_top else (corner_z[-1] if len(corner_z) > 1 and lift < 1e-6 else z0)) - z0)
    if lift > 1e-6 and len(corner_z) > 1:
        fascia = fascia or float(corner_z[1] - corner_z[0])
    start = max(wide_levels)
    line = [[round(float(z), 4), round(float(max(sx, sy)), 4)] for z, sx, sy in profile if z > start + 1e-6]
    if not line:
        return None
    return {"lift": round(lift, 4), "flat": round(min(flat, 0.95) if lift > 1e-6 else 0.0, 3), "fascia": round(fascia, 4), "eave": round(eave, 4), "line": line}


def _tightest_yaw(xy: np.ndarray) -> float:
    """The turn about the vertical (radians, 0 to under 90 degrees) that makes the outline's bounding box smallest."""
    if len(xy) < 3:
        return 0.0
    best, best_area = 0.0, float(np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]))
    for deg in range(1, 90):
        a = np.radians(deg)
        c, s_ = np.cos(-a), np.sin(-a)
        r = xy @ np.array([[c, s_], [-s_, c]])
        area = float(np.ptp(r[:, 0]) * np.ptp(r[:, 1]))
        if area < best_area * 0.97:
            best, best_area = a, area
    return best


def allowed(piece: dict[str, Any]) -> list[str]:
    """What a piece can be rebuilt as: a builder is offered only when the piece really has that builder's shape."""
    kinds = ["keep", "drop", "stairs"]
    if piece["triangles_count"] >= 12:
        kinds += ["organic"]
    if piece["fits"]["rect"]:
        kinds += ["block", "body", "roof"]
    elif piece["fits"].get("shell"):
        kinds += ["body"]
    if piece["fits"]["round"]:
        kinds += ["lathe"]
    if piece["fits"].get("sweep"):
        kinds += ["sweep"]
    if piece["fits"].get("tiled"):
        kinds += ["tiled"]
    return kinds


def guess(piece: dict[str, Any], size_mm: float) -> tuple[str, dict[str, Any]]:
    """What a piece is, from its shape alone."""
    if piece["dropped"]:
        return "drop", {}
    prof = piece["profile"]
    sx, sy, sz = piece["size"]
    wide = max(sx, sy, 1e-9)
    fits = piece["fits"]
    if piece.get("figure"):
        return "organic", {}
    polar = piece.get("polar")
    if polar and fits["round"] and polar["sides"] <= 8 and (len(prof) <= 3 or polar["hole"] > 0) and not (polar["sides"] == 4 and fits["rect"]):   # a prism or a ring, corners and all
        return "lathe", {"sides": polar["sides"]}
    if fits.get("sweep"):
        return "sweep", {}
    stout = min(sx, sy) > 0.35 * wide
    if len(prof) >= 2 and not fits["rect"] and fits.get("shell") and wide > 8 and stout:      # walls with posts standing out of them
        return "body", {"bays_x": int(np.clip(round(sx / 9), 1, 7)), "bays_y": int(np.clip(round(sy / 9), 1, 7)), "zones": ["panel"] * (len(prof) - 1)}
    if len(prof) < 2 or min(sx, sy) < 1e-6 or not (fits["rect"] or fits["round"]):
        if fits.get("tiled") and piece.get("roof_like") and min(sx, sy) > 0.5 * wide:     # a roof with more to it than four slopes: itself, tiled
            return "tiled", {"tiles": int(np.clip(round(wide / 2.4), 6, 36))}
        return "keep", {}
    radii = [max(r[1], r[2]) for r in prof]
    ups = sum(1 for a, b in zip(radii, radii[1:]) if b > a * 1.05)
    downs = sum(1 for a, b in zip(radii, radii[1:]) if b < a * 0.95)
    if fits["round"] and ((len(prof) >= 5 and ups >= 2 and downs >= 2) or (sz > 2.2 * wide and len(prof) >= 3)):
        return "lathe", {"segments": 24 if wide > 3 else 16}
    if not fits["rect"]:
        return ("lathe", {"segments": 24}) if fits["round"] and len(prof) >= 3 else ("keep", {})
    shrink = min(radii[1:]) / max(radii[0], 1e-9)
    if shrink < 0.8 and radii[0] >= max(radii) * 0.95 and sz < 0.9 * wide and wide > 6 and piece.get("roof"):
        return "roof", {"tiles": int(np.clip(round(wide / 2.4), 6, 36)), "rows": 6}
    constant = max(radii) - min(radii) < 0.02 * wide
    if constant and sz > 0.18 * wide and wide > 8 and stout:
        return "body", {"bays_x": int(np.clip(round(sx / 9), 1, 7)), "bays_y": int(np.clip(round(sy / 9), 1, 7)), "zones": ["panel"] * (len(prof) - 1)}
    return "block", {"bevel": round(float(np.clip(min(wide, sz) * 0.04, 0.05, 0.6)), 2)}


_NUM = {"roof": {"tiles": (4, 60, int), "rows": (3, 12, int), "concave": (1.0, 2.2, float), "corner_lift": (0.0, 0.9, float),
                 "tile_radius": (0.0, 5.0, float), "bevel": (0.0, 1.0, float)},
        "body": {"bays_x": (1, 12, int), "bays_y": (1, 12, int), "bars": (2, 12, int), "post": (0.0, 10.0, float), "beam": (0.0, 10.0, float),
                 "panel_depth": (0.0, 5.0, float)},
        "block": {"bevel": (0.0, 2.0, float), "emboss": (0, 1, int)}, "lathe": {"segments": (8, 64, int), "flutes": (0, 32, int), "sides": (0, 12, int), "bevel": (0.0, 1.0, float)},
        "tiled": {"tiles": (4, 60, int), "tile_radius": (0.0, 5.0, float), "bevel": (0.0, 1.0, float), "emboss": (0, 1, int)},
        "organic": {"level": (1, 4, int), "emboss": (0, 1, int)},
        "sweep": {"smooth": (0, 8, int), "bevel": (0.0, 1.0, float), "emboss": (0, 1, int)},
        "stairs": {"steps": (2, 40, int), "bevel": (0.0, 1.0, float)}, "keep": {"bevel": (0.0, 1.0, float), "emboss": (0, 1, int)}, "drop": {}}


def clean_params(kind: str, params: Any) -> dict[str, Any]:
    """Only the parameters a builder has, inside their ranges."""
    out: dict[str, Any] = {}
    if not isinstance(params, dict):
        return out
    for key, (lo, hi, typ) in _NUM.get(kind, {}).items():
        if params.get(key) is not None:
            try:
                out[key] = typ(min(max(float(params[key]), lo), hi))
            except (TypeError, ValueError):
                pass
    if kind == "body":
        if isinstance(params.get("zones"), list):
            out["zones"] = [z if z in ("panel", "lattice", "plain") else "panel" for z in params["zones"][:8]]
        if "door_bay" in params:
            try:
                out["door_bay"] = None if params["door_bay"] is None else int(min(max(int(params["door_bay"]), 0), 11))
            except (TypeError, ValueError):
                pass
        if params.get("post_section") in ("square", "round"):
            out["post_section"] = params["post_section"]
    if kind == "stairs":
        if params.get("axis") in ("x", "y"):
            out["axis"] = params["axis"]
        if isinstance(params.get("flip"), bool):
            out["flip"] = params["flip"]
    return out
