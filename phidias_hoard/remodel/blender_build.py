"""Runs inside Blender (``blender -b --python blender_build.py -- plan.json out_dir``) or with the ``bpy`` module.

Builds every piece of a plan with real modelling operations and clean quads, one object per piece, grouped in
collections by kind. Writes ``remodel.blend`` (bevels left as live modifiers), ``remodel.obj`` (modifiers applied,
quads kept) and ``remodel.json`` (counts per piece). Units are millimetres, Z up."""

import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
PLAN = json.load(open(argv[0], encoding="utf-8"))
OUT = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 0.001
scene.unit_settings.length_unit = "MILLIMETERS"
COLLECTIONS = {}
STATS = []
PLACE = {"center": (0.0, 0.0), "yaw": 0.0}       # where the piece being built goes: it is modelled around its own axis


def collection(kind):
    if kind not in COLLECTIONS:
        c = bpy.data.collections.new(kind)
        scene.collection.children.link(c)
        COLLECTIONS[kind] = c
    return COLLECTIONS[kind]


def finish(bm, name, kind, bevel=0.0, smooth=False, weld=True, weighted=False):
    if weld:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    counts = {"verts": len(bm.verts), "quads": sum(1 for f in bm.faces if len(f.verts) == 4),
              "tris": sum(1 for f in bm.faces if len(f.verts) == 3), "ngons": sum(1 for f in bm.faces if len(f.verts) > 4)}
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection(kind).objects.link(obj)
    obj.location = (PLACE["center"][0], PLACE["center"][1], 0.0)
    obj.rotation_euler = (0.0, 0.0, PLACE["yaw"])
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    if bevel > 0:
        m = obj.modifiers.new("Bisel", "BEVEL")
        m.width = bevel
        m.segments = 2
        m.limit_method = "WEIGHT" if weighted else "ANGLE"
        m.angle_limit = math.radians(35)
        m.harden_normals = True
    STATS.append({"name": name, "kind": kind, **counts})
    return obj


def box(bm, x0, y0, z0, x1, y1, z1):
    v = [bm.verts.new(p) for p in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))]
    for q in ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in q])


def loft_rect(bm, rings):
    """Rectangular rings [(cx, cy, z, sx, sy)] joined by quads, with a quad lid at each end."""
    loops = []
    for cx, cy, z, sx, sy in rings:
        loops.append([bm.verts.new((cx + dx * sx / 2, cy + dy * sy / 2, z)) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for a, b in zip(loops, loops[1:]):
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.faces.new(loops[0][::-1])
    bm.faces.new(loops[-1])


# ------------------------------------------------------------------------------------------------------------ block
def build_block(piece, name):
    if len(piece["profile"]) == 2 and not piece.get("completed") and piece["params"].get("emboss", 1) and _drawn(piece):
        return build_keep(piece, name, kind="bloques")      # a plain box whose faces carry a drawing: the box itself, drawn in
    cx, cy = 0.0, 0.0
    rings = [(cx, cy, z, sx, sy) for z, sx, sy in piece["profile"]]
    if len(rings) < 2:
        return
    bm = bmesh.new()
    loft_rect(bm, rings)
    finish(bm, name, "bloques", bevel=float(piece["params"].get("bevel", 0.25)))


# ------------------------------------------------------------------------------------------------------------ lathe
def build_lathe(piece, name):
    cx, cy = 0.0, 0.0
    seg = int(piece["params"].get("segments", 24))
    prof = [(max(sx, sy) / 2, z) for z, sx, sy in piece["profile"]]
    polar = piece.get("polar") or {}
    sides = int(piece["params"].get("sides", 0))
    turn, hole = 0.0, 0.0
    if polar:
        prof = [(r, z) for z, r in polar["radii"]]
        hole = float(polar.get("hole", 0.0))
    if sides >= 3:                                         # a prism: the corners it has, where it has them
        seg = sides
        if polar and polar.get("sides") == sides:
            turn = math.radians(polar["phase"])
    bm = bmesh.new()
    rings = []
    flutes = int(piece["params"].get("flutes", 0))
    if flutes > 0:                                         # a fluted shaft: channels between the base and the capital
        seg = flutes * 6
        zmin, zmax = prof[0][1], prof[-1][1]
        shaft = [(r, z) for r, z in prof if r > 1e-6]
        body_r = sorted(r for r, _ in shaft)[len(shaft) // 2] if shaft else 0.0
        extra = []
        for (r0, za), (r1, zb) in zip(prof, prof[1:]):     # more rings along the shaft so the channels end softly
            extra.append((r0, za))
            if zb - za > 0.3 * (zmax - zmin):
                for f in (0.06, 0.94):
                    extra.append((r0 + (r1 - r0) * f, za + (zb - za) * f))
        prof = extra + [prof[-1]]

    def radius(r, z, i):
        if flutes <= 0 or r < 1e-6:
            return r
        t = (z - prof[0][1]) / max(prof[-1][1] - prof[0][1], 1e-9)
        inside = 0.0 if (t < 0.04 or t > 0.96 or r > body_r * 1.12) else 1.0
        phase = (i % 6) / 6.0
        return r * (1.0 - inside * 0.09 * math.sin(math.pi * phase) ** 0.8)

    for r, z in prof:
        if r < 1e-6:
            rings.append([bm.verts.new((cx, cy, z))])
        else:
            rings.append([bm.verts.new((cx + radius(r, z, i) * math.cos(turn + 2 * math.pi * i / seg), cy + radius(r, z, i) * math.sin(turn + 2 * math.pi * i / seg), z)) for i in range(seg)])
    for a, b in zip(rings, rings[1:]):
        for i in range(seg):
            j = (i + 1) % seg
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                bm.faces.new((a[0], b[j], b[i]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[j], b[0]))
            else:
                bm.faces.new((a[i], a[j], b[j], b[i]))
    ends = [(ring, k == 0) for k, ring in enumerate((rings[0], rings[-1])) if len(ring) > 1]
    if hole > 1e-6 and len(ends) == 2:                     # a ring: an inner wall and a flat band at each end
        inner = [[bm.verts.new((v.co.x * hole / max(math.hypot(v.co.x, v.co.y), 1e-9), v.co.y * hole / max(math.hypot(v.co.x, v.co.y), 1e-9), v.co.z))
                  for v in ring] for ring, _ in ends]
        for i in range(seg):
            j = (i + 1) % seg
            bm.faces.new((inner[0][j], inner[0][i], inner[1][i], inner[1][j]))
            bm.faces.new((rings[0][j], rings[0][i], inner[0][i], inner[0][j]))
            bm.faces.new((rings[-1][i], rings[-1][j], inner[1][j], inner[1][i]))
    else:
        for ring, bottom in ends:                          # each end as a band of quads around a small middle
            inner = [bm.verts.new((cx + 0.35 * (v.co.x - cx), cy + 0.35 * (v.co.y - cy), v.co.z)) for v in ring]
            for i in range(seg):
                j = (i + 1) % seg
                bm.faces.new((ring[j], ring[i], inner[i], inner[j]) if bottom else (ring[i], ring[j], inner[j], inner[i]))
            if seg == 4:
                bm.faces.new(inner[::-1] if bottom else inner)
            else:
                mid = bm.verts.new((cx, cy, ring[0].co.z))
                if seg % 2 == 0:                           # quads all the way to the middle
                    for i in range(0, seg, 2):
                        q = (inner[i], inner[(i + 1) % seg], inner[(i + 2) % seg], mid)
                        bm.faces.new(q[::-1] if bottom else q)
                else:
                    for i in range(seg):
                        q = (inner[i], inner[(i + 1) % seg], mid)
                        bm.faces.new(q[::-1] if bottom else q)
    finish(bm, name, "torneados", bevel=float(piece["params"].get("bevel", 0.12)), smooth=sides < 3)


# ------------------------------------------------------------------------------------------------------------ sweep
def build_sweep(piece, name):
    """A beam along its own line, straight or curved: box sections joined by quads, a quad at each end."""
    if piece["params"].get("emboss", 1) and "smooth" not in piece["params"] and _drawn(piece):
        return build_keep(piece, name, kind="vigas")       # its texture draws on it: the beam itself, drawn in
    info = piece["sections"]
    st = [list(v) for v in info["stations"]]
    extra = int(piece["params"].get("smooth", 3 if len(st) >= 4 else 0))
    if extra > 0 and len(st) >= 3:                         # a curved beam gets sections in between, on a smooth line through its own
        fine = []
        for k in range(len(st) - 1):
            p0, p1, p2, p3 = st[max(k - 1, 0)], st[k], st[k + 1], st[min(k + 2, len(st) - 1)]
            for j in range(extra + 1):
                t = j / (extra + 1)
                fine.append([0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t ** 3)
                             for a, b, c, d in zip(p0, p1, p2, p3)])
        st = fine + [st[-1]]
    bm = bmesh.new()
    loops = []
    for x, y0, y1, z0, z1 in st:
        if y1 - y0 < 1e-6:
            y0, y1 = y0 - 0.02, y1 + 0.02
        if z1 - z0 < 1e-6:
            z0, z1 = z0 - 0.02, z1 + 0.02
        ring = ((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))
        loops.append([bm.verts.new(p if info["axis"] == "x" else (p[1], p[0], p[2])) for p in ring])
    for a, b in zip(loops, loops[1:]):
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.faces.new(loops[0][::-1])
    bm.faces.new(loops[-1])
    finish(bm, name, "vigas", bevel=float(piece["params"].get("bevel", 0.15)))


# ------------------------------------------------------------------------------------------------------------- roof
def build_roof(piece, name):
    """A hipped roof: fascia, a concave slope with upturned corners, half-round tiles down every side, hip ridges."""
    P = piece["params"]
    prof = piece["profile"]
    cx, cy = 0.0, 0.0
    z0, ex, ey = prof[0]
    # the fascia ends where the size starts to shrink; the slope ends at the smallest ring
    k1 = max(i for i, (z, sx, sy) in enumerate(prof) if abs(sx - ex) < 1e-3 * max(ex, 1) and abs(sy - ey) < 1e-3 * max(ey, 1))
    z1 = prof[k1][0] if k1 > 0 else z0 + 0.12 * (prof[-1][0] - z0)
    top = min(prof[k1:], key=lambda r: r[1] * r[2])
    z2, tx, ty = top
    H = max(z2 - z1, 1e-6)
    n = max(4, int(P.get("tiles", 14)))                    # tiles (and quads) per side
    rows = max(3, int(P.get("rows", 6)))
    power = float(P.get("concave", 1.45))
    lift = float(P.get("corner_lift", 0.35)) * H
    real = piece.get("roof") or {}
    measured = bool(real) and real.get("lift", 0) > 1e-6 and "corner_lift" not in P and "concave" not in P
    if measured:                                           # the model has its own eaves: its rise, its level stretch, its slope
        ex = ey = real["eave"]
        z1 = z0 + max(real["fascia"], 0.02 * H)
        lift, flat = real["lift"], real["flat"]
        line = [(ex, z1)] + [(w, z) for z, w in real["line"]]
        tx = ty = line[-1][0]
        H = max(line[-1][1] - z1, 1e-6)

    def surf(xn, yn, t, dz=0.0):
        """A point of the roof: xn, yn in -1..1 across the ring at height t (0 eave, 1 top)."""
        wx, wy = ex + (tx - ex) * t, ey + (ty - ey) * t
        if measured:
            k = max(i for i in range(len(line) - 1) if line[i][0] >= wx - 1e-9)
            (wa, za), (wb, zb) = line[k], line[k + 1]
            f = (wa - wx) / max(wa - wb, 1e-9)
            f = f * f * (3 - 2 * f) if 0 < k < len(line) - 2 else f
            zl = za + (zb - za) * min(max(f, 0.0), 1.0)
            u = min(abs(xn), abs(yn)) if max(abs(xn), abs(yn)) > 0.999 else abs(xn) * abs(yn)
            rise = min(max((u - flat) / max(1 - flat, 1e-6), 0.0), 1.0)
            z = zl + lift * rise * rise * (3 - 2 * rise) * (1 - t) ** 2 + dz
        else:
            z = z1 + H * (t ** power) + lift * (abs(xn) * abs(yn)) ** 1.5 * (1 - t) ** 2 + dz
        return Vector((cx + xn * wx / 2, cy + yn * wy / 2, z))

    def ring_params():
        out = []
        for side in range(4):
            for i in range(n):
                u = -1 + 2 * i / n
                out.append(((u, -1), (1, u), (-u, 1), (-1, -u))[side])
        return out

    params = ring_params()
    bm = bmesh.new()
    rings = [[bm.verts.new(surf(xn, yn, 0, dz=-(z1 - z0))) for xn, yn in params]]          # under the fascia
    rings += [[bm.verts.new(surf(xn, yn, j / rows)) for xn, yn in params] for j in range(rows + 1)]
    m = len(params)
    for a, b in zip(rings, rings[1:]):
        for i in range(m):
            j = (i + 1) % m
            bm.faces.new((a[i], a[j], b[j], b[i]))
    # soffit: a grid of quads under the eaves that shares the fascia's lower edge
    grid = {}
    for iy in range(n + 1):
        for ix in range(n + 1):
            xn, yn = -1 + 2 * ix / n, -1 + 2 * iy / n
            if ix in (0, n) or iy in (0, n):
                continue
            grid[(ix, iy)] = bm.verts.new(surf(xn, yn, 0, dz=-(z1 - z0)))
    low = rings[0]
    for i in range(n):
        grid[(i, 0)] = low[i]; grid[(n, i)] = low[n + i]; grid[(n - i, n)] = low[2 * n + i]; grid[(0, n - i)] = low[3 * n + i]
    for iy in range(n):
        for ix in range(n):
            bm.faces.new((grid[(ix, iy)], grid[(ix, iy + 1)], grid[(ix + 1, iy + 1)], grid[(ix + 1, iy)]))
    top = rings[-1]                                        # where the next storey sits: a grid of quads too
    tg = {}
    for iy in range(1, n):
        for ix in range(1, n):
            tg[(ix, iy)] = bm.verts.new(surf(-1 + 2 * ix / n, -1 + 2 * iy / n, 1.0))
    for i in range(n):
        tg[(i, 0)] = top[i]; tg[(n, i)] = top[n + i]; tg[(n - i, n)] = top[2 * n + i]; tg[(0, n - i)] = top[3 * n + i]
    for iy in range(n):
        for ix in range(n):
            bm.faces.new((tg[(ix, iy)], tg[(ix + 1, iy)], tg[(ix + 1, iy + 1)], tg[(ix, iy + 1)]))
    finish(bm, name, "tejados", bevel=float(P.get("bevel", 0.15)))

    # tiles: half-round ribs running down each side, shorter towards the hips
    r = float(P.get("tile_radius", 0.0)) or min(ex, ey) / n * 0.22
    arc = 6
    bm = bmesh.new()
    for side in range(4):
        for i in range(n):
            u = -1 + 2 * (i + 0.5) / n                     # across the eave
            reach = (1 - abs(u)) * (ex if side % 2 == 0 else ey) / max((ex - tx) if side % 2 == 0 else (ey - ty), 1e-6)
            t_end = min(1.0, reach)
            if t_end < 0.08:
                continue
            steps = max(2, int(round(rows * t_end)))
            sections = []
            for s in range(steps + 1):
                t = t_end * s / steps
                w = (ex + (tx - ex) * t) if side % 2 == 0 else (ey + (ty - ey) * t)
                un = max(-1.0, min(1.0, u * (ex if side % 2 == 0 else ey) / max(w, 1e-6)))   # the rib keeps its distance from the middle
                at = lambda tt, uu=un: surf(*(((uu, -1), (1, uu), (-uu, 1), (-1, -uu))[side]), tt)
                p = at(t)
                ahead = at(min(t + 0.02, 1.0)) - at(max(t - 0.02, 0.0))
                across = Vector(((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))[side])
                normal = across.cross(ahead).normalized()
                if normal.z < 0:
                    normal = -normal
                if s == 0:
                    p = p - ahead.normalized() * r * 0.6   # the tile end hangs a little over the fascia
                sections.append([bm.verts.new(p + across * (r * math.cos(math.pi * a / arc)) + normal * (r * math.sin(math.pi * a / arc))) for a in range(arc + 1)])
            for a, b in zip(sections, sections[1:]):
                for q in range(arc):
                    bm.faces.new((a[q], a[q + 1], b[q + 1], b[q]))
            e = sections[0]                                # the round end of the tile: two quads and a triangle
            bm.faces.new((e[6], e[5], e[1], e[0])); bm.faces.new((e[5], e[4], e[2], e[1])); bm.faces.new((e[4], e[3], e[2]))
    finish(bm, name + " tejas", "tejados", smooth=True, weld=False)

    # hip ridges: a tube down every corner
    bm = bmesh.new()
    rr, seg = r * 1.9, 8
    for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        path = [surf(sx_, sy_, j / rows) for j in range(rows + 1)]
        path[0] = path[0] - (path[1] - path[0]).normalized() * rr * 0.6
        loops = []
        for j, p in enumerate(path):
            d = (path[min(j + 1, rows)] - path[max(j - 1, 0)]).normalized()
            side = d.cross(Vector((0, 0, 1))).normalized()
            up = side.cross(d).normalized()
            loops.append([bm.verts.new(p + up * (rr * 0.55) + side * (rr * math.cos(2 * math.pi * q / seg)) + up * (rr * math.sin(2 * math.pi * q / seg))) for q in range(seg)])
        for a, b in zip(loops, loops[1:]):
            for q in range(seg):
                bm.faces.new((a[q], a[(q + 1) % seg], b[(q + 1) % seg], b[q]))
        for lp, flip in ((loops[0], True), (loops[-1], False)):
            for q in ((0, 1, 2, 3), (0, 3, 4, 7), (4, 5, 6, 7)):
                vs = [lp[i] for i in q]
                bm.faces.new(vs[::-1] if flip else vs)
    finish(bm, name + " limatesas", "tejados", smooth=True, weld=False)


# ------------------------------------------------------------------------------------------------------------- body
def build_body(piece, name):
    """A storey: posts at the corners and between bays, beams, and a recessed panel (or lattice, or door) in every bay."""
    P = piece["params"]
    prof = piece["profile"]
    cx, cy = 0.0, 0.0
    z0, sx, sy = prof[0]
    z1 = prof[-1][0]
    bays = (max(1, int(P.get("bays_x", 3))), max(1, int(P.get("bays_y", 3))))
    post = float(P.get("post", 0.0)) or min(sx, sy) * 0.075
    beam = float(P.get("beam", 0.0)) or post * 0.8
    recess = post * 0.35
    depth = float(P.get("panel_depth", 0.0)) or post * 0.3
    levels = sorted(set([z0, z1] + [z for z, _, _ in prof[1:-1]]))
    bands = [(a, b) for a, b in zip(levels, levels[1:])]
    zone_kind = list(P.get("zones", []))                   # per band, bottom to top: panel, lattice, plain
    door_bay = P.get("door_bay", None)

    bm = bmesh.new()                                       # posts
    xs = [cx - sx / 2 + post / 2 + (sx - post) * i / bays[0] for i in range(bays[0] + 1)]
    ys = [cy - sy / 2 + post / 2 + (sy - post) * i / bays[1] for i in range(bays[1] + 1)]
    spots = {(x, y) for x in xs for y in (ys[0], ys[-1])} | {(x, y) for y in ys for x in (xs[0], xs[-1])}
    round_posts = str(P.get("post_section", "square")) == "round"
    for x, y in sorted(spots):
        if round_posts:
            seg = 12
            a = [bm.verts.new((x + post / 2 * math.cos(2 * math.pi * i / seg), y + post / 2 * math.sin(2 * math.pi * i / seg), z0)) for i in range(seg)]
            b = [bm.verts.new((v.co.x, v.co.y, z1)) for v in a]
            for i in range(seg):
                bm.faces.new((a[i], a[(i + 1) % seg], b[(i + 1) % seg], b[i]))
        else:
            box(bm, x - post / 2, y - post / 2, z0, x + post / 2, y + post / 2, z1)
    finish(bm, name + " pilares", "cuerpos", bevel=post * 0.12, smooth=round_posts, weld=False)

    bm = bmesh.new()                                       # beams: a ring at the foot, at the head and at every level between
    for i, z in enumerate(levels):
        za, zb = (z, z + beam) if i == 0 else ((z - beam, z) if i == len(levels) - 1 else (z - beam / 2, z + beam / 2))
        o = post * 0.12                                    # beams stand a little proud of the posts
        x0, x1, y0, y1 = cx - sx / 2 - o, cx + sx / 2 + o, cy - sy / 2 - o, cy + sy / 2 + o
        box(bm, x0, y0, za, x1, y0 + beam, zb); box(bm, x0, y1 - beam, za, x1, y1, zb)
        box(bm, x0, y0 + beam, za, x0 + beam, y1 - beam, zb); box(bm, x1 - beam, y0 + beam, za, x1, y1 - beam, zb)
    finish(bm, name + " vigas", "cuerpos", bevel=beam * 0.12, weld=False)

    bm = bmesh.new()                                       # walls: one recessed panel per bay and band
    bars = bmesh.new()
    walls = []
    for i in range(bays[0]):
        walls.append(((xs[i] + post / 2, cy - sy / 2 + recess), (xs[i + 1] - post / 2, cy - sy / 2 + recess), (0, -1), ("x", i)))
        walls.append(((xs[i + 1] - post / 2, cy + sy / 2 - recess), (xs[i] + post / 2, cy + sy / 2 - recess), (0, 1), ("x2", i)))
    for i in range(bays[1]):
        walls.append(((cx + sx / 2 - recess, ys[i] + post / 2), (cx + sx / 2 - recess, ys[i + 1] - post / 2), (1, 0), ("y", i)))
        walls.append(((cx - sx / 2 + recess, ys[i + 1] - post / 2), (cx - sx / 2 + recess, ys[i] + post / 2), (-1, 0), ("y2", i)))
    for (ax, ay), (bx, by), normal, (wall_id, bay) in walls:
        for band, (za, zb) in enumerate(bands):
            za2 = za + (beam if band == 0 else beam / 2)
            zb2 = zb - (beam if band == len(bands) - 1 else beam / 2)
            if zb2 - za2 < 1e-4:
                continue
            kind = zone_kind[band] if band < len(zone_kind) else "panel"
            is_door = door_bay is not None and wall_id == "x" and bay == int(door_bay) and band == 0
            face = bm.faces.new([bm.verts.new(p) for p in ((ax, ay, za2), (bx, by, za2), (bx, by, zb2), (ax, ay, zb2))])
            if kind == "plain" and not is_door:
                continue
            width = math.hypot(bx - ax, by - ay)
            margin = min(width, zb2 - za2) * (0.08 if is_door else 0.14)
            d = depth * (2.2 if is_door else 1.0)
            res = bmesh.ops.inset_region(bm, faces=[face], thickness=margin, depth=0.0)
            for v in face.verts:
                v.co.x -= normal[0] * d
                v.co.y -= normal[1] * d
            if kind == "lattice" and not is_door:
                count = max(2, int(P.get("bars", 5)))
                t = min(margin * 0.5, width / (count * 3))
                for k in range(count):
                    f = (k + 1) / (count + 1)
                    px, py = ax + (bx - ax) * f, ay + (by - ay) * f
                    hx, hy = (t / 2 if normal[1] else d / 2), (t / 2 if normal[0] else d / 2)
                    mx, my = px - normal[0] * d / 2, py - normal[1] * d / 2
                    box(bars, mx - hx, my - hy, za2 + margin, mx + hx, my + hy, zb2 - margin)
    finish(bm, name + " paneles", "cuerpos", bevel=0.0)
    if len(bars.verts):
        finish(bars, name + " celosías", "cuerpos", bevel=0.0, weld=False)
    else:
        bars.free()


# ----------------------------------------------------------------------------------------------------------- stairs
def build_stairs(piece, name):
    cx, cy = 0.0, 0.0
    z0, sx, sy = piece["profile"][0]
    z1 = piece["profile"][-1][0]
    n = max(2, int(piece["params"].get("steps", 5)))
    along_x = sx >= sy if piece["params"].get("axis") is None else piece["params"]["axis"] == "x"
    flip = bool(piece["params"].get("flip", False))
    bm = bmesh.new()
    for i in range(n):
        f0, f1 = i / n, 1.0
        if flip:
            f0, f1 = 0.0, 1 - i / n
        zt = z0 + (z1 - z0) * (i + 1) / n
        zb = z0 + (z1 - z0) * i / n
        if along_x:
            box(bm, cx - sx / 2 + sx * f0, cy - sy / 2, zb, cx - sx / 2 + sx * f1, cy + sy / 2, zt)
        else:
            box(bm, cx - sx / 2, cy - sy / 2 + sy * f0, zb, cx + sx / 2, cy - sy / 2 + sy * f1, zt)
    finish(bm, name, "escaleras", bevel=float(piece["params"].get("bevel", 0.15)), weld=False)


# ------------------------------------------------------------------------------------------------------------- keep
TEXTURES = PLAN.get("textures") or {}
MIN_TEXEL = 0.22                                           # mm: a texel smaller than this at print size is merged with its neighbours
_SHRUNK = {}


def _drawn(piece):
    return any(m in TEXTURES for m in piece.get("triangle_materials") or [])


def _levels(mat, factor):
    """A texture's height steps with ``factor`` texels merged into one."""
    key = (mat, factor)
    if key not in _SHRUNK:
        src = TEXTURES[mat]["levels"]
        h, w = len(src) // factor, len(src[0]) // factor
        out = []
        for y in range(h):
            row = []
            for x in range(w):
                cell = [src[y * factor + j][x * factor + i] for j in range(factor) for i in range(factor)]
                row.append(int(round(sum(cell) / len(cell))))
            out.append(row)
        _SHRUNK[key] = out
    return _SHRUNK[key]


def _cuts(lv, along_u, a, b, lo, hi):
    """Where a face running from a to b across the texture crosses a change of height: positions 0..1 along it, each
    doubled so the step gets a short sloped wall, and the size of a texel in the same measure."""
    h, w = len(lv), len(lv[0])
    n = w if along_u else h
    span = b - a
    if abs(span) < 1e-9:
        return None, 0.0
    other = range(h) if along_u else range(w)
    if hi - lo < 0.999:                                    # only the rows or columns this face really covers
        m = h if along_u else w
        first, last = math.floor(min(lo, hi) * m + 1e-6), math.ceil(max(lo, hi) * m - 1e-6)
        other = [k % m for k in range(first, last)]
    texel = 1.0 / (n * abs(span))
    out = []
    for k in range(math.floor(min(a, b) * n) + 1, math.ceil(max(a, b) * n)):
        c0, c1 = (k - 1) % n, k % n
        if along_u:
            changed = any(lv[(h - 1 - r) % h][c0] != lv[(h - 1 - r) % h][c1] for r in other)
        else:
            changed = any(lv[(h - 1 - c0) % h][r] != lv[(h - 1 - c1) % h][r] for r in other)
        if changed:
            s = (k / n - a) / span
            e = 0.22 * texel
            if e < s < 1 - e:
                out += [s - e, s + e]
    return sorted(out), texel


def _height(mat, uv):
    """A texture's height (in steps) at a point of it, blended between texels."""
    lv = TEXTURES[mat]["levels"]
    h, w = len(lv), len(lv[0])
    x, y = uv.x * w - 0.5, (1 - uv.y) * h - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    tx, ty = x - x0, y - y0
    a, b = lv[y0 % h][x0 % w], lv[y0 % h][(x0 + 1) % w]
    c, d = lv[(y0 + 1) % h][x0 % w], lv[(y0 + 1) % h][(x0 + 1) % w]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def _source(piece):
    """The piece as the game has it: welded, with its UVs and a material index per face."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("uv")
    verts = [bm.verts.new(p) for p in piece["points"]]
    names = sorted(set(piece.get("triangle_materials") or []))
    uvs = piece.get("uvs") or []
    for k, tri in enumerate(piece["triangles"]):
        try:
            face = bm.faces.new([verts[i] for i in tri])
        except ValueError:
            continue
        if k < len(uvs):
            face.material_index = names.index(piece["triangle_materials"][k])
            for loop, c in zip(face.loops, uvs[k]):
                loop[uv_layer].uv = c
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=math.radians(2), angle_shape_threshold=math.radians(60))
    return bm, uv_layer, names


def _complete_exterior(bm, blank_material):
    """Cap broad planar borders at the outside of a piece, leaving smaller holes and ambiguous borders alone."""
    if not bm.verts:
        return 0
    lo = [min(v.co[a] for v in bm.verts) for a in range(3)]
    hi = [max(v.co[a] for v in bm.verts) for a in range(3)]
    span = [hi[a] - lo[a] for a in range(3)]
    tol = max(1e-5, 0.001 * max(span))
    adjacent = {}
    for e in bm.edges:
        if e.is_boundary:
            for v in e.verts:
                adjacent.setdefault(v, []).append(e)
    remaining = set(adjacent)
    completed = 0
    while remaining:
        pending = [remaining.pop()]
        loop = set(pending)
        for v in pending:
            for e in adjacent[v]:
                other = e.other_vert(v)
                if other in remaining:
                    remaining.remove(other)
                    pending.append(other)
                    loop.add(other)
        if len(loop) < 3 or any(len(adjacent[v]) != 2 for v in loop):
            continue
        for axis in range(3):
            coords = [v.co[axis] for v in loop]
            flat = max(coords) - min(coords) <= tol
            outer = max(coords) - lo[axis] <= tol or hi[axis] - min(coords) <= tol
            if span[axis] <= 5 * tol or not (flat and outer):
                continue
            in_plane = [a for a in range(3) if a != axis]
            if any(max(v.co[a] for v in loop) - min(v.co[a] for v in loop) < 0.8 * span[a] for a in in_plane):
                continue
            edges = list({e for v in loop for e in adjacent[v]})
            faces = bmesh.ops.holes_fill(bm, edges=edges)["faces"]
            for face in faces:
                face.material_index = blank_material
            large = [face for face in faces if len(face.verts) > 4]
            if large:
                new_tris = bmesh.ops.triangulate(bm, faces=large)["faces"]
                bmesh.ops.join_triangles(bm, faces=new_tris, angle_face_threshold=math.radians(2),
                                         angle_shape_threshold=math.radians(60))
            completed += int(bool(faces))
            break
    return completed


def build_subdivided(piece, name, kind, rounded, cap=False):
    """A piece whose faces are divided into small quads (every triangle into quads too) and then carry what the
    texture draws, blended between texels. ``rounded`` also smooths the shape: a figure the game made of flat facets
    comes out as a soft surface."""
    bm, _, names = _source(piece)
    completed = _complete_exterior(bm, len(names)) if cap and not piece["params"].get("thickness_mm", 0) else 0
    edges = [e.calc_length() for e in bm.edges]
    faces = len(bm.faces)
    if not edges or not faces:
        bm.free()
        return
    drawn = [m for m in names if m in TEXTURES] if piece["params"].get("emboss", 1) else []
    texel = MIN_TEXEL
    mean = sum(edges) / len(edges)
    level = int(piece["params"].get("level", 0)) or max(2 if rounded else 1, min(4, math.ceil(math.log2(max(mean / (1.6 * texel), 1.0)))))
    while level > 1 and faces * 4 ** level > 60000:
        level -= 1
    mesh = bpy.data.meshes.new(name + " base")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name + " base", mesh)
    scene.collection.objects.link(obj)
    steps = [("SIMPLE", 1), ("CATMULL_CLARK", level - 1)] if rounded and level >= 2 else [("CATMULL_CLARK" if rounded else "SIMPLE", level)]
    for how, n in steps:                                   # a plain division first keeps beaks, tails and limbs their length
        mod = obj.modifiers.new("Subdivision", "SUBSURF")
        mod.subdivision_type = how
        mod.levels = mod.render_levels = n
    fine = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)
    bm = bmesh.new()
    bm.from_mesh(fine)
    bpy.data.meshes.remove(fine)
    uv_layer = bm.loops.layers.uv.active
    if drawn and uv_layer is not None:
        bm.normal_update()
        moves = []
        reach = 0.03 * max(piece["size"])                  # a small figure takes a shallow drawing
        for v in bm.verts:
            hs = [(_height(names[l.face.material_index], l[uv_layer].uv) * min(TEXTURES[names[l.face.material_index]]["depth"], reach) / 3.0)
                  for l in v.link_loops if l.face.material_index < len(names) and names[l.face.material_index] in TEXTURES]
            if hs:
                moves.append((v, v.normal * (sum(hs) / len(hs))))
        for v, d in moves:
            v.co += d
    thickened = _thicken(bm, piece)
    # Coincident points at a folded tip can belong to different sides of the new skin. Re-welding them would
    # create a branching edge again; the source and subdivision are already welded where they should be.
    finish(bm, name, kind, smooth=rounded, weld=not thickened["thickness_mm"])
    STATS[-1]["completed_sides"] = completed
    STATS[-1].update(thickened)


def build_organic(piece, name):
    build_subdivided(piece, name, "figuras", True)


def build_rock(piece, name):
    # SIMPLE turns even triangular facets into quads without shrinking ridges or filling terrain borders.
    build_subdivided(piece, name, "roca", False)


def _thicken(bm, piece):
    """Give an explicitly selected open surface a centred skin and quad rims; closed pieces stay as they are.

    This is a shell, not a boolean union: it must not voxel-remesh away the wing tips or the rock's ridges.
    """
    thickness = float(piece["params"].get("thickness_mm", 0.0))
    boundary = sum(e.is_boundary for e in bm.edges)
    report = {"thickness_mm": 0.0, "boundary_edges_before": boundary, "boundary_edges_after": boundary}
    if thickness <= 0 or not boundary:
        return report
    # Some game ornaments put three surface fans on one edge. Separate those fans before making their rims;
    # they remain separate shells, rather than giving Solidify an ambiguous inside/outside at the junction.
    branches = [e for e in bm.edges if len(e.link_faces) > 2]
    if branches:
        bmesh.ops.split_edges(bm, edges=branches)
    report["split_branch_edges"] = len(branches)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("shell source")
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new("shell source", mesh)
    scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Sheet thickness", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0
    # A mitered/even offset explodes at the near-180-degree folds in game wings. Unit-normal offsets are bounded
    # to half the requested thickness on each side, even at those folded tips.
    mod.use_even_offset = False
    mod.use_quality_normals = True
    fine = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bm.clear()
    bm.from_mesh(fine)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    bpy.data.meshes.remove(fine)
    report.update(thickness_mm=thickness, boundary_edges_after=sum(e.is_boundary for e in bm.edges),
                  branch_edges_after=sum(len(e.link_faces) > 2 for e in bm.edges))
    return report


def emboss(bm, uv_layer, piece, skip=()):
    """What the textures draw, modelled into the faces: every quad whose texture lies square on it becomes a grid of
    quads that follows the texels where the height changes, raised and sunk in steps with a short slope between."""
    names = sorted(set(piece.get("triangle_materials") or []))
    done = 0
    for face in list(bm.faces):
        if len(face.verts) != 4 or face.material_index >= len(names):
            continue
        mat = names[face.material_index]
        if mat in skip or mat not in TEXTURES:
            continue
        loops = list(face.loops)
        v = [l.vert for l in loops]
        uv = [l[uv_layer].uv.copy() for l in loops]
        ea, eb = uv[1] - uv[0], uv[3] - uv[0]
        scale = max(ea.length, eb.length, 1e-9)
        if (uv[0] + ea + eb - uv[2]).length > 0.02 * scale:
            continue
        if abs(ea.y) < 0.02 * scale and abs(eb.x) < 0.02 * scale:
            a_u = True
        elif abs(ea.x) < 0.02 * scale and abs(eb.y) < 0.02 * scale:
            a_u = False
        else:
            continue
        src = TEXTURES[mat]["levels"]
        la, lb = (v[1].co - v[0].co).length, (v[3].co - v[0].co).length
        da, db = (abs(ea.x), abs(eb.y)) if a_u else (abs(ea.y), abs(eb.x))
        wa, wb = (len(src[0]), len(src)) if a_u else (len(src), len(src[0]))
        if da < 1e-9 or db < 1e-9:
            continue
        texel_mm = min(la / (da * wa), lb / (db * wb))
        factor = max(1, math.ceil(MIN_TEXEL / max(texel_mm, 1e-9)))
        if min(len(src), len(src[0])) // factor < 2:
            continue
        lv = _levels(mat, factor)
        h, w = len(lv), len(lv[0])
        if a_u:
            ss, _ = _cuts(lv, True, uv[0].x, uv[1].x, uv[0].y, uv[3].y)
            ts, _ = _cuts(lv, False, 1 - uv[0].y, 1 - uv[3].y, uv[0].x, uv[1].x)
        else:
            ss, _ = _cuts(lv, False, 1 - uv[0].y, 1 - uv[1].y, uv[0].x, uv[3].x)
            ts, _ = _cuts(lv, True, uv[0].x, uv[3].x, uv[0].y, uv[1].y)
        if ss is None or ts is None or (not ss and not ts):
            continue
        ss, ts = [0.0] + ss + [1.0], [0.0] + ts + [1.0]
        if (len(ss) - 1) * (len(ts) - 1) > 6000:
            continue
        step = TEXTURES[mat]["depth"] / 3.0
        normal = face.normal.copy()
        grid = []
        for j, t in enumerate(ts):
            row = []
            for i, s in enumerate(ss):
                corner = {(0, 0): v[0], (len(ss) - 1, 0): v[1], (len(ss) - 1, len(ts) - 1): v[2], (0, len(ts) - 1): v[3]}.get((i, j))
                if corner is not None:
                    row.append(corner)
                    continue
                p = v[0].co.lerp(v[1].co, s).lerp(v[3].co.lerp(v[2].co, s), t)
                if 0 < i < len(ss) - 1 and 0 < j < len(ts) - 1:
                    q = uv[0] + ea * s + eb * t
                    p = p + normal * (lv[int(math.floor((1 - q.y) * h)) % h][int(math.floor(q.x * w)) % w] * step)
                row.append(bm.verts.new(p))
            grid.append(row)
        index = face.material_index
        bmesh.ops.delete(bm, geom=[face], context="FACES_ONLY")
        for j in range(len(ts) - 1):
            for i in range(len(ss) - 1):
                f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
                f.material_index = index
        done += 1
    return done


def _square_share(bm, uv_layer, piece, skip=()):
    """How much of the drawn surface of a piece lies on quads the texture sits square on (the ones emboss can cut)."""
    names = sorted(set(piece.get("triangle_materials") or []))
    square = total = 0.0
    for face in bm.faces:
        if face.material_index >= len(names) or names[face.material_index] in skip or names[face.material_index] not in TEXTURES:
            continue
        area = face.calc_area()
        total += area
        if len(face.verts) == 4:
            uv = [l[uv_layer].uv for l in face.loops]
            ea, eb = uv[1] - uv[0], uv[3] - uv[0]
            scale = max(ea.length, eb.length, 1e-9)
            if (uv[0] + ea + eb - uv[2]).length <= 0.02 * scale and ((abs(ea.y) < 0.02 * scale and abs(eb.x) < 0.02 * scale) or (abs(ea.x) < 0.02 * scale and abs(eb.y) < 0.02 * scale)):
                square += area
    return square / total if total > 0 else 1.0


def build_keep(piece, name, kind="originales", skip=()):
    """The original piece, cleaned: welded, triangles paired back into quads, what its textures draw modelled into
    its faces, sharp edges bevelled."""
    bm, uv_layer, names = _source(piece)
    uvs = piece.get("uvs") or []
    weighted = False
    cap = bool(piece["params"].get("cap", 1))
    if uvs and piece["params"].get("emboss", 1) and _drawn(piece) and not skip and len(bm.faces) <= 600 and _square_share(bm, uv_layer, piece) < 0.5:
        bm.free()                                          # triangles and turned mappings: the drawing goes in on a fine grid instead
        return build_subdivided(piece, name, kind, False, cap=cap)
    # Explicit thickness must operate on the open surface, before any automatic exterior cap.
    completed = _complete_exterior(bm, len(names)) if cap and not piece["params"].get("thickness_mm", 0) else 0
    if uvs and piece["params"].get("emboss", 1) and _drawn(piece):
        bm.normal_update()
        weight = bm.edges.layers.float.new("bevel_weight_edge")
        for e in bm.edges:                                 # the bevel stays on the piece's own corners, not on the drawing
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(35):
                e[weight] = 1.0
        weighted = emboss(bm, uv_layer, piece, skip) > 0
    thickened = _thicken(bm, piece)
    finish(bm, name, kind, bevel=float(piece["params"].get("bevel", 0.15)), weighted=weighted,
           weld=not thickened["thickness_mm"])
    STATS[-1]["completed_sides"] = completed
    STATS[-1].update(thickened)


# ------------------------------------------------------------------------------------------------------------ tiled
def build_tiled(piece, name):
    """A roof kept as the model has it, with half-round tiles laid down its slopes: each tile follows the faces under
    it from the top of the slope to the eave and stops where they stop (a hip, a hatch, a platform)."""
    P = piece["params"]
    on = set(piece.get("tiles_on") or [])
    build_keep(piece, name, kind="tejados", skip=on)
    pts = [Vector(p) for p in piece["points"]]
    wide = max(piece["size"][0], piece["size"][1])
    n = max(4, int(P.get("tiles", 14)))
    pitch = wide / n
    r = float(P.get("tile_radius", 0.0)) or pitch * 0.22
    groups = {}
    for tri, mat in zip(piece["triangles"], piece.get("triangle_materials") or []):
        if mat not in on:
            continue
        a, b, c = (pts[i] for i in tri)
        nrm = (b - a).cross(c - a)
        if nrm.length < 1e-9:
            continue
        nrm.normalize()
        if nrm.z < 0:
            nrm = -nrm
        if not 0.2 < nrm.z < 0.98:
            continue
        d = 0 if abs(nrm.x) >= abs(nrm.y) else 1           # the slope runs down along this axis
        groups.setdefault((d, 1 if nrm[d] > 0 else -1), []).append((a, b, c))
    arc = 6
    bm = bmesh.new()
    for (d, sign), faces in groups.items():
        o = 1 - d
        lo = min(v[o] for f in faces for v in f)
        hi = max(v[o] for f in faces for v in f)
        k = math.ceil((lo + r) / pitch - 0.5)
        while (k + 0.5) * pitch < hi - r:
            at = (k + 0.5) * pitch
            k += 1
            segs = []
            for f in faces:
                cut = []
                for p, q in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    if (p[o] - at) * (q[o] - at) < 0:
                        cut.append(p.lerp(q, (at - p[o]) / (q[o] - p[o])))
                if len(cut) == 2 and (cut[0] - cut[1]).length > 1e-6:
                    segs.append(sorted(cut, key=lambda v: sign * v[d]))          # from the top of the slope to the eave
            segs.sort(key=lambda sg: sign * sg[0][d])
            lines = []
            for sg in segs:
                if lines and (lines[-1][-1] - sg[0]).length < 0.02 * pitch:
                    lines[-1].append(sg[1])
                else:
                    lines.append(list(sg))
            across = Vector((1, 0, 0)) if o == 0 else Vector((0, 1, 0))
            for line in lines:
                if sum((b - a).length for a, b in zip(line, line[1:])) < 4 * r:
                    continue
                path = [line[0]]
                for p in line[1:]:                         # no stations closer than a tile is wide
                    if (p - path[-1]).length > 1.5 * r:
                        path.append(p)
                    else:
                        path[-1] = p if p is line[-1] else path[-1]
                if len(path) < 2:
                    continue
                path[-1] = path[-1] + (path[-1] - path[-2]).normalized() * r * 0.6   # the tile end hangs a little over the eave
                sections = []
                for j, p in enumerate(path):
                    ahead = (path[min(j + 1, len(path) - 1)] - path[max(j - 1, 0)]).normalized()
                    normal = across.cross(ahead).normalized()
                    if normal.z < 0:
                        normal = -normal
                    sections.append([bm.verts.new(p + across * (r * math.cos(math.pi * a / arc)) + normal * (r * math.sin(math.pi * a / arc))) for a in range(arc + 1)])
                for a, b in zip(sections, sections[1:]):
                    for q in range(arc):
                        bm.faces.new((a[q], a[q + 1], b[q + 1], b[q]))
                for e, flip in ((sections[-1], False), (sections[0], True)):       # round ends: two quads and a triangle
                    for q in ((6, 5, 1, 0), (5, 4, 2, 1), (4, 3, 2)):
                        vs = [e[i] for i in q]
                        bm.faces.new(vs[::-1] if flip else vs)
    if len(bm.verts):
        finish(bm, name + " tejas", "tejados", smooth=True, weld=False)
    else:
        bm.free()


BUILDERS = {"organic": build_organic, "rock": build_rock, "tiled": build_tiled, "block": build_block, "lathe": build_lathe, "sweep": build_sweep, "roof": build_roof, "body": build_body, "stairs": build_stairs, "keep": build_keep}
for piece in PLAN["pieces"]:
    kind = piece.get("kind", "keep")
    if kind == "drop":
        continue
    before = len(STATS)
    PLACE["center"], PLACE["yaw"] = tuple(piece["center"][:2]), float(piece.get("yaw", 0.0))
    try:
        BUILDERS.get(kind, build_keep)(piece, piece["name"])
    except Exception as exc:                                # one piece never stops the rest: it falls back to the original
        print("PHIDIAS piece failed", piece["name"], kind, repr(exc), flush=True)
        if "points" in piece:
            build_keep(piece, piece["name"])
    for s in STATS[before:]:
        s["piece"] = piece["id"]

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "remodel.blend"))
bpy.ops.object.select_all(action="SELECT")
bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "remodel.obj"), export_selected_objects=False, apply_modifiers=True,
                      export_triangulated_mesh=False, export_materials=False, export_uv=False, export_normals=False,
                      forward_axis="Y", up_axis="Z")
total = {"objects": len(STATS)}
json.dump({"pieces": STATS, "total": total}, open(os.path.join(OUT, "remodel.json"), "w", encoding="utf-8"))
print("PHIDIAS built", len(STATS), flush=True)
