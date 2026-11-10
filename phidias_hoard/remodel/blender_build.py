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


def collection(kind):
    if kind not in COLLECTIONS:
        c = bpy.data.collections.new(kind)
        scene.collection.children.link(c)
        COLLECTIONS[kind] = c
    return COLLECTIONS[kind]


def finish(bm, name, kind, bevel=0.0, smooth=False, weld=True):
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
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    if bevel > 0:
        m = obj.modifiers.new("Bisel", "BEVEL")
        m.width = bevel
        m.segments = 2
        m.limit_method = "ANGLE"
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
    cx, cy = piece["center"][:2]
    rings = [(cx, cy, z, sx, sy) for z, sx, sy in piece["profile"]]
    if len(rings) < 2:
        return
    bm = bmesh.new()
    loft_rect(bm, rings)
    finish(bm, name, "bloques", bevel=float(piece["params"].get("bevel", 0.25)))


# ------------------------------------------------------------------------------------------------------------ lathe
def build_lathe(piece, name):
    cx, cy = piece["center"][:2]
    seg = int(piece["params"].get("segments", 24))
    prof = [(max(sx, sy) / 2, z) for z, sx, sy in piece["profile"]]
    bm = bmesh.new()
    rings = []
    for r, z in prof:
        if r < 1e-6:
            rings.append([bm.verts.new((cx, cy, z))])
        else:
            rings.append([bm.verts.new((cx + r * math.cos(2 * math.pi * i / seg), cy + r * math.sin(2 * math.pi * i / seg), z)) for i in range(seg)])
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
    if len(rings[0]) > 1:                                  # the bottom as a fan of quads around a small centre ring
        inner = [bm.verts.new((cx + 0.35 * (v.co.x - cx), cy + 0.35 * (v.co.y - cy), v.co.z)) for v in rings[0]]
        for i in range(seg):
            j = (i + 1) % seg
            bm.faces.new((rings[0][j], rings[0][i], inner[i], inner[j]))
        bm.faces.new(inner[::-1])
    finish(bm, name, "torneados", bevel=float(piece["params"].get("bevel", 0.12)), smooth=True)


# ------------------------------------------------------------------------------------------------------------- roof
def build_roof(piece, name):
    """A hipped roof: fascia, a concave slope with upturned corners, half-round tiles down every side, hip ridges."""
    P = piece["params"]
    prof = piece["profile"]
    cx, cy = piece["center"][:2]
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

    def surf(xn, yn, t, dz=0.0):
        """A point of the roof: xn, yn in -1..1 across the ring at height t (0 eave, 1 top)."""
        wx, wy = ex + (tx - ex) * t, ey + (ty - ey) * t
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
    cx, cy = piece["center"][:2]
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
    cx, cy = piece["center"][:2]
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
def build_keep(piece, name):
    """The original piece, cleaned: welded, triangles paired back into quads, sharp edges bevelled."""
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in piece["points"]]
    for tri in piece["triangles"]:
        try:
            bm.faces.new([verts[i] for i in tri])
        except ValueError:
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=math.radians(2), angle_shape_threshold=math.radians(60))
    finish(bm, name, "originales", bevel=float(piece["params"].get("bevel", 0.15)))


BUILDERS = {"block": build_block, "lathe": build_lathe, "roof": build_roof, "body": build_body, "stairs": build_stairs, "keep": build_keep}
for piece in PLAN["pieces"]:
    kind = piece.get("kind", "keep")
    if kind == "drop":
        continue
    before = len(STATS)
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
