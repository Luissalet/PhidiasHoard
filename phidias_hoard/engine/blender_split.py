"""Cut a watertight object on a plane and add mating pegs in Blender."""

import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


root = Path(sys.argv[sys.argv.index("--") + 1])
spec = json.loads((root / "cut.json").read_text(encoding="utf-8"))
axis = spec["axis"]
cut = spec["position"]
radius, depth, clearance = spec["diameter"] / 2, spec["depth"], spec["clearance"]
bpy.ops.wm.read_factory_settings(use_empty=True)
if hasattr(bpy.ops.wm, "ply_import"):
    bpy.ops.wm.ply_import(filepath=str(root / "source.ply"))
else:
    bpy.ops.import_mesh.ply(filepath=str(root / "source.ply"))
source = bpy.context.selected_objects[0]
lo, hi = [min(c[i] for c in source.bound_box) for i in range(3)], [max(c[i] for c in source.bound_box) for i in range(3)]
cross = [i for i in range(3) if i != axis]


def choose_pegs():
    count = spec["pins"]
    if count == 0:
        return []
    tree = BVHTree.FromObject(source, bpy.context.evaluated_depsgraph_get())
    points = []
    for x in range(1, 12):
        for y in range(1, 12):
            p = [0.0, 0.0, 0.0]
            p[axis] = cut
            for i, step in zip(cross, (x, y)):
                p[i] = lo[i] + (hi[i] - lo[i]) * step / 12
            distance = min((tree.find_nearest(Vector(p))[3] or 0), 100)
            points.append((distance, p))
    chosen = []
    for _, p in sorted(points, reverse=True):
        if any(math.dist(p, previous) < spec["diameter"] + 2 * clearance + 1 for previous in chosen):
            continue
        for offset in (-min(depth, 1.0), 0, depth + clearance):
            probe = Vector(p)
            probe[axis] += offset
            near, normal, _index, dist = tree.find_nearest(probe)
            if near is None or dist < radius + clearance + 0.25 or (probe - near).dot(normal) >= 0:
                break
        else:
            chosen.append(p)
            if len(chosen) == count:
                return chosen
    raise RuntimeError("No hay material suficiente en el plano para esos encajes. Cambia el plano o el diámetro.")


def apply_boolean(part, cutter, operation):
    bpy.context.view_layer.objects.active = part
    mod = part.modifiers.new("Cut", "BOOLEAN")
    mod.operation, mod.solver, mod.object = operation, "EXACT", cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def export(part, path):
    bpy.ops.object.select_all(action="DESELECT")
    part.select_set(True)
    bpy.context.view_layer.objects.active = part
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True)
    else:
        bpy.ops.export_mesh.stl(filepath=str(path), use_selection=True)


pegs = choose_pegs()
for lower in (True, False):
    part = source.copy()
    part.data = source.data.copy()
    bpy.context.collection.objects.link(part)
    a, b = [v - 2 for v in lo], [v + 2 for v in hi]
    if lower:
        b[axis] = cut
    else:
        a[axis] = cut
    bpy.ops.mesh.primitive_cube_add(size=1, location=[(x + y) / 2 for x, y in zip(a, b)])
    cutter = bpy.context.object
    cutter.scale = [y - x for x, y in zip(a, b)]
    apply_boolean(part, cutter, "INTERSECT")
    for p in pegs:
        length = depth if lower else depth + clearance
        center = p.copy()
        center[axis] += (length - 0.15) / 2
        rotation = ((0, math.pi / 2, 0), (math.pi / 2, 0, 0), (0, 0, 0))[axis]
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius if lower else radius + clearance,
                                           depth=length + 0.15, location=center, rotation=rotation)
        apply_boolean(part, bpy.context.object, "UNION" if lower else "DIFFERENCE")
    export(part, root / ("piece-a.stl" if lower else "piece-b.stl"))
