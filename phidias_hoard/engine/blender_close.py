"""Runs inside Blender (``blender -b --python blender_close.py -- in.ply out.stl voxel_mm shell_mm max_faces``).

Gives thickness to loose sheets, fuses everything into one closed skin with a voxel remesh and writes an STL. The
result is a single watertight solid whatever the game geometry was (open walls, crossing props, single-sided cards)."""

import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
voxel, shell, max_faces = float(argv[2]), float(argv[3]), int(argv[4])

bpy.ops.wm.read_factory_settings(use_empty=True)
if hasattr(bpy.ops.wm, "ply_import"):
    bpy.ops.wm.ply_import(filepath=src)
else:                                                   # Blender 3.x
    bpy.ops.import_mesh.ply(filepath=src)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print("PHIDIAS in", len(obj.data.polygons), flush=True)

if shell > 0:
    m = obj.modifiers.new("shell", "SOLIDIFY")
    m.thickness = shell
    m.offset = -1.0
    m.use_even_offset = False
    m.use_rim = True
    bpy.ops.object.modifier_apply(modifier=m.name)

m = obj.modifiers.new("fuse", "REMESH")
m.mode = "VOXEL"
m.voxel_size = voxel
m.adaptivity = 0.0
m.use_smooth_shade = False
bpy.ops.object.modifier_apply(modifier=m.name)
faces = len(obj.data.polygons)
print("PHIDIAS voxel", faces, flush=True)

tris = faces * 2
if max_faces > 0 and tris > max_faces:
    m = obj.modifiers.new("thin", "DECIMATE")
    m.ratio = max_faces / tris
    bpy.ops.object.modifier_apply(modifier=m.name)

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
if hasattr(bpy.ops.wm, "stl_export"):
    bpy.ops.wm.stl_export(filepath=dst, export_selected_objects=True, apply_modifiers=True)
else:
    bpy.ops.export_mesh.stl(filepath=dst, use_selection=True)
print("PHIDIAS out", len(obj.data.polygons), flush=True)
