"""Cut a text engraving and a measured figure recess into a closed plinth in Blender."""

import json
import sys
from pathlib import Path

import bpy


root = Path(sys.argv[sys.argv.index("--") + 1])
cuts = json.loads((root / "cuts.json").read_text(encoding="utf-8"))
top = cuts["top"]
bpy.ops.wm.read_factory_settings(use_empty=True)
if hasattr(bpy.ops.wm, "ply_import"):
    bpy.ops.wm.ply_import(filepath=str(root / "plate.ply"))
else:
    bpy.ops.import_mesh.ply(filepath=str(root / "plate.ply"))
plate = bpy.context.selected_objects[0]


def subtract(cutter):
    bpy.context.view_layer.objects.active = plate
    mod = plate.modifiers.new("Plinth cut", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


if socket := cuts.get("socket"):
    depth = socket["depth"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=socket["radius"], depth=depth + 0.2,
                                       location=(*socket["center"], top - depth / 2 + 0.1))
    subtract(bpy.context.object)

if engraving := cuts.get("engraving"):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.text_add(location=(*engraving["center"], top - 0.2))
    text = bpy.context.object
    text.data.body = engraving["name"]
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = engraving["font_size"]
    text.data.extrude = 0.4
    bpy.context.view_layer.update()
    if text.dimensions.x > engraving["max_width"]:
        scale = engraving["max_width"] / text.dimensions.x
        text.scale.x = text.scale.y = scale
    bpy.ops.object.convert(target="MESH")
    subtract(bpy.context.object)

bpy.ops.object.select_all(action="DESELECT")
plate.select_set(True)
bpy.context.view_layer.objects.active = plate
if hasattr(bpy.ops.wm, "stl_export"):
    bpy.ops.wm.stl_export(filepath=str(root / "plate.stl"), export_selected_objects=True)
else:
    bpy.ops.export_mesh.stl(filepath=str(root / "plate.stl"), use_selection=True)
