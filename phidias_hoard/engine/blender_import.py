"""Run inside background Blender. Export the active scene as a self-contained GLB."""

import sys
from pathlib import Path

import bpy


source, destination = map(Path, sys.argv[sys.argv.index("--") + 1:])
if source.suffix.lower() == ".blend":
    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.ops.import_scene.fbx(filepath=str(source)) != {"FINISHED"}:
        raise RuntimeError("FBX import cancelled")

if not any(obj.type == "MESH" and len(obj.data.polygons) for obj in bpy.context.scene.objects):
    raise RuntimeError("The active scene has no mesh faces")

if bpy.ops.export_scene.gltf(filepath=str(destination), export_format="GLB", export_animations=False) != {"FINISHED"}:
    raise RuntimeError("glTF export cancelled")
