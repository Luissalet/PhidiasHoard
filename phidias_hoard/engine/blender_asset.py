"""Runs inside Blender (``blender -b --python blender_asset.py -- <folder>``).

Turns a high-resolution mesh into a game asset: retopology (QuadriFlow, or edge-collapse decimation), automatic UVs,
tangent-space normals and ambient occlusion baked from the high mesh, and glTF / FBX export. ``<folder>`` holds
``job.json`` and ``high.ply``; the script writes the textures, the models, ``low_P.npy`` / ``low_F.npy`` (the low mesh in
the print's millimetres) and ``report.json``."""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

root = Path(sys.argv[sys.argv.index("--") + 1])
spec = json.loads((root / "job.json").read_text(encoding="utf-8"))
report = {"notes": []}


def say(text):
    print("PHIDIAS " + text, flush=True)


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


bpy.ops.wm.read_factory_settings(use_empty=True)
if hasattr(bpy.ops.wm, "ply_import"):
    bpy.ops.wm.ply_import(filepath=str(root / "high.ply"))
else:                                                   # Blender 3.x
    bpy.ops.import_mesh.ply(filepath=str(root / "high.ply"))
high = bpy.context.selected_objects[0]
high.name = "high"
report["high_faces"] = len(high.data.polygons)
extent = max(high.dimensions)
say(f"high {report['high_faces']}")

# ------------------------------------------------------------------ retopology
low = high.copy()
low.data = high.data.copy()
low.name = "low"
bpy.context.collection.objects.link(low)


def collapse(obj, faces):
    """Edge-collapse the object's mesh to about ``faces`` triangles."""
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if tris <= faces:
        return
    select_only(obj)
    m = obj.modifiers.new("collapse", "DECIMATE")
    m.ratio = max(faces / tris, 1e-4)
    bpy.ops.object.modifier_apply(modifier=m.name)


def merge_and_clean(obj):
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=extent * 1e-7)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


target = int(spec["target_faces"])
method = spec["method"]
if method == "quadriflow":
    try:
        collapse(low, 400_000)                           # QuadriFlow is slow and hungry on millions of triangles
        merge_and_clean(low)
        select_only(low)
        before = len(low.data.polygons)
        bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=target, use_mesh_symmetry=False,
                                         use_preserve_sharp=False, use_preserve_boundary=False, smooth_normals=False,
                                         seed=0)
        if len(low.data.polygons) < 4 or len(low.data.polygons) > max(target * 4, 1000):
            raise RuntimeError(f"QuadriFlow returned {len(low.data.polygons)} faces from {before}")
        report["method"] = "quadriflow"
    except Exception as exc:                            # open or non-manifold meshes, or a Blender without the operator
        report["notes"].append(f"QuadriFlow no pudo rehacer la malla ({str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__}); se ha reducido por colapso de aristas.")
        low.data = high.data.copy()
        method = "decimate"
if method == "decimate":
    collapse(low, target)
    report["method"] = "decimate"
merge_and_clean(low)
select_only(low)
bpy.ops.object.shade_smooth()
low_poly = low.data.polygons
report["low_faces"] = len(low_poly)
report["low_quads"] = sum(1 for p in low_poly if len(p.vertices) == 4)
report["low_tris"] = sum(len(p.vertices) - 2 for p in low_poly)
say(f"low {report['low_faces']} ({report['method']})")

# ------------------------------------------------------------------ UVs
select_only(low)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
if not low.data.uv_layers:
    low.data.uv_layers.new(name="UVMap")
margin = 16.0 / spec["texture_px"]
bpy.ops.uv.smart_project(angle_limit=math.radians(float(spec["uv_angle"])), island_margin=margin)
bpy.ops.object.mode_set(mode="OBJECT")
uvs = low.data.uv_layers.active
low.data.calc_loop_triangles()
uv = np.zeros(len(low.data.loops) * 2, dtype=np.float64)
uvs.data.foreach_get("uv", uv)
uv = uv.reshape(-1, 2)
tri_loops = np.zeros(len(low.data.loop_triangles) * 3, dtype=np.int64)
low.data.loop_triangles.foreach_get("loops", tri_loops)
a, b, c = (uv[tri_loops.reshape(-1, 3)[:, i]] for i in range(3))
area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum() / 2
report["uv_coverage"] = round(float(area), 4)
say(f"uv coverage {report['uv_coverage']}")

# ------------------------------------------------------------------ the low mesh in the print's millimetres
points = np.zeros(len(low.data.vertices) * 3, dtype=np.float64)
low.data.vertices.foreach_get("co", points)
tri_vertices = np.zeros(len(low.data.loop_triangles) * 3, dtype=np.int64)
low.data.loop_triangles.foreach_get("vertices", tri_vertices)
np.save(root / "low_P.npy", points.reshape(-1, 3).astype(np.float32))
np.save(root / "low_F.npy", tri_vertices.reshape(-1, 3).astype(np.int32))

# ------------------------------------------------------------------ bake
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.use_denoising = False
size = int(spec["texture_px"])
material = bpy.data.materials.new("asset")
material.use_nodes = True
low.data.materials.clear()
low.data.materials.append(material)
tree = material.node_tree
for node in list(tree.nodes):
    if node.type != "OUTPUT_MATERIAL":
        tree.nodes.remove(node)
output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
shader.inputs["Base Color"].default_value = (0.8, 0.78, 0.74, 1.0)
shader.inputs["Roughness"].default_value = 0.8
tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])


def target_image(name):
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    image.colorspace_settings.name = "Non-Color"
    image.filepath_raw = str(root / f"{name}.png")
    image.file_format = "PNG"
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    node.interpolation = "Linear"
    tree.nodes.active = node
    return image, node


def run_bake(kind, samples):
    scene.cycles.samples = samples
    settings = scene.render.bake
    wanted = {"use_selected_to_active": True, "cage_extrusion": extent * float(spec["cage_pct"]) / 100,
              "max_ray_distance": extent * float(spec["cage_pct"]) / 50, "margin": 16, "margin_type": "EXTEND",
              "normal_space": "TANGENT", "use_clear": True, "use_cage": False, "target": "IMAGE_TEXTURES"}
    for key, value in wanted.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    select_only(high)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type=kind)


normal_image, normal_node = target_image("asset_normal")
say("baking normals")
run_bake("NORMAL", 1)
normal_image.save()
ao_image, ao_node = target_image("asset_ao")
say("baking occlusion")
run_bake("AO", int(spec["ao_samples"]))
ao_image.save()


def mean_of(image):
    pixels = np.zeros(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(-1, 4)[:, :3]


normals = mean_of(normal_image)
ao = mean_of(ao_image)[:, 0]
baked = ao > 0                                          # pixels no island covers stay black in the occlusion map
normals, ao = normals[baked], ao[baked]
report["normal_flat_share"] = round(float(np.mean(np.abs(normals - np.array([0.5, 0.5, 1.0])).max(1) < 0.02)), 3) if ao.size else 0.0
report["ao_mean"] = round(float(ao.mean()), 3) if ao.size else 0.0
report["ao_range"] = [round(float(ao.min()), 3), round(float(ao.max()), 3)] if ao.size else [0.0, 0.0]

# ------------------------------------------------------------------ the material that travels with the model
tree.nodes.remove(ao_node)
ao_node = tree.nodes.new("ShaderNodeTexImage")
ao_node.image = ao_image
bump = tree.nodes.new("ShaderNodeNormalMap")
bump.space = "TANGENT"
bump.uv_map = uvs.name
tree.links.new(normal_node.outputs["Color"], bump.inputs["Color"])
tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
group = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")   # where the glTF exporter reads occlusion
if hasattr(group, "interface"):
    group.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
else:                                                                         # Blender 3.x
    group.inputs.new("NodeSocketFloat", "Occlusion")
holder = tree.nodes.new("ShaderNodeGroup")
holder.node_tree = group
holder.location = (300, -400)
separate = tree.nodes.new("ShaderNodeSeparateColor") if hasattr(bpy.types, "ShaderNodeSeparateColor") else tree.nodes.new("ShaderNodeSeparateRGB")
tree.links.new(ao_node.outputs["Color"], separate.inputs[0])
tree.links.new(separate.outputs[0], holder.inputs["Occlusion"])
ao_node.image.colorspace_settings.name = "Non-Color"

# ------------------------------------------------------------------ game scale: metres, base on the floor, centred
bpy.data.objects.remove(high, do_unlink=True)
select_only(low)
low.data.update()
lo, hi = points.reshape(-1, 3).min(0), points.reshape(-1, 3).max(0)
scale = float(spec["size_m"]) / max(float((hi - lo).max()), 1e-9)
low.data.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
low.data.transform(Matrix.Scale(scale, 4))
low.data.update()
report["size_m"] = [round(float(v), 4) for v in (hi - lo) * scale]
low.name = spec["name"]

formats = []
select_only(low)
if "glb" in spec["formats"]:
    bpy.ops.export_scene.gltf(filepath=str(root / "asset.glb"), export_format="GLB", use_selection=True,
                              export_apply=False, export_yup=True)
    formats.append("glb")
if "fbx" in spec["formats"]:
    bpy.ops.export_scene.fbx(filepath=str(root / "asset.fbx"), use_selection=True, path_mode="COPY", embed_textures=True,
                             mesh_smooth_type="FACE", add_leaf_bones=False)
    formats.append("fbx")
report["formats"] = formats
report["triangles"] = report["low_tris"]
(root / "report.json").write_text(json.dumps(report), encoding="utf-8")
say("done")
