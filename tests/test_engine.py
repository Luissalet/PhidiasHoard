"""The sculpting engine on small made-up models: loading, relief from textures, refinement that never opens the mesh,
displacement in millimetres, the ground block and the STL."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image

from phidias_hoard.engine import export, heights, meshio, recipes, surface
from phidias_hoard.engine import sculpt as S
from phidias_hoard.engine.refine import refine


def test_fill_cavities_keeps_open_tunnel_and_separate_solid():
    from phidias_hoard.engine.finish import fill_cavities

    cube = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    tunnel = []
    for i in range(4):
        j = (i + 1) % 4
        tunnel += [(i, j, j+4, i+4), (i+8, i+12, j+12, j+8),
                   (i, i+8, j+8, j), (i+4, j+4, j+12, i+12)]
    tunnel = np.array(tunnel, dtype=np.int32)
    tunnel_f = np.concatenate([tunnel[:, [0, 1, 2]], tunnel[:, [0, 2, 3]]])
    tunnel_p = np.concatenate([cube * [6, 6, 2], cube * [3, 3, 2]]) + [30, 0, 0]
    P = np.concatenate([cube * 8, cube * 5, cube * 2, cube * 2 + [16, 0, 0], tunnel_p])
    F = np.concatenate([faces, faces[:, ::-1] + 8, faces + 16, faces + 24, tunnel_f + 32])
    points, triangles, count = fill_cavities(P, F)
    assert count == 1 and S.open_edges(triangles, len(points)) == 0
    assert len(triangles) == 2 * len(faces) + len(tunnel_f)
    assert not (np.abs(points) < 6).all(axis=1).any()
    assert (points[:, 0] == 18).any()
    assert np.array_equal(points[triangles[-len(tunnel_f):]], tunnel_p[tunnel_f])


def brick_texture(path, size=32):
    """Light bricks with dark joints every 8 pixels."""
    a = np.full((size, size, 3), 200, np.uint8)
    a[::8, :] = 40
    a[:, ::8] = 40
    Image.fromarray(a).save(path)


def test_directional_detail_follows_a_tilted_shaft_and_roof():
    from phidias_hoard.engine import patterns

    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.column_stack([3 * np.cos(angles), 3 * np.sin(angles), np.zeros(len(angles))])
    turn = np.array([[0.6, 0, 0.8], [0, 1, 0], [-0.8, 0, 0.6]])
    points = np.concatenate([ring, ring + [0, 0, 22]]) @ turn
    normals = np.concatenate([ring, ring]) @ turn / 3
    i = np.arange(len(ring))
    j = np.roll(i, -1)
    n = len(ring)
    faces = np.concatenate([np.column_stack([i, j, i + n]), np.column_stack([j, j + n, i + n])])
    flutes = patterns.oriented("flutes", points, faces, normals, 1.8)
    assert np.ptp(flutes) > 0.5
    assert np.allclose(flutes[:n], flutes[n:], atol=1e-8)

    # The upper edge is four millimetres away along the slope, exactly two staggered rows.
    roof = np.array([[0, 0, 0], [7, 0, 0], [7, 2.4, 3.2], [0, 2.4, 3.2]])
    normal = np.tile([0, -0.8, 0.6], (len(roof), 1))
    tiles = patterns.oriented("tiles", roof, np.array([[0, 1, 2], [0, 2, 3]]), normal, 2)
    assert np.allclose(tiles[[0, 1]], tiles[[3, 2]], atol=1e-8)
    for name in ("carved_wood", "scales", "moss"):
        assert np.ptp(patterns.pattern(name)) > 0.4


def test_texture_counts_combine_image_periods_with_uv_repeats():
    from phidias_hoard.remodel import pieces, texture_counts

    x = np.arange(120) / 120
    light = np.uint8(128 + 45 * np.cos(2 * np.pi * 3 * x) + 33 * np.cos(2 * np.pi * 12 * x))
    texture = np.broadcast_to(light[None, :, None], (40, 120, 3)).copy()
    counts = texture_counts.texture_periods(texture)
    assert counts == ((3, 12), (0, 0))
    assert texture_counts.texture_periods(np.full_like(texture, 120)) == ((0, 0), (0, 0))

    tri = np.array([[[0, 0, 0], [30, 0, 0], [30, 0, 12]]])
    uv = np.array([[[0, 0], [2, 0], [2, 1]]])
    wall = texture_counts.estimate(tri, uv, np.array([0]), np.array([[0, -1, 0]]), np.array([180]),
                                   [30, 15, 12], lambda _: counts, roof=False)
    assert wall == {"bays_x": 6, "bars": 4}
    piece = {"dropped": False, "fits": {"rect": True, "round": False}, "size": [30, 15, 12],
             "profile": [[0, 15, 7.5], [12, 15, 7.5]], "texture_counts": wall}
    kind, params = pieces.guess(piece, 30)
    assert kind == "body" and params == {"bays_x": 6, "bays_y": 2, "bars": 4, "zones": ["lattice"]}

    tri[0, 2, 1] = 9
    roof = texture_counts.estimate(tri, uv, np.array([0]), np.array([[0, -0.8, 0.6]]), np.array([180]),
                                   [30, 15, 12], lambda _: counts, roof=True)
    assert roof == {"tiles": 24}


def test_remodel_completes_missing_back_but_keeps_window(tmp_path):
    from phidias_hoard.engine.finish import find_blender
    from phidias_hoard.remodel import build

    blender = find_blender()
    if not (blender or build.has_bpy()):
        pytest.skip("Blender is not installed")
    pts = [[-5, -4, 0], [5, -4, 0], [5, 4, 0], [-5, 4, 0], [-5, -4, 6], [5, -4, 6], [5, 4, 6], [-5, 4, 6],
           [-1, -4, 2], [1, -4, 2], [1, -4, 4], [-1, -4, 4]]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5),
             (0, 1, 9, 8), (1, 5, 10, 9), (5, 4, 11, 10), (4, 0, 8, 11)]
    faces = [[a, b, c] for a, b, c, _ in quads] + [[a, c, d] for a, _, c, d in quads]
    base = {"id": 1, "name": "Window", "center": [0, 0, 0], "size": [10, 8, 6], "points": pts, "triangles": faces}
    for kind, cap, expected in (("keep", 1, 4), ("tiled", 1, 4), ("keep", 0, 8)):
        out = tmp_path / f"{kind}-{cap}"
        piece = {**base, "kind": kind, "params": {"cap": cap, "bevel": 0}}
        stats = build.run({"pieces": [piece]}, out, blender, timeout_s=60)
        assert stats["pieces"][0]["completed_sides"] == cap
        scene = meshio.load_model(out / "remodel.obj")
        assert S.open_edges(scene.F, len(scene.P)) == expected


def test_plinth_engraving_and_figure_recess_are_cut_at_print_size(tmp_path):
    from phidias_hoard.engine import plinth_detail
    from phidias_hoard.engine.finish import find_blender

    blender = find_blender()
    if not blender:
        pytest.skip("Blender is not installed")
    scene = meshio.load_model(write_wall(tmp_path))
    st = S.SculptSettings(ground=False, plinth="round", size_mm=45, plinth_name="Lugia",
                          plinth_socket_mm=12, plinth_socket_depth_mm=1.5)
    result = S.sculpt(scene, {}, st)
    points, triangles = plinth_detail.decorate(result, st, blender, tmp_path)
    assert S.open_edges(triangles, len(points)) == 0
    top = result.P[:, 2].min() + 0.01
    engraved = points[np.isclose(points[:, 2], top - 0.6, atol=1e-4)]
    pocket = points[np.isclose(points[:, 2], top - 1.5, atol=1e-4)]
    assert len(engraved) > 20 and engraved[:, 1].max() < result.P[:, 1].min()
    assert np.ptp(pocket[:, 0]) == pytest.approx(12, abs=0.01)
    assert pocket[:, 0].min() > result.P[:, 0].max()


def test_learned_relief_keeps_atlas_islands_separate_and_reuses_cache(tmp_path, monkeypatch):
    from phidias_hoard.engine import learned_relief

    atlas = np.zeros((64, 64, 4), dtype=np.uint8)
    atlas[5:27, 5:27] = (128, 130, 90, 255)
    atlas[37:59, 37:59] = (140, 80, 120, 255)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    uv = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]]])
    scene = meshio.Scene(pos, faces, uv, np.zeros(2, dtype=int), [meshio.Material("atlas", albedo=atlas, atlas=True)])
    model = tmp_path / "weights"
    model.mkdir()
    (model / "model.safetensors").touch()
    (model / "config.json").touch()
    calls = []

    def predict(args, **kwargs):
        from pathlib import Path
        from types import SimpleNamespace

        paths = list(Path(args[-1]).glob("*.png"))
        calls.append(len(paths))
        for path in paths:
            with Image.open(path) as img:
                x, y = np.meshgrid(np.arange(img.width), np.arange(img.height))
            np.save(path.with_suffix(".npy"), (np.sin(x / 2) + np.cos(y / 3)).astype(np.float32))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(learned_relief.subprocess, "run", predict)
    cache = tmp_path / "cache"
    h = learned_relief.height_map(scene, 0, str(model), "", cache)
    assert np.ptp(h[5:27, 5:27]) > 0.5 and np.ptp(h[37:59, 37:59]) > 0.5
    assert not h[27:37].any()
    assert np.array_equal(learned_relief.height_map(scene, 0, str(model), "", cache), h) and calls == [2]


def test_split_creates_mating_watertight_parts_and_a_3mf(tmp_path):
    import xml.etree.ElementTree as ET
    import zipfile

    from phidias_hoard.engine import finish, print_split

    blender = finish.find_blender()
    if not blender:
        pytest.skip("Blender is not installed")
    pts = np.array([[0, 0, 0], [30, 0, 0], [30, 20, 0], [0, 20, 0], [0, 0, 12], [30, 0, 12], [30, 20, 12], [0, 20, 12]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    spec = {"axis": "x", "position_mm": None, "pins": 2, "diameter_mm": 4, "depth_mm": 3, "clearance_mm": 0.25}
    result = print_split.create(pts, faces, spec, tmp_path, blender)
    assert result["position_mm"] == 15
    a, fa = finish.read_stl(tmp_path / "piece-a.stl")
    b, fb = finish.read_stl(tmp_path / "piece-b.stl")
    assert S.open_edges(fa, len(a)) == S.open_edges(fb, len(b)) == 0
    assert a[:, 0].max() == pytest.approx(18)
    assert b[:, 0].max() == pytest.approx(15)
    with zipfile.ZipFile(tmp_path / "pieces.3mf") as package:
        model = ET.fromstring(package.read("3D/3dmodel.model"))
    assert model.attrib["unit"] == "millimeter" and len(model.findall(".//{*}item")) == 2


def write_wall(tmp_path, with_shadow=True):
    """A floor, a wall standing on it and (optionally) a shadow plane, as an OBJ with its MTL and textures."""
    brick_texture(tmp_path / "brick.png")
    Image.new("RGB", (8, 8), (90, 120, 90)).save(tmp_path / "grass.png")
    Image.new("RGBA", (8, 8), (0, 0, 0, 90)).save(tmp_path / "h_kage.png")
    mtl = "newmtl grass\nmap_Kd grass.png\nnewmtl brick\nmap_Kd brick.png\nnewmtl h_kage\nmap_Kd h_kage.png\n"
    (tmp_path / "wall.mtl").write_text(mtl)
    obj = ["mtllib wall.mtl",
           "v 0 0 0", "v 10 0 0", "v 10 0 10", "v 0 0 10",          # floor (y up)
           "v 2 0 5", "v 8 0 5", "v 8 6 5", "v 2 6 5",              # wall
           "v 1 0.05 1", "v 3 0.05 1", "v 3 0.05 3", "v 1 0.05 3",  # shadow
           "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "vt 3 0", "vt 3 3", "vt 0 3",
           "usemtl grass", "f 1/1 3/3 2/2", "f 1/1 4/4 3/3",
           "usemtl brick", "f 5/1 6/5 7/6", "f 5/1 7/6 8/7"]
    if with_shadow:
        obj += ["usemtl h_kage", "f 9/1 11/3 10/2", "f 9/1 12/4 11/3"]
    (tmp_path / "wall.obj").write_text("\n".join(obj) + "\n")
    return tmp_path / "wall.obj"


def test_obj_loads_z_up_with_materials(tmp_path):
    scene = meshio.load_model(write_wall(tmp_path))
    assert len(scene.F) == 6 and len(scene.materials) == 3
    assert scene.extents[2] == pytest.approx(6.0)            # the wall's height became z
    brick = next(m for m in scene.materials if m.name == "brick")
    assert brick.size == (32, 32)
    assert scene.UV.max() == pytest.approx(3.0)              # tiling coordinates are kept


def test_obj_with_wrong_mtllib_and_backslash_paths(tmp_path):
    (tmp_path / "Textures").mkdir()
    brick_texture(tmp_path / "Textures" / "wall.png")
    (tmp_path / "scene.mtl").write_text("newmtl wall.png\nmap_Kd Textures\\\\wall.png\n")
    (tmp_path / "scene.obj").write_text("mtllib other name.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl wall.png\nf 1/1 2/2 3/3\n")
    scene = meshio.load_model(tmp_path / "scene.obj")
    assert scene.materials[0].size == (32, 32)


def test_lines_are_carved_and_flat_colour_is_not(tmp_path):
    brick_texture(tmp_path / "b.png")
    albedo = np.asarray(Image.open(tmp_path / "b.png").convert("RGBA"))
    h = heights.height_map(albedo, None, "lines")
    assert h.shape[0] >= 256 and h.min() < -0.6 and h.max() <= 0.0
    joint = heights.sample(h, np.array([[0.5 / 32, 0.5]]))[0]      # on a joint
    face = heights.sample(h, np.array([[4.5 / 32, 4.5 / 32]]))[0]  # middle of a brick
    assert joint < -0.5 and abs(face) < 0.1
    plain = np.full((16, 16, 4), 128, np.uint8)
    assert np.abs(heights.height_map(plain, None, "relief")).max() == 0.0


def test_normal_map_becomes_a_ridge():
    n = np.zeros((64, 64, 3), np.uint8)
    n[...] = (128, 128, 255)
    n[:, 24:32, 0] = 200                                     # surface tilts one way...
    n[:, 32:40, 0] = 56                                      # ...then back: a ridge or a groove along the column
    h = heights.normal_to_height(n)
    profile = h.mean(axis=0)
    assert abs(profile[32]) > 5 * abs(profile[5])
    assert heights.has_normal_detail(n) and not heights.has_normal_detail(np.full((8, 8, 3), (128, 128, 255), np.uint8))


def test_refinement_keeps_the_mesh_closed():
    # a tetrahedron: closed, every edge shared by two triangles
    P = np.array([[0, 0, 0], [10, 0, 0], [5, 9, 0], [5, 3, 8]], float)
    F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32)
    UV = np.random.default_rng(1).random((4, 3, 2)).astype(np.float32)
    want_some = lambda P_, F_, UV_, parent: parent < 2            # only two of the four faces ask for detail
    P2, F2, UV2, parent, info = refine(P, F, UV, want_some, 0.8)
    assert len(F2) > 200 and S.open_edges(F2, len(P2)) == 0
    assert set(np.unique(parent)) == {0, 1, 2, 3}
    a, b, c = P2[F2[:, 0]], P2[F2[:, 1]], P2[F2[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    assert area == pytest.approx(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def test_refinement_respects_the_budget():
    P = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], float)
    F = np.array([[0, 1, 2]], np.int32)
    UV = np.zeros((1, 3, 2), np.float32)
    _, F2, _, _, info = refine(P, F, UV, lambda P_, F_, UV_, p: np.ones(len(F_), bool), 0.01, max_faces=5000)
    assert len(F2) <= 5200 and info["stopped"] == "budget"


def test_sculpt_carves_the_wall_drops_the_shadow_and_closes_the_ground(tmp_path):
    scene = meshio.load_model(write_wall(tmp_path))
    rec = {m.name: S.MaterialRecipe.from_dict(recipes.guess(m, 0.5)) for m in scene.materials}
    assert rec["h_kage"].recipe == "drop" and rec["brick"].recipe == "relief" and rec["grass"].recipe == "flat"
    st = S.SculptSettings(size_mm=100, min_edge_mm=0.6, max_faces=400_000)
    out = S.sculpt(scene, rec, st)
    assert out.info["size_mm"][0] == pytest.approx(100, abs=0.5)
    assert 0.3 < out.info["max_relief_mm"] <= 0.5 + 1e-6
    assert out.info["triangles"] > 20_000
    assert set(np.unique(out.M)) == {0, 1}                    # the shadow plane is gone
    assert out.ground.sum() >= 2 and not out.ground[out.M == 1].any()   # the floor is ground, the wall is not
    # the floor was not refined: nothing to carve there
    assert (out.M == 0).sum() < 50
    P, F = S.assemble(out, st)
    assert P[:, 2].min() == pytest.approx(0.0)
    block = S.close_ground(out, st.base_mm)[1]
    ground_and_block = np.concatenate([out.F[out.ground], block])
    assert S.open_edges(ground_and_block, len(P)) == 0        # the ground block is a closed solid
    size = export.write_stl(tmp_path / "out.stl", P, F)
    assert size == 84 + 50 * len(F)
    assert struct.unpack("<I", (tmp_path / "out.stl").read_bytes()[80:84])[0] == len(F)


def test_normal_integration_on_the_mesh_follows_the_slopes():
    # a flat square split into a grid; the "normal map" tilts the middle band, so the heights must rise and fall there
    n = 24
    xs, ys = np.meshgrid(np.linspace(0, 10, n + 1), np.linspace(0, 10, n + 1))
    P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)
    UV = (P[F][..., :2] / 10.0).astype(np.float32)
    normal = np.zeros((64, 64, 3), np.uint8)
    normal[...] = (128, 128, 255)
    normal[:, 20:32, 0] = 60                                  # leaning towards -u: the surface rises along +u
    normal[:, 32:44, 0] = 196                                 # and comes back down
    sx, sy = surface.slope_maps(normal)
    used, h = surface.integrate(P, F, UV, sx, sy, cutoff=50.0)
    full = np.zeros(len(P)); full[used] = h
    grid = full.reshape(n + 1, n + 1)
    middle, edge = grid[:, n // 2].mean(), grid[:, 1].mean()
    assert middle - edge > 0.5                                # a ridge down the middle
    assert np.abs(grid[:, 1] - grid[:, -2]).max() < 0.3       # both sides end at the same level


def test_gltf_uses_the_uv_set_the_texture_names(tmp_path):
    import base64
    import io
    import json

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4")
    uv0 = np.array([[5, 5], [6, 5], [5, 6]], "<f4")
    uv1 = np.array([[0, 0], [1, 0], [0, 1]], "<f4")
    idx = np.array([0, 1, 2], "<u2")
    png = io.BytesIO()
    Image.new("RGB", (4, 4), (128, 128, 255)).save(png, "PNG")
    blob = pos.tobytes() + uv0.tobytes() + uv1.tobytes() + idx.tobytes() + b"\0\0" + png.getvalue()
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 24},
             {"buffer": 0, "byteOffset": 60, "byteLength": 24}, {"buffer": 0, "byteOffset": 84, "byteLength": 6},
             {"buffer": 0, "byteOffset": 92, "byteLength": len(png.getvalue())}]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "scale": [2, 2, 2]}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}, "indices": 3, "material": 0}]}],
           "materials": [{"name": "baked", "normalTexture": {"index": 0, "texCoord": 1}}],
           "textures": [{"source": 0}], "images": [{"bufferView": 4, "mimeType": "image/png"}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                         {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
                         {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
                         {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}],
           "bufferViews": views, "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]}
    (tmp_path / "s.gltf").write_text(json.dumps(doc))
    scene = meshio.load_model(tmp_path / "s.gltf")
    assert scene.materials[0].atlas and scene.materials[0].normal is not None
    assert scene.UV.max() <= 1.0 + 1e-6                       # TEXCOORD_1, not the tiling set
    assert scene.extents.max() == pytest.approx(2.0)          # the node's scale is applied


def test_zip_listing_and_safe_extraction(tmp_path):
    import zipfile

    z = tmp_path / "pack.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Scene/OBJ/scene.obj", "v 0 0 0\n")
        f.writestr("Scene/scene.glb", b"glTF")
        f.writestr("../evil.txt", "no")
    models = meshio.zip_models(z)
    assert meshio.pick_model(models).endswith(".glb")
    meshio.extract_zip(z, tmp_path / "out")
    assert (tmp_path / "out" / "Scene" / "scene.glb").is_file() and not (tmp_path / "evil.txt").exists()


def test_cutout_removes_the_transparent_part_and_gives_the_rest_thickness(tmp_path):
    # a standing card whose texture is opaque only in its left half
    a = np.zeros((16, 16, 4), np.uint8)
    a[:, :8] = (60, 140, 60, 255)
    a[4:8, 2:6, :3] = (20, 60, 20)
    Image.fromarray(a).save(tmp_path / "leaf.png")
    (tmp_path / "card.mtl").write_text("newmtl leaf\nmap_Kd leaf.png\n")
    (tmp_path / "card.obj").write_text("mtllib card.mtl\nv 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                                       "usemtl leaf\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    scene = meshio.load_model(tmp_path / "card.obj")
    guess = recipes.guess(scene.materials[0], 0.4)
    assert guess["cutout"] is True
    st = S.SculptSettings(size_mm=40, min_edge_mm=0.5, ground=False, sheet_mm=1.0)
    out = S.sculpt(scene, {"leaf": S.MaterialRecipe.from_dict(guess)}, st)
    assert out.P[:, 0].max() < 22.0                           # the see-through half is gone (the card was 40 mm wide)
    assert out.info["sheet_triangles"] > 100
    assert S.open_edges(out.F, len(out.P)) == 0               # front, back and rim: a closed sheet
    assert np.ptp(out.P[:, 1]) == pytest.approx(1.0, abs=0.5) # about a millimetre thick


def test_collada_scene_with_transform_and_texture(tmp_path):
    brick_texture(tmp_path / "wall.png")
    (tmp_path / "s.dae").write_text("""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><up_axis>Y_UP</up_axis></asset>
 <library_images><image id="img"><init_from>wall.png</init_from></image></library_images>
 <library_effects><effect id="fx"><profile_COMMON>
   <newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>
   <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
   <technique sid="c"><phong><diffuse><texture texture="samp" texcoord="UV"/></diffuse></phong></technique>
 </profile_COMMON></effect></library_effects>
 <library_materials><material id="mat" name="wall"><instance_effect url="#fx"/></material></library_materials>
 <library_geometries><geometry id="g"><mesh>
   <source id="p"><float_array id="pa" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
     <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
   <source id="t"><float_array id="ta" count="8">0 0 2 0 2 2 0 2</float_array>
     <technique_common><accessor source="#ta" count="4" stride="2"/></technique_common></source>
   <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
   <polylist material="m0" count="1"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="TEXCOORD" source="#t" offset="1" set="0"/>
     <vcount>4</vcount><p>0 0 1 1 2 2 3 3</p></polylist>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="vs"><node id="n"><scale>3 3 3</scale>
   <instance_geometry url="#g"><bind_material><technique_common><instance_material symbol="m0" target="#mat"/></technique_common></bind_material></instance_geometry>
 </node></visual_scene></library_visual_scenes>
</COLLADA>""")
    scene = meshio.load_model(tmp_path / "s.dae")
    assert len(scene.F) == 2 and scene.materials[0].name == "wall" and scene.materials[0].size == (32, 32)
    assert scene.extents.max() == pytest.approx(3.0) and scene.extents[2] == pytest.approx(3.0)   # scaled, and y became z
    assert scene.UV.max() == pytest.approx(2.0)
    assert meshio.distinct_models(["A/x.glb", "A/OBJ/x.obj", "B/x.dae", "B/y.dae"]) == ["A/x.glb", "B/x.dae", "B/y.dae"]


def test_steps_turn_a_ramp_into_treads_and_risers(tmp_path):
    # a texture with 4 dark step edges across v, on a ramp that climbs 8 units over one tile
    a = np.full((32, 32, 3), 190, np.uint8)
    a[::8, :] = 30
    Image.fromarray(a).save(tmp_path / "stair.png")
    assert heights.step_pattern(np.asarray(Image.open(tmp_path / "stair.png").convert("RGBA")))["v"][0] == 4
    assert heights.step_pattern(np.asarray(Image.open(tmp_path / "stair.png").convert("RGBA")))["u"][0] == 0
    (tmp_path / "r.mtl").write_text("newmtl roof_staire\nmap_Kd stair.png\n")
    (tmp_path / "r.obj").write_text("mtllib r.mtl\nv 0 0 0\nv 10 0 0\nv 10 8 -8\nv 0 8 -8\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                                    "usemtl roof_staire\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    scene = meshio.load_model(tmp_path / "r.obj")
    guess = recipes.guess(scene.materials[0], 0.5)
    assert guess["recipe"] == "steps"
    st = S.SculptSettings(size_mm=50, min_edge_mm=0.4, ground=False)
    out = S.sculpt(scene, {"roof_staire": S.MaterialRecipe.from_dict(guess)}, st)
    assert out.info["stair_triangles"] > 1000
    z = out.P[:, 2]
    rise = np.ptp(z) / 4                                      # about four steps over the whole climb
    levels = np.round((z - z.min()) / (rise * 0.25)) * 0.25   # quarter-step bins
    counts = np.bincount((levels * 4).astype(int))
    flat_share = np.sort(counts)[-4:].sum() / counts.sum()
    assert flat_share > 0.6                                   # most points sit on four flat treads
    a3, b3, c3 = out.P[out.F[:, 0]], out.P[out.F[:, 1]], out.P[out.F[:, 2]]
    n = np.cross(b3 - a3, c3 - a3)
    area = 0.5 * np.linalg.norm(n, axis=1)
    level = np.abs(n[:, 2]) / np.maximum(2 * area, 1e-12) > 0.97
    assert area[level].sum() / area.sum() > 0.45              # treads are level, the rest are the risers


def test_enrich_adds_sized_detail_to_a_plain_material(tmp_path):
    from phidias_hoard.engine import patterns

    for name in patterns.PATTERNS:
        tile = patterns.pattern(name)
        assert tile.shape == (512, 512) and tile.min() < -0.3 and tile.max() > (-0.01 if name == "flutes" else 0.05), name
    Image.new("RGB", (8, 8), (120, 120, 120)).save(tmp_path / "plain.png")
    (tmp_path / "w.mtl").write_text("newmtl stone\nmap_Kd plain.png\n")
    (tmp_path / "w.obj").write_text("mtllib w.mtl\nv 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                                    "usemtl stone\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    scene = meshio.load_model(tmp_path / "w.obj")
    st = S.SculptSettings(size_mm=60, min_edge_mm=0.4, ground=False)
    plain = S.sculpt(scene, {"stone": S.MaterialRecipe("flat")}, st)
    assert plain.info["triangles"] == 2 and plain.info["max_relief_mm"] == 0
    rec = S.MaterialRecipe.from_dict({"recipe": "flat", "enrich": {"pattern": "bricks", "size_mm": 4, "depth_mm": 0.5}})
    out = S.sculpt(scene, {"stone": rec}, st)
    assert out.info["triangles"] > 5000 and 0.3 < out.info["max_relief_mm"] <= 0.5 + 1e-6
    assert np.ptp(out.P[:, 1]) > 0.5                          # the wall (facing y) now has relief across it
    assert S.MaterialRecipe.from_dict({"enrich": {"pattern": "nope"}}).enrich is None


def test_budget_with_cutouts_and_sheets_reports_its_real_detail(tmp_path):
    a = np.zeros((16, 16, 4), np.uint8)
    a[:, :8] = (60, 140, 60, 255)
    Image.fromarray(a).save(tmp_path / "leaf.png")
    (tmp_path / "card.mtl").write_text("newmtl leaf\nmap_Kd leaf.png\n")
    (tmp_path / "card.obj").write_text("mtllib card.mtl\nv 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nvt 0 0\nvt 4 0\nvt 4 4\nvt 0 4\nusemtl leaf\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    scene = meshio.load_model(tmp_path / "card.obj")
    rec = S.MaterialRecipe.from_dict({"recipe": "flat", "cutout": True, "enrich": {"pattern": "bark", "size_mm": 1, "depth_mm": 0.2}})
    out = S.sculpt(scene, {"leaf": rec}, S.SculptSettings(size_mm=80, min_edge_mm=0.1, max_faces=20_000, ground=False))
    assert out.info["stopped"] == "budget" and out.info["edge_mm"] > 0.1 and out.info["sheet_triangles"] > 0


def _scene(points, tris, uv=None, materials=("m",), mat_of=None, albedo=None):
    from phidias_hoard.engine.meshio import Material, Scene

    P = np.array(points, np.float64)
    F = np.array(tris, np.int64)
    UV = np.zeros((len(F), 3, 2), np.float32) if uv is None else np.array(uv, np.float32)
    M = np.zeros(len(F), np.int64) if mat_of is None else np.array(mat_of, np.int64)
    return Scene(P=P, F=F, UV=UV, M=M, materials=[Material(n, albedo=albedo) for n in materials])


def test_pieces_are_measured_for_what_they_are():
    from phidias_hoard.remodel import pieces

    # a hexagonal ring
    pts, tris = [], []
    for z in (0.0, 1.0):
        for r in (2.0, 1.6):
            pts += [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a)), z) for a in range(0, 360, 60)]
    for i in range(6):
        j = (i + 1) % 6
        tris += [(i, j, 12 + j), (i, 12 + j, 12 + i), (6 + i, 18 + j, 6 + j), (6 + i, 18 + i, 18 + j), (12 + i, 12 + j, 18 + j), (12 + i, 18 + j, 18 + i)]
    ring = pieces.split(_scene(pts, tris), 5.0, set())[0]
    assert ring["polar"]["sides"] == 6 and ring["polar"]["hole"] == pytest.approx(8.0, rel=0.02)
    assert pieces.guess(ring, 20)[0] == "lathe" and pieces.guess(ring, 20)[1]["sides"] == 6

    # a beam that climbs along a curve
    pts, tris = [], []
    for k, (x, z) in enumerate(((0, 0), (10, 1), (20, 4), (30, 9))):
        pts += [(x, -1, z), (x, 1, z), (x, 1, z + 1.5), (x, -1, z + 1.5)]
        if k:
            a, b = 4 * (k - 1), 4 * k
            for i in range(4):
                j = (i + 1) % 4
                tris += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    beam = pieces.split(_scene(pts, tris), 1.0, set())[0]
    assert beam["fits"]["sweep"] and not beam["fits"]["rect"] and pieces.guess(beam, 40)[0] == "sweep"
    assert len(beam["sections"]["stations"]) == 4

    # ground with relief inside a square outline is not a block
    side = np.linspace(-10, 10, 5)
    pts = [(x, y, 0.0 if abs(x) == 10 or abs(y) == 10 else 3.0 * ((i + j) % 2) + 1) for i, x in enumerate(side) for j, y in enumerate(side)]
    tris = []
    for i in range(4):
        for j in range(4):
            a = i * 5 + j
            tris += [(a, a + 5, a + 6), (a, a + 6, a + 1)]
    ground = pieces.split(_scene(pts, tris), 1.0, set())[0]
    assert not ground["fits"]["rect"] and pieces.guess(ground, 40)[0] == "keep"


def test_texture_becomes_height_steps():
    from phidias_hoard.engine import heights

    tex = np.full((8, 8, 4), 200, np.uint8)
    tex[2:6, 2:6, :3] = 60                                  # a dark panel in a light frame
    lv = heights.texel_levels(tex)
    assert lv[0, 0] == 0 and lv[3, 3] < 0 and set(np.unique(lv)) == {int(lv[3, 3]), 0}
    assert heights.texel_levels(np.full((8, 8, 4), 128, np.uint8)) is None


def test_a_figure_with_its_own_texture_is_rounded_not_kept():
    from phidias_hoard.remodel import pieces

    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    pts = [(0.0, 0.0, 0.0)] + [(np.cos(a) * (1 + 0.4 * rng.random()), np.sin(a) * (1 + 0.4 * rng.random()), z + 0.2 * rng.random())
                               for z in np.linspace(0.5, 4, 8) for a in ang] + [(0.1, 0.0, 4.8)]
    tris = [(0, 1 + (i + 1) % 8, 1 + i) for i in range(8)]
    for r in range(7):
        for i in range(8):
            a, b = 1 + r * 8 + i, 1 + r * 8 + (i + 1) % 8
            tris += [(a, b, b + 8), (a, b + 8, a + 8)]
    tris += [(57 + i, 57 + (i + 1) % 8, 65) for i in range(8)]
    uv = rng.random((len(tris), 3, 2))
    uv[0] = [[0, 0], [1, 0], [1, 1]]
    figure = pieces.split(_scene(pts, tris, uv=uv, albedo=np.zeros((8, 8, 4), np.uint8)), 3.0, set())[0]
    assert figure["figure"] and pieces.guess(figure, 20)[0] == "organic" and "organic" in pieces.allowed(figure)
