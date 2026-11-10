"""Reading game models into one simple structure.

A :class:`Scene` holds welded positions ``P`` (n, 3), triangles ``F`` (m, 3) into ``P``, texture coordinates per corner
``UV`` (m, 3, 2; origin bottom-left, they may tile outside 0..1), a material index per triangle ``M`` (m,) and the
materials with their pictures. Z is up (game files are Y up and are turned on load)."""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

MODEL_EXTS = (".obj", ".glb", ".gltf", ".dae")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".webp", ".dds")


@dataclass
class Material:
    name: str
    albedo: Optional[np.ndarray] = None      # (h, w, 4) uint8
    normal: Optional[np.ndarray] = None      # (h, w, 3) uint8, tangent space
    color: tuple[int, int, int] = (200, 200, 200)
    atlas: bool = False                       # a baked atlas (unrelated islands side by side), not a tiling texture

    @property
    def size(self) -> tuple[int, int]:
        return (0, 0) if self.albedo is None else (self.albedo.shape[1], self.albedo.shape[0])


@dataclass
class Scene:
    P: np.ndarray
    F: np.ndarray
    UV: np.ndarray
    M: np.ndarray
    materials: list[Material] = field(default_factory=list)
    name: str = ""

    @property
    def extents(self) -> np.ndarray:
        return self.P.max(0) - self.P.min(0) if len(self.P) else np.zeros(3)

    def face_areas(self) -> np.ndarray:
        a, b, c = self.P[self.F[:, 0]], self.P[self.F[:, 1]], self.P[self.F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _image(path_or_img: Any, mode: str) -> Optional[np.ndarray]:
    try:
        img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
        return np.asarray(img.convert(mode))
    except Exception:
        return None


def weld(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Unique positions within ``tol`` and the index of each input point in them."""
    if len(points) == 0:
        return points.reshape(0, 3), np.zeros(0, np.int64)
    q = np.round(points / tol).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return points[first].astype(np.float64), inverse.reshape(-1)


def _finish(pos: np.ndarray, tri_uv: np.ndarray, mat: np.ndarray, materials: list[Material], name: str, y_up: bool) -> Scene:
    """``pos`` is (m, 3, 3): one position per corner. Turns to Z up, welds and drops degenerate triangles."""
    pos = np.asarray(pos, np.float64).reshape(-1, 3, 3)
    if y_up:
        pos = np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], axis=-1)
    flat = pos.reshape(-1, 3)
    ext = float((flat.max(0) - flat.min(0)).max()) if len(flat) else 1.0
    P, inv = weld(flat, max(ext, 1e-9) * 1e-6)
    F = inv.reshape(-1, 3).astype(np.int32)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    scene = Scene(P, F[keep], np.asarray(tri_uv, np.float32).reshape(-1, 3, 2)[keep], np.asarray(mat, np.int16)[keep], materials, name)
    areas = scene.face_areas()
    ok = areas > (ext * 1e-7) ** 2
    scene.F, scene.UV, scene.M = scene.F[ok], scene.UV[ok], scene.M[ok]
    return scene


# ------------------------------------------------------------------------------------------------------------- OBJ
def _find_file(base: Path, ref: str) -> Optional[Path]:
    ref = ref.strip().strip('"').replace("\\", "/")
    if not ref:
        return None
    cand = base / ref
    if cand.is_file():
        return cand
    target = os.path.basename(ref).lower()
    stem = os.path.splitext(target)[0]
    best = None
    for root, _dirs, files in os.walk(base):
        for f in files:
            low = f.lower()
            if low == target:
                return Path(root) / f
            if best is None and os.path.splitext(low)[0] == stem and low.endswith(IMAGE_EXTS):
                best = Path(root) / f
    return best


def _read_mtl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cur: Optional[dict[str, Any]] = None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        key = key.lower()
        rest = rest.strip()
        if key == "newmtl":
            cur = out.setdefault(rest, {})
        elif cur is None:
            continue
        elif key == "kd":
            try:
                cur["kd"] = tuple(int(max(0.0, min(1.0, float(v))) * 255) for v in rest.split()[:3])
            except ValueError:
                pass
        elif key in ("map_kd", "map_bump", "bump", "norm", "map_kn"):
            # options such as "-bm 1.0" come before the file name
            parts = rest.split()
            while parts and parts[0].startswith("-"):
                parts = parts[2:]
            cur["albedo" if key == "map_kd" else "normal"] = " ".join(parts)
    return out


def load_obj(path: Path) -> Scene:
    path = Path(path)
    v: list[tuple[float, float, float]] = []
    vt: list[tuple[float, float]] = []
    tri_p: list[int] = []
    tri_t: list[int] = []
    tri_m: list[int] = []
    names: list[str] = []
    index: dict[str, int] = {}
    libs: list[str] = []
    cur = -1
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("v "):
                p = raw.split()
                v.append((float(p[1]), float(p[2]), float(p[3])))
            elif raw.startswith("vt "):
                p = raw.split()
                vt.append((float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
            elif raw.startswith("f "):
                if cur < 0:
                    cur = index.setdefault("", len(names))
                    if cur == len(names):
                        names.append("")
                corners = []
                for tok in raw.split()[1:]:
                    a = tok.split("/")
                    pi = int(a[0])
                    ti = int(a[1]) if len(a) > 1 and a[1] else 0
                    corners.append((pi - 1 if pi > 0 else len(v) + pi, ti - 1 if ti > 0 else (len(vt) + ti if ti < 0 else -1)))
                for k in range(1, len(corners) - 1):
                    for c in (corners[0], corners[k], corners[k + 1]):
                        tri_p.append(c[0])
                        tri_t.append(c[1])
                    tri_m.append(cur)
            elif raw.startswith("usemtl"):
                name = raw[6:].strip()
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                cur = index[name]
            elif raw.startswith("mtllib"):
                libs.append(raw[6:].strip())
    mtl: dict[str, dict[str, Any]] = {}
    for lib in libs + [path.stem + ".mtl"]:
        found = _find_file(path.parent, lib)
        if found:
            mtl.update(_read_mtl(found))
    if not mtl:                                           # a mtllib line that names a file that is not there
        for other in sorted(path.parent.glob("*.mtl")):
            mtl.update(_read_mtl(other))
    materials = []
    for name in names:
        info = mtl.get(name, {})
        mat = Material(name or "sin material", color=info.get("kd", (200, 200, 200)))
        for key, mode in (("albedo", "RGBA"), ("normal", "RGB")):
            ref = info.get(key) or (name if key == "albedo" and name.lower().endswith(IMAGE_EXTS) else None)
            file = _find_file(path.parent, ref) if ref else None
            if file is not None:
                setattr(mat, key, _image(file, mode))
        materials.append(mat)
    V = np.asarray(v, np.float64).reshape(-1, 3)
    T = np.asarray(vt + [(0.0, 0.0)], np.float32).reshape(-1, 2)
    ti = np.asarray(tri_t, np.int64)
    ti[ti < 0] = len(T) - 1
    return _finish(V[np.asarray(tri_p, np.int64)].reshape(-1, 3, 3), T[ti].reshape(-1, 3, 2), np.asarray(tri_m), materials, path.stem, True)


# ------------------------------------------------------------------------------------------------------------ glTF
_COMPONENT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class _Gltf:
    """Just enough glTF 2.0: meshes with their node transforms, the UV set each texture really uses (baked scenes keep
    the atlas in TEXCOORD_1 or _2), base colour and normal pictures, embedded or beside the file."""

    def __init__(self, path: Path):
        import base64
        import json
        import struct

        self.path = path
        raw = path.read_bytes()
        self.bin: list[Optional[bytes]] = []
        if raw[:4] == b"glTF":
            _magic, _version, total = struct.unpack_from("<III", raw, 0)
            off, chunks = 12, []
            while off + 8 <= min(total, len(raw)):
                size, kind = struct.unpack_from("<II", raw, off)
                chunks.append((kind, raw[off + 8: off + 8 + size]))
                off += 8 + size + (-size % 4)
            self.doc = json.loads(next(c for k, c in chunks if k == 0x4E4F534A))
            glb_bin = next((c for k, c in chunks if k == 0x004E4942), None)
        else:
            self.doc = json.loads(raw.decode("utf-8", "replace"))
            glb_bin = None
        for buf in self.doc.get("buffers", []):
            uri = buf.get("uri")
            if uri is None:
                self.bin.append(glb_bin)
            elif uri.startswith("data:"):
                self.bin.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                file = _find_file(path.parent, _unquote(uri))
                self.bin.append(file.read_bytes() if file else None)

    def view(self, index: int) -> bytes:
        v = self.doc["bufferViews"][index]
        data = self.bin[v["buffer"]] or b""
        return data[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.doc["accessors"][index]
        dtype, width, count = np.dtype(_COMPONENT[acc["componentType"]]), _WIDTH[acc["type"]], acc["count"]
        if "bufferView" not in acc:
            return np.zeros((count, width), dtype)
        v = self.doc["bufferViews"][acc["bufferView"]]
        data = self.bin[v["buffer"]] or b""
        start = v.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = v.get("byteStride") or dtype.itemsize * width
        if stride == dtype.itemsize * width:
            arr = np.frombuffer(data, dtype, count * width, start).reshape(count, width)
        else:
            arr = np.ndarray((count, width), dtype, data, start, (stride, dtype.itemsize))
        arr = arr.astype(np.float64 if dtype.kind == "f" else np.int64)
        if acc.get("normalized") and dtype.kind != "f":
            arr = arr / float(np.iinfo(dtype).max)
        return arr

    def picture(self, texture_index: Optional[int], mode: str) -> Optional[np.ndarray]:
        if texture_index is None:
            return None
        try:
            tex = self.doc["textures"][texture_index]
            src = tex.get("source", tex.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
            img = self.doc["images"][src]
            if "bufferView" in img:
                return _image(Image.open(io.BytesIO(self.view(img["bufferView"]))), mode)
            uri = img.get("uri", "")
            if uri.startswith("data:"):
                import base64
                return _image(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))), mode)
            file = _find_file(self.path.parent, _unquote(uri))
            return _image(file, mode) if file else None
        except Exception:
            return None

    def node_matrix(self, node: dict[str, Any]) -> np.ndarray:
        if "matrix" in node:
            return np.asarray(node["matrix"], float).reshape(4, 4).T
        m = np.eye(4)
        if "scale" in node:
            m = np.diag([*node["scale"], 1.0]) @ m
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            rm = np.eye(4); rm[:3, :3] = r
            m = rm @ m
        if "translation" in node:
            tm = np.eye(4); tm[:3, 3] = node["translation"]
            m = tm @ m
        return m


def _unquote(uri: str) -> str:
    from urllib.parse import unquote
    return unquote(uri)


def load_gltf(path: Path) -> Scene:
    path = Path(path)
    g = _Gltf(path)
    doc = g.doc
    pos, uvs, mats = [], [], []
    materials: list[Material] = []
    mat_index: dict[int, int] = {}
    mat_uvset: dict[int, int] = {}

    def material(i: Optional[int]) -> int:
        key = -1 if i is None else i
        if key in mat_index:
            return mat_index[key]
        info = doc["materials"][i] if i is not None else {}
        pbr = info.get("pbrMetallicRoughness", {})
        base, normal = pbr.get("baseColorTexture"), info.get("normalTexture")
        mat = Material(str(info.get("name") or f"material {len(materials)}"))
        if base:
            mat.albedo = g.picture(base.get("index"), "RGBA")
        if normal:
            mat.normal = g.picture(normal.get("index"), "RGB")
        factor = pbr.get("baseColorFactor")
        if factor:
            mat.color = tuple(int(max(0.0, min(1.0, float(v))) * 255) for v in factor[:3])
        uvset = (normal or base or {}).get("texCoord", 0)
        mat.atlas = uvset > 0
        mat_uvset[key] = uvset
        mat_index[key] = len(materials)
        materials.append(mat)
        return mat_index[key]

    def walk(node_index: int, parent: np.ndarray, depth: int = 0) -> None:
        node = doc["nodes"][node_index]
        m = parent @ g.node_matrix(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                    continue
                attrs = prim["attributes"]
                v = g.accessor(attrs["POSITION"])[:, :3]
                v = v @ m[:3, :3].T + m[:3, 3]
                idx = g.accessor(prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(v))
                faces = idx[: len(idx) // 3 * 3].reshape(-1, 3)
                if np.linalg.det(m[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                mi = material(prim.get("material"))
                key = -1 if prim.get("material") is None else prim["material"]
                name = f"TEXCOORD_{mat_uvset[key]}"
                uv = g.accessor(attrs[name])[:, :2] if name in attrs else (
                    g.accessor(attrs["TEXCOORD_0"])[:, :2] if "TEXCOORD_0" in attrs else np.zeros((len(v), 2)))
                uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)          # glTF's origin is top-left
                pos.append(v[faces]); uvs.append(uv[faces]); mats.append(np.full(len(faces), mi, np.int16))
        if depth < 64:
            for child in node.get("children", []):
                walk(child, m, depth + 1)

    scenes = doc.get("scenes") or [{"nodes": list(range(len(doc.get("nodes", []))))}]
    for root in scenes[doc.get("scene", 0) if doc.get("scene", 0) < len(scenes) else 0].get("nodes", []):
        walk(root, np.eye(4))
    if not pos:
        raise ValueError("the file has no triangles")
    return _finish(np.concatenate(pos), np.concatenate(uvs), np.concatenate(mats), materials, path.stem, True)


# --------------------------------------------------------------------------------------------------------- Collada
def _dae_matrix(node) -> np.ndarray:
    m = np.eye(4)
    for child in node:
        tag = child.tag
        try:
            vals = [float(v) for v in (child.text or "").split()]
        except ValueError:
            continue
        if tag == "matrix" and len(vals) == 16:
            m = m @ np.asarray(vals).reshape(4, 4)
        elif tag == "translate" and len(vals) == 3:
            t = np.eye(4); t[:3, 3] = vals
            m = m @ t
        elif tag == "scale" and len(vals) == 3:
            m = m @ np.diag([*vals, 1.0])
        elif tag == "rotate" and len(vals) == 4:
            axis = np.asarray(vals[:3]); n = np.linalg.norm(axis)
            if n > 1e-12:
                x, y, z = axis / n
                c, s_ = np.cos(np.radians(vals[3])), np.sin(np.radians(vals[3]))
                r = np.eye(4)
                r[:3, :3] = [[c + x * x * (1 - c), x * y * (1 - c) - z * s_, x * z * (1 - c) + y * s_],
                             [y * x * (1 - c) + z * s_, c + y * y * (1 - c), y * z * (1 - c) - x * s_],
                             [z * x * (1 - c) - y * s_, z * y * (1 - c) + x * s_, c + z * z * (1 - c)]]
                m = m @ r
    return m


def load_dae(path: Path) -> Scene:
    import xml.etree.ElementTree as ET

    path = Path(path)
    root = ET.parse(str(path)).getroot()
    for node in root.iter():                              # drop the XML namespace: plain tag names from here on
        if isinstance(node.tag, str) and "}" in node.tag:
            node.tag = node.tag.split("}", 1)[1]
    ref = lambda url: (url or "").lstrip("#")
    up = (root.findtext("asset/up_axis") or "Y_UP").strip().upper()

    images = {}
    for img in root.iterfind("library_images/image"):
        src = (img.findtext("init_from") or img.findtext("init_from/ref") or "").strip()
        if src.startswith("file://"):
            src = src[7:].lstrip("/") if re.match(r"file:///[A-Za-z]:", src) else src[7:]
        images[img.get("id")] = _unquote(src)
    effects = {}
    for eff in root.iterfind("library_effects/effect"):
        params = {}
        for np_ in eff.iter("newparam"):
            init = np_.findtext("surface/init_from") or np_.findtext("sampler2D/instance_image") or ""
            src = np_.findtext("sampler2D/source") or ""
            params[np_.get("sid")] = (init.strip(), src.strip())
            inst = np_.find("sampler2D/instance_image")
            if inst is not None and inst.get("url"):
                params[np_.get("sid")] = (ref(inst.get("url")), "")

        def resolve(name: str, depth: int = 0) -> str:
            if name in images:
                return name
            if name in params and depth < 4:
                init, src = params[name]
                return resolve(init or src, depth + 1) if (init or src) else ""
            return ""

        tex = eff.find(".//diffuse/texture")
        image = resolve(tex.get("texture", "")) if tex is not None else ""
        if not image:
            image = next((resolve(v[0]) for v in params.values() if resolve(v[0])), "")
        colour = eff.findtext(".//diffuse/color")
        rgb = (200, 200, 200)
        if colour:
            try:
                rgb = tuple(int(max(0.0, min(1.0, float(v))) * 255) for v in colour.split()[:3])
            except ValueError:
                pass
        effects[eff.get("id")] = (image, rgb)
    mat_effect = {m.get("id"): (ref(m.find("instance_effect").get("url")) if m.find("instance_effect") is not None else "", m.get("name") or m.get("id"))
                  for m in root.iterfind("library_materials/material")}

    materials: list[Material] = []
    mat_index: dict[str, int] = {}

    def material(mat_id: str) -> int:
        if mat_id in mat_index:
            return mat_index[mat_id]
        effect, name = mat_effect.get(mat_id, ("", mat_id or "sin material"))
        image, rgb = effects.get(effect, ("", (200, 200, 200)))
        mat = Material(str(name or mat_id or "sin material"), color=rgb)
        file = _find_file(path.parent, images.get(image, "")) if image else None
        if file is None and name:                         # rips often name the material after its picture
            file = _find_file(path.parent, str(name) + ".png")
        if file is not None:
            mat.albedo = _image(file, "RGBA")
        mat_index[mat_id] = len(materials)
        materials.append(mat)
        return mat_index[mat_id]

    geoms: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {}
    for geo in root.iterfind("library_geometries/geometry"):
        mesh = geo.find("mesh")
        if mesh is None:
            continue
        sources = {}
        for src in mesh.iterfind("source"):
            arr = src.find("float_array")
            if arr is None or not (arr.text or "").strip():
                continue
            acc = src.find("technique_common/accessor")
            stride = int(acc.get("stride", "3")) if acc is not None else 3
            sources[src.get("id")] = np.asarray((arr.text or "").split(), np.float64).reshape(-1, stride)
        vertices = {}
        for v in mesh.iterfind("vertices"):
            for inp in v.iterfind("input"):
                if inp.get("semantic") == "POSITION":
                    vertices[v.get("id")] = ref(inp.get("source"))
        parts = []
        for prim in mesh:
            if prim.tag not in ("triangles", "polylist", "polygons"):
                continue
            inputs = [(i.get("semantic"), int(i.get("offset", "0")), ref(i.get("source")), i.get("set", "0")) for i in prim.iterfind("input")]
            if not inputs:
                continue
            width = max(o for _, o, _, _ in inputs) + 1
            pos = next(((o, vertices.get(s, s)) for sem, o, s, _ in inputs if sem == "VERTEX"), None)
            if pos is None or pos[1] not in sources:
                continue
            uvs = sorted((st, o, s) for sem, o, s, st in inputs if sem == "TEXCOORD")
            if prim.tag == "polygons":
                polys = [np.asarray((p_.text or "").split(), np.int64).reshape(-1, width) for p_ in prim.iterfind("p")]
            else:
                flat = np.asarray(" ".join((p_.text or "") for p_ in prim.iterfind("p")).split(), np.int64).reshape(-1, width)
                if prim.tag == "triangles":
                    polys = None
                    tri = flat[: len(flat) // 3 * 3].reshape(-1, 3, width)
                else:
                    counts = np.asarray((prim.findtext("vcount") or "").split(), np.int64)
                    polys, at = [], 0
                    for c in counts:
                        polys.append(flat[at: at + c]); at += c
            if polys is not None:
                fans = [np.stack([poly[0], poly[k], poly[k + 1]]) for poly in polys for k in range(1, len(poly) - 1)]
                if not fans:
                    continue
                tri = np.stack(fans)
            P3 = sources[pos[1]][:, :3][tri[..., pos[0]]]
            if uvs and uvs[0][2] in sources:
                T = sources[uvs[0][2]][:, :2][tri[..., uvs[0][1]]]
            else:
                T = np.zeros((len(tri), 3, 2))
            parts.append((P3, T, prim.get("material") or ""))
        geoms[geo.get("id")] = parts
    skins = {c.get("id"): ref(c.find("skin").get("source")) for c in root.iterfind("library_controllers/controller") if c.find("skin") is not None}

    pos, uvs, mats = [], [], []

    def walk(node, parent: np.ndarray, depth: int = 0) -> None:
        m = parent @ _dae_matrix(node)
        for inst in list(node.iterfind("instance_geometry")) + list(node.iterfind("instance_controller")):
            gid = ref(inst.get("url"))
            gid = skins.get(gid, gid)
            bind = {im.get("symbol"): ref(im.get("target")) for im in inst.iter("instance_material")}
            flip = np.linalg.det(m[:3, :3]) < 0
            for P3, T, symbol in geoms.get(gid, []):
                world = P3 @ m[:3, :3].T + m[:3, 3]
                if flip:
                    world, T = world[:, ::-1], T[:, ::-1]
                pos.append(world); uvs.append(T)
                mats.append(np.full(len(world), material(bind.get(symbol, symbol)), np.int16))
        if depth < 64:
            for child in node.iterfind("node"):
                walk(child, m, depth + 1)

    scenes = list(root.iterfind("library_visual_scenes/visual_scene"))
    for vs in scenes[:1]:
        for node in vs.iterfind("node"):
            walk(node, np.eye(4))
    if not pos:                                           # no scene: every geometry as it is
        for parts in geoms.values():
            for P3, T, symbol in parts:
                pos.append(P3); uvs.append(T); mats.append(np.full(len(P3), material(symbol), np.int16))
    if not pos:
        raise ValueError("the file has no triangles")
    return _finish(np.concatenate(pos), np.concatenate(uvs), np.concatenate(mats), materials, path.stem, up != "Z_UP")


def load_model(path: Path) -> Scene:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".obj":
        return load_obj(path)
    if ext in (".glb", ".gltf"):
        return load_gltf(path)
    if ext == ".dae":
        return load_dae(path)
    raise ValueError(f"unsupported model format: {ext}")


# ----------------------------------------------------------------------------------------------------- discovering
def zip_models(zip_path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(zip_path) as z:
            return [n for n in z.namelist() if n.lower().endswith(MODEL_EXTS) and not os.path.basename(n).startswith(".")]
    except (zipfile.BadZipFile, OSError):
        return []


def extract_zip(zip_path: Path, dest: Path, only_under: str = "") -> None:
    """Unpack a zip (only the entries under the folder ``only_under`` when given) without ever leaving ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    prefix = only_under.replace("\\", "/").strip("/")
    prefix = prefix + "/" if prefix else ""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if prefix and not info.filename.replace("\\", "/").startswith(prefix):
                continue
            target = (dest / info.filename).resolve()
            if root not in target.parents and target != root:
                continue                                  # an entry that would leave the folder
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                out.write(src.read())


_RANK = {".glb": 0, ".gltf": 1, ".dae": 2, ".obj": 3}


def distinct_models(candidates: list[str]) -> list[str]:
    """Packages repeat each scene in several formats (GLB and OBJ, DAE and FBX): one per scene, the richest format.
    A scene is told apart by its name and its folder (an ``OBJ`` subfolder counts as its parent)."""
    best: dict[tuple[str, str], str] = {}
    for c in candidates:
        norm = c.replace("\\", "/")
        folder, _, file = norm.rpartition("/")
        if folder.lower().endswith("/obj") or folder.lower() == "obj":
            folder = folder[:-4] if "/" in folder else ""
        stem, ext = os.path.splitext(file)
        key = (folder.lower(), stem.lower())
        if key not in best or _RANK.get(ext.lower(), 9) < _RANK.get(os.path.splitext(best[key])[1].lower(), 9):
            best[key] = c
    return sorted(best.values(), key=lambda c: c.lower())


def pick_model(candidates: list[str]) -> str:
    """Of several models in one package, the one to open when none is named."""
    return sorted(candidates, key=lambda c: (_RANK.get(os.path.splitext(c)[1].lower(), 9), len(c)))[0]


def save_npz(scene: Scene, path: Path) -> None:
    np.savez_compressed(path, P=scene.P, F=scene.F, UV=scene.UV, M=scene.M)


def texture_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()
