"""Write millimetre meshes and base materials to a 3MF Core package."""

from __future__ import annotations

from hashlib import file_digest
from xml.etree import ElementTree
from xml.sax.saxutils import quoteattr
import zipfile

import numpy as np


_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_CONTENT = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""
_RELS = """<?xml version="1.0" encoding="utf-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _name(value):
    return quoteattr("".join(c for c in str(value) if ord(c) >= 32 and ord(c) not in (0xFFFE, 0xFFFF)))


def write(path, parts, palette):
    """Each part is (name, points, faces, optional material index per face)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        z.writestr("[Content_Types].xml", _CONTENT)
        z.writestr("_rels/.rels", _RELS)
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as out:
            def line(text):
                out.write(text.encode("utf-8"))

            line(f'<?xml version="1.0" encoding="utf-8"?><model unit="millimeter" xml:lang="es-ES" xmlns="{_NS}"><resources>')
            line('<basematerials id="1">')
            for name, rgb in palette:
                color = "#" + "".join(f"{int(channel):02X}" for channel in rgb)
                line(f'<base name={_name(name)} displaycolor="{color}"/>')
            line('</basematerials>')
            for ident, (name, points, faces, materials) in enumerate(parts, 2):
                line(f'<object id="{ident}" type="model" name={_name(name)} pid="1" pindex="0"><mesh><vertices>')
                for start in range(0, len(points), 8192):
                    line("".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in points[start:start+8192]))
                line('</vertices><triangles>')
                for start in range(0, len(faces), 8192):
                    block = faces[start:start+8192]
                    if materials is None:
                        line("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in block))
                    else:
                        line("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{mat}"/>'
                                     for (a, b, c), mat in zip(block, materials[start:start+8192])))
                line('</triangles></mesh></object>')
            line('</resources><build>')
            line("".join(f'<item objectid="{ident}"/>' for ident in range(2, len(parts) + 2)))
            line('</build></model>')


def _mesh_topology(points, faces):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(points) or not len(faces) or not np.isfinite(points).all() or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Invalid mesh vertices or triangle indices.")
    sides = np.roll(faces, -1, axis=1)
    keys = (np.minimum(faces, sides) * len(points) + np.maximum(faces, sides)).ravel()
    order = np.argsort(keys)
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]
    counts = np.diff(np.r_[starts, len(keys)])
    orientations = np.add.reduceat(np.where((faces < sides).ravel()[order], 1, -1), starts)
    a, b, c = points[faces].transpose(1, 0, 2)
    degenerate = int(np.count_nonzero(~np.any(np.cross(b - a, c - a), axis=1)))
    return {"triangles": len(faces), "boundary_edges": int(np.count_nonzero(counts == 1)),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "winding_conflicts": int(np.count_nonzero((counts == 2) & (orientations != 0))),
            "degenerate_triangles": degenerate}


def check(path):
    """Inspect the stored, indexed meshes of a Phidias 3MF in its actual export precision."""
    prefix = "{" + _NS + "}"
    objects, build, unit = {}, [], None
    with path.open("rb") as raw:
        digest = file_digest(raw, "sha256").hexdigest()
        raw.seek(0)
        try:
            with zipfile.ZipFile(raw) as package, package.open("3D/3dmodel.model") as model:
                for event, element in ElementTree.iterparse(model, events=("start", "end")):
                    tag = element.tag.removeprefix(prefix)
                    if event == "start":
                        if tag == "model":
                            unit = element.get("unit", "millimeter")
                        elif tag == "object":
                            points, faces = [], []
                        continue
                    if tag == "vertex":
                        points.append(tuple(float(element.attrib[k]) for k in ("x", "y", "z")))
                    elif tag == "triangle":
                        faces.append(tuple(int(element.attrib[k]) for k in ("v1", "v2", "v3")))
                    elif tag == "object" and faces:
                        objects[element.attrib["id"]] = {
                            "name": element.get("name", ""), **_mesh_topology(points, faces)}
                    elif tag == "item":
                        build.append(element.attrib["objectid"])
                    element.clear()
        except (ElementTree.ParseError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError("Unreadable 3MF Core model.") from exc
    if unit is None or not build or any(ident not in objects for ident in build):
        raise ValueError("Expected a 3MF build containing direct mesh objects.")
    parts = [{"id": ident, **objects[ident]} for ident in dict.fromkeys(build)]
    totals = {key: sum(objects[ident][key] for ident in build)
              for key in ("triangles", "boundary_edges", "nonmanifold_edges", "winding_conflicts", "degenerate_triangles")}
    return {"sha256": digest, "unit": unit, "objects": parts, **totals,
            "closed_oriented_edges": not any(totals[key] for key in totals if key != "triangles")}
