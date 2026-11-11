"""Write millimetre meshes and base materials to a 3MF Core package."""

from __future__ import annotations

from xml.sax.saxutils import quoteattr
import zipfile


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
