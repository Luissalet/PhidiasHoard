"""The whole app over HTTP: library, a project from a zip, materials, a sculpt, pictures, export and the agent tools."""

from __future__ import annotations

import zipfile

import pytest

from .test_engine import write_wall


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from phidias_hoard.config import Config
    from phidias_hoard.main import create_app
    from phidias_hoard.services import Services

    lib = tmp_path / "lib" / "Torre"
    lib.mkdir(parents=True)
    write_wall(lib)
    with zipfile.ZipFile(tmp_path / "lib" / "pack.zip", "w") as z:
        for f in lib.iterdir():
            z.write(f, f"Pack/{f.name}")
    svc = Services(Config(data_dir=tmp_path / "data", background=False))
    svc.settings.update({"library_dirs": [str(tmp_path / "lib")], "size_mm": 80, "min_edge_mm": 0.8, "max_faces": 200_000})
    with TestClient(create_app(services=svc), base_url="http://127.0.0.1") as c:
        c.svc = svc
        yield c


def call(c, name, **arguments):
    r = c.post("/api/ui/call", json={"name": name, "arguments": arguments})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_catalogue(client):
    assert client.get("/api/health").json()["service"] == "phidias-hoard"
    tools = client.get("/api/agent/tools").json()["tools"]
    names = {t["name"] for t in tools}
    assert {"library_scan", "project_create", "material_look", "material_set", "sculpt_start", "model_look", "result_export"} <= names
    assert client.post("/api/agent/call", json={"name": "status"}).status_code in (401, 403)
    r = client.post("/api/agent/call", json={"name": "status", "arguments": {}}, headers={"Authorization": f"Bearer {client.svc.token}"})
    assert r.status_code == 200 and "can_close" in r.text


def test_library_lists_models_and_zips(client):
    lib = call(client, "library_scan")
    kinds = sorted((e["format"], bool(e["inner"])) for e in lib["entries"])
    assert kinds == [("obj", False), ("obj", True)]
    assert call(client, "library_scan", query="pack")["count"] == 1
    assert lib["groups"] == {"Torre": 1, "lib": 1}
    assert call(client, "library_scan", group="torre", limit=1)["entries"][0]["format"] == "obj"


def test_project_from_zip_to_stl(client, tmp_path):
    entry = next(e for e in call(client, "library_scan")["entries"] if e["inner"])
    p = call(client, "project_create", source=entry["path"], inner=entry["inner"])
    pid = p["id"]
    assert p["materials"]["h_kage"]["recipe"] == "drop" and p["materials"]["brick"]["recipe"] == "relief"
    assert p["source"]["triangles"] == 6
    assert call(client, "library_scan", refresh=True)["entries"][1]["project_id"] == pid or \
        any(e["project_id"] == pid for e in call(client, "library_scan")["entries"])
    assert client.get(f"/api/projects/{pid}/mesh/before.bin").status_code == 200
    assert client.get(f"/api/projects/{pid}/materials/thumb", params={"name": "brick"}).headers["content-type"] == "image/png"
    sheet = client.get(f"/api/projects/{pid}/materials/sheet", params={"name": "brick", "recipe": "lines"})
    assert sheet.status_code == 200 and sheet.content[:4] == b"\x89PNG"

    call(client, "material_set", project_id=pid, materials={"brick": {"recipe": "lines", "depth_mm": 0.8}})
    p = call(client, "project_set", project_id=pid, size_mm=60, plinth="rect")
    assert p["settings"]["size_mm"] == 60 and p["materials"]["brick"]["depth_mm"] == 0.8

    job = call(client, "sculpt_start", project_id=pid)        # runs inline in tests
    done = call(client, "job_status", job_id=job["id"])
    assert done["state"] == "done", done
    res = done["result"]
    assert res["size_mm"][0] == pytest.approx(60, abs=0.5) and res["triangles_total"] > 5000
    assert 0.5 < res["max_relief_mm"] <= 0.8 + 1e-6
    for path in (f"/api/projects/{pid}/mesh/sculpt.bin", f"/api/projects/{pid}/picture/sculpt.png", f"/api/projects/{pid}/download/sculpt.stl"):
        assert client.get(path).status_code == 200, path
    look = client.get(f"/api/projects/{pid}/look.png", params={"result": "sculpt", "size": 300, "compare": True})
    assert look.status_code == 200 and look.content[:4] == b"\x89PNG"

    out = call(client, "result_export", project_id=pid, folder=str(tmp_path / "stl"))
    assert out["path"].endswith("60mm.stl") and (tmp_path / "stl").is_dir()
    assert call(client, "project_list")["projects"][0]["results"]["sculpt"]["triangles"] == res["triangles_total"]
    call(client, "project_delete", project_id=pid)
    assert call(client, "project_list")["projects"] == []


def test_errors_are_explained(client):
    r = client.post("/api/ui/call", json={"name": "project_get", "arguments": {"project_id": "nope"}})
    assert r.status_code == 404 and "proyecto" in r.json()["error"]
    r = client.post("/api/ui/call", json={"name": "project_create", "arguments": {"source": "C:/no/such.obj"}})
    assert r.status_code == 404
    lib = call(client, "library_scan")
    p = call(client, "project_create", source=lib["entries"][0]["path"])
    r = client.post("/api/ui/call", json={"name": "close_start", "arguments": {"project_id": p["id"]}})
    assert r.status_code == 400 and "esculpir" in r.json()["error"]
    r = client.post("/api/ui/call", json={"name": "material_set", "arguments": {"project_id": p["id"], "materials": {"nada": {"recipe": "flat"}}}})
    assert r.status_code == 404


class FakeModel:
    """Answers like the vision model would: first the materials, then one redo, then ok."""

    def __init__(self):
        self.calls = []

    def status(self):
        return {"available": True, "model": "fake"}

    def ask(self, system, user, images, max_tokens=6000, timeout=None):
        self.calls.append((user, len(images)))
        assert images and all(i[:4] == b"\x89PNG" for i in images)
        if len(self.calls) == 1:
            return {"scene": "a brick wall on a lawn", "materials": {
                "brick": {"what": "brick wall", "recipe": "lines", "depth_mm": 0.7, "cutout": False, "enrich": None},
                "grass": {"what": "lawn", "recipe": "flat", "enrich": {"pattern": "rock", "size_mm": 3, "depth_mm": 0.2}},
                "h_kage": {"what": "shadow", "recipe": "drop"}, "ghost": {"recipe": "flat"}}}
        if len(self.calls) == 2:
            return {"verdict": "redo", "notes": "The joints are too shallow.", "materials": {"brick": {"depth_mm": 5, "recipe": "normal"}}}
        return {"verdict": "ok", "notes": "Reads well now.", "materials": {}}


def test_the_model_directs_looks_and_redoes(client):
    lib = call(client, "library_scan")
    p = call(client, "project_create", source=lib["entries"][0]["path"])
    pid = p["id"]
    fake = FakeModel()
    client.svc.llm = fake
    job = call(client, "direct_start", project_id=pid, rounds=3, wish="juntas marcadas")
    done = call(client, "job_status", job_id=job["id"])
    assert done["state"] == "done", done
    assert done["result"]["director"] == {"rounds": 2, "verdict": "ok", "notes": "Reads well now."}
    assert len(fake.calls) == 3 and "juntas marcadas" in fake.calls[0][0] and fake.calls[1][1] >= 2
    p = call(client, "project_get", project_id=pid)
    brick, grass = p["materials"]["brick"], p["materials"]["grass"]
    assert brick["depth_mm"] == 2.0                           # clamped: the model asked for 5
    assert brick["recipe"] == "relief"                        # it asked for normal, but there is no normal map
    assert grass["enrich"] == {"pattern": "rock", "size_mm": 3.0, "depth_mm": 0.2} and grass["what"] == "lawn"
    assert [s["step"] for s in p["director"]["log"]] == ["materiales", "prueba 1", "prueba 2"]
    assert p["results"]["sculpt"]["enriched_materials"] == 1
    # by hand: an enrichment set and removed
    call(client, "material_set", project_id=pid, materials={"brick": {"enrich": {"pattern": "cracks", "size_mm": 2, "depth_mm": 0.3}}})
    assert call(client, "project_get", project_id=pid)["materials"]["brick"]["enrich"]["pattern"] == "cracks"
    call(client, "material_set", project_id=pid, materials={"brick": {"no_enrich": True}})
    assert call(client, "project_get", project_id=pid)["materials"]["brick"]["enrich"] is None


def test_answers_with_thinking_or_fences_are_read():
    from phidias_hoard.llm import extract_json

    assert extract_json('<think>hmm {"a": 0}</think>\nSure:\n```json\n{"verdict": "ok"}\n```') == {"verdict": "ok"}
    assert extract_json('text {"a": {"b": 1}} trailing') == {"a": {"b": 1}}


def test_remodel_rebuilds_the_pieces_in_quads(client, tmp_path):
    from phidias_hoard.engine import finish
    from phidias_hoard.remodel import build

    if not (finish.find_blender("") or build.has_bpy()):
        pytest.skip("needs Blender")
    # a little pagoda: plinth, body, roof
    obj, base = [], 1

    def ring_box(levels):
        nonlocal base
        lines, rings = [], []
        for z, s in levels:
            rings.append(base + len(rings) * 4)
            lines += [f"v {-s} {z} {s}", f"v {s} {z} {s}", f"v {s} {z} {-s}", f"v {-s} {z} {-s}"]
        for a, b in zip(rings, rings[1:]):
            for i in range(4):
                j = (i + 1) % 4
                lines += [f"f {a + i} {a + j} {b + j}", f"f {a + i} {b + j} {b + i}"]
        base += len(rings) * 4
        return lines

    obj += ring_box([(0, 10), (1, 9)]) + ring_box([(1, 6), (7, 6)]) + ring_box([(7, 11), (8, 11), (12, 5)])
    (tmp_path / "lib" / "pagoda.obj").write_text("\n".join(obj) + "\n")
    p = call(client, "project_create", source=str(tmp_path / "lib" / "pagoda.obj"))
    pid = p["id"]
    plan = call(client, "remodel_plan", project_id=pid)
    assert [t["kind"] for t in plan["types"]] == ["block", "body", "roof"]
    roof = plan["types"][2]["type"]
    plan = call(client, "remodel_set", project_id=pid, types={str(roof): {"params": {"tiles": 8, "rows": 4}}, str(plan["types"][1]["type"]): {"params": {"bays_x": 2, "zones": ["lattice"], "door_bay": 0}}})
    assert plan["types"][2]["params"]["tiles"] == 8 and plan["types"][2]["chosen"]
    job = call(client, "remodel_start", project_id=pid)
    done = call(client, "job_status", job_id=job["id"])
    assert done["state"] == "done", done
    r = done["result"]
    assert r["quads"] > 400 and r["ngons"] == 0 and r["tris"] <= 4 * 8 + 4 and r["kinds"] == {"block": 1, "body": 1, "roof": 1}
    assert client.get(f"/api/projects/{pid}/mesh/remodel.bin").status_code == 200
    assert client.get(f"/api/projects/{pid}/mesh/remodel_edges.bin").status_code == 200
    assert len(client.get(f"/api/projects/{pid}/download/remodel.blend").content) > 2000
    look = client.get(f"/api/projects/{pid}/look.png", params={"result": "remodel", "size": 300})
    assert look.status_code == 200
    # the model decides the pieces, looks at the build and corrects it; what a piece cannot be is refused
    class PieceModel:
        calls = 0

        def status(self):
            return {"available": True, "model": "fake"}

        def ask(self, system, user, images, max_tokens=6000, timeout=None):
            PieceModel.calls += 1
            assert images and "allowed:" in user
            if PieceModel.calls == 1:
                return {"scene": "una pagoda", "types": {str(roof): {"what": "tejado", "kind": "roof", "params": {"tiles": 500, "corner_lift": 0.6}},
                                                           "type 1": {"kind": "spaceship"}, "99": {"kind": "block"}}}
            if PieceModel.calls == 2:
                return {"verdict": "redo", "notes": "Faltan celosías.", "types": {str(plan["types"][1]["type"]): {"params": {"zones": ["lattice"], "bars": 4}}}}
            return {"verdict": "ok", "notes": "Bien.", "types": {}}

    client.svc.llm = PieceModel()
    job = call(client, "remodel_start", project_id=pid, direct=True, rounds=3)
    done = call(client, "job_status", job_id=job["id"])
    assert done["state"] == "done", done
    assert done["result"]["director"] == {"rounds": 3, "verdict": "ok", "notes": "Bien."}
    plan2 = call(client, "remodel_plan", project_id=pid)
    assert plan2["types"][2]["params"]["tiles"] == 60 and plan2["types"][2]["params"]["corner_lift"] == 0.6   # clamped
    assert plan2["types"][0]["kind"] == "block"               # a builder that does not exist is refused: it stayed what it was
    assert plan2["types"][1]["params"]["bars"] == 4
    assert [s["step"] for s in call(client, "project_get", project_id=pid)["remodel_director"]["log"]] == ["piezas", "construcción 1", "construcción 2"]
    r = call(client, "remodel_set", project_id=pid, types={str(roof): {"params": {"tiles": 8}}})
    out = call(client, "result_export", project_id=pid, folder=str(tmp_path / "out"))
    assert out["result"] == "remodel" and out["paths"][0].endswith("remodelado.blend") and (tmp_path / "out").is_dir()


def _library(tmp_path):
    lib = tmp_path / "faustus" / "skills" / "library"
    for slug, tags, body in (("game-scenery-remodel", "[3d, blender]", "Step one."), ("blender-asset-production", "[blender]", "Model it."),
                             ("fastapi-patterns", "[python, api]", "Routers."), ("mesh-lods", "[3d, mesh]", "Levels of detail.")):
        (lib / slug / "references").mkdir(parents=True)
        (lib / slug / "SKILL.md").write_text(f"---\nname: {slug}\ndescription: About {slug}. Use when needed.\nversion: 1.0.0\n"
                                             f"category: engineering\ntags: {tags}\nstatus: published\n---\n\n## Procedure\n\n{body}\n", encoding="utf-8")
        (lib / slug / "references" / "example.md").write_text("An example.", encoding="utf-8")
    return lib


def test_agents_get_the_skills_the_app_asks_the_library_for(client, tmp_path):
    from phidias_hoard.api import agent as agent_api
    from phidias_hoard.hoard_link import skills as family_skills

    lib = _library(tmp_path)
    shelf = agent_api.SHELF
    old = (shelf.library, shelf.wants, shelf._checked)
    shelf.library, shelf._checked = str(lib), 0.0
    shelf.wants = {"slugs": ["game-scenery-remodel", "blender-asset-production", "not-there"], "tags": ["3d", "mesh"], "limit": 8}
    try:
        c, svc = client, client.svc
        listing = c.get("/api/agent/tools").json()
        names = [t["name"] for t in listing["tools"]]
        assert {"skills_list", "skill_read", "skills_refresh"} <= set(names) and "remodel_start" in names
        slugs = [s["slug"] for s in listing["skills"]]
        assert slugs == ["game-scenery-remodel", "blender-asset-production", "mesh-lods"]      # named first, then by shared tags
        assert "game-scenery-remodel: About game-scenery-remodel" in listing["instructions"]
        auth = {"Authorization": "Bearer " + svc.token}
        read = c.post("/api/agent/call", json={"name": "skill_read", "arguments": {"slug": "game-scenery-remodel"}}, headers=auth).json()
        assert "Step one." in read["text"] and read["files"] == ["references/example.md"]
        ref = c.post("/api/agent/call", json={"name": "skill_read", "arguments": {"slug": "mesh-lods", "file": "references/example.md"}}, headers=auth).json()
        assert ref["text"] == "An example."
        assert c.post("/api/agent/call", json={"name": "skill_read", "arguments": {"slug": "fastapi-patterns"}}, headers=auth).status_code == 404
        # the library improves a skill and drops another: the copy follows
        (lib / "mesh-lods" / "SKILL.md").write_text((lib / "mesh-lods" / "SKILL.md").read_text(encoding="utf-8").replace("Levels of detail.", "Better."), encoding="utf-8")
        (lib / "blender-asset-production" / "SKILL.md").unlink()
        again = c.post("/api/agent/call", json={"name": "skills_refresh", "arguments": {}}, headers=auth).json()
        assert again["updated"] == ["mesh-lods"] and again["removed"] == ["blender-asset-production"]
        assert "Better." in c.get("/api/skills/mesh-lods").json()["text"]
        # with the library gone the agents still have the copy
        shelf.library, shelf._checked = str(tmp_path / "nowhere"), 0.0
        import phidias_hoard.hoard_link.fam_mail as fam_mail
        real, fam_mail.faustus_dir = fam_mail.faustus_dir, lambda *a, **k: None
        try:
            assert [s["slug"] for s in c.get("/api/skills").json()["skills"]] == ["game-scenery-remodel", "mesh-lods"]
        finally:
            fam_mail.faustus_dir = real
    finally:
        shelf.library, shelf.wants, shelf._checked = old
    assert family_skills.parse("no front matter")[0] == {}
