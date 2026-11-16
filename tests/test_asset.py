"""From the sculpt to a game asset: retopology, UVs, baked normal and occlusion maps, glTF and FBX."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from phidias_hoard.engine import asset, finish

from .test_app import call, client  # noqa: F401 - the app fixture

BLENDER = finish.find_blender("")
needs_blender = pytest.mark.skipif(not BLENDER, reason="Blender is not installed")
SMALL = {"target_faces": 600, "texture_px": 256, "ao_samples": 4}


def _glb_json(path):
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


def _cube_sphere(n=24):
    """A closed, welded, outward-facing sphere with a groove round the middle."""
    from phidias_hoard.engine import meshio

    axes = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    points, faces, base = [], [], 0
    grid = np.linspace(-1, 1, n + 1)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    ids = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    for normal, u, v in axes:
        points.append((np.array(normal) + a[..., None] * np.array(u) + b[..., None] * np.array(v)).reshape(-1, 3))
        quad = np.stack([ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]], -1).reshape(-1, 4) + base
        faces.append(np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
        base += (n + 1) ** 2
    P, inverse = meshio.weld(np.concatenate(points), 1e-9)
    F = inverse[np.concatenate(faces)].astype(np.int32)
    tri = P[F]
    inward = (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * tri.mean(1)).sum(1) < 0
    F[inward] = F[inward][:, [0, 2, 1]]
    radius = np.linalg.norm(P, axis=1, keepdims=True)
    groove = np.exp(-((P[:, 2:3] / radius) ** 2) / 0.004)
    P = P / radius * (30 - 3 * groove)
    return P - P.min(0), F


@needs_blender
def test_pipeline_retopologises_unwraps_bakes_and_exports(tmp_path):
    P, F = _cube_sphere()
    spec = {"method": "quadriflow", "size_m": 2.0, "formats": ["glb", "fbx"], "cage_pct": 3.0, "uv_angle": 66.0, **SMALL}
    report = asset.create(P, F, spec, tmp_path, BLENDER, "Esfera con surco")
    assert report["method"] == "quadriflow" and not report["notes"]
    assert 200 < report["low_faces"] < 1500 and report["low_faces"] == report["low_quads"]
    assert report["low_faces"] < len(F) / 2
    assert report["size_m"][2] == pytest.approx(2.0, abs=0.01)             # longest side, in metres
    assert 0.05 < report["uv_coverage"] <= 1.0
    assert report["ao_range"][0] < 0.6 < report["ao_mean"] < 1.0            # the groove is darker than the rest
    assert report["normal_flat_share"] < 0.95                               # the maps hold more than a flat surface
    for name in ("asset.glb", "asset.fbx", "asset_normal.png", "asset_ao.png"):
        assert (tmp_path / name).stat().st_size > 1000, name
    gltf = _glb_json(tmp_path / "asset.glb")
    material = gltf["materials"][0]
    assert "normalTexture" in material and "occlusionTexture" in material
    assert {"POSITION", "NORMAL", "TEXCOORD_0"} <= set(gltf["meshes"][0]["primitives"][0]["attributes"])
    low = np.load(tmp_path / "low_P.npy")
    assert len(low) and np.ptp(low, axis=0).max() == pytest.approx(60, abs=3)        # still in the print's millimetres


@needs_blender
def test_pipeline_falls_back_to_decimation_on_an_open_mesh(tmp_path):
    P, F = _cube_sphere(16)
    P = np.vstack([P, P[F[0, 0]] + [0, 0, 5.0]])                              # a fin on one edge: three faces share it,
    F = np.vstack([F, [[F[0, 0], F[0, 1], len(P) - 1]]]).astype(np.int32)     # which QuadriFlow rejects
    spec = {"method": "quadriflow", "size_m": 1.0, "formats": ["glb"], "cage_pct": 3.0, "uv_angle": 66.0, **SMALL}
    report = asset.create(P, F, spec, tmp_path, BLENDER, "abierta")
    assert report["method"] == "decimate" and report["notes"] and "QuadriFlow" in report["notes"][0]
    assert report["low_tris"] <= 700 and (tmp_path / "asset.glb").is_file() and not (tmp_path / "asset.fbx").exists()


def test_formats_are_required(tmp_path):
    with pytest.raises(ValueError):
        asset.create(np.zeros((3, 3)), np.zeros((1, 3), dtype=np.int32), {"formats": []}, tmp_path, "blender")


def test_asset_start_explains_what_is_missing(client, monkeypatch):                # noqa: F811
    entry = call(client, "library_scan")["entries"][0]
    p = call(client, "project_create", source=entry["path"], inner=entry["inner"])
    r = client.post("/api/ui/call", json={"name": "asset_start", "arguments": {"project_id": p["id"]}})
    assert r.status_code == 400 and "esculpir" in r.json()["error"]
    call(client, "sculpt_start", project_id=p["id"])
    r = client.post("/api/ui/call", json={"name": "asset_start", "arguments": {"project_id": p["id"], "source": "closed"}})
    assert r.status_code == 400 and "cerrar" in r.json()["error"]
    monkeypatch.setattr("phidias_hoard.engine.finish.find_blender", lambda _: None)
    r = client.post("/api/ui/call", json={"name": "asset_start", "arguments": {"project_id": p["id"]}})
    assert r.status_code == 503 and "Blender" in r.json()["error"]
    r = client.post("/api/ui/call", json={"name": "asset_start", "arguments": {"project_id": p["id"], "formats": ["obj"]}})
    assert r.status_code in (400, 422)


@needs_blender
def test_asset_over_http_and_mcp(client, tmp_path):                                # noqa: F811
    entry = call(client, "library_scan")["entries"][0]
    pid = call(client, "project_create", source=entry["path"], inner=entry["inner"])["id"]
    assert call(client, "sculpt_start", project_id=pid)["state"] == "done"
    tools = {t["name"] for t in client.get("/api/agent/tools").json()["tools"]}
    assert "asset_start" in tools
    job = client.post("/api/agent/call", json={"name": "asset_start", "arguments": {"project_id": pid, **SMALL}},
                      headers={"Authorization": f"Bearer {client.svc.token}"})
    assert job.status_code == 200, job.text
    done = call(client, "job_status", job_id=job.json()["id"])
    assert done["state"] == "done", done
    info = done["result"]
    assert info["triangles_total"] > 50 and info["files"] == ["asset.fbx", "asset.glb", "asset_ao.png", "asset_normal.png"]
    assert info["method"] in ("quadriflow", "decimate") and 0 < info["uv_coverage"] <= 1
    project = call(client, "project_get", project_id=pid)
    assert project["results"]["asset"]["triangles_total"] == info["triangles_total"]
    for path in (f"/api/projects/{pid}/mesh/asset.bin", f"/api/projects/{pid}/picture/asset.png"):
        assert client.get(path).status_code == 200, path
    glb = client.get(f"/api/projects/{pid}/asset/asset.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get(f"/api/projects/{pid}/asset/asset_normal.png").content[:4] == b"\x89PNG"
    assert client.get(f"/api/projects/{pid}/asset/..%2Fproject.json").status_code in (400, 404)
    look = client.get(f"/api/projects/{pid}/look.png", params={"result": "asset", "size": 300, "compare": True})
    assert look.status_code == 200 and look.content[:4] == b"\x89PNG"
    out = call(client, "result_export", project_id=pid, result="asset", folder=str(tmp_path / "game"))
    assert out["result"] == "asset" and len(out["paths"]) == 4
    assert sorted(p.name for p in (tmp_path / "game").glob("* asset/*")) == ["asset.fbx", "asset.glb", "asset_ao.png", "asset_normal.png"]
    assert call(client, "project_list")["projects"][0]["results"]["asset"]["triangles"] == info["triangles_total"]
    call(client, "sculpt_start", project_id=pid)                                # a new sculpt makes the asset stale
    assert "asset" not in call(client, "project_get", project_id=pid)["results"]
    assert client.get(f"/api/projects/{pid}/asset/asset.glb").status_code == 404
