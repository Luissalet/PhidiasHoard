"""Everything the app runs, built once per process: settings, the project store, the library of game models and the
worker that sculpts and closes meshes one job at a time."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import __version__
from .config import Config, Settings
from . import director
from .engine import export, finish, heights, meshio, patterns, preview, recipes
from .llm import ModelClient
from .remodel import build as remodel_build
from .remodel import pieces as remodel_pieces
from .engine import sculpt as sculpting
from .errors import PhidiasError
from .hoard_link.tokens import read_or_create_token
from .store import Store

log = logging.getLogger("phidias")

SKIP_DIRS = {"trabajo", "originales", "herramientas", "node_modules", ".git", "__pycache__", "phidias"}
PROJECT_SETTINGS = ("size_mm", "min_edge_mm", "tolerance_mm", "max_faces", "ground", "base_mm", "plinth", "plinth_margin_mm",
                    "normal_cutoff_mm", "sheet_mm")
RESULTS = ("sculpt", "closed", "remodel")
CANCELLABLE_JOBS = {"sculpt", "direct"}


class _JobCancelled(Exception):
    pass


class Services:
    def __init__(self, config: Config):
        self.config = config
        config.paths.ensure()
        config.projects_dir.mkdir(parents=True, exist_ok=True)
        self.settings = Settings(config.settings_path)
        self.token = read_or_create_token(config.token_path)
        self.store = Store(config.projects_dir)
        self.llm = ModelClient(self.settings)
        self.started = time.time()
        self.jobs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._scenes: "OrderedDict[str, meshio.Scene]" = OrderedDict()
        self._library: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.config.background and self._worker is None:
            self._worker = threading.Thread(target=self._run, name="phidias-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker = None

    def health(self) -> dict[str, Any]:
        return {"projects": len(self.store.list()), "jobs_running": sum(1 for j in self.jobs.values() if j["state"] == "running")}

    def status(self) -> dict[str, Any]:
        blender = finish.find_blender(self.settings["blender"])
        return {"version": __version__, "projects": len(self.store.list()), "library_dirs": self.settings["library_dirs"],
                "blender": blender or "", "can_close": bool(blender), "patterns": list(patterns.PATTERNS), "pattern_labels": patterns.LABELS, "model": self.llm.status(),
                "jobs": [self._job_public(j) for j in self.jobs.values() if j["state"] in ("queued", "running")]}

    # ------------------------------------------------------------------ library
    def library(self, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._library is not None and not refresh:
                return self._library
        used = {str(p.get("source", {}).get("path", "")).lower() + "|" + str(p.get("source", {}).get("inner", "")).lower(): p["id"]
                for p in self.store.list()}
        entries: list[dict[str, Any]] = []
        missing: list[str] = []
        for root in self.settings["library_dirs"]:
            base = Path(root)
            if not base.is_dir():
                missing.append(root)
                continue
            for dirpath, dirs, files in os.walk(base):
                rel = Path(dirpath).relative_to(base)
                dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith("."))
                if len(rel.parts) >= 7:
                    dirs[:] = []
                for f in sorted(files):
                    low = f.lower()
                    full = Path(dirpath) / f
                    group = rel.parts[0] if rel.parts else base.name
                    if low.endswith(meshio.MODEL_EXTS):
                        entries.append(self._entry(full, "", group, used))
                    elif low.endswith(".zip"):
                        try:
                            if full.stat().st_size > 2_000_000_000:
                                continue
                        except OSError:
                            continue
                        for inner in meshio.distinct_models(meshio.zip_models(full))[:60]:
                            entries.append(self._entry(full, inner, group, used))
                    if len(entries) >= 4000:
                        break
        # a package and its unpacked copy side by side: keep the unpacked one; and one format per loose scene
        loose_best: dict[tuple[str, str, str], dict[str, Any]] = {}
        rank = {"glb": 0, "gltf": 1, "dae": 2, "obj": 3}
        for e in entries:
            if e["inner"]:
                continue
            folder = Path(e["path"]).parent
            if folder.name.lower() == "obj":
                folder = folder.parent
            key = (e["group"], str(folder).lower(), Path(e["path"]).stem.lower())
            if key not in loose_best or rank.get(e["format"], 9) < rank.get(loose_best[key]["format"], 9):
                loose_best[key] = e
        keep_loose = {id(e) for e in loose_best.values()}
        loose_names = {(e["group"], Path(e["path"]).stem.lower()) for e in loose_best.values()}
        entries = [e for e in entries if (id(e) in keep_loose) or (e["inner"] and (e["group"], Path(e["inner"]).stem.lower()) not in loose_names)]
        lib = {"entries": entries, "missing": missing, "scanned": time.time(), "dirs": self.settings["library_dirs"]}
        with self._lock:
            self._library = lib
        return lib

    @staticmethod
    def _entry(path: Path, inner: str, group: str, used: dict[str, str], models: int = 1) -> dict[str, Any]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        name = Path(inner).stem if inner else path.stem
        return {"path": str(path), "inner": inner, "name": name.replace("_", " "), "group": group, "bytes": size,
                "format": (Path(inner).suffix if inner else path.suffix).lower().lstrip("."), "models": models,
                "project_id": used.get(str(path).lower() + "|" + inner.lower(), "")}

    # ------------------------------------------------------------------ projects
    def _model_path(self, project: dict[str, Any]) -> Path:
        src = project.get("source", {})
        if src.get("inner"):
            return self.store.dir(project["id"]) / "src" / src["inner"]
        return Path(src.get("path", ""))

    def scene(self, pid: str) -> meshio.Scene:
        with self._lock:
            if pid in self._scenes:
                self._scenes.move_to_end(pid)
                return self._scenes[pid]
        project = self.store.get(pid)
        path = self._model_path(project)
        if not path.is_file():
            raise PhidiasError("not_found", f"El modelo de origen ya no está en {path}.", "Vuelve a crear el proyecto desde la biblioteca.")
        scene = meshio.load_model(path)
        with self._lock:
            self._scenes[pid] = scene
            while len(self._scenes) > 2:
                self._scenes.popitem(last=False)
        return scene

    def create_project(self, source: str, inner: str = "", name: str = "") -> dict[str, Any]:
        path = Path(source).expanduser()
        if not path.is_file():
            raise PhidiasError("not_found", f"No existe el archivo {path}.", "Pasa la ruta de un .obj, .glb, .gltf o un .zip que los contenga.")
        pid = self.store.new_id()
        pdir = self.store.dir(pid)
        low = path.suffix.lower()
        if low == ".zip":
            models = meshio.zip_models(path)
            if not models:
                raise PhidiasError("invalid", "Ese zip no trae ningún modelo OBJ, DAE, GLB o glTF.")
            inner = inner if inner in models else meshio.pick_model(models)
            pdir.mkdir(parents=True, exist_ok=True)
            folder = inner.replace("\\", "/").rpartition("/")[0]
            if folder.lower().endswith("/obj"):            # textures of an OBJ folder may sit one level up
                folder = folder[:-4]
            meshio.extract_zip(path, pdir / "src", folder)
        elif low not in meshio.MODEL_EXTS:
            raise PhidiasError("invalid", f"Formato no admitido: {low}.", "Formatos: .obj (con su .mtl y texturas), .dae, .glb, .gltf o .zip.")
        else:
            inner = ""
        project = {"id": pid, "name": (name or (Path(inner).stem if inner else path.stem)).replace("_", " ").strip()[:120],
                   "created": time.time(), "source": {"path": str(path), "inner": inner}, "materials": {}, "results": {},
                   "settings": {k: self.settings[k] for k in ("size_mm", "min_edge_mm", "tolerance_mm", "max_faces", "base_mm", "plinth")}}
        project["settings"].update({"ground": True})
        self.store.save(project)
        try:
            scene = self.scene(pid)
        except PhidiasError:
            self.store.delete(pid)
            raise
        except Exception as exc:
            self.store.delete(pid)
            raise PhidiasError("invalid", f"No se pudo leer el modelo: {exc}") from exc
        areas = scene.face_areas()
        total = float(areas.sum()) or 1.0
        mats = {}
        for i, mat in enumerate(scene.materials):
            sel = scene.M == i
            if not sel.any():
                continue
            info = recipes.describe(mat)
            info.update(recipes.guess(mat, self.settings["depth_mm"]))
            info.update({"index": i, "triangles": int(sel.sum()), "share": round(float(areas[sel].sum()) / total, 4)})
            mats[mat.name] = info
        project["materials"] = mats
        ext = scene.extents
        project["source"].update({"triangles": int(len(scene.F)), "points": int(len(scene.P)),
                                  "extents": [round(float(v), 4) for v in ext], "format": self._model_path(project).suffix.lower().lstrip(".")})
        self.store.save(project)
        self._library = None
        try:
            self._write_before(project, scene)
        except Exception:                                  # a picture is never a reason to lose the project
            log.exception("before view")
        return self.project(pid)

    def _settings_of(self, project: dict[str, Any]) -> sculpting.SculptSettings:
        return sculpting.SculptSettings.from_dict(project.get("settings"))

    def _write_before(self, project: dict[str, Any], scene: meshio.Scene) -> None:
        st = self._settings_of(project)
        scale = st.size_mm / max(float(scene.extents.max()), 1e-9)
        P = (scene.P - scene.P.min(0)) * scale
        pdir = self.store.dir(project["id"])
        (pdir / "before.bin").write_bytes(export.viewer_buffers(P, scene.F))
        preview.render(P, scene.F, size=640).save(pdir / "before.png")

    def project(self, pid: str) -> dict[str, Any]:
        project = self.store.get(pid)
        project["job"] = next((self._job_public(j) for j in reversed(self.jobs.values())
                               if j["project_id"] == pid and j["state"] in ("queued", "running")), None)
        return project

    def update_project(self, pid: str, name: Optional[str] = None, settings: Optional[dict[str, Any]] = None,
                       materials: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
        project = self.store.get(pid)
        if name:
            project["name"] = name.strip()[:120]
        if settings:
            unknown = [k for k in settings if k not in PROJECT_SETTINGS]
            if unknown:
                raise PhidiasError("invalid", f"Ajuste desconocido: {', '.join(unknown)}.", "Ajustes: " + ", ".join(PROJECT_SETTINGS))
            merged = {**project.get("settings", {}), **{k: v for k, v in settings.items() if v is not None}}
            if merged.get("plinth", "none") not in ("none", "rect", "round"):
                raise PhidiasError("invalid", "plinth debe ser none, rect o round.")
            st = sculpting.SculptSettings.from_dict(merged)
            resized = st.size_mm != self._settings_of(project).size_mm
            project["settings"] = {k: getattr(st, k) for k in PROJECT_SETTINGS}
            if resized:
                try:
                    self._write_before(project, self.scene(pid))
                except Exception:
                    log.exception("before view")
        if materials:
            for mname, change in materials.items():
                targets = list(project["materials"]) if mname == "*" else [mname]
                for target in targets:
                    if target not in project["materials"]:
                        raise PhidiasError("not_found", f"El proyecto no tiene el material «{target}».",
                                           "Materiales: " + ", ".join(list(project["materials"])[:40]))
                    cur = project["materials"][target]
                    if "recipe" in change and change["recipe"] is not None:
                        if change["recipe"] not in ("auto", "normal", "lines", "relief", "steps", "flat", "drop"):
                            raise PhidiasError("invalid", "recipe debe ser auto, normal, lines, relief, steps, flat o drop.")
                        cur["recipe"] = change["recipe"]
                        cur["why"] = ""
                    if change.get("depth_mm") is not None:
                        cur["depth_mm"] = float(min(max(float(change["depth_mm"]), 0.0), 20.0))
                    if change.get("invert") is not None:
                        cur["invert"] = bool(change["invert"])
                    if change.get("cutout") is not None:
                        cur["cutout"] = bool(change["cutout"])
                    if "enrich" in change:
                        cur["enrich"] = sculpting.clean_enrich(change["enrich"])
                    if change.get("steps") is not None:
                        cur["steps"] = int(min(max(int(change["steps"]), 0), 200))
                    if change.get("smooth") is not None:
                        cur["smooth"] = float(min(max(float(change["smooth"]), 0.0), 8.0))
        self.store.save(project)
        return self.project(pid)

    def delete_project(self, pid: str) -> dict[str, Any]:
        if any(j["project_id"] == pid and j["state"] in ("queued", "running") for j in self.jobs.values()):
            raise PhidiasError("busy", "Ese proyecto tiene un trabajo en marcha.", "Espera a que acabe.")
        self.store.delete(pid)
        self._scenes.pop(pid, None)
        self._library = None
        return {"deleted": pid}

    def material_sheet(self, pid: str, material: str, recipe: Optional[str] = None) -> bytes:
        project = self.store.get(pid)
        info = project["materials"].get(material)
        if info is None:
            raise PhidiasError("not_found", f"El proyecto no tiene el material «{material}».")
        scene = self.scene(pid)
        mat = scene.materials[info["index"]]
        arr = recipes.sheet(mat, recipe or info["recipe"], info.get("invert", False), info.get("smooth", 0.0))
        return meshio.texture_png(arr)

    def material_thumb(self, pid: str, material: str) -> bytes:
        from PIL import Image
        project = self.store.get(pid)
        info = project["materials"].get(material)
        if info is None:
            raise PhidiasError("not_found", f"El proyecto no tiene el material «{material}».")
        mat = self.scene(pid).materials[info["index"]]
        if mat.albedo is None:
            arr = np.full((64, 64, 3), mat.color, np.uint8)
        else:
            img = Image.fromarray(mat.albedo[..., :3])
            arr = np.asarray(img.resize((64, 64), Image.NEAREST if max(img.size) <= 64 else Image.BILINEAR))
        return meshio.texture_png(arr)

    # ------------------------------------------------------------------ jobs
    # ------------------------------------------------------------------ remodel
    def remodel_plan(self, pid: str, full: bool = False) -> dict[str, Any]:
        """The model taken apart into pieces, each with what it will be built as (saved choices, else a guess)."""
        project = self.store.get(pid)
        scene = self.scene(pid)
        st = self._settings_of(project)
        scale = st.size_mm / max(float(scene.extents.max()), 1e-9)
        dropped = {n for n, m in project["materials"].items() if m.get("recipe") == "drop"}
        parts = remodel_pieces.split(scene, scale, dropped)
        saved = (project.get("remodel") or {}).get("types", {})
        types: dict[str, dict[str, Any]] = {}
        for p in parts:
            key = str(p["type"])
            choice = saved.get(key)
            if choice and choice.get("kind") in remodel_pieces.allowed(p):
                p["kind"], p["params"] = choice["kind"], dict(choice.get("params") or {})
            else:
                p["kind"], p["params"] = remodel_pieces.guess(p, st.size_mm)
            p["name"] = f'{p["id"]:03d} {p["kind"]}'
            t = types.setdefault(key, {"type": p["type"], "kind": p["kind"], "params": p["params"], "count": 0, "pieces": [],
                                       "size_mm": p["size"], "triangles": p["triangles_count"], "levels": len(p["profile"]),
                                       "materials": p["materials"], "what": (choice or {}).get("what", ""),
                                       "chosen": bool(choice), "allowed": remodel_pieces.allowed(p)})
            t["count"] += 1
            t["pieces"].append(p["id"])
        out: dict[str, Any] = {"project_id": pid, "pieces": len(parts), "types": list(types.values()), "kinds": list(remodel_pieces.KINDS)}
        if full:
            out["_parts"] = parts
            out["_textures"] = {}                          # what each texture draws, as height steps to model into the faces
            for name, info in project["materials"].items():
                mat = scene.materials[info["index"]]
                if info.get("recipe") in ("flat", "drop") or mat.atlas:
                    continue
                lv = heights.texel_levels(mat.albedo)
                if lv is not None:
                    out["_textures"][name] = {"levels": (-lv if info.get("invert") else lv).tolist(), "depth": float(info.get("depth_mm", 0.6))}
        return out

    def remodel_set(self, pid: str, changes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        project = self.store.get(pid)
        plan = self.remodel_plan(pid)
        known = {str(t["type"]): t for t in plan["types"]}
        saved = project.setdefault("remodel", {}).setdefault("types", {})
        for key, change in changes.items():
            if key not in known:
                raise PhidiasError("not_found", f"No hay ningún tipo de pieza {key}.", "Tipos: " + ", ".join(known))
            cur = saved.get(key) or {"kind": known[key]["kind"], "params": dict(known[key]["params"])}
            kind = change.get("kind")
            if kind is not None:
                if kind not in known[key]["allowed"]:
                    raise PhidiasError("invalid", f"La pieza {key} no tiene forma de «{kind}».",
                                       "Para esta pieza valen: " + ", ".join(known[key]["allowed"]) + ".")
                if kind != cur["kind"]:
                    cur = {"kind": kind, "params": {}}
            if isinstance(change.get("params"), dict):
                cur["params"] = remodel_pieces.clean_params(cur["kind"], {**cur.get("params", {}), **change["params"]})
            if isinstance(change.get("what"), str):
                cur["what"] = change["what"][:160]
            saved[key] = cur
        self.store.save(project)
        return self.remodel_plan(pid)

    def piece_sheets(self, pid: str, types: list[dict[str, Any]], per_sheet: int = 6) -> list[bytes]:
        """Pictures of piece types: the original piece from two sides and the textures on it, with its number."""
        from PIL import Image, ImageDraw
        plan = self.remodel_plan(pid, full=True)
        parts = {p["type"]: p for p in reversed(plan["_parts"])}
        scene = self.scene(pid)
        by_name = {m.name: m for m in scene.materials}
        sheets = []
        for at in range(0, len(types), per_sheet):
            rows = []
            for t in types[at: at + per_sheet]:
                p = parts[t["type"]]
                P, F = np.asarray(p["points"], float), np.asarray(p["triangles"], np.int32)
                tiles = [preview.render(P, F, yaw=30, pitch=25, size=200), preview.render(P, F, yaw=120, pitch=12, size=200)]
                for name in list(p["materials"])[:2]:
                    mat = by_name.get(name)
                    if mat is not None and mat.albedo is not None:
                        img = Image.fromarray(mat.albedo[..., :3])
                        tiles.append(img.resize((200, 200), Image.NEAREST if max(img.size) < 200 else Image.BILINEAR))
                row = Image.new("RGB", (4 * 206, 220), (24, 22, 28))
                for i, tile in enumerate(tiles[:4]):
                    row.paste(tile, (i * 206, 20))
                ImageDraw.Draw(row).text((4, 4), f"type {t['type']}  x{t['count']}  {' x '.join(f'{v:.0f}' for v in t['size_mm'])} mm   (piece from two sides | its textures)", fill=(240, 230, 210))
                rows.append(row)
            sheet = Image.new("RGB", (rows[0].width, 220 * len(rows)), (24, 22, 28))
            for i, r in enumerate(rows):
                sheet.paste(r, (0, i * 220))
            sheets.append(preview.png(sheet))
        return sheets

    def start_remodel(self, pid: str, direct: bool = False, rounds: int = 0, wish: str = "") -> dict[str, Any]:
        if direct:
            status = self.llm.status()
            if not status.get("available"):
                raise PhidiasError("unavailable", status.get("reason") or "No hay modelo con visión disponible.",
                                   "Comprueba que Faustus y las Sparks están en marcha, o indica un servidor en Ajustes.")
            if not (finish.find_blender(self.settings["blender"]) or remodel_build.has_bpy()):
                raise PhidiasError("unavailable", "No encuentro Blender en este equipo.", "Instálalo o indica su ruta en Ajustes.")
            return self._enqueue("remodel_direct", pid, {"rounds": int(rounds or self.settings["director_rounds"]), "wish": wish})
        return self._start_remodel_plain(pid)

    def _start_remodel_plain(self, pid: str) -> dict[str, Any]:
        if not (finish.find_blender(self.settings["blender"]) or remodel_build.has_bpy()):
            raise PhidiasError("unavailable", "No encuentro Blender en este equipo.", "Instálalo o indica su ruta en Ajustes.")
        return self._enqueue("remodel", pid)

    def _do_remodel(self, pid: str, say) -> dict[str, Any]:
        t0 = time.time()
        say("separando piezas", 0.05)
        plan = self.remodel_plan(pid, full=True)
        parts = plan.pop("_parts")
        textures = plan.pop("_textures")
        pdir = self.store.dir(pid)
        say("Blender: construyendo las piezas", 0.2)
        try:
            stats = remodel_build.run({"pieces": parts, "textures": textures}, pdir / "remodel", finish.find_blender(self.settings["blender"]))
        except Exception as exc:
            raise PhidiasError("failed", str(exc)) from exc
        say("vista", 0.85)
        P, F, E = preview.read_polygons(pdir / "remodel" / "remodel.obj")
        np.save(pdir / "remodel_P.npy", P.astype(np.float32)); np.save(pdir / "remodel_F.npy", F); np.save(pdir / "remodel_E.npy", E)
        (pdir / "remodel.bin").write_bytes(export.viewer_buffers(P, F))
        (pdir / "remodel_edges.bin").write_bytes(np.array([len(E)], "<u4").tobytes() + P[E].astype("<f4").tobytes())
        preview.render(P, F, size=640, edges=E).save(pdir / "remodel.png")
        total = {k: int(sum(s[k] for s in stats["pieces"])) for k in ("verts", "quads", "tris", "ngons")}
        kinds: dict[str, int] = {}
        for p in parts:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        info = {**total, "objects": len(stats["pieces"]), "pieces": len(parts), "kinds": kinds, "triangles_total": int(len(F)),
                "open_edges": None, "size_mm": [round(float(v), 2) for v in (P.max(0) - P.min(0))] if len(P) else [0, 0, 0],
                "file_bytes": (pdir / "remodel" / "remodel.obj").stat().st_size, "seconds": round(time.time() - t0, 1), "at": time.time(),
                "log": stats["log"]}
        project = self.store.get(pid)
        project.setdefault("results", {})["remodel"] = info
        self.store.save(project)
        return info

    def start_direct(self, pid: str, rounds: int = 0, wish: str = "") -> dict[str, Any]:
        status = self.llm.status()
        if not status.get("available"):
            raise PhidiasError("unavailable", status.get("reason") or "No hay modelo con visión disponible.",
                               "Comprueba que Faustus y las Sparks están en marcha, o indica un servidor en Ajustes.")
        return self._enqueue("direct", pid, {"rounds": int(rounds or self.settings["director_rounds"]), "wish": wish})

    def _do_direct(self, pid: str, say, params: dict[str, Any], begin_publish) -> dict[str, Any]:
        outcome = director.run(self, pid, params["rounds"], params["wish"], lambda stage, frac: say(stage, frac * 0.7))
        final = self._do_sculpt(pid, lambda stage, frac: say("esculpido final: " + stage, 0.7 + frac * 0.3), begin_publish)
        return {**final, "director": {k: outcome[k] for k in ("rounds", "verdict", "notes")}}

    def _job_public(self, job: dict[str, Any]) -> dict[str, Any]:
        return {k: job[k] for k in ("id", "kind", "project_id", "state", "stage", "progress", "error", "created",
                                    "finished", "result", "cancellable", "cancel_requested")}

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise PhidiasError("not_found", "No hay ningún trabajo con ese id.")
            return self._job_public(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise PhidiasError("not_found", "No hay ningún trabajo con ese id.")
            if job["state"] == "cancelled" or job["cancel_requested"]:
                return self._job_public(job)
            if not job["cancellable"] or job["state"] not in ("queued", "running"):
                raise PhidiasError("conflict", "Ese trabajo ya no se puede cancelar.")
            job.update(cancel_requested=True, cancellable=False, stage="cancelando")
            if job["state"] == "queued":
                self._mark_cancelled(job)
            return self._job_public(job)

    @staticmethod
    def _mark_cancelled(job: dict[str, Any]) -> None:
        job.update(state="cancelled", stage="cancelado", finished=time.time(), cancellable=False)

    def _enqueue(self, kind: str, pid: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self.store.get(pid)
        with self._lock:
            for j in self.jobs.values():
                if j["project_id"] == pid and j["state"] in ("queued", "running"):
                    raise PhidiasError("busy", "Ese proyecto ya tiene un trabajo en marcha.", "Consulta job_status y espera a que acabe.")
            job = {"id": uuid.uuid4().hex[:10], "kind": kind, "project_id": pid, "state": "queued", "stage": "en cola", "progress": 0.0,
                   "error": "", "created": time.time(), "finished": None, "result": None, "params": params or {},
                   "cancellable": kind in CANCELLABLE_JOBS, "cancel_requested": False}
            self.jobs[job["id"]] = job
            while len(self.jobs) > 60:
                expired = next((k for k, v in self.jobs.items() if v["state"] not in ("queued", "running")), None)
                if expired is None:
                    break
                self.jobs.pop(expired)
        if self.config.background and self._worker is not None:
            self._queue.put(job["id"])
        else:
            self._execute(job)
        return self._job_public(job)

    def start_sculpt(self, pid: str) -> dict[str, Any]:
        return self._enqueue("sculpt", pid)

    def start_close(self, pid: str, fill_interior: bool = True) -> dict[str, Any]:
        project = self.store.get(pid)
        if "sculpt" not in project.get("results", {}):
            raise PhidiasError("invalid", "Primero hay que esculpir el proyecto.", "Lanza sculpt_start y espera a que acabe.")
        if not finish.find_blender(self.settings["blender"]):
            raise PhidiasError("unavailable", "No encuentro Blender en este equipo.", "Instálalo o indica su ruta en el ajuste blender.")
        return self._enqueue("close", pid, {"fill_interior": fill_interior})

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.jobs.get(job_id)
            if job is not None:
                self._execute(job)

    def _execute(self, job: dict[str, Any]) -> None:
        with self._lock:
            if job["state"] != "queued":
                return
            job["state"] = "running"

        def say(stage: str, frac: float) -> None:
            with self._lock:
                if job["cancel_requested"]:
                    raise _JobCancelled()
                job["stage"], job["progress"] = stage, round(float(frac), 3)

        def begin_publish() -> None:
            with self._lock:
                if job["cancel_requested"]:
                    raise _JobCancelled()
                job.update(cancellable=False, stage="guardando el resultado")

        try:
            if job["kind"] == "sculpt":
                job["result"] = self._do_sculpt(job["project_id"], say, begin_publish)
            elif job["kind"] == "remodel":
                job["result"] = self._do_remodel(job["project_id"], say)
            elif job["kind"] == "remodel_direct":
                job["result"] = director.run_remodel(self, job["project_id"], job["params"]["rounds"], job["params"]["wish"], say)
            elif job["kind"] == "direct":
                job["result"] = self._do_direct(job["project_id"], say, job["params"], begin_publish)
            else:
                job["result"] = self._do_close(job["project_id"], say, job["params"]["fill_interior"])
            job["state"] = "done"
            job["stage"], job["progress"] = "hecho", 1.0
        except _JobCancelled:
            self._mark_cancelled(job)
        except PhidiasError as exc:
            job["state"], job["error"] = "error", str(exc)
        except MemoryError:
            job["state"], job["error"] = "error", "No hay memoria para tantos triángulos: baja el presupuesto o sube el detalle mínimo."
        except Exception as exc:
            log.exception("job %s", job["id"])
            job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
        with self._lock:
            job.update(cancellable=False, finished=job["finished"] or time.time())

    def _do_sculpt(self, pid: str, say, begin_publish) -> dict[str, Any]:
        project = self.store.get(pid)
        scene = self.scene(pid)
        st = self._settings_of(project)
        rec = {name: sculpting.MaterialRecipe.from_dict(m) for name, m in project["materials"].items()}
        t0 = time.time()
        result = sculpting.sculpt(scene, rec, st, lambda stage, frac: say(stage, frac * 0.8))
        say("cerrando el suelo", 0.82)
        P, F = sculpting.assemble(result, st)
        pdir = self.store.dir(pid)
        with tempfile.TemporaryDirectory(prefix="sculpt-", dir=pdir) as scratch:
            draft = Path(scratch)
            say("escribiendo STL", 0.86)
            size = export.write_stl(draft / "sculpt.stl", P, F, project["name"])
            np.save(draft / "sculpt_P.npy", P.astype(np.float32))
            np.save(draft / "sculpt_F.npy", F)
            say("vista 3D", 0.9)
            limit = int(self.settings["preview_faces"])
            view = result
            if len(result.F) > limit:
                light = sculpting.SculptSettings.from_dict({**project["settings"], "max_faces": limit, "ground": False})
                view = sculpting.sculpt(scene, rec, light, lambda stage, frac: say(f"vista 3D: {stage}", 0.9 + frac * 0.05))
            (draft / "sculpt.bin").write_bytes(export.viewer_buffers(view.P, view.F))
            say("imagen", 0.95)
            preview.render(result.P, result.F, size=640).save(draft / "sculpt.png")
            info = {**result.info, "file_bytes": size, "triangles_total": int(len(F)), "open_edges": sculpting.open_edges(F, len(P)),
                    "seconds": round(time.time() - t0, 1), "at": time.time(), "settings": {k: getattr(st, k) for k in PROJECT_SETTINGS}}
            begin_publish()
            for artifact in draft.iterdir():
                artifact.replace(pdir / artifact.name)
        project = self.store.get(pid)
        project.setdefault("results", {})["sculpt"] = info
        project["results"].pop("closed", None)
        for stale in ("closed.stl", "closed.bin", "closed.png", "closed_P.npy", "closed_F.npy"):
            (pdir / stale).unlink(missing_ok=True)
        self.store.save(project)
        return info

    def _do_close(self, pid: str, say, fill_interior: bool) -> dict[str, Any]:
        project = self.store.get(pid)
        pdir = self.store.dir(pid)
        blender = finish.find_blender(self.settings["blender"])
        if not blender:
            raise PhidiasError("unavailable", "No encuentro Blender en este equipo.")
        t0 = time.time()
        say("preparando", 0.05)
        P, F = np.load(pdir / "sculpt_P.npy"), np.load(pdir / "sculpt_F.npy")
        voxel, shell = float(self.settings["voxel_mm"]), float(self.settings["shell_mm"])
        with tempfile.TemporaryDirectory(prefix="close-", dir=pdir) as scratch:
            ply = Path(scratch) / "sculpt.ply"
            raw = Path(scratch) / "closed.stl"
            export.write_ply(ply, P, F)
            say("Blender: fundiendo en un sólido", 0.15)
            try:
                run = finish.close_mesh(blender, ply, raw, voxel, shell, int(project["settings"].get("max_faces", 8_000_000)))
                say("comprobando", 0.85)
                CP, CF = finish.read_stl(raw)
            except Exception as exc:
                raise PhidiasError("failed", str(exc)) from exc
        cavities = 0
        if fill_interior:
            say("rellenando cavidades", 0.88)
            CP, CF, cavities = finish.fill_cavities(CP, CF)
        CP -= CP.min(0)
        size = export.write_stl(pdir / "closed.stl", CP, CF, project["name"])
        np.save(pdir / "closed_P.npy", CP.astype(np.float32))
        np.save(pdir / "closed_F.npy", CF)
        if len(CF) <= int(self.settings["preview_faces"]) * 3:
            (pdir / "closed.bin").write_bytes(export.viewer_buffers(CP, CF))
        preview.render(CP, CF, size=640).save(pdir / "closed.png")
        info = {"triangles_total": int(len(CF)), "open_edges": sculpting.open_edges(CF, len(CP)), "file_bytes": size,
                "size_mm": [round(float(v), 2) for v in CP.max(0)], "voxel_mm": voxel, "shell_mm": shell,
                "seconds": round(time.time() - t0, 1), "at": time.time(), "log": run["log"],
                "fill_interior": fill_interior, "cavities_filled": cavities}
        info["watertight"] = info["open_edges"] == 0
        project = self.store.get(pid)
        project.setdefault("results", {})["closed"] = info
        self.store.save(project)
        return info

    # ------------------------------------------------------------------ results
    def mesh(self, pid: str, which: str) -> tuple[np.ndarray, np.ndarray]:
        pdir = self.store.dir(pid)
        if which == "before":
            project = self.store.get(pid)
            scene = self.scene(pid)
            scale = self._settings_of(project).size_mm / max(float(scene.extents.max()), 1e-9)
            return (scene.P - scene.P.min(0)) * scale, scene.F
        if which not in RESULTS or not (pdir / f"{which}_P.npy").is_file():
            raise PhidiasError("not_found", "Ese resultado todavía no existe.", "Lanza sculpt_start (y close_start para el sólido cerrado).")
        return np.load(pdir / f"{which}_P.npy").astype(np.float64), np.load(pdir / f"{which}_F.npy")

    def look(self, pid: str, which: str = "sculpt", view: str = "iso", yaw: Optional[float] = None, pitch: Optional[float] = None,
             size: int = 900, region: Optional[list[float]] = None, compare: bool = False, wire: bool = True) -> bytes:
        y, p = preview.VIEWS.get(view, preview.VIEWS["iso"])
        y = y if yaw is None else yaw
        p = p if pitch is None else pitch
        size = int(min(max(size, 200), 2000))
        P, F = self.mesh(pid, which)
        edges = None
        if which == "remodel" and wire and (self.store.dir(pid) / "remodel_E.npy").is_file():
            edges = np.load(self.store.dir(pid) / "remodel_E.npy")
        img = preview.render(P, F, yaw=y, pitch=p, size=size, region=region, edges=edges)
        if compare and which != "before":
            BP, BF = self.mesh(pid, "before")
            if region is None:                             # the result sits on its base: line the original up with it
                BP = BP + np.array([0, 0, float(P[:, 2].max() - BP[:, 2].max())])
            img = preview.side_by_side([preview.render(BP, BF, yaw=y, pitch=p, size=size, region=region), img], ["original", "remodelado" if which == "remodel" else "esculpido"])
        return preview.png(img)

    def _export_remodel(self, project: dict[str, Any], folder: str) -> dict[str, Any]:
        src = self.store.dir(project["id"]) / "remodel"
        if not (src / "remodel.obj").is_file():
            raise PhidiasError("not_found", "Todavía no hay remodelado.", "Lanza remodel_start y espera a que acabe.")
        dest = Path(folder).expanduser() if folder else (Path(self.settings["export_dir"]) if self.settings["export_dir"]
                                                         else Path(project["source"]["path"]).parent / "Phidias")
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError:
            dest = self.config.data_dir / "exports"
            dest.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in project["name"]).strip() or project["id"]
        size = int(round(float(project["settings"].get("size_mm", 0))))
        paths = []
        for ext in ("blend", "obj"):
            out = dest / f"{safe} {size}mm remodelado.{ext}"
            shutil.copyfile(src / f"remodel.{ext}", out)
            paths.append(str(out))
        r = project["results"]["remodel"]
        return {"path": paths[0], "paths": paths, "result": "remodel", "quads": r["quads"], "tris": r["tris"], "ngons": r["ngons"]}

    def export_result(self, pid: str, which: str = "", folder: str = "") -> dict[str, Any]:
        project = self.store.get(pid)
        results = project.get("results", {})
        which = which or ("remodel" if "remodel" in results else ("closed" if "closed" in results else "sculpt"))
        if which == "remodel":
            return self._export_remodel(project, folder)
        src = self.store.dir(pid) / f"{which}.stl"
        if which not in RESULTS or not src.is_file():
            raise PhidiasError("not_found", "Ese resultado todavía no existe.", "Lanza sculpt_start y espera a que acabe.")
        if folder:
            dest = Path(folder).expanduser()
        elif self.settings["export_dir"]:
            dest = Path(self.settings["export_dir"])
        else:
            dest = Path(project["source"]["path"]).parent / "Phidias"
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError:
            dest = self.config.data_dir / "exports"
            dest.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in project["name"]).strip() or project["id"]
        size = int(round(float(project["settings"].get("size_mm", 0))))
        out = dest / f"{safe} {size}mm{' cerrado' if which == 'closed' else ''}.stl"
        shutil.copyfile(src, out)
        return {"path": str(out), "bytes": out.stat().st_size, "result": which, "triangles": results[which].get("triangles_total")}
