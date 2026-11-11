"""Process configuration (environment) and the editable settings (``<data>/settings.json``)."""

from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import APP_ID, DEFAULT_PORT
from .hoard_link.appconfig import AppPaths, env_int, env_str
from .hoard_link.atomic import read_json, write_json_atomic
from .hoard_link.guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTINGS: dict[str, Any] = {
    "library_dirs": [],          # folders with game models (OBJ, GLB, glTF, or zips of them) offered for sculpting
    "size_mm": 150.0,            # longest side of a new project's print
    "min_edge_mm": 0.25,         # finest triangle edge: the detail the printer can show
    "tolerance_mm": 0.03,        # relief smaller than this inside a triangle does not get more triangles
    "max_faces": 8_000_000,      # triangle budget of a sculpt
    "depth_mm": 0.6,             # default depth of the relief of a material
    "base_mm": 2.0,              # solid thickness under the lowest point of the ground
    "plinth": "none",            # none, rect, round
    "preview_faces": 1_500_000,  # the 3-D view gets at most this many triangles
    "blender": "",               # path of blender.exe; empty: found by itself
    "depth_model": "",           # local Transformers depth checkpoint
    "depth_python": "",          # optional Python with Torch and Transformers
    "voxel_mm": 0.2,             # closing a mesh with Blender: voxel size
    "shell_mm": 1.2,             # closing a mesh with Blender: thickness given to loose sheets
    "llm_url": "",               # empty: the family's vision model through Hoard Link; else an OpenAI-compatible URL
    "llm_model": "",
    "llm_timeout_s": 1800,       # per question to the model
    "director_rounds": 3,        # test sculpts the model may look at and redo before the final one
    "director_faces": 2_500_000, # triangle budget of those test sculpts
    "export_dir": "",            # where "save next to the source" does not apply; empty: <data>/exports
}
PLINTHS = ("none", "rect", "round")


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    port: int = DEFAULT_PORT
    allowed_hosts: tuple[str, ...] = ()
    data_dir_configured: bool = False
    background: bool = True          # the sculpting worker; tests switch it off and run jobs inline

    @property
    def paths(self) -> AppPaths:
        return AppPaths(APP_ID, REPO_ROOT, self.data_dir, self.data_dir_configured)

    @property
    def token_path(self) -> Path:
        return self.paths.token_path

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @classmethod
    def from_env(cls) -> "Config":
        data_env = env_str("PHIDIAS_DATA_DIR")
        return cls(
            data_dir=Path(data_env).expanduser() if data_env else REPO_ROOT / "data",
            port=env_int("PHIDIAS_PORT", default=DEFAULT_PORT) or DEFAULT_PORT,
            allowed_hosts=tuple(parse_allowed_hosts(os.environ.get("PHIDIAS_ALLOWED_HOSTS", ""))),
            data_dir_configured=bool(data_env),
        )


class Settings:
    """Thread-safe JSON settings with defaults; ``update`` validates and writes atomically."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        raw = read_json(self.path, default={}) or {}
        data = copy.deepcopy(DEFAULT_SETTINGS)
        if isinstance(raw, dict):
            data.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        try:
            return normalize(data)
        except (ValueError, TypeError):
            return copy.deepcopy(DEFAULT_SETTINGS)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return copy.deepcopy(self._data[key])

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = copy.deepcopy(self._data)
            for key, value in dict(changes).items():
                if key not in DEFAULT_SETTINGS:
                    raise ValueError(f"Unknown setting: {key}")
                if value is None:
                    continue
                data[key] = value
            data = normalize(data)
            self._data = data
            write_json_atomic(self.path, data)
        return self.get()


def _clamp(v: Any, lo: float, hi: float, kind=float):
    return kind(min(max(kind(v), lo), hi))


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    dirs = data.get("library_dirs") or []
    if isinstance(dirs, str):
        dirs = [dirs]
    data["library_dirs"] = [str(d).strip() for d in dirs if str(d).strip()][:32]
    data["size_mm"] = _clamp(data["size_mm"], 10, 2000)
    data["min_edge_mm"] = _clamp(data["min_edge_mm"], 0.05, 20)
    data["tolerance_mm"] = _clamp(data["tolerance_mm"], 0.005, 2)
    data["max_faces"] = _clamp(data["max_faces"], 10_000, 60_000_000, int)
    data["depth_mm"] = _clamp(data["depth_mm"], 0, 20)
    data["base_mm"] = _clamp(data["base_mm"], 0, 100)
    if data["plinth"] not in PLINTHS:
        raise ValueError("plinth must be none, rect or round")
    data["preview_faces"] = _clamp(data["preview_faces"], 50_000, 20_000_000, int)
    data["blender"] = str(data.get("blender") or "").strip()
    data["depth_model"] = str(data.get("depth_model") or "").strip()
    data["depth_python"] = str(data.get("depth_python") or "").strip()
    data["voxel_mm"] = _clamp(data["voxel_mm"], 0.05, 5)
    data["shell_mm"] = _clamp(data["shell_mm"], 0, 20)
    data["export_dir"] = str(data.get("export_dir") or "").strip()
    data["llm_url"] = str(data.get("llm_url") or "").strip()
    data["llm_model"] = str(data.get("llm_model") or "").strip()
    data["llm_timeout_s"] = _clamp(data["llm_timeout_s"], 30, 14400, int)
    data["director_rounds"] = _clamp(data["director_rounds"], 1, 6, int)
    data["director_faces"] = _clamp(data["director_faces"], 200_000, 20_000_000, int)
    return data
