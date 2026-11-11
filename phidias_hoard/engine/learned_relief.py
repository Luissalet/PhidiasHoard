"""Turn local relative depth into bounded relief, keeping atlas islands separate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..errors import PhidiasError


SCRIPT = Path(__file__).with_name("depth_worker.py")


def _inputs(scene, index: int):
    mat = scene.materials[index]
    if mat.albedo is None:
        raise PhidiasError("invalid", "El material no tiene una textura de color para estimar el relieve.")
    image = Image.fromarray(mat.albedo)
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    width, height = image.size
    pixels = np.asarray(image)
    mask = np.ones((height, width), dtype=bool)
    if pixels.shape[2] == 4:
        mask &= pixels[..., 3] >= 128
    if mat.atlas:
        used = Image.new("1", image.size)
        draw = ImageDraw.Draw(used)
        uv = np.clip(scene.UV[scene.M == index], 0, 1)
        for tri in uv:
            draw.polygon([(float(u) * (width - 1), (1 - float(v)) * (height - 1)) for u, v in tri], fill=1)
        mask &= np.asarray(used)
    return pixels[..., :3], mask


def _relief(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sigma = max(1.5, min(raw.shape) / 16)
    blurred = ndimage.gaussian_filter(np.where(mask, raw, 0), sigma)
    weight = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    residual = raw - blurred / np.maximum(weight, 1e-6)
    scale = np.percentile(np.abs(residual[mask]), 95)
    return np.where(mask, np.clip(residual / max(scale, 1e-5), -1, 1), 0).astype(np.float32)


def height_map(scene, index: int, model: str, python: str, cache: Path) -> np.ndarray:
    model_dir = Path(model).expanduser()
    executable = Path(python or sys.executable).expanduser()
    weights = model_dir / "model.safetensors"
    if not model or not weights.is_file() or not (model_dir / "config.json").is_file():
        raise PhidiasError("unavailable", "Configura la carpeta local del modelo de relieve en Ajustes.")
    if not executable.is_file():
        raise PhidiasError("unavailable", "No encuentro el Python del modelo de relieve configurado.")
    pixels, mask = _inputs(scene, index)
    identity = hashlib.sha256()
    identity.update(f"v1:{model_dir.resolve()}:{weights.stat().st_mtime_ns}:{mask.shape}".encode())
    identity.update(pixels.tobytes())
    identity.update(mask.tobytes())
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / (identity.hexdigest() + ".npy")
    if saved.is_file():
        try:
            return np.load(saved, allow_pickle=False)
        except (OSError, ValueError):
            pass
    labels, _ = ndimage.label(mask)
    areas = np.bincount(labels.reshape(-1))
    regions = ndimage.find_objects(labels)
    result = np.zeros(mask.shape, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="inference-", dir=cache) as temp:
        root = Path(temp)
        selected = {}
        for ident in sorted(range(1, len(areas)), key=lambda i: -areas[i])[:24]:
            bounds = regions[ident - 1]
            if not bounds or areas[ident] < 64 or min(s.stop - s.start for s in bounds) < 8:
                continue
            island = labels[bounds] == ident
            rgb = pixels[bounds].copy()
            rgb[~island] = np.median(rgb[island], axis=0)
            Image.fromarray(rgb).save(root / f"{ident}.png")
            selected[ident] = (bounds, island)
        if not selected:
            raise PhidiasError("invalid", "La textura no tiene ninguna isla suficientemente grande para calcular relieve.")
        try:
            run = subprocess.run([str(executable.resolve()), str(SCRIPT), str(model_dir.resolve()), str(root.resolve())],
                                 capture_output=True, text=True, errors="replace", timeout=240,
                                 env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise PhidiasError("failed", "El modelo de relieve tardó más de cuatro minutos.") from exc
        if run.returncode:
            detail = (run.stderr or run.stdout).strip().splitlines()[-1:]
            raise PhidiasError("failed", "No se pudo ejecutar el modelo de relieve. " + " ".join(detail)[:260],
                               "Comprueba el Python con Torch y Transformers y la carpeta del modelo.")
        for ident, (bounds, island) in selected.items():
            raw = np.load(root / f"{ident}.npy", allow_pickle=False)
            if raw.shape != island.shape or not np.isfinite(raw).all():
                raise PhidiasError("failed", "El modelo devolvió un mapa de relieve inválido.")
            result[bounds][island] = _relief(raw, island)[island]
        draft = root / "height.npy"
        np.save(draft, result)
        draft.replace(saved)
    return result
