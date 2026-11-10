"""A first guess of what to do with each material, from its name and its pictures. The user (or an assistant looking
at ``material_look``) corrects it afterwards."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from . import heights
from .meshio import Material

# planes games use for light and atmosphere: they are not part of the place
_DROP = re.compile(r"(^|[_\-. ])(h_)?(kage|shadow|flame|fire|sparks?|overlay|glow|light|fog|mist|cloud|sky|lens|flare|godray|reflect)", re.I)
_STEPS = re.compile(r"(stair|staire|steps?($|[_\-. 0-9])|kaidan|escal)", re.I)
_FLAT = re.compile(r"(water|mizu|sea|umi|lava|magma|glass|mirror)", re.I)


def describe(mat: Material) -> dict[str, Any]:
    """What a material's pictures hold: size, how much of it is see-through, how much it varies."""
    out: dict[str, Any] = {"name": mat.name, "texture": list(mat.size), "normal_map": heights.has_normal_detail(mat.normal),
                           "atlas": bool(mat.atlas), "transparent": 0.0, "contrast": 0.0}
    if mat.albedo is not None:
        out["transparent"] = round(float((mat.albedo[..., 3] < 128).mean()), 3)
        out["contrast"] = round(float(heights.luminance(mat.albedo).std()), 3)
        rgb = mat.albedo[..., :3].reshape(-1, 3)
        solid = rgb[mat.albedo[..., 3].reshape(-1) >= 128]
        out["color"] = [int(v) for v in (solid if len(solid) else rgb).mean(0)]
    else:
        out["color"] = list(mat.color)
    return out


def guess(mat: Material, depth_mm: float) -> dict[str, Any]:
    info = describe(mat)
    recipe, why = "auto", ""
    if _DROP.search(mat.name):
        recipe, why = "drop", "luz, sombra o atmósfera"
    elif _FLAT.search(mat.name):
        recipe, why = "flat", "agua, lava o cristal"
    elif _STEPS.search(mat.name) and mat.albedo is not None and not mat.atlas and any(c for c, _ in heights.step_pattern(mat.albedo).values()):
        recipe, why = "steps", "escalera dibujada en la textura"
    elif info["normal_map"]:
        recipe, why = "normal", "tiene mapa de normales"
    elif mat.albedo is None or info["contrast"] < 0.012:
        recipe, why = "flat", "color liso"
    elif mat.atlas:
        recipe, why = "relief", "dibujo del atlas horneado"
    else:
        recipe, why = "relief", "dibujo de la textura"
    cutout = recipe != "drop" and not mat.atlas and 0.02 < info["transparent"] < 0.98
    if cutout:
        why = (why + ", recortado por su transparencia").lstrip(", ")
    return {"recipe": recipe, "depth_mm": float(depth_mm), "invert": False, "smooth": 0.0, "cutout": cutout, "why": why}


def sheet(mat: Material, recipe: str, invert: bool = False, smooth: float = 0.0, side: int = 384) -> "np.ndarray":
    """Albedo | normal map | height map side by side (uint8 RGB) to judge a recipe by eye."""
    from PIL import Image

    def fit(arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.full((side, side, 3), 40, np.uint8)
        img = Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr).convert("RGB")
        return np.asarray(img.resize((side, side), Image.NEAREST if max(img.size) < side else Image.BILINEAR))

    if recipe in ("flat", "drop"):
        h = np.zeros((4, 4), np.float32)
    elif recipe == "steps" and mat.albedo is not None:      # the saw-tooth the ramp will follow, along the axis found
        pattern = heights.step_pattern(mat.albedo)
        axis = "u" if pattern["u"][0] > pattern["v"][0] else "v"
        k, off = pattern[axis]
        t = (np.arange(256) + 0.5) / 256.0
        saw = ((t * k - off * k) % 1.0) * 2 - 1 if k else np.zeros(256)
        h = np.tile(saw[None, :], (256, 1)) if axis == "u" else np.tile(saw[::-1, None], (1, 256))
        h = h.astype(np.float32)
    else:
        h = heights.height_map(mat.albedo, mat.normal, recipe, invert=invert, smooth=smooth, atlas=mat.atlas)
    grey = ((h * 0.5 + 0.5) * 255).astype(np.uint8)
    gap = np.full((side, 6, 3), 24, np.uint8)
    return np.concatenate([fit(mat.albedo), gap, fit(mat.normal), gap, fit(grey)], axis=1)
