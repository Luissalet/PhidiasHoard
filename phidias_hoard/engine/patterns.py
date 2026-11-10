"""Sculpted detail a texture does not hold: rock, cracks, cobbles, bricks, planks, roof tiles, bark and ripples as
tileable height maps (-1..1), laid on the model by its position in millimetres (three projections blended by the
surface's direction), so the detail has a real size and does not depend on the UV layout."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import heights

N = 512
# pattern -> how many features (cells, courses, planks) one tile holds across
CELLS = {"rock": 4, "cracks": 5, "cobble": 6, "bricks": 4, "planks": 6, "tiles": 4, "bark": 6, "ripples": 6, "flutes": 6}
PATTERNS = tuple(CELLS)
LABELS = {"rock": "roca", "cracks": "grietas", "cobble": "adoquines", "bricks": "ladrillo", "planks": "tablas", "tiles": "tejas",
          "bark": "corteza", "ripples": "ondas", "flutes": "estrías"}


def _noise(rng: np.random.Generator, beta: float, stretch: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    fy = np.fft.fftfreq(N).reshape(N, 1) * stretch[0]
    fx = np.fft.fftfreq(N).reshape(1, N) * stretch[1]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / f ** beta
    spec[0, 0] = 0
    h = np.real(np.fft.ifft2(spec))
    return (h / np.percentile(np.abs(h), 99.5)).clip(-1, 1).astype(np.float32)


def _voronoi(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance to the nearest and second nearest of ``count`` points scattered on the tile (wrapping), and which."""
    pts = rng.random((count, 2))
    tiled = np.concatenate([pts + (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    t = (np.arange(N) + 0.5) / N
    grid = np.stack(np.meshgrid(t, t), -1).reshape(-1, 2)
    d, i = cKDTree(tiled).query(grid, k=2)
    return d[:, 0].reshape(N, N), d[:, 1].reshape(N, N), (i[:, 0] % count).reshape(N, N)


@lru_cache(maxsize=16)
def pattern(name: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31 if name not in CELLS else list(CELLS).index(name) + 11)
    t = (np.arange(N) + 0.5) / N
    u, v = np.meshgrid(t, t[::-1])                       # v grows upwards, like UVs
    if name == "rock":
        h = 0.7 * _noise(rng, 1.7) + 0.3 * np.abs(_noise(rng, 1.2)) * -1
    elif name == "cracks":
        f1, f2, _ = _voronoi(rng, 24)
        h = -np.clip(1.0 - (f2 - f1) / 0.018, 0, 1) + 0.12 * _noise(rng, 1.5)
    elif name == "cobble":
        f1, f2, cell = _voronoi(rng, 36)
        edge = np.clip((f2 - f1) / 0.035, 0, 1)
        lift = rng.uniform(-0.25, 0.25, 36)[cell]
        h = edge * (0.75 + lift) - 0.9 * (1 - edge) - 0.25 * (f1 / f1.max()) ** 2
    elif name == "bricks":
        rows, cols = 4, 2
        row = np.floor(v * rows).astype(int)
        uu = (u * cols + 0.5 * (row % 2)) % cols
        fu, fv = uu % 1.0, (v * rows) % 1.0
        joint = np.minimum(np.minimum(fu, 1 - fu) * (rows / cols) * 2, np.minimum(fv, 1 - fv))
        groove = np.clip(joint / 0.07, 0, 1)
        lift = rng.uniform(-0.2, 0.2, (rows, cols))[row % rows, np.floor(uu).astype(int) % cols]
        h = groove * (0.8 + lift) - (1 - groove) + 0.1 * _noise(rng, 1.4)
    elif name == "planks":
        k = CELLS["planks"]
        fu = (u * k) % 1.0
        groove = np.clip(np.minimum(fu, 1 - fu) / 0.06, 0, 1)
        lift = rng.uniform(-0.2, 0.2, k)[np.floor(u * k).astype(int) % k]
        h = groove * (0.7 + lift) - (1 - groove) + 0.22 * _noise(rng, 1.3, (1.0, 0.12))
    elif name == "tiles":
        k = CELLS["tiles"]
        row = np.floor(v * k).astype(int)
        fu, fv = (u * k + 0.5 * (row % 2)) % 1.0, (v * k) % 1.0
        round_edge = np.sqrt(np.clip(1 - (2 * fu - 1) ** 2, 0, 1))        # each tile ends in a rounded lower edge
        h = (1 - fv) * 1.6 - 0.8 - 0.9 * np.clip(1 - (fv - (1 - round_edge) * 0.45) / 0.1, 0, 1) * (fv < 0.6)
    elif name == "bark":
        h = 1 - 2 * np.abs(_noise(rng, 1.5, (1.0, 0.15)))
    elif name == "flutes":                                  # the channels of a column: round grooves with a thin flat between
        fu = (u * CELLS["flutes"]) % 1.0
        h = -np.sqrt(np.clip(1 - ((fu - 0.5) / 0.42) ** 2, 0, 1)) * 1.0 + 0.0 * v
    elif name == "ripples":
        h = np.sin(2 * np.pi * (v * CELLS["ripples"] + 0.6 * _noise(rng, 2.2)))
    else:
        raise ValueError(f"unknown pattern: {name}")
    h = ndimage.gaussian_filter(h.astype(np.float32), 1.0, mode="wrap")
    return np.clip(h, -1, 1).astype(np.float32)


def triplanar(name: str, points: np.ndarray, normals: np.ndarray, size_mm: float) -> np.ndarray:
    """The pattern's height (-1..1) at each point (millimetres), features about ``size_mm`` across. Walls take it
    with its rows horizontal; floors and roofs take it seen from above."""
    tile = pattern(name)
    period = max(size_mm, 0.05) * CELLS[name]
    p = points / period
    w = np.abs(normals) ** 6 + 1e-9
    w /= w.sum(axis=1, keepdims=True)
    out = w[:, 0] * heights.sample(tile, np.stack([p[:, 1], p[:, 2]], 1))
    out += w[:, 1] * heights.sample(tile, np.stack([p[:, 0], p[:, 2]], 1))
    out += w[:, 2] * heights.sample(tile, np.stack([p[:, 0], p[:, 1]], 1))
    return out
