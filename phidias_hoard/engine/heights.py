"""Relief read out of a material's pictures.

Every recipe returns a tileable height map in -1..1 (0 = the original surface, negative = carved in) at a resolution
high enough to sculpt from, whatever the size of the game texture:

* ``normal``  integrates the tangent-space normal map (what the game itself uses to fake cracks, joints and flutes).
* ``lines``   carves what the texture draws darker than its surroundings: joints between blocks, cracks, the lines of
  a column, the edge of each step.
* ``relief``  lines plus the soft light and dark areas as gentle bumps.
* ``steps``   a ramp whose texture draws steps becomes real steps (see :func:`step_pattern`).
* ``flat``    no relief.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

RECIPES = ("auto", "normal", "lines", "relief", "steps", "flat", "drop")
TARGET = 512          # small textures are enlarged to about this many pixels a side
MAX_SIDE = 2048       # big ones are reduced to this


def luminance(albedo: np.ndarray) -> np.ndarray:
    rgb = albedo[..., :3].astype(np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def _enlarge(h: np.ndarray, soften: float = 0.45) -> np.ndarray:
    """Pixel art up to a sculptable size: each pixel becomes a block, then the blocks are rounded off so a one-pixel
    line turns into a groove with sloped walls instead of a square trench."""
    side = max(h.shape)
    if side > MAX_SIDE:
        f = MAX_SIDE / side
        return ndimage.zoom(h, f, order=1, mode="grid-wrap").astype(np.float32)
    k = int(np.clip(round(TARGET / side), 1, 32))
    if k == 1:
        return h.astype(np.float32)
    big = np.repeat(np.repeat(h, k, axis=0), k, axis=1)
    return ndimage.gaussian_filter(big, k * soften, mode="wrap").astype(np.float32)


def _norm(h: np.ndarray, pct: float = 99.5) -> np.ndarray:
    scale = float(np.percentile(np.abs(h), pct)) if h.size else 0.0
    return np.clip(h / scale, -1.0, 1.0).astype(np.float32) if scale > 1e-6 else np.zeros_like(h, np.float32)


def _local_mean(lum: np.ndarray, atlas_mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """The brightness around each pixel and how much the picture varies. On a baked atlas (``atlas_mask`` = the pixels
    that hold something) the neighbourhood is small and never mixes in the empty space between islands."""
    side = max(lum.shape)
    if atlas_mask is None:
        return ndimage.gaussian_filter(lum, max(1.0, side / 10.0), mode="wrap"), float(lum.std())
    m = atlas_mask.astype(np.float32)
    radius = max(2.0, side / 220.0)
    weight = ndimage.gaussian_filter(m, radius, mode="nearest")
    local = ndimage.gaussian_filter(lum * m, radius, mode="nearest") / np.maximum(weight, 1e-3)
    return local, float(lum[atlas_mask].std()) if atlas_mask.any() else 0.0


def _atlas_mask(albedo: np.ndarray, atlas: bool) -> np.ndarray | None:
    if not atlas:
        return None
    return albedo[..., 3] >= 128 if albedo.shape[2] == 4 else np.ones(albedo.shape[:2], bool)


def lines_map(albedo: np.ndarray, strength: float = 1.0, atlas: bool = False) -> np.ndarray:
    """0 where the surface stays, down to -1 where the texture draws a dark line."""
    lum = luminance(albedo)
    mask = _atlas_mask(albedo, atlas)
    local, spread = _local_mean(lum, mask)
    if spread < 0.012:                                    # a plain colour: nothing drawn
        return np.zeros(lum.shape, np.float32)
    dark = np.clip((local - lum) / (1.6 * spread), 0.0, 1.0)
    dark = np.where(dark < 0.25, 0.0, dark)               # texture noise is not a line
    if mask is not None:
        dark = np.where(mask, dark, 0.0)
    return (-np.clip(dark * strength, 0.0, 1.0)).astype(np.float32)


def relief_map(albedo: np.ndarray, atlas: bool = False) -> np.ndarray:
    lum = luminance(albedo)
    side = max(lum.shape)
    mask = _atlas_mask(albedo, atlas)
    if float(lum.std()) < 0.012:
        return np.zeros(lum.shape, np.float32)
    if mask is None:
        broad = ndimage.gaussian_filter(lum, max(1.0, side / 6.0), mode="wrap")
        soft = ndimage.gaussian_filter(lum, max(0.6, side / 48.0), mode="wrap") - broad
    else:
        broad, _ = _local_mean(lum, mask)
        soft = np.where(mask, ndimage.gaussian_filter(lum, max(0.6, side / 1400.0), mode="nearest") - broad, 0.0)
    return np.clip(0.45 * _norm(soft, 99.0) + lines_map(albedo, atlas=atlas), -1.0, 1.0).astype(np.float32)


def normal_to_height(normal: np.ndarray) -> np.ndarray:
    """Height whose slopes match a tangent-space normal map (Frankot-Chellappa, periodic). The slow undulation an atlas
    picks up from unrelated neighbours is removed: only detail smaller than a tenth of the picture is kept."""
    n = normal.astype(np.float32) / 127.5 - 1.0
    nz = np.clip(n[..., 2], 0.08, None)
    gx = -n[..., 0] / nz                                  # dh/dx
    gy = n[..., 1] / nz                                   # image rows grow downwards, green points up
    h, w = gx.shape
    fx = np.fft.fftfreq(w).reshape(1, w) * 2 * np.pi
    fy = np.fft.fftfreq(h).reshape(h, 1) * 2 * np.pi
    denom = fx ** 2 + fy ** 2
    denom[0, 0] = 1.0
    z = (-1j * fx * np.fft.fft2(gx) - 1j * fy * np.fft.fft2(gy)) / denom
    z[0, 0] = 0.0
    height = np.real(np.fft.ifft2(z)).astype(np.float32)
    height -= ndimage.gaussian_filter(height, max(h, w) / 10.0, mode="wrap")
    return height


def has_normal_detail(normal: np.ndarray | None) -> bool:
    if normal is None:
        return False
    xy = normal[..., :2].astype(np.float32) / 127.5 - 1.0
    return float(np.abs(xy).mean()) > 0.01 and float(xy.std()) > 0.01


def height_map(albedo: np.ndarray | None, normal: np.ndarray | None, recipe: str = "auto", *, invert: bool = False,
               smooth: float = 0.0, atlas: bool = False) -> np.ndarray:
    """The height map for one material (-1..1, float32). ``smooth`` blurs it (fraction of a texture pixel block)."""
    if recipe == "auto":
        recipe = "normal" if has_normal_detail(normal) else "relief"
    if recipe == "normal" and has_normal_detail(normal):
        side = max(normal.shape[:2])
        src = normal
        if side > MAX_SIDE:
            f = MAX_SIDE / side
            src = np.clip(ndimage.zoom(normal.astype(np.float32), (f, f, 1), order=1, mode="grid-wrap"), 0, 255)
        h = _norm(normal_to_height(src), 99.7)
    elif recipe in ("lines", "relief", "normal") and albedo is not None:
        h = lines_map(albedo, atlas=atlas) if recipe == "lines" else relief_map(albedo, atlas=atlas)
        if atlas:                                         # an atlas is sampled as it is: resizing would blur its islands together
            h = ndimage.gaussian_filter(h, 0.6, mode="nearest").astype(np.float32)
        else:
            h = _enlarge(h)
    else:
        return np.zeros((4, 4), np.float32)
    if smooth > 0:
        h = ndimage.gaussian_filter(h, smooth * max(1.0, max(h.shape) / 64.0), mode="wrap")
    return (-h if invert else h).astype(np.float32)


def texel_levels(albedo: np.ndarray | None, levels: int = 4, max_px: int = 64) -> np.ndarray | None:
    """A pixel-art texture as a few height steps, one value per texel: 0 is the level most of the texture sits at,
    positive is raised (lighter), negative is sunk (darker). None when the texture is one flat tone."""
    if albedo is None or min(albedo.shape[:2]) < 2:
        return None
    lum = luminance(albedo)
    while max(lum.shape) > max_px and min(lum.shape) >= 4:
        h, w = lum.shape[0] // 2 * 2, lum.shape[1] // 2 * 2
        lum = lum[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    lo, hi = np.percentile(lum, [2, 98])
    if hi - lo < 0.06:
        return None
    idx = np.clip(((lum - lo) / (hi - lo) * levels).astype(np.int64), 0, levels - 1)
    base = int(np.bincount(idx.ravel(), minlength=levels).argmax())
    return (idx - base).astype(np.int8)


def sample(height: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear, tiling. ``uv`` (..., 2) with the origin bottom-left."""
    h, w = height.shape
    x = uv[..., 0].astype(np.float64) * w - 0.5
    y = (1.0 - uv[..., 1].astype(np.float64)) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    return (height[y0, x0] * (1 - tx) * (1 - ty) + height[y0, x1] * tx * (1 - ty)
            + height[y1, x0] * (1 - tx) * ty + height[y1, x1] * tx * ty)


class RangePyramid:
    """How much a height map varies inside any square of it, for deciding where a triangle needs more triangles."""

    def __init__(self, height: np.ndarray):
        self.shape = height.shape
        lo = hi = height
        self.levels = [(lo, hi)]
        while min(lo.shape) > 1:
            h2, w2 = (lo.shape[0] + 1) // 2 * 2, (lo.shape[1] + 1) // 2 * 2
            lo = np.pad(lo, ((0, h2 - lo.shape[0]), (0, w2 - lo.shape[1])), mode="wrap")
            hi = np.pad(hi, ((0, h2 - hi.shape[0]), (0, w2 - hi.shape[1])), mode="wrap")
            lo = lo.reshape(h2 // 2, 2, w2 // 2, 2).min(axis=(1, 3))
            hi = hi.reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))
            self.levels.append((lo, hi))
        self.total = float(height.max() - height.min()) if height.size else 0.0

    def variation(self, uv: np.ndarray) -> np.ndarray:
        """Upper bound of max - min of the height inside each triangle's UV bounding box. ``uv`` is (m, 3, 2)."""
        h, w = self.shape
        lo = uv.min(axis=1).astype(np.float64)
        hi = uv.max(axis=1).astype(np.float64)
        span = np.maximum((hi[:, 0] - lo[:, 0]) * w, (hi[:, 1] - lo[:, 1]) * h)
        level = np.clip(np.ceil(np.log2(np.maximum(span, 1e-9))).astype(np.int64) + 1, 0, len(self.levels) - 1)
        out = np.zeros(len(uv), np.float32)
        for lv in np.unique(level):
            sel = np.nonzero(level == lv)[0]
            lmin, lmax = self.levels[lv]
            lh, lw = lmin.shape
            cell = float(2 ** lv)
            vmin = np.full(len(sel), np.inf, np.float32)
            vmax = np.full(len(sel), -np.inf, np.float32)
            for cx in (lo[sel, 0], hi[sel, 0]):
                for cy in (lo[sel, 1], hi[sel, 1]):
                    px = np.floor(cx * w / cell).astype(np.int64)
                    py = np.floor((1.0 - cy) * h / cell).astype(np.int64)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            ix, iy = (px + dx) % lw, (py + dy) % lh
                            vmin = np.minimum(vmin, lmin[iy, ix])
                            vmax = np.maximum(vmax, lmax[iy, ix])
            out[sel] = vmax - vmin
        return out


def step_pattern(albedo: np.ndarray) -> dict[str, tuple[int, float]]:
    """How many step edges a texture draws across each UV axis, and where the first one is (fraction of the tile).
    ``{"u": (count, offset), "v": (count, offset)}``; a count of 0 means no lines across that axis."""
    dark = -lines_map(albedo)
    out: dict[str, tuple[int, float]] = {}
    for name, axis in (("u", 0), ("v", 1)):
        # lines across u are columns (profile along x); lines across v are rows (profile along y, read upwards)
        profile = dark.mean(axis=axis)
        if name == "v":
            profile = profile[::-1]
        n = len(profile)
        top = float(profile.max()) if n else 0.0
        if top < 0.3 or float(profile.mean()) > 0.6 * top:
            out[name] = (0, 0.0)
            continue
        on = profile > 0.5 * top
        starts = [i for i in range(n) if on[i] and not on[i - 1]]
        if not starts or len(starts) > n // 2:
            out[name] = (0, 0.0)
            continue
        if len(starts) > 1:                               # evenly spaced lines only: bricks and noise are not steps
            gaps = np.diff(starts + [starts[0] + n])
            if gaps.max() > 1.6 * gaps.min() + 1:
                out[name] = (0, 0.0)
                continue
        ends = [next(j for j in range(i, i + n) if not on[j % n]) for i in starts]
        out[name] = (len(starts), ((starts[0] + ends[0]) / 2.0) / n)
    return out
