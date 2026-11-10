"""Pictures of a mesh without a GPU: the surface is sampled into a depth image and shaded from the depth itself, so
what you see is the geometry that will be printed (carved lines, steps, cracks), not a texture."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

VIEWS = {"front": (0, 12), "back": (180, 12), "left": (90, 12), "right": (-90, 12), "top": (0, 89), "iso": (35, 30),
         "iso_back": (215, 30), "low": (25, 8)}


def _camera(yaw: float, pitch: float) -> np.ndarray:
    y, p = np.radians(yaw), np.radians(pitch)
    forward = -np.array([np.sin(y) * np.cos(p), -np.cos(y) * np.cos(p), np.sin(p)])   # camera looks at the model
    right = np.cross(forward, [0, 0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward])


def read_polygons(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An OBJ written with quads: points, fan triangles for shading, and the polygons' own edges (the topology)."""
    pts, tris, edges = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                a = line.split()
                pts.append((float(a[1]), float(a[2]), float(a[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
                for k in range(len(idx)):
                    a, b = idx[k], idx[(k + 1) % len(idx)]
                    edges.append((a, b) if a < b else (b, a))
    E = np.unique(np.asarray(edges, np.int64).reshape(-1, 2), axis=0) if edges else np.zeros((0, 2), np.int64)
    return np.asarray(pts, np.float64).reshape(-1, 3), np.asarray(tris, np.int32).reshape(-1, 3), E


def render(P: np.ndarray, F: np.ndarray, *, yaw: float = 35, pitch: float = 30, size: int = 900, region=None,
           seed: int = 3, edges: Optional[np.ndarray] = None) -> Image.Image:
    """``region`` = (cx, cy, cz, radius) in model millimetres zooms on a detail."""
    R = _camera(yaw, pitch)
    if region is not None:
        c, rad = np.asarray(region[:3], float), float(region[3])
        near = np.linalg.norm(P[F].mean(axis=1) - c, axis=1) < rad * 1.6
        F = F[near] if near.any() else F
    Q = P @ R.T
    a, b, c3 = Q[F[:, 0]], Q[F[:, 1]], Q[F[:, 2]]
    if region is not None:
        cq = np.asarray(region[:3], float) @ R.T
        lo = np.array([cq[0] - region[3], cq[1] - region[3]]); span = 2 * float(region[3])
    else:
        used = Q[np.unique(F)]
        low, high = used[:, :2].min(0), used[:, :2].max(0)
        span = float((high - low).max()) * 1.04
        lo = 0.5 * (low + high) - 0.5 * span               # centred in the frame
    px = span / size
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c3[:, 1] - a[:, 1]) - (c3[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    count = np.clip(np.ceil(area / (px * px) * 5.0).astype(np.int64), 1, 200_000)
    total = int(count.sum())
    limit = 60_000_000
    if total > limit:
        count = np.maximum((count * (limit / total)).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(F)), count)
    rng = np.random.default_rng(seed)
    r1, r2 = np.sqrt(rng.random(len(owner), dtype=np.float32)), rng.random(len(owner), dtype=np.float32)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    x = a[owner, 0] * w0 + b[owner, 0] * w1 + c3[owner, 0] * w2
    y = a[owner, 1] * w0 + b[owner, 1] * w1 + c3[owner, 1] * w2
    z = a[owner, 2] * w0 + b[owner, 2] * w1 + c3[owner, 2] * w2
    ix = np.floor((x - lo[0]) / px).astype(np.int64)
    iy = size - 1 - np.floor((y - lo[1]) / px).astype(np.int64)
    ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
    depth = np.full(size * size, -np.inf, np.float32)
    np.maximum.at(depth, iy[ok] * size + ix[ok], z[ok].astype(np.float32))      # nearest to the camera = largest z
    depth = depth.reshape(size, size)
    hit = np.isfinite(depth)
    if not hit.any():
        return Image.new("RGB", (size, size), (24, 22, 28))
    raw = np.where(hit, depth, -1e30)
    closed = ndimage.minimum_filter(ndimage.maximum_filter(raw, 3), 3)     # a pixel no sample reached takes its neighbours
    filled = np.maximum(raw, closed)
    hit2 = filled > -1e29
    d = np.where(hit2, filled, filled[hit2].min())
    gy, gx = np.gradient(d / px)
    normal = np.stack([-gx, gy, np.ones_like(d)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.45, 0.55, 0.7]); light /= np.linalg.norm(light)
    fill = np.array([0.6, -0.2, 0.5]); fill /= np.linalg.norm(fill)
    shade = 0.18 + 0.72 * np.clip(normal @ light, 0, 1) + 0.18 * np.clip(normal @ fill, 0, 1)
    cavity = d - ndimage.gaussian_filter(d, 2.5)
    shade *= np.clip(1.0 + cavity / (px * 6.0), 0.55, 1.15)
    base = np.array([222, 205, 180], np.float32)
    img = np.clip(shade[..., None] * base, 0, 255)
    img[~hit2] = (24, 22, 28)
    if edges is not None and len(edges):                   # the mesh's own edges, drawn where they are in front
        a2, b2 = Q[edges[:, 0]], Q[edges[:, 1]]
        length = np.linalg.norm((b2 - a2)[:, :2], axis=1) / px
        count = np.clip(np.ceil(length * 1.5).astype(np.int64), 2, 4000)
        owner = np.repeat(np.arange(len(edges)), count)
        start = np.cumsum(count) - count
        f = (np.arange(len(owner)) - start[owner]) / np.maximum(count[owner] - 1, 1)
        pts = a2[owner] * (1 - f[:, None]) + b2[owner] * f[:, None]
        ex = np.floor((pts[:, 0] - lo[0]) / px).astype(np.int64)
        ey = size - 1 - np.floor((pts[:, 1] - lo[1]) / px).astype(np.int64)
        ok = (ex >= 0) & (ex < size) & (ey >= 0) & (ey < size)
        ex, ey, ez = ex[ok], ey[ok], pts[ok, 2]
        front = ez >= d[ey, ex] - px * 2.5
        img[ey[front], ex[front]] = img[ey[front], ex[front]] * 0.25
    return Image.fromarray(img.astype(np.uint8))


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def side_by_side(images: list[Image.Image], labels: list[str] | None = None) -> Image.Image:
    from PIL import ImageDraw
    h = max(i.height for i in images)
    sheet = Image.new("RGB", (sum(i.width for i in images) + 6 * (len(images) - 1), h), (24, 22, 28))
    x = 0
    for k, im in enumerate(images):
        sheet.paste(im, (x, 0))
        if labels:
            ImageDraw.Draw(sheet).text((x + 8, 6), labels[k], fill=(240, 230, 210))
        x += im.width + 6
    return sheet
