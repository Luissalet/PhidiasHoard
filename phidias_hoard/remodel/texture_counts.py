"""Estimate architectural repetitions from a tiled texture and its mapping to a piece."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.signal import find_peaks


def texture_periods(albedo: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (coarse, fine) repeats per UV tile, in U and V. Zero means no clear repetition."""
    image = Image.fromarray(albedo)
    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    light = pixels[..., :3].mean(axis=2)
    if pixels.shape[2] == 4:
        light *= pixels[..., 3] / 255
    return _periods(light.T), _periods(light)


def _periods(lines: np.ndarray) -> tuple[int, int]:
    if len(lines) < 12:
        return 0, 0
    signal = lines - lines.mean(axis=0)
    if float(signal.std()) < 5:
        return 0, 0
    power = (np.abs(np.fft.rfft(signal, axis=0)) ** 2).mean(axis=1)
    peaks = find_peaks(power)[0]
    peaks = peaks[(peaks >= 2) & (peaks <= min(48, len(lines) // 4))]
    if not len(peaks):
        return 0, 0
    primary = int(peaks[np.argmax(power[peaks])])
    if power[primary] < 0.24 * power[1:].sum():
        return 0, 0
    coarse = primary
    for n in peaks:
        if n >= primary:
            break
        if power[n] >= 0.34 * power[primary] and primary % n == 0:
            coarse = int(n)
            break
    finer = [int(n) for n in peaks if 2 * coarse <= n <= 12 * coarse and n % coarse == 0
             and power[n] >= 0.34 * power[coarse]]
    fine = max(finer, key=lambda n: power[n]) if finer else 0
    return coarse, fine


def estimate(tri, uv, material_ids, normals, area, size, periods, *, roof: bool) -> dict[str, int]:
    """Measure repeats along each façade or along the horizontal width of each roof slope."""
    vertical = np.abs(normals[:, 2])
    eligible = ((vertical > 0.2) & (vertical < 0.98)) if roof else (vertical < 0.35)
    mostly_x = np.abs(normals[:, 0]) >= np.abs(normals[:, 1])
    out = {}
    roof_counts, bar_counts = [], []
    for axis, key in enumerate(("bays_x", "bays_y")):
        side = eligible & (mostly_x if axis == 1 else ~mostly_x)
        if not side.any():
            continue
        mat = max(np.unique(material_ids[side]), key=lambda i: float(area[side & (material_ids == i)].sum()))
        counts = periods(int(mat))
        if counts is None:
            continue
        side &= material_ids == mat
        a, b = (tri[side, 1] - tri[side, 0]), (tri[side, 2] - tri[side, 0])
        det = a[:, axis] * b[:, 2] - b[:, axis] * a[:, 2]
        usable = np.abs(det) > 1e-9
        if not usable.any():
            continue
        uva, uvb = (uv[side, 1] - uv[side, 0]), (uv[side, 2] - uv[side, 0])
        deriv = (b[usable, 2:3] * uva[usable] - a[usable, 2:3] * uvb[usable]) / det[usable, None]
        density = np.median(np.abs(deriv), axis=0)
        uv_axis = int(np.argmax(density))
        if density[uv_axis] < 1e-8 or density[uv_axis] < 1.35 * density[1 - uv_axis]:
            continue
        coarse, fine = counts[uv_axis]
        repeats = (fine or coarse) if roof else coarse
        amount = round(float(density[uv_axis] * size[axis] * repeats))
        if roof and 4 <= amount <= 60:
            roof_counts.append(amount)
        if not roof and 2 <= amount <= 12 and size[axis] / amount >= 3:
            out[key] = amount
            if fine:
                bar_counts.append(fine // coarse)
    if roof_counts:
        out["tiles"] = round(float(np.median(roof_counts)))
    if bar_counts:
        out["bars"] = round(float(np.median(bar_counts)))
    return out
