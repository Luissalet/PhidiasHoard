"""The model as art director of a sculpt.

It looks at every material (texture and the relief each recipe would carve), says what each one is and how to sculpt
it, including detail the texture does not hold (rock, cracks, bricks...). Then the project is sculpted, the model
looks at the result next to the original from several sides and either accepts it or changes recipes and has it
sculpted again, up to a number of rounds. What it decided and why is kept with the project."""

from __future__ import annotations

import io
import time
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

from .engine import patterns, preview, recipes
from .engine import sculpt as sculpting

RECIPES = ("normal", "relief", "lines", "steps", "flat", "drop")

SYSTEM = """You direct the sculpting of low-poly video-game scenery into a detailed model for 3D printing.
The game only draws relief in its textures; the sculptor carves it into the mesh. Each material has:
- recipe: normal (relief from its normal map; only when has_normal_map), relief (dark lines carved plus soft bumps from light and shade), lines (only the dark lines: joints, cracks, flutes), steps (a ramp whose texture draws stair steps becomes real steps), flat (leave smooth: water, glass, skin of a plain colour, things that must stay clean), drop (remove: shadow planes, light shafts, flames, glows, fog, sky).
- depth_mm: depth of that relief on the print (0.2 subtle, 0.5 normal, 0.9 strong). Stone joints and roof tiles deserve more; fine ornament less.
- cutout: true cuts away what the texture leaves transparent (fences, railings, leaves, tree cards, grilles).
- enrich: optional sculpted detail the texture does NOT hold, laid by real size: {"pattern": one of rock, cracks, cobble, bricks, planks, tiles, bark, ripples, flutes, "size_mm": size of one feature on the print (1-6; flutes are the vertical channels of a column shaft, size = one channel), "depth_mm": 0.1-0.6}. Use it where the texture is plain or too poor for what the surface is (bare rock, cliffs, tree trunks, sand, old stone that should be cracked), and to add wear to stone. Do not put it on surfaces whose texture already draws that same thing well, on water, or on anything that should stay smooth. null for none.
Think about what each material IS in the place (a temple wall, a cave floor, a roof) and sculpt it as a craftsman would for a diorama that will be painted. Write the texts of "scene", "what" and "notes" in Spanish (Spain). Answer with one JSON object and nothing else."""


def _label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 18), (24, 22, 28))
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text[:120], fill=(240, 230, 210))
    return out


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def material_sheets(scene, project: dict[str, Any], names: list[str], per_sheet: int = 6) -> list[bytes]:
    """Pictures with several materials each: name on top, then texture | normal map | height map of its recipe."""
    out = []
    for at in range(0, len(names), per_sheet):
        rows = []
        for name in names[at: at + per_sheet]:
            info = project["materials"][name]
            arr = recipes.sheet(scene.materials[info["index"]], info["recipe"], info.get("invert", False), info.get("smooth", 0.0), side=200)
            rows.append(_label(Image.fromarray(arr), f"{name}   (texture | normal map | height: white raised, black carved)"))
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (24, 22, 28))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height
        out.append(_png(sheet))
    return out


def _table(project: dict[str, Any], names: list[str]) -> str:
    lines = []
    for name in names:
        m = project["materials"][name]
        e = m.get("enrich")
        lines.append(f'- "{name}": {m["share"] * 100:.1f}% of the surface, texture {m["texture"][0]}x{m["texture"][1]}, '
                     f'has_normal_map={str(bool(m["normal_map"])).lower()}, baked_atlas={str(bool(m["atlas"])).lower()}, '
                     f'transparent={m["transparent"]:.2f}, contrast={m["contrast"]:.2f}; now recipe={m["recipe"]}, depth_mm={m["depth_mm"]}, '
                     f'cutout={str(bool(m.get("cutout"))).lower()}, enrich={e if e else "null"}')
    return "\n".join(lines)


def clean_change(raw: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Only what a material can take from the model's answer."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    recipe = raw.get("recipe")
    if recipe in RECIPES:
        if recipe == "normal" and not info.get("normal_map"):
            recipe = "relief"
        out["recipe"] = recipe
    for key, lo, hi in (("depth_mm", 0.0, 2.0), ("smooth", 0.0, 8.0)):
        try:
            if raw.get(key) is not None:
                out[key] = float(np.clip(float(raw[key]), lo, hi))
        except (TypeError, ValueError):
            pass
    for key in ("cutout", "invert"):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key] and not (key == "cutout" and info.get("atlas"))
    if "steps" in raw:
        try:
            out["steps"] = int(np.clip(int(raw["steps"] or 0), 0, 200))
        except (TypeError, ValueError):
            pass
    if "enrich" in raw:
        out["enrich"] = sculpting.clean_enrich(raw["enrich"])
        if out["enrich"]:
            out["enrich"]["size_mm"] = float(np.clip(out["enrich"]["size_mm"], 0.5, 12.0))
            out["enrich"]["depth_mm"] = float(np.clip(out["enrich"]["depth_mm"], 0.05, 1.0))
    if isinstance(raw.get("what"), str):
        out["what"] = raw["what"].strip()[:120]
    return out


def result_views(before: tuple[np.ndarray, np.ndarray], after: tuple[np.ndarray, np.ndarray], spots: list[tuple[str, np.ndarray, float]]) -> list[bytes]:
    """The sculpt next to the original from two sides, plus close-ups of the biggest materials."""
    BP, BF = before
    P, F = after
    out = []
    for yaw, pitch, label in ((35, 30, "three-quarter view"), (215, 22, "from behind, lower")):
        pair = preview.side_by_side([preview.render(BP, BF, yaw=yaw, pitch=pitch, size=560), preview.render(P, F, yaw=yaw, pitch=pitch, size=560)],
                                    ["game model", "sculpt"])
        out.append(_png(_label(pair, label)))
    closeups = []
    for name, centre, radius in spots[:4]:
        img = preview.render(P, F, yaw=30, pitch=28, size=380, region=[*centre, radius])
        closeups.append(_label(img, f"close-up: {name}"))
    if closeups:
        out.append(_png(preview.side_by_side(closeups)))
    return out


def run(svc, pid: str, rounds: int, wish: str, say: Callable[[str, float], None]) -> dict[str, Any]:
    project = svc.store.get(pid)
    scene = svc.scene(pid)
    llm = svc.llm
    log: list[dict[str, Any]] = []
    names = [n for n, m in sorted(project["materials"].items(), key=lambda kv: -kv[1]["share"])][:36]
    scale = f'The print is {project["settings"]["size_mm"]:.0f} mm on its longest side; the finest detail is {project["settings"]["min_edge_mm"]} mm.'
    wish_text = f"\nThe owner asks: {wish.strip()}" if wish.strip() else ""

    # ---- 1. what each material is and how to sculpt it
    say("el modelo mira los materiales", 0.03)
    answer = llm.ask(SYSTEM, f'Scene: "{project["name"]}". {scale}{wish_text}\nMaterials (biggest first):\n{_table(project, names)}\n\n'
                     'The pictures show each material by name. Decide every material. Answer: {"scene": "what this place is, one line", '
                     '"materials": {"<name>": {"what": "what it is", "recipe": ..., "depth_mm": ..., "cutout": true|false, "enrich": {...}|null}}}',
                     material_sheets(scene, project, names))
    changes = {n: clean_change(c, project["materials"][n]) for n, c in (answer.get("materials") or {}).items() if n in project["materials"]}
    _apply(svc, pid, changes)
    log.append({"step": "materiales", "at": time.time(), "scene": str(answer.get("scene", ""))[:300],
                "materials": {n: c for n, c in changes.items() if c}})

    # ---- 2. sculpt, look, and sculpt again while the model asks for it
    verdict, notes = "ok", ""
    for attempt in range(max(1, rounds)):
        frac = 0.1 + 0.7 * attempt / max(1, rounds)
        say(f"esculpiendo la prueba {attempt + 1}", frac)
        project = svc.store.get(pid)
        st = sculpting.SculptSettings.from_dict({**project["settings"], "max_faces": min(int(project["settings"]["max_faces"]), int(svc.settings["director_faces"]))})
        rec = {n: sculpting.MaterialRecipe.from_dict(m) for n, m in project["materials"].items()}
        trial = sculpting.sculpt(scene, rec, st)
        if attempt == rounds - 1 and rounds > 1:
            break                                          # the last round's changes go straight to the final sculpt
        say(f"el modelo mira la prueba {attempt + 1}", frac + 0.25 / max(1, rounds))
        spots = []
        for n in names[:8]:
            idx = project["materials"][n]["index"]
            sel = trial.M == idx
            if project["materials"][n]["recipe"] != "drop" and sel.sum() > 30:
                pts = trial.P[trial.F[sel]].reshape(-1, 3)
                centre = np.median(pts, axis=0)
                spots.append((n, centre, float(np.clip(np.percentile(np.linalg.norm(pts - centre, axis=1), 60), 6.0, st.size_mm / 4))))
        scale0 = st.size_mm / max(float(scene.extents.max()), 1e-9)
        before = ((scene.P - scene.P.min(0)) * scale0, scene.F)
        answer = llm.ask(SYSTEM, f'Scene: "{project["name"]}". {scale}{wish_text}\nThis is test sculpt {attempt + 1}. Current materials:\n{_table(project, names)}\n\n'
                         'Look at the sculpt next to the game model and at the close-ups. Judge it as the maker of a diorama: is the relief readable '
                         'and deep enough at this size, is anything noisy, blobby or damaged, is a surface still bare that should have texture, '
                         'is something sculpted that should be smooth or removed? If it is good, accept it. If not, change only what needs changing. '
                         'Answer: {"verdict": "ok" | "redo", "notes": "what you see and why, two or three sentences", '
                         '"materials": {"<name>": {fields to change}}}', result_views(before, (trial.P, trial.F), spots))
        verdict = "redo" if str(answer.get("verdict", "")).lower() == "redo" else "ok"
        notes = str(answer.get("notes", ""))[:800]
        changes = {n: clean_change(c, project["materials"][n]) for n, c in (answer.get("materials") or {}).items() if n in project["materials"]}
        changes = {n: c for n, c in changes.items() if c}
        log.append({"step": f"prueba {attempt + 1}", "at": time.time(), "verdict": verdict, "notes": notes, "materials": changes,
                    "triangles": trial.info["triangles"]})
        if verdict == "ok" or not changes:
            break
        _apply(svc, pid, changes)
    project = svc.store.get(pid)
    project["director"] = {"at": time.time(), "wish": wish.strip()[:500], "model": llm.status().get("model", ""), "log": log}
    svc.store.save(project)
    return {"rounds": len(log) - 1, "verdict": verdict, "notes": notes, "log": log}


def _apply(svc, pid: str, changes: dict[str, dict[str, Any]]) -> None:
    project = svc.store.get(pid)
    for name, change in changes.items():
        cur = project["materials"].get(name)
        if cur is None:
            continue
        for key in ("recipe", "depth_mm", "smooth", "cutout", "invert", "steps", "what"):
            if key in change:
                cur[key] = change[key]
        if "enrich" in change:
            cur["enrich"] = change["enrich"]
        if "recipe" in change:
            cur["why"] = "decidido por el modelo"
    svc.store.save(project)
