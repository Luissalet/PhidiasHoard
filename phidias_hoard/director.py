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
- enrich: optional sculpted detail the texture does NOT hold, laid by real size: {"pattern": one of rock, cracks, cobble, bricks, planks, tiles, bark, ripples, flutes, carved_wood, scales, moss, "size_mm": size of one feature on the print (1-6; flutes follow the shaft, tiles follow the slope), "depth_mm": 0.1-0.6}. Use it where the texture is plain or too poor for what the surface is (bare rock, cliffs, tree trunks, sand, old stone that should be cracked), and to add wear to stone. Do not put it on surfaces whose texture already draws that same thing well, on water, or on anything that should stay smooth. null for none.
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


REVIEW_ANGLES = ((35, 30, "three-quarter view"), (215, 22, "from behind, lower"))
CLOSEUP_INSTRUCTIONS = (
    'If you need a closer look before deciding, return {"closeups": ["<exact material name>", '
    '{"center": [x, y, z], "radius": 0.1}]}. Coordinates are fractions of the current mesh bounds (0 to 1), '
    'with Z up; radius is a fraction of the longest dimension. Request at most three close-ups.'
)


def _review_views(before, after, previous):
    BP, BF = before
    P, F = after
    images, current = [], []
    for i, (yaw, pitch, label) in enumerate(REVIEW_ANGLES):
        base = preview.render(BP, BF, yaw=yaw, pitch=pitch, size=500)
        now = preview.render(P, F, yaw=yaw, pitch=pitch, size=500)
        panels = [base, previous[i], now] if previous else [base, now]
        labels = ["game model", "previous trial", "current trial"] if previous else ["game model", "current trial"]
        images.append(_png(_label(preview.side_by_side(panels, labels), label)))
        current.append(now)
    return images, current


def _material_spot(trial, info: dict[str, Any], size_mm: float):
    sel = trial.M == info["index"]
    if not sel.any():
        return None
    pts = trial.P[trial.F[sel]].reshape(-1, 3)
    centre = np.median(pts, axis=0)
    radius = float(np.clip(np.percentile(np.linalg.norm(pts - centre, axis=1), 60), size_mm * 0.035, size_mm * 0.25))
    return centre, radius


def _requested_spots(request, trial, materials, size_mm):
    if not isinstance(request, list):
        return []
    lo = trial.P.min(0)
    span = trial.P.max(0) - lo
    spots = []
    for item in request[:3]:
        if isinstance(item, str):
            if item in materials and (spot := _material_spot(trial, materials[item], size_mm)) is not None:
                spots.append((item, *spot))
            continue
        if not isinstance(item, dict):
            continue
        try:
            centre = np.asarray(item.get("center"), dtype=float)
            radius = float(item.get("radius", 0.1))
        except (TypeError, ValueError, OverflowError):
            continue
        if centre.shape == (3,) and np.isfinite(centre).all() and np.all((0 <= centre) & (centre <= 1)) and np.isfinite(radius):
            spots.append(("requested region", lo + centre * span, float(np.clip(radius, 0.03, 0.35)) * size_mm))
    return spots


def _closeup_sheet(trial, spots) -> bytes:
    images = [_label(preview.render(trial.P, trial.F, yaw=30, pitch=28, size=380, region=[*centre, radius]), name)
              for name, centre, radius in spots]
    return _png(preview.side_by_side(images))


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
    say("aplicando las recetas", 0.08)
    changes = {n: clean_change(c, project["materials"][n]) for n, c in (answer.get("materials") or {}).items() if n in project["materials"]}
    _apply(svc, pid, changes)
    log.append({"step": "materiales", "at": time.time(), "scene": str(answer.get("scene", ""))[:300],
                "materials": {n: c for n, c in changes.items() if c}})

    # ---- 2. sculpt, look, and sculpt again while the model asks for it
    verdict, notes = "ok", ""
    previous_views = []
    for attempt in range(max(1, rounds)):
        frac = 0.1 + 0.7 * attempt / max(1, rounds)
        say(f"esculpiendo la prueba {attempt + 1}", frac)
        project = svc.store.get(pid)
        st = sculpting.SculptSettings.from_dict({**project["settings"], "max_faces": min(int(project["settings"]["max_faces"]), int(svc.settings["director_faces"]))})
        rec = {n: sculpting.MaterialRecipe.from_dict(m) for n, m in project["materials"].items()}
        inferred = svc._learned_maps(pid, scene, rec, lambda stage, _progress: say(stage, frac))
        trial = sculpting.sculpt(scene, rec, st, lambda stage, progress: say(
            f"prueba {attempt + 1}: {stage}", frac + progress * 0.22 / max(1, rounds)), inferred=inferred)
        say(f"el modelo mira la prueba {attempt + 1}", frac + 0.25 / max(1, rounds))
        materials = project["materials"]
        spots = []
        for n in names[:8]:
            if materials[n]["recipe"] != "drop" and (spot := _material_spot(trial, materials[n], st.size_mm)) is not None:
                spots.append((n, *spot))
                if len(spots) == 4:
                    break
        scale0 = st.size_mm / max(float(scene.extents.max()), 1e-9)
        before = ((scene.P - scene.P.min(0)) * scale0, scene.F)
        images, current_views = _review_views(before, (trial.P, trial.F), previous_views)
        if spots:
            images.append(_closeup_sheet(trial, spots))
        compare = "Compare the previous and current trials in the matching views. " if previous_views else ""
        prompt = (f'Scene: "{project["name"]}". {scale}{wish_text}\nThis is test sculpt {attempt + 1}. Current materials:\n{_table(project, names)}\n\n'
                  f'{compare}Look at the game model, the current sculpt and the close-ups. Judge if the relief is readable and deep enough '
                  'for a printed diorama. Look for noisy or damaged surfaces, missing relief and detail where it should be smooth. '
                  'If the sculpt is ready, accept it. Otherwise change only what needs changing. '
                  'Answer: {"verdict": "ok" | "redo", "notes": "what you see and why", "materials": {"<name>": {fields to change}}}')
        answer = llm.ask(SYSTEM, f"{prompt}\n{CLOSEUP_INSTRUCTIONS}", images)
        requested = _requested_spots(answer.get("closeups"), trial, materials, st.size_mm)
        if requested:
            say(f"el modelo mira primeros planos de la prueba {attempt + 1}", frac + 0.35 / max(1, rounds))
            answer = llm.ask(SYSTEM, f"{prompt}\nThe last image contains the requested close-ups. Give your verdict and changes now.",
                             [*images, _closeup_sheet(trial, requested)])
        previous_views = current_views
        say(f"revisando la prueba {attempt + 1}", frac + 0.42 / max(1, rounds))
        verdict = "redo" if str(answer.get("verdict", "")).lower() == "redo" else "ok"
        notes = str(answer.get("notes", ""))[:800]
        changes = {n: clean_change(c, project["materials"][n]) for n, c in (answer.get("materials") or {}).items() if n in project["materials"]}
        changes = {n: c for n, c in changes.items() if c}
        log.append({"step": f"prueba {attempt + 1}", "at": time.time(), "verdict": verdict, "notes": notes, "materials": changes,
                    "triangles": trial.info["triangles"]})
        if requested:
            log[-1]["closeups"] = [{"label": name, "center_mm": centre.tolist(), "radius_mm": radius}
                                   for name, centre, radius in requested]
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


# ------------------------------------------------------------------------------------------------------------ remodel
REMODEL_SYSTEM = """You direct the remodelling of low-poly video-game scenery. The game model has been taken apart into pieces; pieces of the same shape are one type. Each type is rebuilt in Blender in clean quads by one builder, using the original only for sizes and positions. You decide which builder and its parameters, from the picture of the original piece and of the textures the game puts on it (the textures tell you what the piece really is: a tiled roof, a wall with panels or lattice windows, a door, a fluted column).
Builders (a type can only use the ones listed as allowed for it):
- roof: hipped roof with real half-round tiles running down each side, hip ridges, concave slope, upturned corners. params: tiles (ribs per side: count them in the texture times how often it repeats; 8-30), rows (segments down the slope, 4-8), concave (1 straight, 1.45 oriental curve, up to 2), corner_lift (0 flat eaves, 0.35 gentle, 0.7 strong pagoda upturn).
- body: one storey of a building: posts at the corners and between bays, beams, and in every bay a recessed panel. params: bays_x (bays across the front), bays_y (across the side), zones (list, one per horizontal band from bottom to top: "panel" recessed panel, "lattice" window with vertical bars, "plain" bare wall), bars (bars per lattice), door_bay (index from 0 of the front bay that is a door, or null), post_section ("square" or "round").
- tiled: a roof that is more than four slopes (it has a platform, a hatch, a dormer): the piece itself, with the same half-round tiles laid down its slopes. params: tiles (ribs across its widest side).
- block: a slab, plinth, step or plain box that follows the piece's own outline, with bevelled edges. params: bevel (mm).
- sweep: a beam along its own line, straight or curved (a hip ridge, a rail, a curved eave beam). params: smooth (sections added between its own, 0-8; giving it rebuilds the beam plain, without what its texture draws).
- lathe: a turned piece (spire, finial, round column, urn) revolved from the piece's own profile. params: segments (16-48), flutes (0, or the number of vertical channels of a fluted column shaft: 8-24), sides (0 round; 3-12 a prism with that many flat sides, like a hexagonal ring or an octagonal post). A ring keeps its hole.
- stairs: real steps inside the piece's box. params: steps, axis ("x" or "y": the direction they climb along), flip (true if they climb the other way).
- organic: a figure the game made of flat facets (a statue, an animal, a carved ornament), rounded into a soft surface of small quads that carries what its texture draws (feathers, eyes, folds). params: level (subdivisions, 1-4). Not for crystals, rock or anything meant to have hard edges.
- rock: rock or terrain rebuilt as subdivided quads, retaining every original facet and hard ridge without rounding or capping terrain borders. params: level (1-4), emboss (0 to disable painted relief).
- keep: the original piece as it is, cleaned into quads.
For organic, rock and keep, thickness_mm (0-5, default 0) adds a centred shell and quad rims to open pieces. Only when the owner asks for thickness. It does not fuse parts or guarantee no intersections.
What the textures draw (panels, frames, bands, joints) is modelled into the faces of keep, tiled, sweep and plain block pieces in steps; "emboss": 0 in params turns that off for a type whose texture is only shading or noise.
- drop: remove (shadow planes, light effects, sky).
When the original roof has its own curved eaves they are measured and used: give concave or corner_lift only to override them.
Write "what" and "notes" in Spanish (Spain). Answer with one JSON object and nothing else."""


def _types_table(types: list[dict[str, Any]]) -> str:
    lines = []
    for t in types:
        mats = ", ".join(f'{n} (repeats {m["repeat_u"]} x {m["repeat_v"]})' for n, m in list(t["materials"].items())[:3])
        lines.append(f'- type {t["type"]}: {t["count"]} piece(s), {" x ".join(f"{v:.1f}" for v in t["size_mm"])} mm, {t["triangles"]} triangles, '
                     f'{t["levels"]} levels; textures: {mats}; allowed: {", ".join(t["allowed"])}; now: {t["kind"]} {t["params"]}')
    return "\n".join(lines)


def _apply_types(svc, pid: str, raw: Any) -> dict[str, dict[str, Any]]:
    applied: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return applied
    for key, change in raw.items():
        key = str(key).replace("type", "").strip()
        if not isinstance(change, dict):
            continue
        clean = {k: change[k] for k in ("kind", "params", "what") if k in change}
        try:
            svc.remodel_set(pid, {key: clean})
            applied[key] = clean
        except Exception:                                  # a builder the piece cannot take, or an unknown type: skipped
            continue
    return applied


def run_remodel(svc, pid: str, rounds: int, wish: str, say: Callable[[str, float], None]) -> dict[str, Any]:
    project = svc.store.get(pid)
    llm = svc.llm
    plan = svc.remodel_plan(pid)
    choosable = [t for t in plan["types"] if len(t["allowed"]) > 3 or t["kind"] not in ("keep",)]
    choosable = sorted(choosable, key=lambda t: -t["triangles"] * t["count"])[:30]
    wish_text = f"\nThe owner asks: {wish.strip()}" if wish.strip() else ""
    log: list[dict[str, Any]] = []
    if choosable:
        say("el modelo mira las piezas", 0.05)
        answer = llm.ask(REMODEL_SYSTEM, f'Scene: "{project["name"]}", {project["settings"]["size_mm"]:.0f} mm on its longest side.{wish_text}\n'
                         f'Piece types:\n{_types_table(choosable)}\n\nDecide every type. Answer: {{"scene": "what this place is, one line", '
                         '"types": {"<type number>": {"what": "what the piece is", "kind": ..., "params": {...}}}}', svc.piece_sheets(pid, choosable))
        applied = _apply_types(svc, pid, answer.get("types"))
        log.append({"step": "piezas", "at": time.time(), "scene": str(answer.get("scene", ""))[:300], "types": applied})
    verdict, notes, result = "ok", "", None
    for attempt in range(max(1, rounds)):
        result = svc._do_remodel(pid, lambda stage, frac: say(f"construcción {attempt + 1}: {stage}", 0.15 + 0.8 * (attempt + frac * 0.6) / max(1, rounds)))
        if attempt == rounds - 1 or not choosable:
            break
        say(f"el modelo mira la construcción {attempt + 1}", 0.15 + 0.8 * (attempt + 0.8) / max(1, rounds))
        BP, BF = svc.mesh(pid, "before")
        P, F = svc.mesh(pid, "remodel")
        E = np.load(svc.store.dir(pid) / "remodel_E.npy")
        views = []
        for yaw, pitch in ((35, 25), (215, 15)):
            pair = preview.side_by_side([preview.render(BP, BF, yaw=yaw, pitch=pitch, size=560), preview.render(P, F, yaw=yaw, pitch=pitch, size=560, edges=E)],
                                        ["game model", "remodel (with its edges)"])
            views.append(_png(pair))
        centre, radius = P.mean(0), float(np.linalg.norm(P.max(0) - P.min(0))) / 7
        views.append(_png(_label(preview.render(P, F, yaw=30, pitch=20, size=640, region=[*centre, radius], edges=E), "close-up of the remodel")))
        plan = svc.remodel_plan(pid)
        current = [t for t in plan["types"] if any(t["type"] == c["type"] for c in choosable)]
        answer = llm.ask(REMODEL_SYSTEM, f'Scene: "{project["name"]}".{wish_text}\nThis is build {attempt + 1}. Piece types now:\n{_types_table(current)}\n\n'
                         'Compare the remodel with the game model. Is every piece the right thing, with believable proportions and enough but not too much '
                         'detail for its size (tiles, bays, bars, flutes)? Is something missing that the game model shows? Change only what needs changing. '
                         'Answer: {"verdict": "ok" | "redo", "notes": "two or three sentences", "types": {"<type number>": {fields to change}}}', views)
        verdict = "redo" if str(answer.get("verdict", "")).lower() == "redo" else "ok"
        notes = str(answer.get("notes", ""))[:800]
        applied = _apply_types(svc, pid, answer.get("types")) if verdict == "redo" else {}
        log.append({"step": f"construcción {attempt + 1}", "at": time.time(), "verdict": verdict, "notes": notes, "types": applied})
        if verdict == "ok" or not applied:
            break
    project = svc.store.get(pid)
    project["remodel_director"] = {"at": time.time(), "wish": wish.strip()[:500], "model": llm.status().get("model", ""), "log": log}
    svc.store.save(project)
    return {**(result or {}), "director": {"rounds": len(log), "verdict": verdict, "notes": notes}}
