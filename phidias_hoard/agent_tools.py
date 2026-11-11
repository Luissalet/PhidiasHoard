"""Tools for assistants. One catalogue drives ``/api/agent/*`` (the MCP bridge), the web UI (``/api/ui/call``) and
``mcp_server.py``. ``material_look`` and ``model_look`` return pictures (MCP images) so an assistant can judge the
relief it is asking for before and after sculpting."""

from __future__ import annotations

import base64
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .config import DEFAULT_SETTINGS
from .engine.preview import VIEWS
from .hoard_link.agentkit import Empty, Tool, ann
from .hoard_link.agentkit import call_tool as _call_tool
from .hoard_link.agentkit import tool_catalog as _catalog
from .hoard_link.agentkit import uncapped  # noqa: F401 - re-exported for the UI routes
from .services import PROJECT_SETTINGS, Services

AGENT_INSTRUCTIONS = """Phidias's Hoard turns low-poly game scenery (OBJ with its MTL and textures, GLB/glTF, or zips of them) into high-resolution models for 3D printing: the relief the textures only draw (cracks, joints between blocks, steps, the lines of a column, roof tiles) is sculpted into the mesh.
Two ways of working. Remodel (the main one): remodel_plan takes the model apart into pieces and says what each will be rebuilt as; remodel_set corrects it; remodel_start rebuilds everything in Blender with modelling operations (real roof tiles, posts and panels, turned spires) in clean quads, as a .blend and an OBJ; model_look with result remodel shows it with its edges. Sculpt (displacement, dense triangles): the rest of the tools.
Flow: library_scan lists the game models in the user's folders; project_create opens one (it reads the materials and guesses a recipe for each); material_look shows a material's texture, its normal map and the height map a recipe would carve; material_set changes recipes and depths; project_set changes the print size and detail; sculpt_start sculpts in the background (job_status); model_look shows the result, alone or next to the original, from any angle or zoomed on a region; close_start fuses it into one watertight solid with Blender; result_export writes the STL in millimetres.
Recipes: normal (relief from the normal map, when there is one: the best source), relief (dark lines carved plus soft bumps from light and shade), lines (only what the texture draws darker: joints, cracks, flutes), steps (a ramp whose texture draws steps becomes real steps: flat treads and risers, as many as the texture draws or the number given in steps), flat (leave the surface as it is: water, glass, plain colours), drop (remove it: shadow planes, flames, glows, sky), auto. depth_mm is how deep the relief goes at print size (0.3-1.0 reads well on a 150 mm model). invert swaps carved and raised. enrich adds sculpted detail the texture does not hold (rock, cracks, cobble, bricks, planks, tiles, bark, ripples, flutes) at a real size in millimetres, on top of any recipe. direct_start hands the whole decision to the vision model, which sculpts, looks and redoes. cutout cuts away what the texture leaves see-through (fences, railings, leaves, tree cards) and gives the rest thickness (sheet_mm).
The ground of the scene (faces looking up with nothing below) is closed into a solid block down to a base so it prints without supports under the terrain."""

RECIPE = Literal["auto", "normal", "lines", "relief", "steps", "flat", "drop"]


def _d(first: str, detail: str = "", synonyms: str = "") -> str:
    parts = [first]
    if detail:
        parts.append(detail)
    if synonyms:
        parts.append("Sinónimos: " + synonyms)
    return "\n".join(parts)


def _img(data: bytes) -> dict[str, str]:
    return {"data": base64.b64encode(data).decode(), "mime": "image/png"}


class ProjectId(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=64)


class CloseArgs(ProjectId):
    fill_interior: bool = Field(True, description="Fill sealed cavities remaining after remeshing.")


class ScanArgs(BaseModel):
    refresh: bool = Field(False, description="Walk the folders again instead of using the last listing.")
    query: str = Field("", max_length=120, description="Only entries whose name, group or path contains this text.")
    group: str = Field("", max_length=200, description="Only entries of this group (the top folder inside a library folder).")
    limit: int = Field(80, ge=1, le=400)
    offset: int = Field(0, ge=0)


class CreateArgs(BaseModel):
    source: str = Field(..., min_length=1, max_length=1000, description="Path on this PC of a .obj, .glb, .gltf or a .zip that contains them.")
    inner: str = Field("", max_length=500, description="For a zip with several models: the one to open (as library_scan lists it).")
    name: str = Field("", max_length=120)


class SetArgs(ProjectId):
    name: Optional[str] = Field(None, max_length=120)
    size_mm: Optional[float] = Field(None, ge=10, le=2000, description="Longest side of the print in millimetres.")
    min_edge_mm: Optional[float] = Field(None, ge=0.05, le=20, description="Finest triangle edge (the detail the printer shows; 0.2-0.3 for FDM, 0.1 for resin).")
    tolerance_mm: Optional[float] = Field(None, ge=0.005, le=2, description="Relief below this inside a triangle gets no more triangles.")
    max_faces: Optional[int] = Field(None, ge=10_000, le=60_000_000, description="Triangle budget; when the detail asked does not fit, the finest edge grows evenly.")
    ground: Optional[bool] = Field(None, description="Close the ground into a solid block down to the base.")
    base_mm: Optional[float] = Field(None, ge=0, le=100, description="Solid thickness under the lowest point of the ground.")
    plinth: Optional[Literal["none", "rect", "round"]] = Field(None, description="A plate under the whole scene.")
    plinth_margin_mm: Optional[float] = Field(None, ge=0, le=100)
    normal_cutoff_mm: Optional[float] = Field(None, ge=0.5, le=200, description="Relief from normal maps keeps features up to this size.")
    sheet_mm: Optional[float] = Field(None, ge=0, le=20, description="Thickness given to cut-out sheets (fences, leaves) so they print; 0 leaves a single surface.")


class MaterialChange(BaseModel):
    recipe: Optional[RECIPE] = None
    depth_mm: Optional[float] = Field(None, ge=0, le=20)
    invert: Optional[bool] = None
    smooth: Optional[float] = Field(None, ge=0, le=8)
    steps: Optional[int] = Field(None, ge=0, le=200, description="Recipe steps: step edges per texture tile; 0 counts them from the texture.")
    enrich: Optional[EnrichArgs] = Field(None, description="Sculpted detail the texture does not hold, laid by real size. Send no_enrich=true to remove it.")
    no_enrich: Optional[bool] = Field(None, description="true removes the enrich of the material.")
    cutout: Optional[bool] = Field(None, description="Cut away what the texture leaves see-through (fences, leaves, railings) and give the rest thickness.")


class EnrichArgs(BaseModel):
    pattern: Literal["rock", "cracks", "cobble", "bricks", "planks", "tiles", "bark", "ripples", "flutes"]
    size_mm: float = Field(2.0, ge=0.3, le=60, description="Size of one feature (a stone, a brick course, a plank) on the print.")
    depth_mm: float = Field(0.3, ge=0, le=10)


class DirectArgs(ProjectId):
    rounds: int = Field(0, ge=0, le=6, description="Test sculpts the model may look at and redo (0 = the setting director_rounds).")
    wish: str = Field("", max_length=500, description='What the user wants from this sculpt, in words ("more worn stone", "keep the floor smooth").')


class PieceChange(BaseModel):
    kind: Optional[Literal["roof", "tiled", "body", "block", "lathe", "sweep", "stairs", "organic", "keep", "drop"]] = None
    params: Optional[dict[str, Any]] = Field(None, description="roof: tiles (ribs per side), rows, concave (1-2), corner_lift (0-0.8), tile_radius, bevel. "
                                             "lathe also flutes (channels of a fluted shaft). body: bays_x, bays_y, zones (per band bottom to top: panel, lattice or plain), bars, door_bay (bay of the front, or null), "
                                             "post_section (square, round), post, beam, panel_depth. block: bevel. lathe: segments, sides (flat sides of a prism, 0 round), bevel. tiled: tiles. sweep: smooth. stairs: steps, axis (x, y), flip. "
                                             "keep, tiled, sweep, block: emboss (0 leaves out what the textures draw). Sizes in mm.")
    what: Optional[str] = Field(None, max_length=160)


class RemodelStartArgs(ProjectId):
    direct: bool = Field(False, description="Let the vision model decide what every type of piece is and its parameters from pictures of the pieces and their textures, look at the build next to the original and redo it.")
    rounds: int = Field(0, ge=0, le=6, description="With direct: builds the model may look at and redo (0 = the setting director_rounds).")
    wish: str = Field("", max_length=500, description="With direct: what the user wants, in words.")


class RemodelSetArgs(ProjectId):
    types: dict[str, PieceChange] = Field(..., description='Piece type (the number remodel_plan gives) -> what to build it as. Example: {"4": {"kind": "roof", "params": {"tiles": 20, "corner_lift": 0.5}}}')


class MaterialSetArgs(ProjectId):
    materials: dict[str, MaterialChange] = Field(..., description='Material name -> changes; the name "*" applies to every material. '
                                                                  'Example: {"roof_tou01": {"recipe": "lines", "depth_mm": 0.8}, "h_kage": {"recipe": "drop"}}')


class MaterialLookArgs(ProjectId):
    material: str = Field(..., min_length=1, max_length=300)
    recipe: Optional[RECIPE] = Field(None, description="Preview this recipe instead of the saved one.")


class LookArgs(ProjectId):
    result: Literal["before", "sculpt", "closed", "remodel"] = Field("sculpt", description="before = the game model as it came; remodel = the model rebuilt in quads (drawn with its edges); sculpt = the displaced mesh; closed = the watertight solid.")
    wire: bool = Field(True, description="For remodel: draw the edges of the mesh, to judge the topology.")
    view: Literal[tuple(VIEWS)] = Field("iso")  # type: ignore[valid-type]
    yaw: Optional[float] = Field(None, ge=-360, le=360, description="Degrees around the model (0 = from the front, 90 = from the left side).")
    pitch: Optional[float] = Field(None, ge=-89, le=89, description="Degrees above the horizon.")
    size: int = Field(900, ge=200, le=2000)
    region: Optional[list[float]] = Field(None, min_length=4, max_length=4, description="[x, y, z, radius] in millimetres of the print: zoom on that spot.")
    compare: bool = Field(False, description="Put the original next to the result from the same camera.")


class ExportArgs(ProjectId):
    result: Optional[Literal["remodel", "sculpt", "closed"]] = Field(None, description="Default: the remodel (.blend and .obj in quads) when there is one, else the closed solid, else the sculpt.")
    folder: str = Field("", max_length=1000, description="Where to write it; default: a Phidias folder next to the source model.")


class JobId(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=64)


class SettingsArgs(BaseModel):
    changes: dict[str, Any] = Field(..., description="Keys: " + ", ".join(DEFAULT_SETTINGS))


def _scan(svc: Services, a: ScanArgs) -> dict[str, Any]:
    lib = svc.library(a.refresh)
    entries = lib["entries"]
    groups: dict[str, int] = {}
    for e in entries:
        groups[e["group"]] = groups.get(e["group"], 0) + 1
    if a.query:
        q = a.query.lower()
        entries = [e for e in entries if q in e["name"].lower() or q in e["group"].lower() or q in (e["path"] + "|" + e["inner"]).lower()]
    if a.group:
        entries = [e for e in entries if e["group"].lower() == a.group.lower()]
    page = entries[a.offset: a.offset + a.limit]
    return {"dirs": lib["dirs"], "missing": lib["missing"], "groups": groups, "count": len(entries), "offset": a.offset, "entries": page}


def _summary(p: dict[str, Any]) -> dict[str, Any]:
    return {"id": p["id"], "name": p["name"], "source": p["source"].get("path"), "triangles": p["source"].get("triangles"),
            "materials": len(p.get("materials", {})), "size_mm": p.get("settings", {}).get("size_mm"),
            "results": {k: {"triangles": v.get("triangles_total"), "open_edges": v.get("open_edges")} for k, v in p.get("results", {}).items()}}


def _set(svc: Services, a: SetArgs) -> dict[str, Any]:
    changes = {k: getattr(a, k) for k in PROJECT_SETTINGS if getattr(a, k, None) is not None}
    return svc.update_project(a.project_id, name=a.name, settings=changes or None)


def _material_set(svc: Services, a: MaterialSetArgs) -> dict[str, Any]:
    changes = {}
    for k, v in a.materials.items():
        c = v.model_dump(exclude_none=True)
        if c.pop("no_enrich", False):
            c["enrich"] = None
        changes[k] = c
    p = svc.update_project(a.project_id, materials=changes)
    return {"id": p["id"], "materials": {k: {x: v[x] for x in ("recipe", "depth_mm", "invert", "smooth", "cutout", "steps", "enrich") if x in v} for k, v in p["materials"].items()}}


def _material_look(svc: Services, a: MaterialLookArgs) -> dict[str, Any]:
    info = svc.store.get(a.project_id)["materials"].get(a.material, {})
    return {"material": a.material, "panels": "texture | normal map | height map (white = raised, black = carved)",
            **{k: info.get(k) for k in ("recipe", "depth_mm", "texture", "normal_map", "atlas", "transparent", "share")},
            "_image": _img(svc.material_sheet(a.project_id, a.material, a.recipe))}


def _look(svc: Services, a: LookArgs) -> dict[str, Any]:
    data = svc.look(a.project_id, a.result, a.view, a.yaw, a.pitch, a.size, a.region, a.compare, a.wire)
    return {"project_id": a.project_id, "result": a.result, "view": a.view, "_image": _img(data)}


TOOLS: list[Tool] = [
    Tool("status", _d("What Phidias has: projects, library folders, whether Blender is available and the jobs running.", synonyms="estado"),
         Empty, ann(read_only=True), lambda s, a: s.status()),
    Tool("library_scan", _d("List the game models found in the library folders (and inside their zips), with the project already made from each.",
                            synonyms="biblioteca, modelos de juego, dioramas"), ScanArgs, ann(read_only=True), _scan, timeout_s=120, capped=False),
    Tool("project_list", _d("List sculpting projects, newest first.", synonyms="proyectos"), Empty, ann(read_only=True),
         lambda s, a: {"projects": [_summary(p) for p in s.store.list()]}),
    Tool("project_create", _d("Open a game model as a project: reads its geometry and materials and guesses a recipe for each material.",
                              "A zip is unpacked inside the project; other files are read where they are.", "nuevo proyecto, importar modelo"),
         CreateArgs, ann(draft_safe=True), lambda s, a: s.create_project(a.source, a.inner, a.name), timeout_s=300),
    Tool("project_get", _d("A project: source, print settings, every material with its recipe and what its texture holds, and the results."),
         ProjectId, ann(read_only=True), lambda s, a: s.project(a.project_id)),
    Tool("project_set", _d("Change a project's name, print size, detail, triangle budget, ground block or plinth.", synonyms="tamaño, detalle, peana"),
         SetArgs, ann(draft_safe=True, idempotent=True), _set),
    Tool("material_look", _d("See a material: its texture, its normal map and the height map its recipe would carve (a picture).",
                             synonyms="ver material, mapa de alturas"), MaterialLookArgs, ann(read_only=True), _material_look, timeout_s=60, capped=False),
    Tool("material_set", _d("Change the recipe, depth, inversion or smoothing of one or more materials.", synonyms="receta, profundidad, relieve"),
         MaterialSetArgs, ann(draft_safe=True, idempotent=True), _material_set),
    Tool("sculpt_start", _d("Sculpt the project in the background with its current recipes and settings. Returns a job id for job_status.",
                            "Replaces the previous sculpt of the project.", "esculpir, generar alta resolución"),
         ProjectId, ann(draft_safe=True), lambda s, a: s.start_sculpt(a.project_id)),
    Tool("remodel_plan", _d("The model taken apart into its pieces: each type of piece with how many there are, its size, its materials and what it will be rebuilt as.",
                            "Kinds: roof (hipped roof with real tiles, hip ridges and upturned eaves), body (a storey: posts, beams, recessed panels, lattices, a door), "
                            "block (bevelled slab or plinth following the piece's profile), lathe (turned piece from its profile: spires, columns), stairs, "
                            "keep (the original piece cleaned into quads), drop.", "piezas, despiece"), ProjectId, ann(read_only=True), lambda s, a: s.remodel_plan(a.project_id)),
    Tool("remodel_set", _d("Choose what a type of piece is rebuilt as and with which parameters.", synonyms="cambiar pieza, tejas, vanos"),
         RemodelSetArgs, ann(draft_safe=True, idempotent=True), lambda s, a: s.remodel_set(a.project_id, {k: v.model_dump(exclude_none=True) for k, v in a.types.items()})),
    Tool("remodel_start", _d("Rebuild the model piece by piece in Blender with modelling operations and clean quads, using the game model only as the reference "
                             "for sizes and positions. Returns a job id. Result: a .blend (one object per piece, bevels as live modifiers) and an OBJ in quads.",
                             synonyms="remodelar, topología limpia, reconstruir"), RemodelStartArgs, ann(draft_safe=True), lambda s, a: s.start_remodel(a.project_id, a.direct, a.rounds, a.wish)),
    Tool("direct_start", _d("Let the family's vision model direct the sculpt: it looks at every material, decides recipe, depth, cut-outs and enrichment, "
                            "compares test sculpts with the original and the previous trial, and can request close-ups before deciding what to change. Returns a job id.",
                            "What it decided and why is kept in the project (director). Needs the vision model reachable (status.model).", "dirigir con IA, que lo decida el modelo, rehacer"),
         DirectArgs, ann(draft_safe=True), lambda s, a: s.start_direct(a.project_id, a.rounds, a.wish)),
    Tool("close_start", _d("Fuse the sculpt into one watertight solid with Blender (thickens loose sheets, voxel remesh). Returns a job id.",
                           "Detail finer than the voxel size (setting voxel_mm) is lost; the sculpt stays available.", "cerrar malla, estanco, manifold"),
         CloseArgs, ann(draft_safe=True), lambda s, a: s.start_close(a.project_id, a.fill_interior)),
    Tool("job_status", _d("State of a job: queued, running (stage and progress), done (with result), cancelled or error."), JobId, ann(read_only=True),
         lambda s, a: s.job(a.job_id)),
    Tool("job_cancel", _d("Cancel a queued or running sculpt or directed sculpt. A running job stops at its next checkpoint and keeps the last completed sculpt."),
         JobId, ann(draft_safe=True), lambda s, a: s.cancel_job(a.job_id)),
    Tool("model_look", _d("See the model: the original, the sculpt or the closed solid, from any angle, zoomed on a region, or next to the original.",
                          "The picture is shaded from the geometry itself, so it shows what will be printed.", "ver modelo, render, comparar"),
         LookArgs, ann(read_only=True), _look, timeout_s=180, capped=False),
    Tool("result_export", _d("Write the STL (millimetres, base at z = 0) to a folder; by default a Phidias folder next to the source model.",
                             synonyms="exportar, STL, guardar"), ExportArgs, ann(draft_safe=True), lambda s, a: s.export_result(a.project_id, a.result or "", a.folder),
         timeout_s=300),
    Tool("project_delete", _d("Delete a project and its results (the source model is not touched)."), ProjectId, ann(destructive=True),
         lambda s, a: s.delete_project(a.project_id)),
    Tool("settings_get", _d("Defaults for new projects, the library folders and the Blender options."), Empty, ann(read_only=True),
         lambda s, a: s.settings.get()),
    Tool("settings_set", _d("Change settings (library_dirs is the list of folders with game models)."), SettingsArgs, ann(draft_safe=True, idempotent=True),
         lambda s, a: (s.settings.update(a.changes), setattr(s, "_library", None), s.settings.get())[2]),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_catalog() -> list[dict[str, Any]]:
    return _catalog(TOOLS)


def call_tool(svc: Services, name: str, arguments: Optional[dict[str, Any]], caller: str = "agent") -> Any:
    return _call_tool(TOOLS_BY_NAME, svc, name, arguments)
