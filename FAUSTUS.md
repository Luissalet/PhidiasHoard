# Phidias's Hoard — what it can and cannot do

App id `phidias`, service `phidias-hoard`, port 5251, MCP stdio bridge `mcp_server.py` (20 tools; `material_look` and
`model_look` return MCP images). Connector preset: its own `faustus-plugin.json`.

## Can

- Remodel (`remodel_plan`, `remodel_set`, `remodel_start`): split the model into connected pieces (position, size,
  profile = outline size at every height with corners, materials and texture repeats; equal shapes are one type),
  guess what each is, and build each in Blender (`remodel/blender_build.py`, with the Blender found or the `bpy`
  module) as quads: `roof` (fascia, concave slope, corner lift, half-round tiles shortened at the hips, hip ridges,
  quad soffit; params tiles, rows, concave, corner_lift, tile_radius, bevel), `body` (posts, beams at every level,
  one recessed panel per bay and band; zones panel/lattice/plain, bars, door_bay, post_section, post, beam,
  panel_depth, bays_x, bays_y), `block` (loft of the profile, bevel), `lathe` (profile revolved; segments), `stairs`
  (steps, axis, flip), `keep` (original welded, triangles joined to quads, bevel), `drop`. A piece whose builder
  fails falls back to `keep`. Output in `data/projects/<id>/remodel/`: `remodel.blend` (object per piece in
  collections by kind, bevel modifiers live), `remodel.obj` (modifiers applied, quads), `remodel.json` (verts,
  quads, tris, n-gons per object). `model_look` with `result: remodel` draws the edges (`wire`); `result_export`
  copies the .blend and .obj.

- List the game models in the library folders (`library_scan`: OBJ, Collada, GLB, glTF, loose or inside zips; one
  entry per scene, richest format first; `query`, `group`, `limit`, `offset`; `groups` counts per top folder).
- Open a model as a project (`project_create`): geometry welded and turned Z up, materials with their pictures, a
  recipe guessed per material (shadow planes, flames and glows removed; water and glass flat; normal map when there is
  one; steps for stair textures; relief for everything else; cut-out for textures with see-through parts).
  - OBJ: MTL found even when `mtllib` names a file that is not there; texture paths with backslashes or in subfolders;
    materials named after their picture.
  - Collada: scene transforms, `triangles`/`polylist`/`polygons`, effect → sampler → surface → image chains, Y or Z up.
  - glTF/GLB: node transforms, the UV set each texture names (baked scenes keep the atlas in `TEXCOORD_1`/`_2`), base
    colour and normal pictures embedded or beside the file.
  - Baked atlases are recognised (a texture on a secondary UV set, or a big picture mostly empty with every UV inside
    it) and never read as tiling textures.
- Show a material (`material_look`): texture, normal map and the height map a recipe would carve.
- Recipes (`material_set`; `"*"` = every material): `normal`, `relief`, `lines`, `steps`, `flat`, `drop`, `auto`;
  `depth_mm`, `invert`, `smooth`, `cutout`, `steps` (edges per tile, 0 = counted from the texture).
- Added detail (`material_set` with `enrich: {pattern, size_mm, depth_mm}`, `no_enrich` removes it): rock, cracks,
  cobble, bricks, planks, tiles, bark, ripples, flutes, laid by position in millimetres (three projections blended),
  on top of any recipe; the mesh is refined enough to draw the pattern.
- Direction by the vision model (`direct_start` with optional `rounds` and `wish`): reads the materials from pictures,
  sets recipe, depth, cut-out, steps and enrich per material, sculpts a test inside `director_faces`, looks at it
  next to the original and at close-ups, and redoes it up to `director_rounds` times before the final sculpt. Its
  answers are clamped (depths, sizes, no normal recipe without a normal map, no cut-out on atlases). The project keeps
  `director` (model, wish and the log of each step with verdict, notes and changes) and `what` per material.
  Model: the family's vision model through Hoard Link, or `llm_url`/`llm_model` in settings.
- Print settings (`project_set`): `size_mm` (longest side), `min_edge_mm`, `tolerance_mm`, `max_faces`, `ground`,
  `base_mm`, `plinth` (none, rect, round), `plinth_margin_mm`, `normal_cutoff_mm`, `sheet_mm`.
- Sculpt in the background (`sculpt_start`, `job_status`): adaptive refinement that never opens the mesh, triangle
  budget shared evenly (longest edges first), relief from height maps or from normal maps integrated on the mesh,
  stairs, cut-outs with thickness, ground block and plinth. Result: STL in millimetres, a mesh for the 3D view, a
  picture, measures, triangle counts and open-edge count.
- Fuse the sculpt into one watertight solid with Blender (`close_start`): thickness for loose sheets, voxel remesh,
  open edges counted afterwards (`watertight`).
- Look at any result (`model_look`): `before`, `sculpt` or `closed`; named views or any yaw/pitch; `region`
  `[x, y, z, radius]` in print millimetres; `compare` puts the original next to it.
- Export (`result_export`): the STL into a `Phidias` folder next to the source model, or the folder given.
- Studio UI with the same handlers: library, projects, 3D view of the three results, material list with thumbnails,
  recipe and depth per material, the picture of each recipe, job progress, download and save buttons, settings.

## Cannot

- Remodel pieces it has no builder for: irregular rock, terrain, organic shapes and anything not built on rectangular
  or round outlines stay as the cleaned original (`keep`). The model does not choose the pieces' kinds yet.
- Read a piece's ornament from its texture into the remodel (panel layouts, tile counts and bays are parameters).

- Read FBX, `.blend`, `.smd` or engine-native formats (they have to be converted to one of the four formats).
- Invent relief a texture does not hold: plain colours stay flat; a baked atlas without normal maps gives coarser
  relief than the original textures would.
- Turn foliage cards into volumes (they stay sheets with thickness) or rebuild the inside of buildings.
- Fill enclosed hollow spaces in the watertight solid, or keep detail finer than its voxel size there.
- Recognise stairs by shape: the steps recipe is chosen by material name, by the model or by hand.
- Direct a sculpt without a vision model reachable, or cancel a direction half way.
- Split a model into parts for a small printer, add supports or slice.
- Undo a sculpt: each sculpt replaces the previous result of the project.

## Where things are

- `phidias_hoard/engine/`: `meshio.py` (loaders, atlas detection, zips), `heights.py` (height maps from textures,
  step patterns, sampling, detail pyramids), `surface.py` (normal maps integrated on the mesh), `refine.py` (adaptive
  refinement), `sculpt.py` (the pipeline, ground block, sheets, plinth), `recipes.py` (guesses and the material
  picture), `patterns.py` (added detail), `preview.py` (pictures), `export.py` (STL, PLY, viewer buffers), `finish.py` + `blender_close.py` (the
  watertight solid).
- `phidias_hoard/director.py` (the model's direction), `llm.py` (the vision model client).
- `phidias_hoard/services.py` (library, projects, the job worker), `agent_tools.py` (the catalogue), `api/` (routes),
  `static/` (the studio), `store.py` (projects on disk).
- Data: `data/settings.json`, `data/projects/<id>/` (`project.json`, unpacked zip in `src/`, `sculpt.stl`,
  `closed.stl`, viewer meshes and pictures), `data/mcp-token`.
- Tests: `tests/test_engine.py`, `tests/test_app.py`.
