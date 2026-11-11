# Phidias's Hoard

Sculpting studio of the Hoard family. It opens low-poly scenery taken from games and turns it into a high-resolution
model for 3D printing: the relief that the game only draws in its textures (cracks, joints between blocks, steps,
the lines of a column, roof tiles) is carved into the mesh, at the size you are going to print and in millimetres.

- **Remodelling** (the main way of working): the game model is taken apart into its pieces and each one is built
  again in Blender with modelling operations, in clean quads, using the original only as the reference for sizes and
  positions. Hipped roofs get real half-round tiles, hip ridges, a concave slope and upturned eaves; storeys get
  posts, beams and a recessed panel, lattice or door in every bay; turned pieces (spires, columns) are lathed from
  the piece's own profile, as a round, as a prism with the sides the piece has (a hexagonal ring stays hexagonal)
  or as a ring with its hole; slabs and plinths follow their profile with bevels; beams and hip ridges follow their
  own line, straight or curved; stairs get real steps; anything else is kept as the original piece cleaned into
  quads. A roof whose eaves are curved in the game model is rebuilt with that curve, measured, and whole even when
  the game left out the sides nobody sees; a roof with more to it than four slopes (a platform, a hatch) keeps its
  shape and gets the tiles laid down its slopes. What the textures draw (panels, frames, bands, joints) is modelled
  into the faces: each quad becomes a grid that follows the texels where the height changes, raised and sunk in
  steps; on triangles and on textures that sit turned on the face it goes in on a fine grid of quads instead
  (crowns, brackets, ornaments). Figures the game made of flat facets (statues, animals) are rounded into a soft
  surface of small quads that carries what their texture draws: feathers, eyes, folds. A piece is only offered the builders its shape really fits, so sloping ground is never turned into a
  block or a roof. The family's vision model can choose what each type is and its parameters, look at the result
  next to the original and correct it ("Remodelar con IA"). Pieces of the same shape are one type: you choose what each type
  is and its parameters (tiles per side, bays, bars, segments). The result is a `.blend` with one object per piece
  and the bevels as live modifiers, and an OBJ in quads. The 3D view and the pictures draw the mesh's own edges.
- **Library**: point it at the folders where you keep game models. It lists every OBJ (with its MTL and textures),
  Collada (`.dae`), GLB and glTF it finds, also inside zips, grouped by folder. A scene that comes in several formats
  appears once, in the richest one.
- **Project**: opening a model reads its geometry and materials and guesses a recipe for each material. Zips are
  unpacked inside the project; everything else is read where it is.
- **Recipes per material**
  - *Normal map*: when the game ships one, the relief is integrated from it on the mesh itself, so it works with
    tiling textures, mirrored halves and baked atlases alike.
  - *Relief*: what the texture draws darker than its surroundings is carved (joints, cracks, flutes) and its light
    and dark areas become soft bumps. Pixel-art textures of 8 to 128 pixels are enlarged first so a one-pixel line
    becomes a groove with sloped walls. Baked atlases are read island by island.
  - *Lines only*, *flat* (water, glass, plain colours) and *remove* (shadow planes, flames, glows).
  - *Steps*: a ramp whose texture draws steps becomes real steps, with level treads and risers, as many as the
    texture draws (or the number you give).
  - *Cut out by transparency*: fences, railings, leaves and tree cards lose what the texture leaves see-through and
    the rest gets thickness so it prints.
  - Depth in millimetres, inversion and smoothing per material. A picture shows the texture, the normal map and the
    height map each recipe would carve before you sculpt.
- **Added detail**: on top of any recipe a material can take sculpted detail its texture does not hold, laid by real
  size in millimetres whatever the UVs are: rock, cracks, cobbles, bricks, planks, roof tiles, bark, ripples and
  column flutes, each with the size of its features and its depth.
- **Directed by the model**: one button hands the sculpt to the family's vision model. It looks at every material and
  says what it is, chooses recipe, depth, cut-outs and added detail, and compares each test with the original and
  the previous trial. It can ask for close-ups of a material or an area before deciding whether to redo the sculpt.
  You can tell it in words what you want. What it decided and why stays with the project.
- **Sculpting** (the other way: displacement on a dense triangle mesh): the mesh is refined only where the relief needs it, down to the finest detail you ask for (0.2 to
  0.3 mm for filament, 0.1 mm for resin), inside a triangle budget; when the detail does not fit, it coarsens evenly
  and says so. Neighbouring triangles always agree on their cuts, so the surface never opens.
  Use **Cancel job** in the progress bar to stop a sculpt or directed sculpt at its next checkpoint.
  Phidias keeps the last finished sculpt.
- **Ground block**: the faces that look up with nothing below them are closed into a solid block down to a base, so
  terrain prints without supports underneath. Optional rectangular or round plinth.
- **Watertight solid**: one button fuses everything into a single closed solid with Blender in the background
  (thickness for loose sheets, voxel remesh) and checks that no edge is left open. It fills sealed cavities left
  by the remesh while keeping passages that are open to the outside. Disable **Fill cavities** to keep the raw remesh.
- **3D view and pictures**: original, sculpt and closed solid in the browser; pictures shaded from the geometry
  itself, from any angle, zoomed on a spot, or next to the original.
- **STL** in millimetres with the base at z = 0, downloaded or saved in a `Phidias` folder next to the source model.

Part of the Hoard family (see `faustus-plugin.json`): the same tools the buttons use are available to assistants
through MCP (`mcp_server.py`), with pictures of materials and models so they can judge the relief they ask for.

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m phidias_hoard
```

It listens on http://127.0.0.1:5251. Data lives in `data/` (`PHIDIAS_DATA_DIR` to move it). Blender is found by itself
(or set its path in Ajustes); without it everything works except the watertight solid. Tests: `python -m pytest`.

## Limits

- Formats: OBJ, Collada, GLB and glTF. FBX and `.blend` have to be converted first.
- The relief comes from what the textures hold. A flat colour gives nothing, and a baked atlas without normal maps
  gives less than the game's original textures would.
- The watertight solid loses detail finer than its voxel size.
- Leaves and tree cards stay flat sheets with thickness: they are not turned into volumes.
