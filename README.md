# Phidias's Hoard

Sculpting studio of the Hoard family. It opens low-poly scenery taken from games and turns it into a high-resolution
model for 3D printing: the relief that the game only draws in its textures (cracks, joints between blocks, steps,
the lines of a column, roof tiles) is carved into the mesh, at the size you are going to print and in millimetres.

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
- **Sculpting**: the mesh is refined only where the relief needs it, down to the finest detail you ask for (0.2 to
  0.3 mm for filament, 0.1 mm for resin), inside a triangle budget; when the detail does not fit, it coarsens evenly
  and says so. Neighbouring triangles always agree on their cuts, so the surface never opens.
- **Ground block**: the faces that look up with nothing below them are closed into a solid block down to a base, so
  terrain prints without supports underneath. Optional rectangular or round plinth.
- **Watertight solid**: one button fuses everything into a single closed solid with Blender in the background
  (thickness for loose sheets, voxel remesh) and checks that no edge is left open.
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
- The watertight solid loses detail finer than its voxel size and leaves enclosed spaces hollow.
- Leaves and tree cards stay flat sheets with thickness: they are not turned into volumes.
