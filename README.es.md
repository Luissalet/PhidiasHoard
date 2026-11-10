# Phidias's Hoard

Estudio de esculpido de la familia Hoard. Abre escenarios low-poly sacados de juegos y los convierte en un modelo de
alta resolución para imprimir en 3D: el relieve que el juego solo dibuja en sus texturas (grietas, juntas entre
bloques, escalones, las líneas de una columna, tejas) se talla en la malla, al tamaño al que vas a imprimir y en
milímetros.

- **Biblioteca**: dile en qué carpetas guardas los modelos. Lista cada OBJ (con su MTL y texturas), Collada (`.dae`),
  GLB y glTF que encuentra, también dentro de zips, agrupados por carpeta. Un escenario que viene en varios formatos
  aparece una vez, en el más completo.
- **Proyecto**: al abrir un modelo lee su geometría y sus materiales y propone una receta para cada material. Los
  zips se descomprimen dentro del proyecto; lo demás se lee donde está.
- **Recetas por material**
  - *Mapa de normales*: cuando el juego lo trae, el relieve se integra a partir de él sobre la propia malla, así que
    funciona igual con texturas en mosaico, mitades en espejo y atlas horneados.
  - *Relieve*: se talla lo que la textura dibuja más oscuro que lo que lo rodea (juntas, grietas, estrías) y sus
    zonas claras y oscuras se convierten en bultos suaves. Las texturas pixel art de 8 a 128 píxeles se amplían antes,
    para que una línea de un píxel sea una ranura con paredes inclinadas. Los atlas horneados se leen isla a isla.
  - *Solo líneas*, *liso* (agua, cristal, colores planos) y *quitar* (planos de sombra, llamas, brillos).
  - *Escalones*: una rampa cuya textura dibuja escalones pasa a tener escalones de verdad, con huellas planas y
    contrahuellas, tantos como dibuja la textura (o el número que le des).
  - *Recortar por transparencia*: vallas, barandillas, hojas y árboles en plano pierden lo que la textura deja
    transparente y el resto recibe grosor para que se pueda imprimir.
  - Profundidad en milímetros, inversión y suavizado por material. Una imagen enseña la textura, el mapa de normales
    y el mapa de alturas que tallaría cada receta antes de esculpir.
- **Detalle añadido**: además de su receta, un material puede recibir detalle esculpido que su textura no trae, puesto
  a tamaño real en milímetros sean como sean sus UV: roca, grietas, adoquines, ladrillo, tablas, tejas, corteza, ondas
  y estrías de columna, cada uno con el tamaño de sus piezas y su profundidad.
- **Dirigido por el modelo**: un botón deja el esculpido en manos del modelo con visión de la familia. Mira cada
  material y dice qué es, elige receta, profundidad, recortes y detalle añadido, manda esculpir una prueba, la mira
  junto al original desde varios lados y de cerca, y la rehace mientras no le convenza; después hace el esculpido
  final. Puedes decirle con palabras lo que quieres. Lo que decidió y por qué queda guardado en el proyecto.
- **Esculpido**: la malla se refina solo donde el relieve lo pide, hasta el detalle más fino que le indiques (0,2 a
  0,3 mm para filamento, 0,1 mm para resina), dentro de un presupuesto de triángulos; si el detalle no cabe, se
  reparte por igual y te lo dice. Los triángulos vecinos siempre se cortan de acuerdo, así que la superficie nunca
  se abre.
- **Suelo macizo**: las caras que miran hacia arriba sin nada debajo se cierran en un bloque macizo hasta una base,
  para que el terreno se imprima sin soportes por debajo. Peana rectangular o redonda opcional.
- **Sólido cerrado**: un botón funde todo en un único sólido estanco con Blender en segundo plano (grosor para las
  láminas sueltas y remallado por vóxeles) y comprueba que no queda ningún borde abierto.
- **Vista 3D e imágenes**: original, esculpido y sólido cerrado en el navegador; imágenes sombreadas a partir de la
  propia geometría, desde cualquier ángulo, ampliadas sobre un punto o junto al original.
- **STL** en milímetros con la base en z = 0, descargado o guardado en una carpeta `Phidias` junto al modelo de origen.

Forma parte de la familia Hoard (ver `faustus-plugin.json`): las mismas herramientas que usan los botones están al
alcance de los asistentes por MCP (`mcp_server.py`), con imágenes de los materiales y de los modelos para que puedan
juzgar el relieve que piden.

## Arrancar

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m phidias_hoard
```

Escucha en http://127.0.0.1:5251. Los datos están en `data/` (`PHIDIAS_DATA_DIR` para moverlos). Blender se encuentra
solo (o se indica su ruta en Ajustes); sin él funciona todo menos el sólido cerrado. Pruebas: `python -m pytest`.

## Límites

- Formatos: OBJ, Collada, GLB y glTF. Los FBX y `.blend` hay que convertirlos antes.
- El relieve sale de lo que contienen las texturas. Un color plano no da nada, y un atlas horneado sin mapas de
  normales da menos que las texturas originales del juego.
- El sólido cerrado pierde el detalle más fino que su tamaño de vóxel y deja huecos los espacios cerrados.
- Las hojas y los árboles en plano siguen siendo láminas con grosor: no se convierten en volúmenes.
