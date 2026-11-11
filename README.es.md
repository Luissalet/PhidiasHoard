# Phidias's Hoard

Estudio de esculpido de la familia Hoard. Abre escenarios low-poly sacados de juegos y los convierte en un modelo de
alta resolución para imprimir en 3D: el relieve que el juego solo dibuja en sus texturas (grietas, juntas entre
bloques, escalones, las líneas de una columna, tejas) se talla en la malla, al tamaño al que vas a imprimir y en
milímetros.

- **Remodelado** (la forma principal de trabajar): el modelo del juego se despieza y cada pieza se construye de nuevo
  en Blender con operaciones de modelado, en quads limpios, usando el original solo como referencia de medidas y
  posiciones. Los tejados a cuatro aguas llevan tejas de medio punto de verdad, limatesas, faldón cóncavo y aleros
  levantados; los cuerpos llevan pilares, vigas y un panel rehundido, celosía o puerta en cada vano; las piezas
  torneadas (agujas, columnas) se tornean con el perfil de la propia pieza; losas y peanas siguen su perfil con
  biseles; las vigas y limatesas siguen su propia línea, recta o curva; las escaleras llevan peldaños; lo demás se
  conserva como la pieza original limpiada a quads. Las piezas torneadas salen redondas, como prisma con los lados
  que tiene la pieza (un anillo hexagonal sigue siendo hexagonal) o como anillo con su hueco. Un tejado con los aleros
  curvos en el modelo del juego se reconstruye con esa curva, medida, y entero aunque el juego no traiga los lados
  que no se ven; un tejado con algo más que cuatro faldones (una plataforma, una trampilla) conserva su forma y recibe
  las tejas a lo largo de sus faldones. Las piezas conservadas y esos tejados completan los lados exteriores planos
  cuyo borde abarca toda la pieza. Lo que dibujan las texturas (paneles, marcos, bandas, juntas) se modela en
  las caras: cada quad pasa a ser una retícula que sigue los téxeles donde cambia la altura, con escalones hacia
  fuera y hacia dentro; en triángulos y en texturas giradas sobre la cara entra con una retícula fina de quads
  (coronas, ménsulas, ornamentos). Las figuras que el juego hizo con facetas planas (estatuas, animales) se
  redondean en una superficie suave de quads pequeños que lleva lo que dibuja su textura: plumas, ojos, pliegues. A
  cada pieza solo se le ofrecen los constructores en los que su forma encaja de verdad, así
  que un terreno en pendiente nunca acaba convertido en bloque ni en tejado. El modelo de visión de la familia puede
  elegir qué es cada tipo y sus parámetros, mirar el resultado junto al original y corregirlo («Remodelar con IA»). Las piezas
  con la misma forma son un tipo: eliges qué es cada tipo y sus parámetros (tejas por lado, vanos, barrotes,
  segmentos). Si la textura tiene una repetición clara, Phidias propone el recuento de tejas, vanos y barrotes a
  partir de su mapeado en la pieza. El resultado es un `.blend` con un objeto por pieza y biseles vivos, y un OBJ
  en quads. La vista 3D y las imágenes dibujan las aristas de la malla.
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
  - *Relieve estimado* (`depth`): estima relieve relativo a partir del color con un modelo Depth Anything V2 local.
    Procesa cada isla del atlas por separado y usa la profundidad del material para ajustar la amplitud.
  - *Solo líneas*, *liso* (agua, cristal, colores planos) y *quitar* (planos de sombra, llamas, brillos).
  - *Escalones*: una rampa cuya textura dibuja escalones pasa a tener escalones de verdad, con huellas planas y
    contrahuellas, tantos como dibuja la textura (o el número que le des).
  - *Recortar por transparencia*: vallas, barandillas, hojas y árboles en plano pierden lo que la textura deja
    transparente y el resto recibe grosor para que se pueda imprimir.
  - Profundidad en milímetros, inversión y suavizado por material. Una imagen enseña la textura, el mapa de normales
    y el mapa de alturas que tallaría cada receta antes de esculpir.
- **Detalle añadido**: además de su receta, un material puede recibir detalle esculpido que su textura no trae, puesto
  a tamaño real en milímetros sean como sean sus UV: roca, grietas, adoquines, ladrillo, tablas, tejas, corteza, ondas
  y estrías de columna, madera tallada, escamas y musgo. Las estrías siguen el fuste y las tejas siguen la pendiente.
- **Dirigido por el modelo**: un botón deja el esculpido en manos del modelo con visión de la familia. Mira cada
  material y dice qué es, elige receta, profundidad, recortes y detalle añadido, y compara cada prueba con el
  original y con la prueba anterior. Puede pedir primeros planos de un material o una zona antes de decidir
  si rehace el esculpido. Puedes decirle con palabras lo que quieres. Lo que decidió y por qué queda guardado en el proyecto.
- **Esculpido** (la otra forma: desplazamiento sobre una malla densa de triángulos): la malla se refina solo donde el relieve lo pide, hasta el detalle más fino que le indiques (0,2 a
  0,3 mm para filamento, 0,1 mm para resina), dentro de un presupuesto de triángulos; si el detalle no cabe, se
  reparte por igual y te lo dice. Los triángulos vecinos siempre se cortan de acuerdo, así que la superficie nunca
  se abre.
  **Cancelar trabajo** en la barra de progreso detiene el esculpido o la dirección en el siguiente punto de
  control. Phidias conserva el último esculpido terminado.
- **Suelo macizo**: las caras que miran hacia arriba sin nada debajo se cierran en un bloque macizo hasta una base,
  para que el terreno se imprima sin soportes por debajo. En la peana rectangular o redonda puedes grabar un
  nombre y abrir un alojamiento circular para la figura con el diámetro y la profundidad que indiques en
  milímetros. El grabado y el alojamiento requieren Blender.
- **Sólido cerrado**: un botón funde todo en un único sólido estanco con Blender en segundo plano (grosor para las
  láminas sueltas y remallado por vóxeles) y comprueba que no queda ningún borde abierto. Rellena las cavidades
  selladas que quedan tras el remallado y conserva los pasos abiertos al exterior. Desmarca **Rellenar cavidades al
  cerrar** para conservar el remallado original.
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
solo (o se indica su ruta en Ajustes). Pruebas: `python -m pytest`.

Para **Relieve estimado**, selecciona en Ajustes una carpeta local con los [pesos de Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
Instala las dependencias opcionales con `pip install ".[depth]"` o indica un Python que ya tenga Torch y Transformers.
La inferencia se ejecuta en el PC por CPU.

## Límites

- Formatos: OBJ, Collada, GLB y glTF. Los FBX y `.blend` hay que convertirlos antes.
- El relieve sale de lo que contienen las texturas. Un color plano no da nada, y un atlas horneado sin mapas de
  normales da menos que las texturas originales del juego.
- El sólido cerrado pierde el detalle más fino que su tamaño de vóxel.
- Las hojas y los árboles en plano siguen siendo láminas con grosor: no se convierten en volúmenes.
