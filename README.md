# Traductor LSM

Aplicación de escritorio que reconoce la Lengua de Señas Mexicana (LSM) con una cámara. MediaPipe detecta los 21 puntos de cada mano y el esqueleto del cuerpo; un clasificador ONNX reconoce el **alfabeto estático** (21 letras: A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y), un reconocedor por **DTW** reconoce el **alfabeto dinámico** (J, K, Ñ, Q, X, Z), que llevan movimiento en vez de postura fija, y las **palabras completas** HOLA, GRACIAS, POR FAVOR, AYUDA y MAMÁ. Las tres cosas se detectan a la vez, sin cambiar de modo. Las letras confirmadas forman palabras, y las palabras terminadas se guardan en un historial y se pueden leer en voz alta.

El repositorio incluye también la documentación del **guante instrumentado** diseñado para el reto de LSM de Indivisa Ingenium 2026 (Universidad La Salle Oaxaca).

## Características

- Detección de manos en tiempo real con MediaPipe Hand Landmarker (una o dos manos).
- Alfabeto estático (21 letras) vía modelo ONNX + suavizado temporal de predicciones.
- Alfabeto dinámico (J, K, Ñ, Q, X, Z) vía DTW (Dynamic Time Warping) contra un dataset abierto de LSM (CICESE, CC BY 4.0), con segmentación automática de inicio/fin de seña y sin necesidad de tecla.
- Palabras completas (HOLA, GRACIAS, POR FAVOR, AYUDA, MAMÁ) vía DTW con la forma de las manos y su ubicación respecto al cuerpo. Las plantillas se sacan de videos con `extraer_palabras_videos.py`.
- **Modo automático**: letras estáticas, letras dinámicas y palabras al mismo tiempo, sin botones ni atajos para cambiar de modo (ver "Cómo reconoce el modo automático").
- Esqueleto del cuerpo con MediaPipe Pose (hombros, brazos, cuello y cara) para saber dónde están las manos respecto a la persona, necesario para las palabras completas. Se muestra u oculta con la casilla "Dibujar esqueleto del cuerpo".
- Interfaz gráfica construida con PyQt6.
- Construcción de palabras letra por letra, con historial y lectura en voz alta.

## Contenido

| Archivo | Qué es |
|---|---|
| `senas.py` | Aplicación (PyQt6): cámara, MediaPipe, modo automático (letras estáticas, dinámicas y palabras), ventana y voz |
| `interfaz_lsm.py` | Tema visual, menú inicial, manual de señas, componentes de la ventana y textos de retroalimentación (qué corregir de cada seña) |
| `manual/` | Ilustraciones del manual de señas: `letras/` (abecedario) y `palabras/`, cada una con los puntos de donde se dibujó (.json) |
| `generar_manual.py` | Genera las ilustraciones de `manual/` a partir de datos reales (videos del equipo y plantillas del CICESE) |
| `sign_classifier.py` | Clasificador ONNX del alfabeto estático, suavizado de predicciones y confirmación de letras y espacios |
| `dtw_recognizer.py` | Reconocedor DTW del alfabeto dinámico (J, K, Ñ, Q, X, Z) y de las palabras (con la ubicación respecto al cuerpo) |
| `extraer_palabras_videos.py` | Saca las plantillas JSON de palabras de una carpeta de videos (una subcarpeta por palabra). Corre en la computadora o en Google Colab |
| `extraer_palabras_colab.ipynb` | Cuaderno de Google Colab que corre `extraer_palabras_videos.py` sobre videos en Google Drive |
| `segmentador_automatico.py` | Detecta solo, sin tecla, dónde empieza y termina una seña, y la reconoce o la graba como muestra (`--grabar`). Letras: mientras haya mano. Palabras (`--modo palabras`): mientras una mano esté sobre la línea de reposo, usando la pose |
| `body_tracker.py` | Esqueleto del cuerpo (MediaPipe Pose) y ubicación de las manos respecto a hombros y boca (9 valores aparte del vector de 126) |
| `recolector_estatico.py`, `recolector_dinamico.py` | Herramientas para grabar vocabulario nuevo (letras o palabras), con la ubicación respecto al cuerpo |
| `entrenar_palabras.py`, `word_classifier.py`, `lsm_words.onnx`, `word_labels.json` | Entrenamiento y clasificador de palabras. El modelo incluido es solo de prueba (datos sintéticos, no reconoce palabras reales; ver `ESTADO_PROYECTO_COMPLETO.md`) |
| `procesar_dataset_dinamico.py`, `extraer_landmarks_crudos.py`, `verificar_landmarks_crudos.py` | Conversión del dataset CICESE a plantillas y respaldo de landmarks crudos |
| `evaluar_*.py`, `diagnostico_orientacion.py`, `separar_muestras_cortas.py`, `probar_modelos.py` | Herramientas de evaluación y limpieza de datos usadas para medir el alfabeto dinámico |
| `datos_dinamicas/` | Plantillas DTW del alfabeto dinámico (dataset CICESE procesado) |
| `datos_palabras_dinamicas/` | Plantillas de palabras (una subcarpeta por palabra), sacadas de videos con `extraer_palabras_videos.py` o grabadas con el segmentador. Van aparte de las letras porque el DTW toma cada subcarpeta como una clase |
| `lsm_alphabet.onnx`, `lsm_alphabet.onnx.data` | Modelo entrenado del alfabeto estático (red pequeña, 63 entradas: 21 puntos × 3) |
| `lsm_labels.json` | Etiquetas del modelo estático y tipo de normalización |
| `guante.py` | Guante con ESP32: recibe los datos por WiFi (UDP), los compara con el dataset del guante y decide cuándo escribir la seña |
| `grabar_guante.py` | Graba muestras del guante en `datos_guante/dataset_guante.jsonl` (`--probar` para reconocer sin grabar) |
| `simular_guante.py` | ESP32 falsa: manda muestras del dataset por UDP para probar sin el guante |
| `datos_guante/` | Dataset del guante (una línea JSON por muestra: persona, etiqueta y 2 s de lecturas) |
| `tests/` | Pruebas unitarias: configuración, alfabeto estático, modo automático, DTW con cuerpo, retroalimentación, ventana y guante |
| `probar_modo_dinamico_senas.py` | Pruebas de regresión del alfabeto dinámico integrado en `senas.py` |
| `requirements.txt` | Dependencias de Python |
| `Guante_LSM_Indivisa_Ingenium_2026.pdf` | Guía técnica del guante v2: enlace inalámbrico ESP-NOW, batería LiPo y estación Raspberry Pi |
| `Cronograma_Indivisa_Ingenium_2026.pdf` | Plan de trabajo: días previos y las 24 horas del evento |
| `Ensamblaje_guante_Hall.html` | Animación del montaje con sensores Hall SS49E (se abre con doble clic, funciona sin internet) |
| `ESTADO_PROYECTO.md`, `ESTADO_PROYECTO_COMPLETO.md` | Estado detallado del proyecto: qué es real, qué es prototipo, métricas medidas y riesgos para la demo |

## Instalación

Probado con Python 3.11.

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

En macOS, `requirements.txt` evita mediapipe 1.0.1, que se cae al iniciar los detectores en Mac.

En Raspberry Pi con la cámara oficial, instala además Picamera2 con `sudo apt install python3-picamera2` y crea el entorno virtual con `--system-site-packages`. Para la voz en Linux: `sudo apt install espeak-ng`. macOS usa `say` y Windows la voz del sistema (System.Speech por PowerShell), que ya vienen instalados.

## Uso

```bash
python senas.py
```

La primera vez descarga los modelos de manos y de pose de MediaPipe en `~/.sign_translator/models/` (requiere internet la primera vez). En la Raspberry Pi, si hace falta CPU, el modelo de pose se puede bajar a `"pose_model": "lite"` o apagar con `"body_tracking": false` en la configuración. La configuración se guarda en `~/.sign_translator/config.json`; los valores inválidos se ignoran o se ajustan a su rango.

### Opciones disponibles

| Opción | Descripción |
|---|---|
| `--camera N` | Índice de la cámara a usar (por defecto: 0) |
| `--config RUTA` | Archivo de configuración JSON |
| `--threshold X` | Confianza mínima de detección (0–1) |
| `--max-hands N` | Número máximo de manos a detectar |
| `-v`, `--verbose` | Muestra logs detallados |

Ejemplo:

```bash
python senas.py --camera 1 --threshold 0.7
```

Si la mano se acerca demasiado a la cámara, el video muestra un aviso: MediaPipe sigue la mano mientras no la pierda, pero una vez perdida no la puede volver a detectar si ocupa ~80% de la imagen o más.

### Cómo reconoce el modo automático

La línea punteada del video es la **línea de reposo** (a la altura del ombligo). Una seña empieza cuando una mano sube por encima de ella y termina cuando las manos bajan (o salen de cuadro).

- **Letras estáticas (A-Y):** se fijan cuando la mano está quieta un momento, sola y arriba de la línea. Una mano en movimiento no escribe letras. La primera letra de cada seña se muestra en la tarjeta **Seña** y se escribe al bajar la mano (o en cuanto llega la segunda letra, si estás deletreando): así, la pausa de una palabra (HOLA en la frente) no deja una letra suelta que luego se borra.
- **Letras con movimiento (J, K, Ñ, Q, X, Z):** sube la mano, haz la letra y bájala.
- **Palabras (HOLA, GRACIAS, POR FAVOR, AYUDA, MAMÁ):** sube las manos, haz la seña y bájalas. La palabra se escribe completa y se cierra sola. La confianza está calibrada con personas que no aparecen en las plantillas (temperatura 0.5), y basta un margen de 0.15 sobre la segunda palabra para escribirla.
- **HOLA y MAMÁ:** si son las dos primeras, decide dónde quedó la punta del índice: MAMÁ se hace en la boca y HOLA en la frente (en las plantillas, la punta pasa cerca de la boca 0-20% del tiempo en HOLA y 71-98% en MAMÁ). Hace falta que se vean los hombros y la boca.
- Al bajar las manos, la seña completa se compara con las letras dinámicas y con las palabras. Si en ella se fijaron 3 letras estáticas o más, fue deletreo y se respeta.
- **Los tres tipos tienen la misma prioridad:** ninguno gana por su tipo. Si en la seña se fijaron 1 o 2 letras estáticas (la I al empezar la J, la R en la pausa de HOLA), una letra con movimiento o una palabra las reemplaza solo si lo demuestra: la letra con movimiento, con una distancia DTW de 1.0 o menos (ninguna letra fija sostenida baja de 1.15); la palabra, con el cuerpo a la vista (o, sin cuerpo, a 0.6 o menos). Si no, las letras fijas se quedan.
- **Sin cuerpo visible** (no se ven hombros o boca), las palabras se comparan solo con las manos: con el cuerpo en ceros, todo se parecía a MAMÁ.
- La tarjeta **Seña** muestra el top-3 de cada seña con movimiento, y **Retroalimentación** dice si salió bien y, si no, qué corregir.

### Menú inicial y manual de señas

Al abrir el programa aparece un menú. **Iniciar programa** muestra primero el **manual de señas** y, con **Continuar al traductor**, enciende la cámara. Con el traductor corriendo, **📖 Manual** (o F1) abre el manual en su propia ventana, sin detener la cámara.

El manual tiene tres pestañas:

- **Abecedario:** las 27 letras, en espejo, como te verás en la pantalla. Las letras fijas vienen de un video del equipo deletreando el abecedario: cada una se tomó de una pausa en la que el clasificador del programa la reconoce con confianza y que respeta el orden alfabético. Las letras con movimiento (J, K, Ñ, Q, X, Z) son la plantilla más representativa de cada una (dataset del CICESE, CC BY 4.0), animada; esas plantillas están centradas en la muñeca, así que muestran la forma y el giro de la mano, no el recorrido en el aire.
- **Palabras:** una figura dibujada, sin cara, a partir del esqueleto y las manos del video más representativo de cada palabra, animada y con la trayectoria de la mano.
- **Cómo usar:** los pasos para signar frente al traductor.

La **E** y la **P** todavía no tienen ilustración: en el video del abecedario no se sostuvieron el tiempo suficiente, y no se dibujan de memoria para no enseñar una forma equivocada. Para agregarlas, graba un video corto sosteniendo la letra y corre:

```bash
python generar_manual.py --letra E --video e.mp4
```

Para regenerar todo: `python generar_manual.py --abecedario VIDEO --dinamicas --palabras CARPETA_DE_VIDEOS`, o `--redibujar` para volver a dibujar desde los `.json` sin los videos.

### La interfaz

- **Encabezado:** cámara, 📖 Manual (F1), ⚙ Ajustes (Ctrl+,: umbrales, mano que deletrea, voz, dibujo y diagnóstico) y ▶ Iniciar / ■ Detener.
- **Video:** el marco cambia de color según lo que pasa: gris en reposo, azul mientras haces la seña, morado mientras la reconoce, verde si la reconoció y ámbar si hay que repetirla.
- **Seña:** la letra que se está formando o la última letra o palabra reconocida, con sus 3 candidatas.
- **Texto traducido:** las palabras terminadas en gris, la palabra en curso en blanco y, subrayadas en azul, las letras que la seña en curso todavía puede cambiar. Botones para borrar, terminar la palabra, leer en voz alta, guardar y limpiar.
- **Retroalimentación:** consejos en vivo ("No veo tus hombros", "¿B o P? Ajusta la forma de los dedos", "Para deletrear usa una sola mano", "Mano muy cerca de la cámara") y el resultado de cada seña con movimiento. Si una palabra no sale, compara cómo la hiciste con cómo se hace según sus plantillas: por ejemplo, "¿HOLA o MAMÁ? Tu mano quedó frente al pecho; HOLA se hace a la altura de la cabeza", "AYUDA se hace con las dos manos" o "La hiciste muy rápido".

### Rendimiento (Raspberry Pi 5)

Medido en la laptop de desarrollo; en la Pi 5 todo es unas 3-5 veces más lento, en la misma proporción.

| Qué | Antes | Ahora |
|---|---|---|
| Proceso por cuadro (manos + cuerpo) | 16.4 ms, uno tras otro | ~8 ms: la pose corre en su propio hilo, en paralelo (`"pose_async": true`) |
| Modelo de pose en la Raspberry Pi | full | lite (se detecta la Pi sola) |
| Reconocer una seña con movimiento | 90 ms | 30 ms: las plantillas de letras se comparan a ~15 cuadros por segundo |
| Cargar las plantillas al abrir | 1.75 s | 0.06 s: se guardan en un caché (`.cache_plantillas_*.npz`, se regenera si cambian) y se cargan en segundo plano al abrir la ventana |
| Compilar el DTW (Numba) | en cada arranque | solo la primera vez (`cache=True`) |
| Dibujar el video en la ventana | conversión a RGB + escalado suave de Qt | escalado con OpenCV y la imagen en BGR directo |

Pasando los 15 videos de prueba por la app como cámara, con pose lite, a 30 y a 15 cuadros por segundo, las 15 palabras salen bien en ambos casos. El único costo medido: de 90 letras dinámicas de prueba se escriben 82 en vez de 84 (si se quiere la precisión completa, `LETTER_TEMPLATE_STEP = 1` en `senas.py`).

### Guante (ESP32)

El guante lleva 6 sensores inerciales (pulgar, índice, medio, anular, meñique y dorso de la mano). La ESP32 crea la red WiFi `GUANTE_LSM` (IP `192.168.4.1`) y manda por UDP (puerto 4210) un JSON por lectura, unas 21 por segundo, al equipo que le dice `hola`:

```json
{"pulgar": [ax, ay, az, gx, gy, gz, pitch, roll], "indice": [...], "medio": [...], "anular": [...], "menique": [...], "mano": [...], "err": 0}
```

1. Conecta la Raspberry a la red del guante: `sudo nmcli device wifi connect GUANTE_LSM password lsm12345`
2. Graba muestras de cada seña (6 o más por seña; mejor de varias personas): `python grabar_guante.py`. Cada muestra son 2 s tras la cuenta atrás (2, 1, ¡ya!). Las etiquetas de más de una letra son palabras (`POR FAVOR` se guarda como `POR_FAVOR`).
3. Revisa cómo reconoce: `python grabar_guante.py --probar` (muestra cuántas acierta dejando cada muestra fuera y el umbral de distancia).
4. El traductor conecta el guante solo al abrir; no hay botón ni opción para activarlo. Conectarlo no cambia la pantalla: solo la barra de estado y la caja **Sensores del guante**, que muestra la última lectura de cada sensor.

**Cámara y guante juntos (una sola respuesta):**

- Cuando la cámara ve la mano, su respuesta y la del guante se combinan y **tienen que coincidir**: si coinciden, se escribe con más confianza; si la cámara duda entre dos (A o B) y el guante siente una de ellas, esa se escribe; si cada uno dice otra seña, no se escribe nada y el panel dice «¿B o C? La cámara ve B y el guante siente C». Vale para las letras fijas (en cada momento) y para las letras con movimiento y las palabras (al bajar la mano, con lo que el guante sintió durante toda la seña).
- El guante solo opina de las señas que tiene grabadas: si la cámara ve una letra que el guante no conoce (por ejemplo D), decide la cámara sola.
- Si la cámara no reconoce la seña (no se parece a nada) y el guante está muy seguro (80% o más), se escribe lo del guante.
- Si la cámara no ve la mano (el guante oscuro no se detecta, la mano sale de cuadro o la cámara está apagada), el guante escribe por su cuenta.
- En el video, la mano con guante siempre muestra sus 21 puntos (aunque el dibujo esté apagado en Ajustes) y encima una flecha por sensor: la dirección es su roll y el largo baja con el pitch. Si la cámara no encuentra la mano, las flechas salen en un recuadro «Guante» abajo a la izquierda. La zona de la mano se aclara un poco; el resto de la imagen no cambia.
- Para que el guante escriba palabras hay que grabarlas con `grabar_guante.py` (`HOLA`, `MAMA` o `MAMÁ`, `GRACIAS`...); se escriben igual que las de la cámara.

- **Automático:** sostén la seña ~1 s y se escribe. Para repetir una letra (LL), cambia de postura un momento y vuelve a hacerla.
- **Ctrl+G:** captura con cuenta atrás, igual que al grabar; útil para señas con movimiento o con el automático apagado (Ajustes → Guante).
- El reconocimiento compara la ventana de los últimos 2 s con las muestras del dataset (vecinos más cercanos). Si la distancia a la seña más parecida supera el umbral (calibrado solo con el propio dataset), o si duda entre dos señas, no escribe nada.

Sin el guante: `python simular_guante.py --senas L,A,Y` y, en otra terminal, `python senas.py --guante --glove-ip 127.0.0.1`.

La IP, el puerto y el archivo del dataset se cambian en `~/.sign_translator/config.json` (`glove_ip`, `glove_port`, `glove_dataset`) o con `--glove-ip` y `--glove-dataset`.

### Crear las plantillas de palabras desde videos

Pon los videos en una carpeta con una subcarpeta por palabra (el nombre de la subcarpeta es la palabra; `PORFAVOR` se guarda como `POR_FAVOR` y se muestra como "POR FAVOR"):

```
Entrenamiento/
    HOLA/       video1.mp4, video2.mp4, ...
    GRACIAS/    ...
```

```bash
python extraer_palabras_videos.py ~/Downloads/Entrenamiento --revision revision_palabras
```

Cada video se procesa igual que la cámara en vivo (espejo, manos, cuerpo y corte con la línea de reposo) y se guarda como `datos_palabras_dinamicas/<PALABRA>/muestra_N.json`. Si en el video hay más gente, se sigue a la persona que está al centro. Con `--revision`, guarda una imagen por muestra con el esqueleto, para revisar a ojo que se tomó a la persona correcta. Al final evalúa las plantillas (cada una contra las demás). Volver a correrlo salta los videos ya extraídos (`--sobrescribir` para rehacerlos).

**En Google Colab:** abre `extraer_palabras_colab.ipynb`, sube a Google Drive la carpeta de videos y estos archivos del programa: `extraer_palabras_videos.py`, `body_tracker.py`, `sign_classifier.py`, `segmentador_automatico.py` y `dtw_recognizer.py`. El cuaderno devuelve un `.zip` con la carpeta `datos_palabras_dinamicas/`, que se copia a la carpeta del programa.

### Grabar vocabulario sin tecla

`segmentador_automatico.py` detecta solo cada seña y, con `--grabar`, la guarda como muestra (manos, ubicación respecto al cuerpo y datos crudos):

```bash
python segmentador_automatico.py --modo palabras --grabar HOLA   # sube las manos, haz la seña, bájalas
python segmentador_automatico.py --grabar J                      # letras: la seña dura mientras haya mano
python segmentador_automatico.py --modo palabras                 # reconocer contra las palabras grabadas
```

En modo palabras, la seña termina al bajar las manos por debajo de la línea de reposo punteada, sin sacarlas de cuadro. En la ventana, `d` descarta la última muestra (la mueve a `datos_descartados/`) y `ESC` sale. Las letras se siguen cortando como las plantillas del dataset CICESE (del primer al último frame con mano), para que coincidan con ellas.

### Atajos de teclado

| Atajo (en macOS, Cmd en lugar de Ctrl) | Acción |
|---|---|
| Ctrl+R / Ctrl+T | Iniciar / detener |
| F1 | Manual de señas |
| Ctrl+, | Ajustes |
| Retroceso | Borrar la última letra |
| Ctrl+Retroceso | Borrar la palabra |
| Enter o Ctrl+Espacio | Terminar la palabra (espacio) |
| Ctrl+S | Guardar captura |
| Ctrl+G | Capturar una seña del guante con cuenta atrás |

Para repetir una letra (LL, RR, EE), relaja la mano un instante (o bájala) y vuelve a hacerla.

En **Ajustes**, **Mano que deletrea** elige qué mano se clasifica cuando hay dos en cuadro. Con la izquierda, la seña se refleja para compararla con el modelo y las plantillas, que son de la mano derecha. **Leer palabras en voz alta** lee cada palabra al terminarla. Los valores de los sliders y estas opciones se guardan en la configuración.

## Pruebas

```bash
python -m unittest discover -s tests -v      # configuración, alfabeto estático, modo automático
python probar_modo_dinamico_senas.py          # reglas del alfabeto dinámico (requiere datos_dinamicas/)
```

## Estado y limitaciones

Ver `ESTADO_PROYECTO_COMPLETO.md` para el detalle completo (qué está probado con varias personas, qué sigue siendo prototipo, métricas reales medidas, y riesgos prácticos para la demo). En resumen:

- Alfabeto estático (21 letras) y alfabeto dinámico completo (J, K, Ñ, Q, X, Z) funcionales, probados con varias personas.
- **Palabras completas:** 59 plantillas de 5 palabras, sacadas de videos de 3 personas del equipo. Reconociendo a cada persona solo con las plantillas de las otras dos (como un usuario nuevo): 56/59 bien, y se escriben 54 de esas 56 sin agregar errores; los 3 errores son AYUDA↔GRACIAS. Falta probar con más personas y cámaras. El modelo `lsm_words.onnx` sigue siendo de prueba y el modo automático no lo usa.
- **Guante:** el lector (`guante.py`) y el reconocimiento están integrados en el traductor. El dataset actual tiene 30 muestras de una sola persona (A, B, C, L, Y); dejando cada muestra fuera acierta 30/30, pero falta grabar más señas y a más personas.
- La precisión del alfabeto estático todavía no está medida con personas que no participaron en el entrenamiento original.
- Todo el desarrollo y las métricas de latencia se midieron en la laptop de desarrollo, no en la Raspberry Pi 5 real.

## Autor

Josué Gabriel Cortés Muñoz (alfabeto estático original), con aportes del equipo de software de Indivisa Ingenium 2026.
