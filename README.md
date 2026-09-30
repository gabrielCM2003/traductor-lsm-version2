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
| `interfaz_lsm.py` | Tema visual, componentes de la ventana y textos de retroalimentación (qué corregir de cada seña) |
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
| `tests/` | Pruebas unitarias: configuración, alfabeto estático, modo automático, DTW con cuerpo, retroalimentación y ventana |
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
- Al bajar las manos, la seña completa se compara con las letras dinámicas y con las palabras. Si en ella se fijaron 3 letras estáticas o más, fue deletreo y se respeta.
- La tarjeta **Seña** muestra el top-3 de cada seña con movimiento, y **Retroalimentación** dice si salió bien y, si no, qué corregir.

### La interfaz

- **Encabezado:** cámara, ❔ Guía (F1, cómo se hace cada seña), ⚙ Ajustes (Ctrl+,: umbrales, mano que deletrea, voz, dibujo y diagnóstico) y ▶ Iniciar / ■ Detener.
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
| F1 | Guía rápida |
| Ctrl+, | Ajustes |
| Retroceso | Borrar la última letra |
| Ctrl+Retroceso | Borrar la palabra |
| Enter o Ctrl+Espacio | Terminar la palabra (espacio) |
| Ctrl+S | Guardar captura |

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
- El adaptador de datos del guante (sensores de flexión) todavía no tiene código: el protocolo de datos de mecatrónica sigue sin definirse.
- La precisión del alfabeto estático todavía no está medida con personas que no participaron en el entrenamiento original.
- El guante está documentado, pero su firmware y el lector del guante en la aplicación aún no están escritos: según las reglas del concurso, la programación del dispositivo se hace durante el evento.
- Todo el desarrollo y las métricas de latencia se midieron en la laptop de desarrollo, no en la Raspberry Pi 5 real.

## Autor

Josué Gabriel Cortés Muñoz (alfabeto estático original), con aportes del equipo de software de Indivisa Ingenium 2026.
