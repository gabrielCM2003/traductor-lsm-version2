# Traductor LSM

Aplicación de escritorio que reconoce la Lengua de Señas Mexicana (LSM) con una cámara. MediaPipe detecta los 21 puntos de cada mano; un clasificador ONNX reconoce el **alfabeto estático** (21 letras: A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y) y un reconocedor por **DTW** reconoce el **alfabeto dinámico** (J, K, Ñ, Q, X, Z), que llevan movimiento en vez de postura fija. Las letras confirmadas forman palabras, y las palabras terminadas se guardan en un historial y se pueden leer en voz alta.

El repositorio incluye también la documentación del **guante instrumentado** diseñado para el reto de LSM de Indivisa Ingenium 2026 (Universidad La Salle Oaxaca).

## Características

- Detección de manos en tiempo real con MediaPipe Hand Landmarker (una o dos manos).
- Alfabeto estático (21 letras) vía modelo ONNX + suavizado temporal de predicciones.
- Alfabeto dinámico (J, K, Ñ, Q, X, Z) vía DTW (Dynamic Time Warping) contra un dataset abierto de LSM (CICESE, CC BY 4.0), con segmentación automática de inicio/fin de seña y sin necesidad de tecla. Se alterna con **Ctrl+D** dentro de la app.
- Esqueleto del cuerpo con MediaPipe Pose (hombros, brazos, cuello y cara) para saber dónde están las manos respecto a la persona, necesario para las palabras completas. Se muestra u oculta con la casilla "Dibujar esqueleto del cuerpo".
- Interfaz gráfica construida con PyQt6.
- Construcción de palabras letra por letra, con historial y lectura en voz alta.

## Contenido

| Archivo | Qué es |
|---|---|
| `senas.py` | Aplicación (PyQt6): cámara, MediaPipe, clasificación estática y dinámica, interfaz y voz |
| `sign_classifier.py` | Clasificador ONNX del alfabeto estático, suavizado de predicciones y confirmación de letras y espacios |
| `dtw_recognizer.py` | Reconocedor DTW del alfabeto dinámico (J, K, Ñ, Q, X, Z) |
| `segmentador_automatico.py` | Máquina de estados que detecta inicio/fin de una seña dinámica sin tecla |
| `body_tracker.py` | Esqueleto del cuerpo (MediaPipe Pose) y ubicación de las manos respecto a hombros y boca (9 valores aparte del vector de 126) |
| `recolector_estatico.py`, `recolector_dinamico.py` | Herramientas para grabar vocabulario nuevo (letras o palabras), con la ubicación respecto al cuerpo |
| `entrenar_palabras.py`, `word_classifier.py`, `lsm_words.onnx`, `word_labels.json` | Entrenamiento y clasificador de palabras. El modelo incluido es solo de prueba (datos sintéticos, no reconoce palabras reales; ver `ESTADO_PROYECTO_COMPLETO.md`) |
| `procesar_dataset_dinamico.py`, `extraer_landmarks_crudos.py`, `verificar_landmarks_crudos.py` | Conversión del dataset CICESE a plantillas y respaldo de landmarks crudos |
| `evaluar_*.py`, `diagnostico_orientacion.py`, `separar_muestras_cortas.py`, `probar_modelos.py` | Herramientas de evaluación y limpieza de datos usadas para medir el alfabeto dinámico |
| `datos_dinamicas/` | Plantillas DTW del alfabeto dinámico (dataset CICESE procesado) |
| `lsm_alphabet.onnx`, `lsm_alphabet.onnx.data` | Modelo entrenado del alfabeto estático (red pequeña, 63 entradas: 21 puntos × 3) |
| `lsm_labels.json` | Etiquetas del modelo estático y tipo de normalización |
| `tests/` | Pruebas unitarias del alfabeto estático |
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

En Raspberry Pi con la cámara oficial, instala además Picamera2 con `sudo apt install python3-picamera2` y crea el entorno virtual con `--system-site-packages`. Para la voz en Linux: `sudo apt install espeak-ng` (macOS usa `say`, que ya viene instalado).

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

### Atajos de teclado

| Atajo (en macOS, Cmd en lugar de Ctrl) | Acción |
|---|---|
| Ctrl+R / Ctrl+T | Iniciar / detener |
| Ctrl+D | Alternar entre alfabeto estático y dinámico |
| Retroceso | Borrar la última letra |
| Ctrl+Retroceso | Borrar la palabra |
| Enter | Terminar la palabra (espacio) |
| Ctrl+S | Guardar captura |

Para repetir una letra (LL, RR, EE), relaja la mano un instante y vuelve a hacerla.

## Pruebas

```bash
python -m unittest discover -s tests -v      # alfabeto estático
python probar_modo_dinamico_senas.py          # alfabeto dinámico (requiere datos_dinamicas/)
```

## Estado y limitaciones

Ver `ESTADO_PROYECTO_COMPLETO.md` para el detalle completo (qué está probado con varias personas, qué sigue siendo prototipo, métricas reales medidas, y riesgos prácticos para la demo). En resumen:

- Alfabeto estático (21 letras) y alfabeto dinámico completo (J, K, Ñ, Q, X, Z) funcionales, probados con varias personas.
- El modo de reconocimiento de **palabras completas** todavía usa datos sintéticos de prueba, no vocabulario real grabado — no confundir con un sistema funcional.
- El adaptador de datos del guante (sensores de flexión) todavía no tiene código: el protocolo de datos de mecatrónica sigue sin definirse.
- La precisión del alfabeto estático todavía no está medida con personas que no participaron en el entrenamiento original.
- El guante está documentado, pero su firmware y el lector del guante en la aplicación aún no están escritos: según las reglas del concurso, la programación del dispositivo se hace durante el evento.
- Todo el desarrollo y las métricas de latencia se midieron en la laptop de desarrollo, no en la Raspberry Pi 5 real.

## Autor

Josué Gabriel Cortés Muñoz (alfabeto estático original), con aportes del equipo de software de Indivisa Ingenium 2026.
