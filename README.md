# Traductor LSM

Aplicación de escritorio que reconoce el alfabeto manual de la Lengua de Señas Mexicana (LSM) con una cámara. MediaPipe detecta los 21 puntos de la mano y un clasificador ONNX reconoce **21 letras estáticas** (A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y). Las letras confirmadas forman palabras, y las palabras terminadas se guardan en un historial y se pueden leer en voz alta.

El repositorio incluye también la documentación del **guante instrumentado** diseñado para el reto de LSM de Indivisa Ingenium 2026 (Universidad La Salle Oaxaca).

## Contenido

| Archivo | Qué es |
|---|---|
| `señas.py` | Aplicación (PyQt6): cámara, MediaPipe, clasificación, interfaz y voz |
| `sign_classifier.py` | Clasificador ONNX, suavizado de predicciones y confirmación de letras y espacios |
| `lsm_alphabet.onnx`, `lsm_alphabet.onnx.data` | Modelo entrenado (red pequeña, 63 entradas: 21 puntos × 3) |
| `lsm_labels.json` | Etiquetas del modelo y tipo de normalización |
| `tests/` | Pruebas unitarias |
| `requirements.txt` | Dependencias de Python |
| `Guante_LSM_Indivisa_Ingenium_2026.pdf` | Guía técnica del guante v2: enlace inalámbrico ESP-NOW, batería LiPo y estación Raspberry Pi |
| `Cronograma_Indivisa_Ingenium_2026.pdf` | Plan de trabajo: días previos y las 24 horas del evento |
| `Ensamblaje_guante_Hall.html` | Animación del montaje con sensores Hall SS49E (se abre con doble clic, funciona sin internet) |

## Instalación

Probado con Python 3.11.

```bash
pip install -r requirements.txt
```

En Raspberry Pi con la cámara oficial, instala además Picamera2 con `sudo apt install python3-picamera2` y crea el entorno virtual con `--system-site-packages`. Para la voz en Linux: `sudo apt install espeak-ng` (macOS usa `say`, que ya viene instalado).

## Uso

```bash
python señas.py
```

La primera vez descarga el modelo de manos de MediaPipe en `~/.sign_translator/models/`. La configuración se guarda en `~/.sign_translator/config.json`; los valores inválidos se ignoran o se ajustan a su rango.

Opciones: `--camera N`, `--config RUTA`, `--threshold 0.5`, `--max-hands 1`, `-v` (registro detallado).

| Atajo (en macOS, Cmd en lugar de Ctrl) | Acción |
|---|---|
| Ctrl+R / Ctrl+T | Iniciar / detener |
| Retroceso | Borrar la última letra |
| Ctrl+Retroceso | Borrar la palabra |
| Enter | Terminar la palabra (espacio) |
| Ctrl+S | Guardar captura |

Para repetir una letra (LL, RR, EE), relaja la mano un instante y vuelve a hacerla.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Estado y limitaciones

- Reconoce letras **estáticas** con **una mano**. No reconoce J, K, Ñ, Q, X ni Z, que llevan movimiento, ni señas de palabra completa.
- La precisión del modelo todavía no está medida con personas que no participaron en el entrenamiento.
- El guante está documentado, pero su firmware y el lector del guante en la aplicación aún no están escritos: según las reglas del concurso, la programación del dispositivo se hace durante el evento.
