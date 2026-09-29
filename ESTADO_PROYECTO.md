# Estado del proyecto — Traductor LSM

Documento de estado para el equipo, no un diario de trabajo. Describe el resultado final y cómo llegar a él directamente, no el camino real (con vueltas y correcciones) que se tomó para construirlo.

> ⚠️ **Antes que nada:** todo este trabajo está en commits **locales, sin subir a GitHub** (la rama `main` local está 6 commits adelante de `origin/main`). Si alguien clona el repo desde GitHub o hace `git pull` en otra máquina, **no va a ver nada de esto**. Alguien del equipo tiene que hacer `git push` desde esta máquina antes de que el resto pueda trabajar sobre esto. Ver la sección 7 para la lista de commits.

---

## 1. ANTES (lo que Josué entregó)

- `senas.py` + `sign_classifier.py`: aplicación de escritorio (PyQt6) que reconoce el **alfabeto estático de LSM**, letra por letra, con la cámara.
- Reconoce **21 letras** (todas menos J, K, Ñ, Q, X, Z, que en LSM se hacen con movimiento, no con una postura fija).
- Modelo: una red neuronal pequeña (MLP 63→128→128→21) entrenada aparte y exportada a **ONNX** (`lsm_alphabet.onnx` + `lsm_labels.json`).
- Trabaja con **una sola mano**: el vector de entrada del modelo es de 63 valores (21 landmarks de MediaPipe × x,y,z, normalizados).
- Sin soporte para señas con movimiento ni para palabras completas.

## 2. AHORA (estado actual)

- **Alfabeto dinámico completo** (J, K, Ñ, Q, X, Z) integrado directamente en `senas.py`. Un botón/atajo (**Ctrl+D**) alterna entre el modo estático original (sin cambios) y el modo dinámico.
- **Arquitectura de dos manos** (vector de 126 = 2 × 63) ya definida y en uso — es la misma que necesita el reconocimiento de palabras completas más adelante, no hay que rehacerla.
- El alfabeto dinámico **no usa un modelo entrenado**: compara la secuencia de movimiento en vivo contra plantillas guardadas, usando **DTW** (Dynamic Time Warping, una técnica que compara dos secuencias de distinta duración). Las plantillas salen del dataset abierto de **CICESE** (Zenodo, DOI `10.5281/zenodo.14689869`, licencia **CC BY 4.0** — hay que citarlo en la presentación) más grabaciones propias del equipo.
- **Recolectores de datos** listos para que el equipo grabe su propio vocabulario de palabras:
  - `recolector_estatico.py` — palabras de postura fija (una o dos manos).
  - `recolector_dinamico.py` — palabras/señas con movimiento (inicio/fin explícitos con la tecla `g`, confirmación de guardar/descartar en pantalla).
- **Esqueleto del cuerpo (MediaPipe Pose)** para saber dónde están las manos respecto a la persona, que las palabras necesitan y el vector de 126 no tiene (normalizar en la muñeca borra la posición). Vive en `body_tracker.py`:
  - `senas.py` dibuja el esqueleto (casilla "Dibujar esqueleto del cuerpo"). Con `"body_tracking": false` en la configuración ni siquiera se carga el modelo.
  - Los dos recolectores guardan, además del vector de 126, **9 valores de ubicación** por muestra o frame: columnas `b0..b8` en el CSV, `body_frames` en el JSON y los 33 puntos crudos del cuerpo en el `.npz`.
  - El vector de 126 no cambió: el alfabeto, el DTW y `lsm_words.onnx` funcionan igual.
  - La pose usa **tiempo real** (clase `BodyTracker`), no el contador de +1 ms por frame que se le pasa al detector de manos. Con +1 ms, el suavizado interno de MediaPipe atrasaba el esqueleto unos 4 frames (~130 ms). A las manos ese contador no les afecta.
  - Modelo de pose `full` por defecto (tiembla menos). En la Raspberry, si hace falta CPU, se cambia con `"pose_model": "lite"` en la configuración.
  - Se eligió sobre YOLOv8-pose porque viene en el mismo paquete `mediapipe` (sin torch ni ultralytics, que reinstala `opencv-python` encima de `opencv-contrib-python`) y tiene puntos de la boca.
- Ya existe también una primera versión funcional del **reconocimiento de palabras** (modelo ONNX propio, `word_classifier.py` + `lsm_words.onnx` + `word_labels.json`), probada con 3 palabras de ejemplo (HOLA, GRACIAS, BUENOS_DIAS). Lo que falta es el vocabulario real del equipo (ver sección 5).
- **Confiabilidad real por letra**, medida con varias personas distintas (no solo quien programó):
  - **J, Ñ, X: sólidas.** Se comprometen con la regla normal de confianza (≥55%).
  - **K, Q, Z: se reconocen bien la mayoría de las veces, pero con confianza estructuralmente baja.** Las 6 letras dinámicas quedan muy cerca entre sí en distancia DTW, así que la confianza del top-1 casi nunca cruza ~45%, acierte o no. Por eso estas tres usan una **regla distinta: margen sobre la segunda opción**, no confianza absoluta (detalle técnico en la sección 3e). Cuando el margen es amplio, sí se comprometen — y se marcan distinto en pantalla para que se note que vinieron de esta regla.

---

## 3. Cómo construirlo directo (guía de reconstrucción)

Así es como se arma esto sabiendo ya el resultado — no el orden real en que se fue descubriendo.

### a) Definir el vector de 126 features desde el inicio

Vector = 2 bloques de 63 (mano izquierda + mano derecha), cada bloque = 21 landmarks de MediaPipe (x, y de la imagen + z del mundo), centrados en la muñeca y escalados por el tamaño de la mano. Si falta una mano, su bloque va en ceros.

- **Función que define esta normalización:** `normalize_keypoints` / `hand_to_feature_vector` en **`sign_classifier.py`**. Todo el resto del proyecto reutiliza estas dos funciones — nunca se reimplementan en otro archivo.

### b) Conseguir/generar el dataset de landmarks para las letras dinámicas

Fuente: dataset abierto de CICESE en Zenodo (DOI arriba), dos archivos (`.7z` vista frontal ~2.2 GB y de perfil ~1.4 GB).

- **Descarga + extracción + conversión a vectores de 126 ya normalizados:** `procesar_dataset_dinamico.py`. Llenó `datos_dinamicas/<LETRA>/muestra_<sujeto>_<repeticion>.json` (formato: `{n_frames, n_features, frames}`).
- **Respaldo de los landmarks SIN normalizar** (por si hace falta recalcular otros rasgos — trayectoria de la muñeca, orientación, velocidad — sin volver a descargar/decodificar los videos): `extraer_landmarks_crudos.py`. Guarda en `C:\Proyectos\Dataset_CICESE\landmarks_crudos\<LETRA>\S<sujeto>_<vista>_<repeticion>.npz` (por frame: timestamp, y por mano detectada: etiqueta, score, 21 landmarks de imagen y 21 de world; frames sin mano se guardan vacíos, no se omiten).
- **Verificación de que ambos formatos coinciden:** `verificar_landmarks_crudos.py`.
- **Para ampliar el vocabulario con grabaciones propias** (letras faltantes o palabras nuevas): `recolector_dinamico.py` (señas con movimiento) y `recolector_estatico.py` (posturas fijas). Ambos guardan directamente en el mismo formato que ya usa el proyecto.

### c) Construir el reconocedor DTW contra esas plantillas

Carga todas las plantillas de `datos_dinamicas/`, agrupadas por letra, y compara una secuencia nueva contra todas ellas.

- **Vive en:** `dtw_recognizer.py` (clase `DTWRecognizer`, métodos `predict`/`predict_topk`).

### d) Construir un segmentador automático (inicio/fin de seña por presencia de mano)

Máquina de estados simple: sin mano → esperando; aparece una mano → empieza a grabar; la mano desaparece un tiempo sostenido → termina y entrega la secuencia para clasificar.

Los umbrales de "cuánto tiempo sin mano cuenta como que terminó" están en **milisegundos reales** (`time.perf_counter()`), no en cantidad de frames — así el comportamiento es el mismo sin importar qué tan rápido procese cada máquina (una laptop de desarrollo vs. una Raspberry Pi son velocidades distintas; contar frames los haría sentir distinto en cada una).

- **Vive en:** `segmentador_automatico.py` (clase `AutoSegmenter`) — es el motor genérico, reusable también por consola sin la GUI.
- **Umbrales específicos que usa la GUI** (más tolerantes que el modo consola, porque señas como la J son un trazo largo): `DYN_NO_HAND_MS_TO_END` y `DYN_MAX_SEQUENCE_MS`, al inicio de **`senas.py`**.

### e) Integrar a la interfaz con una regla de commit en dos niveles

- **Vive en:** `senas.py` — función `dynamic_commit_decision()`.
- **Regla normal** (letras sólidas, `DYN_COMMIT_LETTERS = {"J", "Ñ", "X"}`): se compromete si la confianza del top-1 alcanza `DYN_MIN_CONF` (55%) y el margen sobre el top-2 alcanza `DYN_MIN_MARGIN`.
- **Regla experimental** (letras ambiguas — hoy K, Q, Z): la confianza absoluta nunca es buen indicador para estas tres, así que se compromete si el **margen** sobre el top-2 supera `DYN_EXPERIMENTAL_MIN_MARGIN` (12 puntos porcentuales), con `DYN_EXPERIMENTAL_MIN_CONF` (30%) solo como piso de sensatez, no como criterio principal.
- El botón/atajo **Ctrl+D** en la toolbar de `senas.py` alterna entre este modo dinámico y el alfabeto estático original.

---

## 4. Cómo correr todo

Desde la carpeta del proyecto, con el entorno virtual ya creado (`venv/`):

```bat
:: Activar el entorno virtual
venv\Scripts\activate

:: Correr la aplicación principal (alfabeto estático + dinámico con Ctrl+D)
python senas.py

:: Grabar palabras de postura fija (una o dos manos)
python recolector_estatico.py

:: Grabar señas/palabras con movimiento
python recolector_dinamico.py

:: (opcional) elegir camara si no es la 0
python senas.py --camera 1
```

Si no quieres activar el entorno, se puede invocar el Python del venv directamente sin activar nada:

```bat
venv\Scripts\python.exe senas.py
```

**OpenCV duplicado en entornos ya creados.** `requirements.txt` ahora pide `opencv-contrib-python` (el que ya trae `mediapipe`) en vez de `opencv-python`. Tener los dos instalados hace que ambos escriban la misma carpeta `cv2`, y desinstalar o actualizar uno rompe al otro. En un `venv` creado antes de este cambio, dejar solo uno:

```bat
venv\Scripts\pip uninstall -y opencv-python opencv-contrib-python
venv\Scripts\pip install opencv-contrib-python
```

---

## 5. Qué falta (en orden de prioridad)

1. **Vocabulario de palabras + grabación en equipo.** La arquitectura y el modelo de palabras ya funcionan (probado con HOLA/GRACIAS/BUENOS_DIAS), pero falta decidir el vocabulario real de la competencia y grabarlo con `recolector_estatico.py`/`recolector_dinamico.py` entre todo el equipo (más personas grabando = mejor generalización, como ya se vio con las letras dinámicas). Al grabar, que se vean **hombros y boca**: los recolectores muestran "Cuerpo: visible / NO visible", y si no se ven, la ubicación de esa muestra queda en ceros. Con el vocabulario grabado, falta que `entrenar_palabras.py` y `word_classifier.py` usen las columnas `b0..b8`; hoy solo leen `v0..v125` y las ignoran sin problema.
2. **Protocolo de datos del guante con mecatrónica.** Definir cómo van a entregar sus lecturas (formato, frecuencia, qué sensores) para poder integrarlas al mismo esquema de features o a uno paralelo.
3. **Integración fluida de los 3 modos sin Ctrl+D** (estático, dinámico, palabras) — si alcanza el tiempo. Hoy el cambio de modo es manual; lo ideal sería que el sistema detecte solo qué tipo de seña se está haciendo.

---

## 6. Cómo mejorar K, Q y Z (propuestas, no solo diagnóstico)

- **Q es la más débil** (58% de acierto incluso con datos limpios de laboratorio). La hipótesis más fuerte: la normalización actual (centrar en la muñeca y escalar) borra a propósito la posición y trayectoria de la muñeca en el aire — y Q podría depender de ese movimiento más que las otras letras. **Vía a probar:** agregar la posición relativa de la muñeca (no solo la postura de los dedos) como feature extra, además del vector de 126 actual. Los landmarks crudos ya guardados en `Dataset_CICESE/landmarks_crudos/` (ver sección 3b) permiten probar esto sin volver a descargar ni reprocesar ningún video.
- **K y Z generalizan mejor**, pero la confianza absoluta del DTW nunca cruza ~45% con las 6 clases tan cerca entre sí — es una limitación estructural del método de comparación, no solo de los datos. Ya hay un experimento hecho ampliando la galería de plantillas con copias rotadas del dataset (±15°), que sube algo la precisión de Q sin dañar las demás letras, pero cuesta ~3x más tiempo de cómputo por seña clasificada. **Pendiente de decidir:** si ese costo es aceptable corriendo en la Raspberry Pi de destino, o si conviene solo para el laptop de demo.
- **Cualquier cambio aquí debe volver a probarse con varias personas distintas, no solo con quien lo programó.** Ya se vio que el mismo sistema que funciona bien con una persona puede fallar con otra por diferencias de tamaño/forma de mano, ángulo de cámara o estilo personal al hacer la seña — un ajuste que "arregla" a K probado con una sola persona puede no servir de nada (o empeorar) en la demo real.

---

## 7. Checkpoints de git (por si hay que revertir algo)

Del más reciente al más antiguo. Todos son commits **locales** (ver aviso al inicio del documento).

| Hash | Qué representa |
|---|---|
| `2f7b3b638e79abd21a62f428484f2e514be685fc` | Antes de agregar la regla experimental de margen para K/Q/Z. **El código actual en disco va un paso más adelante que este commit** (la regla experimental ya está implementada en los archivos, solo falta confirmarla en un commit nuevo). |
| `f9d769f0d976bc2481c64d2235e75b60dc92d386` | Antes de restringir qué letras dinámicas se comprometen (`DYN_COMMIT_LETTERS`). |
| `ad30cc697ee0f9c0b3be7a340fc505dc31ab3fd4` | Antes de cambiar `recolector_dinamico.py` a inicio/fin explícito con la tecla `g` (en vez de "mantener presionada"). |
| `f12216d411c20f9104ade24db65691ea76370b0b` | Antes de arreglar el choque entre el comando de salida del prompt y las etiquetas de una sola letra (ej. escribir "Q" para grabar esa letra). |
| `f6dcafb3847d42023d5e20d39d56bf24c67a97e0` | Antes de los primeros ajustes de UX del modo dinámico en la GUI (textos de estado, panel de diagnóstico). |
| `eea588aee3d8c2f13f17201eac897927f3229a10` | Antes de integrar el alfabeto dinámico a `senas.py` por primera vez. Este es el punto para volver al alfabeto estático puro de Josué si algo del modo dinámico da problemas graves el día de la competencia. |
| `dedce1136de64ad56d79097d7a53e324caf6da92` | Entrega original de Josué, sin ningún cambio (alfabeto estático de 21 letras). |

Para volver a cualquiera de estos puntos sin perder el resto del historial:

```bat
git checkout <hash> -- senas.py
```

(cambia solo `senas.py`; para otros archivos, apunta el nombre correspondiente en vez de `senas.py`).
