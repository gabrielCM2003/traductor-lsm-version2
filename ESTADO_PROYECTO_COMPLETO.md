# Estado del proyecto — Traductor LSM (Ingenium)

> Este documento complementa (no reemplaza) `ESTADO_PROYECTO.md` generado por Claude Code. Ese archivo describe bien el *qué* y el *cómo reconstruirlo*; este documento agrega el contexto completo, las métricas reales medidas durante el desarrollo, correcciones a afirmaciones que quedaron imprecisas, y los riesgos prácticos para el día de la competencia. Escrito por Claude (chat), que coordinó el trabajo de Code y Antigravity durante todo el proceso — este documento tiene el contexto completo de esa coordinación que ningún agente individual vio por sí solo.

---

## 0. Contexto de la competencia (para quien no lo tenga fresco)

- **Ingenium**: hackatón de 24h, mezcla mecatrónica + software. Equipo de 6 (3 mecatrónica, 3 software).
- **Objetivo del proyecto**: guantes con sensores de flexión que traduzcan Lengua de Señas Mexicana (LSM), **y que funcione tanto con guante como con cámara, de forma independiente** (ninguno depende del otro).
- **Hardware de meca**: Raspberry Pi 5 (8GB), 2× ESP32 (guante↔servidor), par de giroscopios, sensores de flexión. **El protocolo de comunicación (formato de datos, qué manda cada guante) sigue sin definirse** al momento de escribir esto — es el bloqueador más importante para conectar el adaptador del guante.
- **Rol de quien coordina este proyecto (Cesar)**: apoyo físico/mecatrónico + puente de comunicación entre equipos; **no toma decisiones de hardware**. En software, trabajó junto con un compañero (Josué, autor del código base) más el resto del equipo de software, que decidió que "cada quien avanza como sienta mejor" en vez de forzar una organización de carpetas común — este documento existe justamente para que ese avance individual no parta de cero.
- **Todo el trabajo descrito aquí corrió en la laptop de desarrollo (Windows, Ryzen 9 / RTX 4060), nunca en la Raspberry Pi 5 real.** Rendimiento, latencias y hasta la detección de cámara pueden comportarse distinto en el hardware final — ver sección 8.

---

## 1. ANTES (lo que Josué entregó — sin cambios)

- `senas.py` + `sign_classifier.py`: app de escritorio (PyQt6), reconoce el alfabeto estático de LSM con la cámara.
- 21 letras (todas menos **J, K, Ñ, Q, X, Z**, que en LSM llevan movimiento, no postura fija — por eso su modelo original no las incluye, no fue un olvido).
- Modelo: MLP 63→128→128→21, exportado a ONNX (`lsm_alphabet.onnx` + `lsm_labels.json`).
- Una sola mano, vector de 63 (21 landmarks de MediaPipe × x,y,z, normalizados centrando en la muñeca y escalando por tamaño de mano).
- **No hay script de entrenamiento ni dataset en el repo original** — solo el modelo ya entrenado. No se sabe con certeza de dónde salieron los datos de entrenamiento (ver sección 9, pregunta abierta a Josué).
- Sin soporte de movimiento ni de palabras.

---

## 2. AHORA — qué es real y qué NO (léase esta sección con cuidado)

### ✅ Funcional y probado con varias personas

- **Alfabeto estático** (21 letras, Josué): sin cambios, sigue igual.
- **Alfabeto dinámico completo** (J, K, Ñ, Q, X, Z) integrado en `senas.py`, alternando con **Ctrl+D**.
- **Arquitectura de dos manos** (vector de 126 = 2×63) en uso — ver sección 3.
- **Reconocimiento por DTW** (no red neuronal) contra plantillas de un dataset abierto — ver sección 5 para la fuente exacta.
- **Recolectores** (`recolector_estatico.py`, `recolector_dinamico.py`) probados y funcionando, listos para grabar vocabulario nuevo.

### ⚠️ CORRECCIÓN IMPORTANTE — esto NO está listo, aunque el reporte de Code pueda leerse como que sí

El documento de Code dice: *"ya existe también una primera versión funcional del reconocimiento de palabras... probada con 3 palabras de ejemplo (HOLA, GRACIAS, BUENOS_DIAS)"*.

**Esto es engañoso si se lee rápido.** Esas 3 palabras **nunca fueron grabadas por ninguna persona real** — Antigravity generó datos **sintéticos/aleatorios** (ruido gaussiano) únicamente para comprobar que el pipeline de entrenamiento corría sin errores de código. El modelo (`lsm_words.onnx`), el CSV (`datos_palabras/dataset_palabras.csv`) y esas tres etiquetas son **datos de prueba de plomería, no un sistema que reconozca esas palabras**.

**Si alguien del equipo ve esos archivos y asume que "ya reconoce HOLA/GRACIAS/BUENOS_DIAS", va a llevar una sorpresa mala en la demo.** Antes de confiar en el modo de palabras hace falta:
1. Definir el vocabulario real (sección 6 de Code, complementada abajo).
2. Grabar datos reales con `recolector_estatico.py`/`recolector_dinamico.py`.
3. Volver a correr `entrenar_palabras.py` con esos datos reales (esto sobrescribe el modelo mock automáticamente).

**Verificar antes de usar:** revisar si `datos_palabras/dataset_palabras.csv` todavía tiene `quien_grabo: mock_generator` en sus filas — si sí, el modelo de palabras sigue siendo falso.

### 🆕 Integrado, pero todavía sin probar con varias personas

- **Esqueleto del cuerpo (MediaPipe Pose, `body_tracker.py`)** para ubicar las manos respecto a la persona, que es lo que las palabras necesitan y el vector de 126 no tiene.
  - `senas.py` dibuja hombros, brazos, cuello y cara (casilla "Dibujar esqueleto del cuerpo").
  - Los recolectores guardan **9 valores de ubicación** además del vector de 126: columnas `b0..b8` en el CSV, `body_frames` en el JSON y los 33 puntos crudos en el `.npz`.
  - **Ningún modelo los usa todavía**: `entrenar_palabras.py` sigue leyendo solo `v0..v125`.
  - Probado con una persona real y con movimiento simulado: sin retraso (la pose usa tiempo real, no el contador de +1 ms de las manos) y con los brazos completos aunque los codos queden cerca del borde de la imagen.
- **Ojo con el CSV de prueba:** `recolector_estatico.py` se niega a escribir en un `dataset_palabras.csv` con las columnas viejas (sin `b0..b8`), para no desalinearlo. Si alguien todavía tiene el CSV mock, hay que renombrarlo o moverlo antes de grabar vocabulario real.

### ❌ No empezado

- **Adaptador del guante**: solo diseñado en el patrón de arquitectura (adaptador intercambiable que entrega un vector al mismo núcleo), pero **no hay ni una línea de código escrita para él**, porque el protocolo de datos de meca sigue sin definirse.
- **Integración fluida de los 3 modos sin Ctrl+D** (que el sistema decida solo si es letra estática, dinámica o palabra): no empezada.
- **Vocabulario de palabras real**: cero datos grabados por el equipo a la fecha de este documento.

---

## 3. Arquitectura técnica (complementa la sección 3 de Code con detalle exacto)

### El vector de 126 features

```
vector[0:63]   = mano izquierda (21 landmarks × x,y,z), o CEROS si no se detectó
vector[63:126] = mano derecha   (21 landmarks × x,y,z), o CEROS si no se detectó
```

Cada bloque de 63 se normaliza con `normalize_keypoints()` (en `sign_classifier.py`):
1. Resta la posición (x,y) de la muñeca (landmark 0) a todos los puntos → la mano queda centrada en el origen, sin importar dónde esté en la imagen.
2. Divide por la distancia muñeca→base del dedo medio (landmark 9) → sin importar qué tan cerca/lejos esté de la cámara.

**Consecuencia importante para cualquiera que quiera mejorar K/Q/Z (sección 7):** este paso 1 **borra la trayectoria de la muñeca en el espacio** — cada frame "olvida" dónde estaba la mano en el frame anterior. Para una seña estática esto es correcto (es justo lo que se quiere). Para una seña dinámica que se distingue por *hacia dónde se mueve la muñeca* (no solo por la forma de los dedos), esta normalización puede estar destruyendo información necesaria. Esto es una **hipótesis con evidencia parcial**, no un hecho comprobado — ver sección 7.

### Por qué DTW y no una red neuronal para las letras dinámicas

Con pocas decenas de muestras por clase, entrenar una red neuronal desde cero (tipo LSTM) es poco confiable. **DTW (Dynamic Time Warping)** es un algoritmo matemático (no aprende, no se "entrena") que compara dos secuencias de distinta duración midiendo cuánto hay que "estirar/encoger" una para parecerse a la otra. Con solo 3-5 plantillas por clase ya funciona razonablemente — por eso se usó también para las palabras dinámicas del vocabulario futuro, no solo para el alfabeto.

- **Implementación:** `dtw_recognizer.py`, clase `DTWRecognizer`.
- **Optimización de velocidad aplicada:** la implementación original (librería `fastdtw`) tardaba **~4.4 segundos por clasificación** contra las 558 plantillas del dataset — inaceptable para uso en vivo, y además competía por el CPU con el hilo de la cámara, congelando el video. Se reemplazó por cálculo de matriz de costos con `scipy.spatial.distance.cdist` + programación dinámica compilada con `numba` (`@njit`). **Resultado: ~88-114 ms por clasificación (39-45× más rápido)**, verificado dando *exactamente* el mismo ranking que la versión original en varios casos de prueba.
- **Sigue corriendo en un hilo aparte** (no `QThread`, sino `threading.Thread` con una `queue.Queue` para pasar el resultado) para no bloquear el video ni aunque la clasificación tardara más de lo esperado.

### Segmentación automática (inicio/fin de seña sin tecla)

Máquina de estados: sin mano → esperando; aparece mano → grabando (acumula vectores); mano ausente durante un tiempo sostenido → termina, clasifica.

- **Implementación:** `segmentador_automatico.py`, clase `AutoSegmenter`. Diseñada para poder probarse con timestamps sintéticos (sin cámara), lo cual se usó extensamente durante el desarrollo.
- **Los umbrales están en milisegundos reales** (`time.perf_counter()`), no en cantidad de frames — decisión deliberada: contar frames haría que el mismo umbral se sintiera distinto en la laptop de desarrollo vs. la Raspberry Pi (que probablemente procese a menos FPS), porque el mismo número de frames representa distinto tiempo real según qué tan rápido vaya cada máquina.
- **Umbrales de la GUI (`senas.py`)**, más tolerantes que el modo consola porque señas como J son un trazo largo:
  - `DYN_NO_HAND_MS_TO_END` — tiempo sin mano para dar la seña por terminada.
  - `DYN_MAX_SEQUENCE_MS` — tope máximo por seña, para no quedarse esperando indefinidamente.

### Regla de decisión de "commit" (cuándo una letra se agrega a la palabra)

Vive en `dynamic_commit_decision()`, en `senas.py`. **Dos reglas distintas, no una:**

| | Letras | Criterio | Constantes |
|---|---|---|---|
| Regla normal | J, Ñ, X (`DYN_COMMIT_LETTERS`) | Confianza del top-1 ≥ umbral | `DYN_MIN_CONF = 0.55`, `DYN_MIN_MARGIN = 0.0` |
| Regla experimental | K, Q, Z | **Margen** sobre el 2.º lugar ≥ umbral, con un piso mínimo de confianza solo como filtro de sensatez | `DYN_EXPERIMENTAL_MIN_MARGIN = 0.12` (12 puntos porcentuales), `DYN_EXPERIMENTAL_MIN_CONF = 0.30` |

**Por qué dos reglas y no una:** con 6 clases muy cercanas en distancia DTW, la confianza (softmax sobre distancias) de K, Q y Z casi nunca cruza ~45%, acierten o no — es una limitación estructural de tener pocas clases muy parecidas, no un defecto de calibración. Exigirles el mismo 55% que a J/Ñ/X las habría dejado bloqueadas siempre. El margen (qué tanto le gana el 1.º al 2.º lugar) resultó ser un mejor indicador de si el resultado es confiable — ver la validación con datos reales en la sección 6.

Cuando K, Q o Z se comprometen por la regla experimental, se marcan distinto en pantalla (ej. `"K 38.9% (margen alto)"`) para que quede claro que vinieron de un criterio distinto al normal.

> **Nota (2026-09-29):** esta tabla describe el diseño ORIGINAL de dos reglas. Ese mismo día se extendió la regla de margen a las 6 letras (J, Ñ, X ahora también pueden comprometerse por margen amplio, no solo por confianza) y se agregó aislamiento de mano intrusa durante la grabación en vivo. Ver el historial de commits de `senas.py` para el detalle actualizado; esta sección se deja tal cual como registro histórico de la decisión original.

---

## 4. Fuentes de datos utilizadas

### Dataset de letras dinámicas (SÍ usado, es la base de todo el reconocimiento dinámico)

- **Fuente:** dataset abierto de letras dinámicas de LSM (J, K, Ñ, Q, X, Z) publicado en Zenodo por investigadores de **CICESE**.
- **DOI: `10.5281/zenodo.14689869`** — verificar la cita bibliográfica completa (autores, año) directamente en esa página de Zenodo antes de ponerla en la presentación, para no citar mal.
- **Licencia: CC BY 4.0** — de uso libre, **citando a los autores es obligatorio**, no opcional. Vale la pena mencionarlo explícitamente en la demo/presentación, además de sumar seriedad académica al proyecto.
- **Composición:** 20 sujetos, 5 repeticiones por letra, vista frontal y de perfil (45°), fondo verde controlado, 558 videos por vista.
- **Existe un dataset hermano** con el alfabeto **estático** de LSM del mismo grupo (Zenodo DOI `10.5281/zenodo.10067509`) — no se usó porque el estático de Josué ya funcionaba, pero podría servir para comparar/mejorar ese modelo si sobra tiempo después de la competencia.

### Dataset de 249 palabras — encontrado, disponible, NUNCA usado (recurso desaprovechado)

- **Mendeley Data**, Espejel et al. (UAEM), 2023, DOI `10.17632/6rj76z6y3n.1`.
- **249 palabras de LSM** agrupadas en 15 categorías: saludos, tiempo, días, meses, útiles escolares, familia, casa, adjetivos, comida, ropa, partes del cuerpo, vehículos, lugares, pronombres, verbos, profesiones, estados de México.
- Formato: secuencias de imágenes JPG (no video), 11 personas, fondo controlado (tela/ropa negra).
- **Nadie del equipo lo ha descargado ni procesado.** Si el vocabulario de la demo necesita palabras comunes (saludos, números, etc.), este dataset podría ahorrar mucha grabación manual — el mismo patrón que se usó con las letras dinámicas de CICESE (descargar, procesar con MediaPipe, generar plantillas) aplicaría aquí, adaptando `procesar_dataset_dinamico.py` como referencia.

### Datasets descartados (para que nadie pierda tiempo re-intentándolos)

| Dataset | Por qué no sirve |
|---|---|
| Corpus LSM de Mejía-Pérez et al. (2022) | Solo datos ya codificados según criterio propio de los autores, sin acceso a landmarks crudos reprocesables |
| Otro corpus de la misma autora (27 señas) | Capturado con sensor Kinect V2 (nube de puntos de profundidad), esquema de puntos incompatible con los 21 landmarks de MediaPipe |
| Datasets públicos genéricos de lengua de señas (tipo Kaggle) | Casi todos son de **ASL** (americana), no LSM — usarlos dañaría la credibilidad del proyecto ante el jurado |

---

## 5. Métricas reales medidas (esto es lo más valioso que falta en el reporte de Code)

### 5.1 — Precisión "de laboratorio" (LOSO: leave-one-subject-out sobre el dataset CICESE)

Validación cruzada dejando fuera un sujeto a la vez (nunca evalúa con datos que el propio sujeto aportó como plantilla) — 558 muestras, 20 sujetos:

| Letra | Top-1 | Top-3 |
|---|---|---|
| J | **100.00%** (102/102) | 100% |
| Z | **100.00%** (88/88) | 100% |
| Ñ | **98.92%** (92/93) | — |
| X | **92.13%** (82/89) | — |
| K | **84.21%** (80/95) | — |
| **Q** | **58.24%** (53/91) | 88.9% |
| **Global** | **89.07%** (497/558) | **98.57%** (550/558) |

**Matriz de confusión (extracto, quién se confunde con quién):**
- Q se confunde con X (26 de 91, 28.6%) y con K (11 de 91, 12.1%).
- K se confunde con Q (9 de 95) y X (6 de 95).
- J, Z nunca se confundieron con nada en esta validación.

Esto confirma que **Q es estructuralmente la letra más difícil, incluso en condiciones de laboratorio perfectas** (misma cámara, mismo fondo, mismas condiciones que el entrenamiento) — no es solo un problema de "cámara casera vs. laboratorio".

### 5.2 — Prueba real con 3 personas distintas (nunca fueron quien programó el sistema)

Esta prueba se hizo directamente en `senas.py` con cámaras caseras, condiciones reales. Resumen por letra (intentos = veces que se hizo la seña y se leyó el resultado):

| Letra | Intentos observados | Correctas en top-1 | Nota |
|---|---|---|---|
| Ñ | 3 | 3/3 | Siempre se agregó, alta confianza (63.6%-74.4%) |
| J | 3 | 3/3 | Top-1 correcto siempre, pero solo 1/3 se agregó — las otras 2 tenían ~30pp de ventaja sobre el 2.º lugar pero quedaban justo debajo del 55% de confianza |
| X | 2 (+ 1 sin captura, reportada correcta) | 2/2 (+1 reportada) | Mismo patrón que J: correcta pero a veces bajo el umbral |
| Z | 4 | 3/4 | 1 falla fue con velocidad rápida (se confundió con Q); con velocidad normal/lenta, correcta |
| K | 6 | 5/6 | La falla fue en una muestra que "costó mucho trabajo hacer y mantener" según quien probó — dudosa de por sí |
| Q | 5 | 4/5 | La mejor Q de toda la prueba tuvo 20 puntos de margen; la peor perdió contra Z por completo |

**Validación de la regla experimental (12pp de margen) contra estos datos reales:** de 11 intentos correctos con margen medible, el umbral actual **habría comprometido 4 correctamente y rechazado 7 por seguridad** (eran correctos pero con margen dudoso — mejor perderlos que arriesgar un error). **Cero veces habría comprometido algo incorrecto.** Esto es una señal razonablemente buena de que el umbral de 12pp está calibrado del lado seguro, no del lado permisivo.

### 5.3 — Experimento de orientación (por qué Q es particularmente terca)

Se midió el ángulo de la mano en el plano de la imagen (vector muñeca→base del dedo medio):

| | Mediana del ángulo (dataset) | Cuánto gira dentro de una seña (dataset) | Mediana (muestras propias, rígidas) |
|---|---|---|---|
| Q | -48.1° | **56.2°** de variación interna | -102.9° (solo 5.9° de variación) |
| K | -66.0° | 41.8° | -95.5° (solo 3.9°) |
| Z | -74.4° | 14.5° | -91.7° |
| J | -73.1° | 24.9° | -98.0° |

**Hallazgo:** compensar solo el ángulo fijo (rotar toda la seña por la diferencia de mediana) **sí rescató bastante a K** (12.5% → 62.5% en un experimento, aunque con fuga de información — el ángulo se calibró con las mismas muestras que se evaluaron, así que ese número es optimista). **Pero Q nunca mejoró con ningún ángulo probado** (barrido de -20° a +90°) — siempre terminaba cayendo en K o X. La lectura más probable: Q no es solo una cuestión de "está rotada", sino que en el dataset incluye un giro dinámico de muñeca *durante* la seña (56° de variación interna) que las muestras rígidas no tienen — compensar un ángulo fijo no puede arreglar algo que es, en el fondo, un movimiento distinto.

### 5.4 — Experimento de ampliar la galería con rotaciones fijas (sin fuga de información esta vez)

Se agregaron copias rotadas (±15°, ±30°) de las 558 plantillas del dataset, sin mirar ninguna muestra de prueba al elegir los ángulos:

| Galería | Q (LOSO) | Ñ (LOSO) | Latencia por consulta |
|---|---|---|---|
| Original (1×, 558 plantillas) | 58.24% | 98.92% | 111.6 ms |
| ±15° (3×, 1,674 plantillas) | **62.64%** (+4.4 pts) | 97.85% (-1.07 pts, seguro) | 337.1 ms |
| ±30° (5×, 2,790 plantillas) | 60.44% (+2.2 pts) | **94.62%** (-4.3 pts, **rompe el límite de seguridad**) | 558.0 ms |

**Veredicto (de Antigravity, confirmado por el análisis):** la versión de ±30° se descarta (daña a Ñ y cuesta medio segundo por seña). La de ±15° es segura pero **no se integró** — la ganancia (+4.4 puntos en Q) no se consideró suficiente para justificar triplicar la latencia y volver a calibrar toda la regla de margen, a horas de la competencia. **Queda como mejora candidata para después de Ingenium**, no bloqueante.

---

## 6. Cómo correr todo

(Igual que en `ESTADO_PROYECTO.md` de Code — se repite aquí para que este documento sea autocontenido)

```bat
venv\Scripts\activate

python senas.py                    :: app principal (Ctrl+D alterna estático/dinámico)
python recolector_estatico.py      :: grabar palabras de postura fija
python recolector_dinamico.py      :: grabar señas/palabras con movimiento
python senas.py --camera 1         :: si la cámara no es la 0
```

O sin activar el venv: `venv\Scripts\python.exe senas.py`

### ⚠️ Riesgo práctico para mañana: dependencia de internet en el primer arranque

**MediaPipe descarga su modelo de detección de manos (`hand_landmarker.task`) de servidores de Google la primera vez que se corre `senas.py` en una máquina.** Si el lugar de la competencia no tiene wifi confiable, o si se usa una laptop/Raspberry que nunca ha corrido el proyecto antes, **la app puede fallar en el peor momento** (a media demo, sin internet).

**Recomendación:** correr `senas.py` al menos una vez en cada máquina que se vaya a usar en la demo, **antes** de llegar al lugar del evento, para que el modelo quede descargado localmente. Verificar también si esto aplica a la Raspberry Pi si se va a correr el modo cámara ahí.

### Otro detalle práctico: la primera clasificación DTW es más lenta

La compilación de `numba` (que acelera el DTW) ocurre en la primera llamada, no al iniciar el programa — la primera seña dinámica que se haga después de abrir la app puede tardar más de lo normal. Vale la pena "precalentar" haciendo una seña de prueba antes de la demo real, no como el primer intento frente al jurado.

---

## 7. Cómo mejorar K, Q y Z (propuestas concretas, con la evidencia que las respalda)

- **Q es la más débil, incluso en laboratorio (58.2% LOSO).** La hipótesis mejor respaldada (sección 5.3): tiene un componente de giro de muñeca *durante* la seña que la normalización actual borra (ver sección 3, por qué `normalize_keypoints` resta la posición de la muñeca en cada frame). **Vía a probar:** agregar la trayectoria/velocidad relativa de la muñeca como features extra, no solo la postura de los dedos. Los landmarks **sin normalizar** ya están guardados (`C:\Proyectos\Dataset_CICESE\landmarks_crudos\`, ver sección 3b del reporte de Code) — se puede experimentar con nuevos features sin volver a descargar ni procesar ningún video.
- **K y Z generalizan mejor**, pero su confianza absoluta nunca cruza ~45% por diseño (6 clases muy cercanas en distancia DTW) — no es un bug, es una limitación estructural del método. La regla de margen (sección 3) ya compensa esto razonablemente bien, validada con datos reales de 3 personas (sección 5.2).
- **La ampliación de galería con rotaciones (±15°) ya está probada y ayuda un poco a Q** (+4.4 puntos, sección 5.4) sin dañar las demás letras — pendiente de decidir si vale la pena el costo de latencia (3× más lento) en la Raspberry Pi real.
- **Cualquier cambio debe volver a probarse con varias personas**, no solo con quien lo programó — ya se documentó que el mismo sistema se comporta distinto entre personas (sección 5.2), y solo se ha probado con 3 personas externas hasta ahora. Más pruebas = calibración más confiable.

---

## 8. Riesgos y pendientes prácticos (no cubiertos en el reporte de Code)

| Riesgo | Detalle | Urgencia |
|---|---|---|
| **`git push` pendiente** | Todo el trabajo está en commits locales; si otra persona clona el repo desde GitHub, no ve nada de esto. Alguien tiene que hacer `git push` desde esta máquina | 🔴 Antes de que cualquier compañero intente trabajar en otra máquina |
| **Nada probado en la Raspberry Pi 5 real** | Todo el desarrollo y las mediciones de latencia fueron en la laptop de desarrollo | 🔴 Si el modo cámara va a correr en la Pi para la demo |
| **Modo de palabras con datos falsos** | Ver sección 2 — no confundir con un sistema funcional | 🔴 Antes de mostrarlo a nadie |
| **`.gitignore` ausente** | `venv/` está siendo rastreado por git, infla artificialmente el conteo de cambios (~955 archivos en un momento dado) | 🟡 Cosmético, no bloquea nada |
| **Archivo fantasma en el historial de git** | El repo tiene rastro de `señas.py` (con ñ) siendo "borrado" cuando en realidad se renombró a `senas.py` para evitar problemas de codificación en terminal | 🟡 No afecta funcionamiento |
| **Umbrales calibrados con muestra pequeña** | Solo 3 personas externas probaron K/Q/Z hasta ahora | 🟡 Ideal seguir probando con más compañeros antes de confiar 100% en la demo |

---

## 9. Preguntas abiertas sin resolver

1. **¿De dónde sacó Josué los datos para entrenar el alfabeto estático original?** Nunca se confirmó — el repo no tiene script de entrenamiento ni dataset. No afecta el modelo actual (ya funciona y no se ha tocado), pero sería bueno saberlo por si hace falta reentrenarlo o ampliarlo.
2. **Protocolo de datos del guante** — sigue sin definir por parte de mecatrónica (formato, frecuencia, qué sensores exactos). Es el bloqueador principal para empezar el adaptador del guante.
3. **Vocabulario de palabras para la demo** — nunca se decidió en firme. Se consideró "números 1-10" como candidato de bajo riesgo (bien documentado, estructuralmente parecido al alfabeto), y el dataset de 249 palabras de Mendeley (sección 4) como posible atajo — pero nada de esto está confirmado ni grabado.

---

## 10. Checkpoints de git

(Igual que en el reporte de Code — commits **locales**, recordar hacer `git push`)

| Hash | Qué representa |
|---|---|
| `2f7b3b638e79abd21a62f428484f2e514be685fc` | Checkpoint antes de agregar la regla experimental de margen para K/Q/Z (esa regla ya está aplicada en el código actual en disco, pendiente de un commit que la incluya) |
| `f9d769f0d976bc2481c64d2235e75b60dc92d386` | Antes de restringir qué letras dinámicas se comprometen (`DYN_COMMIT_LETTERS`) |
| `ad30cc697ee0f9c0b3be7a340fc505dc31ab3fd4` | Antes de cambiar `recolector_dinamico.py` a inicio/fin explícito con `g` (antes era "mantener presionada") |
| `f12216d411c20f9104ade24db65691ea76370b0b` | Antes de arreglar el choque entre el comando de salida y las etiquetas de una letra (ej. escribir "Q" para grabar esa letra activaba "salir") |
| `f6dcafb3847d42023d5e20d39d56bf24c67a97e0` | Antes de los ajustes de UX del modo dinámico (textos de estado, panel de diagnóstico) |
| `eea588aee3d8c2f13f17201eac897927f3229a10` | Antes de integrar el alfabeto dinámico a `senas.py` por primera vez — **punto de retorno al alfabeto estático puro de Josué** si el modo dinámico da problemas graves el día de la competencia |
| `dedce1136de64ad56d79097d7a53e324caf6da92` | Entrega original de Josué, sin ningún cambio |

```bat
git checkout <hash> -- senas.py
```

---

## 11. Para la presentación / cita académica

Si el jurado pregunta de dónde salió el reconocimiento de letras dinámicas: dataset abierto de CICESE, licencia CC BY 4.0, DOI `10.5281/zenodo.14689869` — **verificar la cita completa (autores, título exacto) directamente en esa página de Zenodo antes de la presentación**, para no improvisar una cita incorrecta frente al jurado.
