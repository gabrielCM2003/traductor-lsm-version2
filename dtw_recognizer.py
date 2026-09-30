"""Reconocedor de señas dinámicas (con movimiento) usando Dynamic Time Warping (DTW) optimizado.

Compara una secuencia temporal de vectores de 126 valores (Left hand + Right hand)
contra un conjunto de plantillas de referencia guardadas en `datos_dinamicas/<palabra>/`.

Optimización de rendimiento:
- Precalcula la matriz de costos con `scipy.spatial.distance.cdist` (BLAS/C).
- Resuelve la programación dinámica y el camino de deformación de forma compilada con `numba`
  (con fallback transparente a NumPy vectorizado en caso de que Numba no esté presente).
- Reduce la latencia de inferencia de ~4.4 segundos a ~0.08 - 0.11 segundos contra 558 plantillas
  (~40x más rápido), liberando el GIL y evitando bloquear el hilo de la cámara.

Mantiene la consistencia de interfaz pública con el resto del proyecto:
    DTWRecognizer.predict_topk(secuencia, k=3) -> list[tuple[palabra, confianza/distancia]]
    DTWRecognizer.predict(secuencia) -> tuple[palabra, confianza/distancia]
"""
from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

# Detección condicional de Numba para máxima aceleración
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

log = logging.getLogger("dtw_recognizer")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"
DEFAULT_LABELS_FILE = Path(__file__).resolve().parent / "labels_dinamicas.json"
N_FEATURES = 126
# Bloque de ubicacion respecto al cuerpo ("body_frames" de las muestras de
# palabras): body_tracker.N_BODY_FEATURES. No se importa de alli porque ese
# modulo carga mediapipe y este no lo necesita.
N_BODY_FEATURES = 9

# Palabras completas: plantillas en datos_palabras_dinamicas/<PALABRA>/ (las
# graba segmentador_automatico.py --modo palabras --grabar o las extrae
# extraer_palabras_videos.py). Se comparan con la ubicacion respecto al cuerpo
# multiplicada por WORD_BODY_WEIGHT: deja-uno-fuera con las 59 muestras de
# video de HOLA, GRACIAS, POR FAVOR, AYUDA y MAMA dio 89.8% solo con manos,
# 93.2% con peso 1, 96.6% con 2 y 98.3% con 4.
DEFAULT_WORDS_DIR = Path(__file__).resolve().parent / "datos_palabras_dinamicas"
WORD_BODY_WEIGHT = 4.0
# Temperatura del softmax de las palabras (distances_to_topk), ajustada para
# que la confianza sea una probabilidad calibrada: minimiza la log-verosimilitud
# negativa de la palabra correcta reconociendo a cada persona de los videos
# SOLO con las plantillas de las otras dos (3 personas). Optimo 0.50 (NLL
# 0.27, contra 0.36 con 1.0). Con 1.0 la confianza quedaba por debajo del
# acierto real y muchas palabras correctas se tomaban como dudosas; con 0.5,
# de las que salen con 80% o mas se acierta mas del 92%. Las letras
# dinamicas siguen con 1.0 (sus reglas se calibraron con esa escala).
WORD_TEMPERATURE = 0.5


# =========================================================================== #
# Programación Dinámica DTW acelerada (Numba / NumPy)
# =========================================================================== #

if HAVE_NUMBA:
    # nogil: senas.py clasifica en un hilo aparte para no congelar el video;
    # sin soltar el candado de Python, ese hilo lo acaparaba igual durante los
    # varios segundos que tarda en maquinas lentas (y el watchdog reiniciaba
    # el hilo de deteccion a medio reconocimiento).
    @njit(fastmath=True, nogil=True)
    def _dtw_dp_numba(cost_matrix: np.ndarray) -> float:
        """Cálculo DTW y longitud de path con Numba a nivel de C."""
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                prev_min = dp[i - 1, j - 1]
                if dp[i - 1, j] < prev_min:
                    prev_min = dp[i - 1, j]
                if dp[i, j - 1] < prev_min:
                    prev_min = dp[i, j - 1]
                dp[i, j] = cost_matrix[i - 1, j - 1] + prev_min

        total_dist = dp[n, m]

        # Backtracking para normalizar exactamente por longitud de camino (warp path length)
        i, j = n, m
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            diag = dp[i - 1, j - 1]
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1
            i -= 1
        while j > 0:
            path_len += 1
            j -= 1

        return total_dist / max(1, path_len)

    def _compute_dtw_distance(cost_matrix: np.ndarray) -> float:
        return float(_dtw_dp_numba(cost_matrix))
else:
    def _compute_dtw_distance(cost_matrix: np.ndarray) -> float:
        """Fallback en NumPy si Numba no está disponible."""
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            c_row = cost_matrix[i - 1]
            d_prev = dp[i - 1]
            d_curr = dp[i]
            for j in range(1, m + 1):
                d_curr[j] = c_row[j - 1] + min(d_prev[j - 1], d_prev[j], d_curr[j - 1])

        total_dist = dp[n, m]

        i, j = n, m
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            diag, up, left = dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1
            i -= 1
        while j > 0:
            path_len += 1
            j -= 1

        return float(total_dist / max(1, path_len))


# =========================================================================== #
# Utilidades de transformación y remuestreo de secuencias
# =========================================================================== #

def mirror_and_swap_hands(sequence: np.ndarray) -> np.ndarray:
    """Intercambia bloque izquierdo (63) y derecho (63) y refleja el eje X de cada mano.
    
    Permite comparar secuencias realizadas con la mano contraria a las plantillas.
    Si la secuencia trae ademas el bloque de cuerpo (126 + 9 columnas), tambien
    intercambia sus dos slots de mano y niega sus dx; la bandera queda igual.
    """
    out = np.array(sequence, copy=True)
    out[:, :63] = sequence[:, 63:126]
    out[:, 63:126] = sequence[:, :63]
    for i in range(21):
        out[:, i * 3] = -out[:, i * 3]
        out[:, 63 + i * 3] = -out[:, 63 + i * 3]
    if sequence.shape[1] == N_FEATURES + N_BODY_FEATURES:
        b = N_FEATURES
        out[:, b:b + 4] = sequence[:, b + 4:b + 8]
        out[:, b + 4:b + 8] = sequence[:, b:b + 4]
        for col in (b, b + 2, b + 4, b + 6):
            out[:, col] = -out[:, col]
    return out


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Interpola linealmente la secuencia a una longitud fija de frames."""
    n_frames, n_feats = sequence.shape
    if n_frames == target_len:
        return sequence
    x_old = np.linspace(0.0, 1.0, n_frames)
    x_new = np.linspace(0.0, 1.0, target_len)
    resampled = np.zeros((target_len, n_feats), dtype=sequence.dtype)
    for f in range(n_feats):
        resampled[:, f] = np.interp(x_new, x_old, sequence[:, f])
    return resampled


def distances_to_topk(
    distances: dict[str, float], k: int = 3, temperature: float = 1.0
) -> list[tuple[str, float]]:
    """{seña: distancia DTW} -> top-k [(seña, confianza)], con la confianza
    como softmax de -distancia/temperatura (la distancia menor gana). Es lo
    que devuelve predict_topk; separado para quien ya calculo las distancias
    (el modo automatico de senas.py las necesita tambien crudas)."""
    sorted_candidates = sorted(distances.items(), key=lambda item: item[1])
    k = min(k, len(sorted_candidates))
    words = [w for w, _ in sorted_candidates]
    dists = np.array([d for _, d in sorted_candidates], dtype=np.float64)
    logits = -dists / max(1e-4, temperature)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return [(words[i], float(probs[i])) for i in range(k)]


# =========================================================================== #
# Clase principal de reconocimiento DTW
# =========================================================================== #

class DTWRecognizer:
    """Reconocedor DTW basado en plantillas multi-muestra para señas dinámicas."""

    def __init__(
        self,
        data_dir: Optional[Union[Path, str]] = None,
        labels_path: Optional[Union[Path, str]] = None,
        auto_save_labels: bool = True,
        resample_len: Optional[int] = None,
        hand_agnostic: bool = False,
        body_weight: Optional[float] = None,
    ):
        """Inicializa el reconocedor y carga en memoria todas las plantillas disponibles.

        Args:
            data_dir: Carpeta raíz donde están las subcarpetas de palabras
                      (ej. datos_dinamicas/<palabra>/muestra_*.json).
            labels_path: Ruta del archivo JSON con las etiquetas soportadas.
            auto_save_labels: Si es True, actualiza automáticamente el JSON de
                              etiquetas con las carpetas de palabras encontradas.
            resample_len: Opcional. Si se especifica, remuestrea las plantillas y
                          consultas a una longitud fija de frames.
            hand_agnostic: Si es True, evalúa la consulta tanto normal como en
                           espejo (con manos intercambiadas) y toma la menor distancia.
                           Por defecto es False.
            body_weight: None (por defecto) compara solo los 126 de las manos,
                         como el alfabeto dinamico. Con un numero, compara
                         tambien la ubicacion respecto al cuerpo ("body_frames",
                         9 por frame) multiplicada por ese peso: lo usan las
                         palabras, que se distinguen por DONDE se hacen. Las
                         consultas traen entonces 126 + 9 columnas y las
                         plantillas sin "body_frames" se saltan.
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_FILE
        self.resample_len = resample_len
        self.hand_agnostic = hand_agnostic
        self.body_weight = body_weight
        self.n_features = N_FEATURES + (N_BODY_FEATURES if body_weight is not None else 0)

        # Diccionario con estructura: { 'palabra': [array(T1, 126), array(T2, 126), ...] }
        self._templates: dict[str, list[np.ndarray]] = {}
        self._labels: list[str] = []

        self.load_templates()

        if auto_save_labels and self._labels:
            self.save_labels_json(self.labels_path)

        # Calentamiento inicial de Numba si está disponible
        if HAVE_NUMBA:
            _dtw_dp_numba(np.zeros((4, 4), dtype=np.float64))

    @classmethod
    def try_load(
        cls,
        data_dir: Optional[Union[Path, str]] = None,
        labels_path: Optional[Union[Path, str]] = None,
        resample_len: Optional[int] = None,
        hand_agnostic: bool = False,
        body_weight: Optional[float] = None,
    ) -> Optional["DTWRecognizer"]:
        """Intenta instanciar DTWRecognizer de manera segura.
        
        Retorna None si el directorio no existe o no tiene plantillas válidas.
        """
        try:
            recognizer = cls(
                data_dir=data_dir,
                labels_path=labels_path,
                resample_len=resample_len,
                hand_agnostic=hand_agnostic,
                body_weight=body_weight,
            )
            if not recognizer.labels:
                log.warning("DTWRecognizer no disponible: no se encontraron plantillas.")
                return None
            return recognizer
        except Exception as e:
            log.warning("No se pudo inicializar DTWRecognizer: %s", e)
            return None

    def load_templates(self) -> None:
        """Escanea `data_dir` y carga todas las secuencias de plantillas disponibles."""
        self._templates.clear()

        if not self.data_dir.exists():
            log.info("Directorio de datos dinámicos no existe: %s", self.data_dir)
            self._labels = []
            return

        for word_dir in sorted(self.data_dir.iterdir()):
            if not word_dir.is_dir():
                continue

            # macOS puede entregar el nombre de la carpeta "Ñ" descompuesto
            # (N + tilde combinable): sin normalizar, esa etiqueta no era
            # igual a la "Ñ" de senas.py (DYN_NORMAL_LETTERS) y se ordenaba
            # distinto que en Windows.
            word = unicodedata.normalize("NFC", word_dir.name)
            samples: list[np.ndarray] = []

            # 1. Buscar archivos JSON (formato estándar de recolector_dinamico.py)
            for json_file in sorted(word_dir.glob("*.json")):
                try:
                    content = json.loads(json_file.read_text(encoding="utf-8"))
                    frames = content.get("frames", content.get("sequence", content))
                    arr = np.array(frames, dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] == N_FEATURES and self.body_weight is not None:
                        body = np.array(content.get("body_frames", []), dtype=np.float64)
                        if body.shape != (len(arr), N_BODY_FEATURES):
                            log.warning(
                                "Archivo %s sin body_frames validos (%s): se salta",
                                json_file.name, body.shape,
                            )
                            continue
                        arr = np.hstack([arr, body])
                    if arr.ndim == 2 and arr.shape[1] == self.n_features and len(arr) > 0:
                        samples.append(self._prepare(arr))
                    else:
                        log.warning(
                            "Archivo %s con dimensiones no válidas: %s",
                            json_file.name,
                            arr.shape,
                        )
                except Exception as e:
                    log.warning("Error al leer %s: %s", json_file, e)

            # 2. Buscar archivos NPY si los hubiera
            for npy_file in sorted(word_dir.glob("*.npy")):
                try:
                    arr = np.load(str(npy_file)).astype(np.float64)
                    if arr.ndim == 2 and arr.shape[1] == self.n_features and len(arr) > 0:
                        samples.append(self._prepare(arr))
                except Exception as e:
                    log.warning("Error al leer %s: %s", npy_file, e)

            # 3. Buscar archivos CSV si los hubiera
            for csv_file in sorted(word_dir.glob("*.csv")):
                try:
                    arr = np.loadtxt(str(csv_file), delimiter=",").astype(np.float64)
                    if arr.ndim == 2 and arr.shape[1] == self.n_features and len(arr) > 0:
                        samples.append(self._prepare(arr))
                except Exception as e:
                    log.warning("Error al leer %s: %s", csv_file, e)

            if samples:
                # extend y no =: en Linux una carpeta "Ñ" compuesta y otra
                # descompuesta son dos carpetas distintas con la misma
                # etiqueta, y la segunda borraba las plantillas de la primera.
                self._templates.setdefault(word, []).extend(samples)
                log.debug("Cargadas %d plantillas para '%s'", len(samples), word)

        self._labels = sorted(list(self._templates.keys()))
        log.info(
            "DTWRecognizer inicializado: %d señas dinámicas (%d plantillas totales, Numba=%s)",
            len(self._labels),
            sum(len(v) for v in self._templates.values()),
            HAVE_NUMBA,
        )

    @property
    def labels(self) -> list[str]:
        """Lista de etiquetas de señas dinámicas soportadas."""
        return list(self._labels)

    @property
    def templates_count(self) -> dict[str, int]:
        """Cantidad de plantillas cargadas por cada seña."""
        return {word: len(tmpls) for word, tmpls in self._templates.items()}

    def save_labels_json(self, output_path: Optional[Path] = None) -> None:
        """Guarda la lista de etiquetas dinámicas en formato estándar JSON."""
        target = Path(output_path) if output_path else self.labels_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labels": self._labels,
            "n_features": self.n_features,
            "normalization": "wrist_centered_middle_scaled",
        }
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log.info("Etiquetas dinámicas guardadas en: %s", target)

    def _prepare(self, arr: np.ndarray) -> np.ndarray:
        """Secuencia cruda (T, n_features) -> lo que compara el DTW: con el
        bloque de cuerpo multiplicado por body_weight y, si se pidio,
        remuestreada. Se aplica igual a plantillas y consultas."""
        arr = np.array(arr, dtype=np.float64)
        if self.body_weight is not None:
            arr[:, N_FEATURES:] *= self.body_weight
        if self.resample_len is not None:
            arr = resample_sequence(arr, self.resample_len)
        return arr

    def _validate_sequence(
        self, sequence: Union[np.ndarray, List[List[float]], List[np.ndarray]]
    ) -> np.ndarray:
        """Convierte y valida que la secuencia de entrada tenga shape (T, n_features)."""
        if not isinstance(sequence, np.ndarray):
            arr = np.array(sequence, dtype=np.float64)
        else:
            arr = sequence.astype(np.float64)

        if arr.ndim != 2:
            raise ValueError(
                f"La secuencia debe tener 2 dimensiones (T, {self.n_features}), recibido ndim={arr.ndim}"
            )
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"Se esperaba {self.n_features} características por frame, recibido {arr.shape[1]}"
            )
        if len(arr) == 0:
            raise ValueError("La secuencia recibida está vacía.")

        return self._prepare(arr)

    def compute_distances(self, sequence: np.ndarray) -> dict[str, float]:
        """Calcula la distancia DTW normalizada mínima hacia cada seña registrada.
        
        Utiliza scipy.spatial.distance.cdist para precalcular la matriz de costos
        una sola vez por plantilla y programación dinámica compilada (Numba/NumPy).
        """
        return self._distances(self._validate_sequence(sequence))

    def leave_one_out(self) -> list[tuple[str, str]]:
        """(etiqueta real, etiqueta predicha) de cada plantilla reconocida
        contra todas las DEMAS: estima el acierto sin grabar datos aparte."""
        results = []
        for word, template_list in self._templates.items():
            for i in range(len(template_list)):
                query = template_list.pop(i)
                try:
                    distances = self._distances(query)
                finally:
                    template_list.insert(i, query)
                results.append((word, min(distances, key=distances.get)))
        return results

    def _distances(self, sec_arr: np.ndarray) -> dict[str, float]:
        mirrored_arr = mirror_and_swap_hands(sec_arr) if self.hand_agnostic else None
        distances: dict[str, float] = {}

        for word, template_list in self._templates.items():
            min_dist = float("inf")

            for tmpl in template_list:
                # Matriz de costos pairwise euclídea vectorizada en C
                cost_mat = cdist(sec_arr, tmpl, metric="euclidean")
                norm_dist = _compute_dtw_distance(cost_mat)

                # Comparación agnóstica a la mano (si está habilitada)
                if self.hand_agnostic and mirrored_arr is not None:
                    cost_mat_m = cdist(mirrored_arr, tmpl, metric="euclidean")
                    norm_dist_m = _compute_dtw_distance(cost_mat_m)
                    if norm_dist_m < norm_dist:
                        norm_dist = norm_dist_m

                if norm_dist < min_dist:
                    min_dist = norm_dist

            distances[word] = min_dist

        return distances

    def predict_topk(
        self,
        sequence: Union[np.ndarray, List[List[float]], List[np.ndarray]],
        k: int = 3,
        return_distance: bool = False,
        temperature: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Predice las top-k señas dinámicas más probables.

        Args:
            sequence: Secuencia temporal de vectores de 126 (forma (T, 126)).
            k: Número de candidatos a retornar.
            return_distance: Si es True, retorna la distancia normalizada en vez de
                             la confianza normalizada (menor distancia = mejor match).
            temperature: Factor de escala de temperatura para convertir distancias
                         en probabilidades tipo softmax.

        Returns:
            Lista de tuplas (palabra, confianza) o (palabra, distancia) si return_distance=True.
        """
        if not self._templates:
            raise RuntimeError(
                "No hay plantillas cargadas en DTWRecognizer. Añade datos a datos_dinamicas/."
            )

        distances = self.compute_distances(sequence)
        # Ordenar de menor distancia a mayor distancia (menor distancia es mejor)
        sorted_candidates = sorted(distances.items(), key=lambda item: item[1])
        k = min(k, len(sorted_candidates))

        if return_distance:
            return [(word, dist) for word, dist in sorted_candidates[:k]]

        return distances_to_topk(distances, k, temperature)

    def predict(
        self,
        sequence: Union[np.ndarray, List[List[float]], List[np.ndarray]],
        return_distance: bool = False,
    ) -> tuple[str, float]:
        """Predice la seña dinámica más probable (Top-1)."""
        topk = self.predict_topk(sequence, k=1, return_distance=return_distance)
        return topk[0]


# =========================================================================== #
# Mini prueba autónoma si se ejecuta directamente
# =========================================================================== #

if __name__ == "__main__":
    import sys
    import time

    print("=== Mini Prueba Autónoma: DTWRecognizer (Optimizado) ===")
    recognizer = DTWRecognizer.try_load()
    if recognizer is None:
        print("ERROR: No se pudo inicializar DTWRecognizer. Verifica datos_dinamicas/.")
        sys.exit(1)

    print(f"Señas disponibles ({len(recognizer.labels)}): {recognizer.labels}")
    print(f"Detalle de plantillas: {recognizer.templates_count}")

    # Probar con una muestra real del dataset y medir latencia
    sample_word = recognizer.labels[0]
    sample_template = recognizer._templates[sample_word][0]
    noisy_query = sample_template + np.random.randn(*sample_template.shape) * 0.05

    print(f"\nClasificando secuencia de prueba (base: '{sample_word}', {len(noisy_query)} frames)...")
    
    t0 = time.perf_counter()
    top_confs = recognizer.predict_topk(noisy_query, k=3, return_distance=False)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    best_word, best_conf = top_confs[0]
    print(f"\n[Confianza] Top 1: {best_word} ({best_conf * 100:.2f}%) | Latencia: {latency_ms:.2f} ms")
    for r, (w, c) in enumerate(top_confs, 1):
        print(f"  {r}. {w:15s} -> Confianza: {c * 100:6.2f}%")

    top_dists = recognizer.predict_topk(noisy_query, k=3, return_distance=True)
    print(f"\n[Distancia DTW] Top 1: {top_dists[0][0]} (Distancia: {top_dists[0][1]:.4f})")
    for r, (w, d) in enumerate(top_dists, 1):
        print(f"  {r}. {w:15s} -> Distancia: {d:8.4f}")

    print("\n Prueba de DTWRecognizer completada con éxito.")
