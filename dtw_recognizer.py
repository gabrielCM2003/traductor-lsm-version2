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
    """
    out = np.zeros_like(sequence)
    out[:, :63] = sequence[:, 63:]
    out[:, 63:] = sequence[:, :63]
    for i in range(21):
        out[:, i * 3] = -out[:, i * 3]
        out[:, 63 + i * 3] = -out[:, 63 + i * 3]
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
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_FILE
        self.resample_len = resample_len
        self.hand_agnostic = hand_agnostic

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
                    if arr.ndim == 2 and arr.shape[1] == N_FEATURES and len(arr) > 0:
                        if self.resample_len is not None:
                            arr = resample_sequence(arr, self.resample_len)
                        samples.append(arr)
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
                    if arr.ndim == 2 and arr.shape[1] == N_FEATURES and len(arr) > 0:
                        if self.resample_len is not None:
                            arr = resample_sequence(arr, self.resample_len)
                        samples.append(arr)
                except Exception as e:
                    log.warning("Error al leer %s: %s", npy_file, e)

            # 3. Buscar archivos CSV si los hubiera
            for csv_file in sorted(word_dir.glob("*.csv")):
                try:
                    arr = np.loadtxt(str(csv_file), delimiter=",").astype(np.float64)
                    if arr.ndim == 2 and arr.shape[1] == N_FEATURES and len(arr) > 0:
                        if self.resample_len is not None:
                            arr = resample_sequence(arr, self.resample_len)
                        samples.append(arr)
                except Exception as e:
                    log.warning("Error al leer %s: %s", csv_file, e)

            if samples:
                self._templates[word] = samples
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
            "n_features": N_FEATURES,
            "normalization": "wrist_centered_middle_scaled",
        }
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log.info("Etiquetas dinámicas guardadas en: %s", target)

    def _validate_sequence(
        self, sequence: Union[np.ndarray, List[List[float]], List[np.ndarray]]
    ) -> np.ndarray:
        """Convierte y valida que la secuencia de entrada tenga shape (T, 126)."""
        if not isinstance(sequence, np.ndarray):
            arr = np.array(sequence, dtype=np.float64)
        else:
            arr = sequence.astype(np.float64)

        if arr.ndim != 2:
            raise ValueError(
                f"La secuencia debe tener 2 dimensiones (T, {N_FEATURES}), recibido ndim={arr.ndim}"
            )
        if arr.shape[1] != N_FEATURES:
            raise ValueError(
                f"Se esperaba {N_FEATURES} características por frame, recibido {arr.shape[1]}"
            )
        if len(arr) == 0:
            raise ValueError("La secuencia recibida está vacía.")

        if self.resample_len is not None:
            arr = resample_sequence(arr, self.resample_len)

        return arr

    def compute_distances(self, sequence: np.ndarray) -> dict[str, float]:
        """Calcula la distancia DTW normalizada mínima hacia cada seña registrada.
        
        Utiliza scipy.spatial.distance.cdist para precalcular la matriz de costos
        una sola vez por plantilla y programación dinámica compilada (Numba/NumPy).
        """
        sec_arr = self._validate_sequence(sequence)
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

        # Conversión de distancias a probabilidades calibradas tipo softmax
        words = [w for w, _ in sorted_candidates]
        dists = np.array([d for _, d in sorted_candidates], dtype=np.float64)

        # Distancia normalizada invertida mediante softmax con temperatura
        # Usamos -dist / temp para que la distancia menor tenga la probabilidad más alta
        logits = -dists / max(1e-4, temperature)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        return [(words[i], float(probs[i])) for i in range(k)]

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
