"""Clasificador ONNX para señas estáticas (palabras de 1 o 2 manos).

Proporciona la clase WordClassifier con la MISMA interfaz que SignClassifier,
adaptada a vectores de características de 126 valores (63 por cada mano:
izquierda y derecha).

Mantiene total consistencia con `sign_classifier.py`:
- Carga `lsm_words.onnx` y `word_labels.json`.
- Método `predict_topk(vector, k=3) -> list[tuple[palabra, confianza]]`.
- Método `predict(vector) -> tuple[palabra, confianza]`.
- Normalización idempotente por mano (wrist centered, middle finger scaled).
- Opcionalmente incluye PredictionSmoother para estabilizar predicciones entre frames.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

log = logging.getLogger("word_classifier")

MODEL_FILENAME = "lsm_words.onnx"
LABELS_FILENAME = "word_labels.json"
N_FEATURES = 126
N_FEATURES_PER_HAND = 63


# =========================================================================== #
# Normalización de Keypoints para 2 manos (126 features)
# =========================================================================== #

def normalize_hand_keypoints(vec: np.ndarray) -> np.ndarray:
    """Normaliza un vector de 63 features correspondiente a una sola mano.
    
    Centra las coordenadas en la muñeca (landmark 0) y escala por la distancia
    a la base del dedo medio (landmark 9). Idéntico a normalize_keypoints de
    sign_classifier.py.
    """
    out = vec.astype(np.float32).copy()
    pts = out.reshape(21, 3)

    wrist_xy = pts[0, :2].copy()
    pts[:, :2] -= wrist_xy

    base_middle_xy = pts[9, :2]
    scale = float(np.linalg.norm(base_middle_xy))
    if scale > 1e-6:
        pts[:, :2] /= scale

    return pts.reshape(63)


def normalize_word_keypoints(vec: np.ndarray) -> np.ndarray:
    """Normaliza un vector completo de 126 features [Mano Izq (63) + Mano Der (63)].
    
    Cada mano se normaliza por separado solo si está presente (valores no nulos).
    Si una mano no fue detectada (llena de ceros), se mantiene en ceros.
    Es una operación idempotente.
    """
    if vec.shape != (N_FEATURES,):
        raise ValueError(
            f"Se esperaba vector de dimension ({N_FEATURES},), recibido {vec.shape}"
        )

    out = vec.astype(np.float32).copy()
    left_hand = out[:N_FEATURES_PER_HAND]
    if np.any(left_hand):
        out[:N_FEATURES_PER_HAND] = normalize_hand_keypoints(left_hand)

    right_hand = out[N_FEATURES_PER_HAND:]
    if np.any(right_hand):
        out[N_FEATURES_PER_HAND:] = normalize_hand_keypoints(right_hand)

    return out


def hands_to_feature_vector(hands: dict[str, Any]) -> np.ndarray:
    """Convierte un diccionario {'Left': HandDetection, 'Right': HandDetection} a vector 126."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for slot_idx, handedness in enumerate(("Left", "Right")):
        hand = hands.get(handedness)
        if hand is None:
            continue
        if hasattr(hand, "landmarks_2d") and hasattr(hand, "landmarks_3d"):
            lm_2d = hand.landmarks_2d
            lm_3d = hand.landmarks_3d
            raw = np.zeros(N_FEATURES_PER_HAND, dtype=np.float32)
            for i in range(21):
                raw[i * 3] = float(lm_2d[i, 0])
                raw[i * 3 + 1] = float(lm_2d[i, 1])
                raw[i * 3 + 2] = float(lm_3d[i, 2])
            norm = normalize_hand_keypoints(raw)
            offset = slot_idx * N_FEATURES_PER_HAND
            vec[offset : offset + N_FEATURES_PER_HAND] = norm
    return vec


# =========================================================================== #
# Estabilizador temporal de predicciones (Smoothing)
# =========================================================================== #

@dataclass
class SmoothedPrediction:
    word: Optional[str]
    confidence: float
    raw_top1: Optional[tuple[str, float]] = None
    raw_top2: Optional[tuple[str, float]] = None
    raw_top3: Optional[tuple[str, float]] = None
    margin: float = 0.0


class WordPredictionSmoother:
    """Estabilizador por ventana deslizante idéntico a PredictionSmoother."""

    def __init__(
        self,
        window_size: int = 7,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
        per_word_confidence: Optional[dict[str, float]] = None,
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.per_word_confidence = dict(per_word_confidence) if per_word_confidence else {}
        self._preds: deque[tuple[str, float, float]] = deque(maxlen=window_size)
        self._last_topk: list[tuple[str, float]] = []

    def reset(self) -> None:
        self._preds.clear()
        self._last_topk = []

    def push(self, topk: list[tuple[str, float]]) -> SmoothedPrediction:
        if len(topk) < 2:
            raise ValueError("topk debe tener al menos 2 entradas")

        top1_word, top1_conf = topk[0]
        top2_word, top2_conf = topk[1]
        margin = top1_conf - top2_conf
        self._preds.append((top1_word, top1_conf, top2_conf))
        self._last_topk = list(topk)

        raw_top1 = (top1_word, top1_conf)
        raw_top2 = (top2_word, top2_conf)
        raw_top3 = topk[2] if len(topk) >= 3 else None

        if not self._preds:
            return SmoothedPrediction(
                word=None,
                confidence=0.0,
                raw_top1=raw_top1,
                raw_top2=raw_top2,
                raw_top3=raw_top3,
                margin=margin,
            )

        counter = Counter(p[0] for p in self._preds)
        winner, votes = counter.most_common(1)[0]

        winner_records = [p for p in self._preds if p[0] == winner]
        avg_conf = sum(r[1] for r in winner_records) / len(winner_records)
        avg_margin = sum(r[1] - r[2] for r in winner_records) / len(winner_records)

        majority = votes > (self.window_size // 2)
        conf_threshold = self.per_word_confidence.get(winner, self.min_confidence)
        confident_enough = avg_conf >= conf_threshold
        unambiguous = avg_margin >= self.min_margin

        result_word = winner if (majority and confident_enough and unambiguous) else None

        return SmoothedPrediction(
            word=result_word,
            confidence=avg_conf,
            raw_top1=raw_top1,
            raw_top2=raw_top2,
            raw_top3=raw_top3,
            margin=margin,
        )

    @property
    def last_topk(self) -> list[tuple[str, float]]:
        return list(self._last_topk)


# =========================================================================== #
# Clasificador principal de palabras
# =========================================================================== #

class WordClassifier:
    """Clasificador ONNX para palabras estáticas (1 o 2 manos, 126 features)."""

    def __init__(self, session: Any, labels: list[str]):
        self._session = session
        self._labels = labels
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    @classmethod
    def try_load(
        cls,
        models_dir: Path | None = None,
        model_filename: str = MODEL_FILENAME,
        labels_filename: str = LABELS_FILENAME,
    ) -> Optional["WordClassifier"]:
        """Intenta cargar el clasificador desde los archivos ONNX y JSON correspondientes."""
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent

        onnx_path = models_dir / model_filename
        labels_path = models_dir / labels_filename

        if not onnx_path.exists():
            log.info("WordClassifier no disponible: falta %s", onnx_path)
            return None
        if not labels_path.exists():
            log.info("WordClassifier no disponible: falta %s", labels_path)
            return None

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            log.warning(
                "onnxruntime no está instalado. Instálalo con: pip install onnxruntime"
            )
            return None

        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            labels = data["labels"]
            if not isinstance(labels, list) or not labels:
                raise ValueError("labels debe ser una lista no vacía")
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            log.warning("No se pudo leer %s: %s", labels_path, e)
            return None

        try:
            session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            log.warning("No se pudo cargar el modelo ONNX %s: %s", onnx_path, e)
            return None

        log.info("WordClassifier LSM cargado: %d palabras", len(labels))
        return cls(session, labels)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def predict_topk(
        self, feature_vec: np.ndarray, k: int = 3
    ) -> list[tuple[str, float]]:
        """Predice las k palabras más probables junto con su nivel de confianza."""
        if feature_vec.shape != (N_FEATURES,):
            raise ValueError(
                f"Se esperaba shape ({N_FEATURES},), recibido {feature_vec.shape}"
            )

        normalized = normalize_word_keypoints(feature_vec)
        batch = normalized.reshape(1, N_FEATURES).astype(np.float32)

        logits = self._session.run([self._output_name], {self._input_name: batch})[0]
        logits = logits[0]  # (n_classes,)

        logits_max = float(np.max(logits))
        exp = np.exp(logits - logits_max)
        probs = exp / exp.sum()

        k = min(k, len(probs))
        top_idx = np.argsort(probs)[::-1][:k]
        return [(self._labels[int(i)], float(probs[int(i)])) for i in top_idx]

    def predict(self, feature_vec: np.ndarray) -> tuple[str, float]:
        """Predice la palabra más probable (Top-1) y su confianza."""
        topk = self.predict_topk(feature_vec, k=1)
        return topk[0]

    def predict_from_hands(self, hands: dict[str, Any]) -> tuple[str, float]:
        """Predice a partir de un diccionario de manos {'Left': ..., 'Right': ...}."""
        vec = hands_to_feature_vector(hands)
        return self.predict(vec)

    def predict_topk_from_hands(
        self, hands: dict[str, Any], k: int = 3
    ) -> list[tuple[str, float]]:
        """Predice top-k a partir de un diccionario de manos."""
        vec = hands_to_feature_vector(hands)
        return self.predict_topk(vec, k=k)


# =========================================================================== #
# Mini prueba autónoma si se ejecuta directamente
# =========================================================================== #

if __name__ == "__main__":
    import sys

    print("=== Mini Prueba Autónoma: WordClassifier ===")
    classifier = WordClassifier.try_load()
    if classifier is None:
        print("ERROR: No se pudo cargar WordClassifier. ¿Ya corriste entrenar_palabras.py?")
        sys.exit(1)

    print(f"Palabras soportadas ({len(classifier.labels)}): {classifier.labels}")

    # Vector aleatorio simulando postura de 2 manos
    test_vec = np.random.randn(126).astype(np.float32)
    top3 = classifier.predict_topk(test_vec, k=3)
    best_word, best_conf = classifier.predict(test_vec)

    print("\nResultado para vector de prueba:")
    print(f"  Top 1: {best_word} (Confianza: {best_conf * 100:.2f}%)")
    print("  Top K:")
    for rank, (word, conf) in enumerate(top3, 1):
        print(f"    {rank}. {word:15s} -> {conf * 100:6.2f}%")
    print("\n Prueba de WordClassifier completada con éxito.")
