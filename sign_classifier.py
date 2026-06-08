from __future__ import annotations

import json
import logging
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger("sign_classifier")


MODEL_FILENAME = "lsm_alphabet.onnx"
LABELS_FILENAME = "lsm_labels.json"


def normalize_keypoints(vec: np.ndarray) -> np.ndarray:
    out = vec.astype(np.float32).copy()
    pts = out.reshape(21, 3)

    wrist_xy = pts[0, :2].copy()
    pts[:, :2] -= wrist_xy

    base_middle_xy = pts[9, :2]
    scale = float(np.linalg.norm(base_middle_xy))
    if scale > 1e-6:
        pts[:, :2] /= scale

    return pts.reshape(63)



def hand_to_feature_vector(landmarks_2d: np.ndarray, landmarks_3d: np.ndarray) -> np.ndarray:
    vec = np.zeros(63, dtype=np.float32)
    for i in range(21):
        vec[i * 3]     = float(landmarks_2d[i, 0])
        vec[i * 3 + 1] = float(landmarks_2d[i, 1])
        vec[i * 3 + 2] = float(landmarks_3d[i, 2])
    return vec



@dataclass
class SmoothedPrediction:
    letter: Optional[str]   
    confidence: float      
    raw_top1: Optional[tuple[str, float]] = None    
    raw_top2: Optional[tuple[str, float]] = None    
    raw_top3: Optional[tuple[str, float]] = None    
    margin: float = 0.0     

class PredictionSmoother:
    
    def __init__(
        self,
        window_size: int = 7,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
        per_letter_confidence: Optional[dict[str, float]] = None,
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.per_letter_confidence = dict(per_letter_confidence) if per_letter_confidence else {}
        self._preds: deque[tuple[str, float, float]] = deque(maxlen=window_size)
        self._last_topk: list[tuple[str, float]] = []

    def reset(self) -> None:
        self._preds.clear()
        self._last_topk = []

    def push(
        self,
        topk: list[tuple[str, float]],
    ) -> SmoothedPrediction:
        if len(topk) < 2:
            raise ValueError("topk debe tener al menos 2 entradas")

        top1_letter, top1_conf = topk[0]
        top2_letter, top2_conf = topk[1]
        margin = top1_conf - top2_conf
        self._preds.append((top1_letter, top1_conf, top2_conf))
        self._last_topk = list(topk)

        raw_top1 = (top1_letter, top1_conf)
        raw_top2 = (top2_letter, top2_conf)
        raw_top3 = topk[2] if len(topk) >= 3 else None

        if not self._preds:
            return SmoothedPrediction(
                letter=None, confidence=0.0,
                raw_top1=raw_top1, raw_top2=raw_top2, raw_top3=raw_top3,
                margin=margin,
            )

        counter = Counter(p[0] for p in self._preds)
        winner, votes = counter.most_common(1)[0]

        winner_records = [p for p in self._preds if p[0] == winner]
        avg_conf = sum(r[1] for r in winner_records) / len(winner_records)
        avg_margin = sum(r[1] - r[2] for r in winner_records) / len(winner_records)

        majority = votes > (self.window_size // 2)
        conf_threshold = self.per_letter_confidence.get(winner, self.min_confidence)
        confident_enough = avg_conf >= conf_threshold
        unambiguous = avg_margin >= self.min_margin

        result_letter = winner if (majority and confident_enough and unambiguous) else None

        return SmoothedPrediction(
            letter=result_letter,
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
# Clasificador principal
# =========================================================================== #

class SignClassifier:
    
    def __init__(self, session, labels: list[str]):
        self._session = session
        self._labels = labels
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    @classmethod
    def try_load(cls, models_dir: Path | None = None) -> Optional["SignClassifier"]:
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent

        onnx_path = models_dir / MODEL_FILENAME
        labels_path = models_dir / LABELS_FILENAME

        if not onnx_path.exists():
            log.info("Clasificador no disponible: falta %s", onnx_path)
            return None
        if not labels_path.exists():
            log.info("Clasificador no disponible: falta %s", labels_path)
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
                raise ValueError("labels debe ser lista no vacía")
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

        log.info("Clasificador LSM cargado: %d clases", len(labels))
        return cls(session, labels)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def predict_topk(self, feature_vec: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        if feature_vec.shape != (63,):
            raise ValueError(f"Se esperaba shape (63,), recibido {feature_vec.shape}")

        normalized = normalize_keypoints(feature_vec)
        batch = normalized.reshape(1, 63).astype(np.float32)

        logits = self._session.run([self._output_name], {self._input_name: batch})[0]
        logits = logits[0]                          # (n_classes,)

        logits_max = float(np.max(logits))
        exp = np.exp(logits - logits_max)
        probs = exp / exp.sum()

        k = min(k, len(probs))
        top_idx = np.argsort(probs)[::-1][:k]
        return [(self._labels[int(i)], float(probs[int(i)])) for i in top_idx]

    def predict(self, feature_vec: np.ndarray) -> tuple[str, float]:
        topk = self.predict_topk(feature_vec, k=1)
        return topk[0]

    def predict_from_hand(self, landmarks_2d: np.ndarray, landmarks_3d: np.ndarray) -> tuple[str, float]:
        vec = hand_to_feature_vector(landmarks_2d, landmarks_3d)
        return self.predict(vec)

    def predict_topk_from_hand(
        self,
        landmarks_2d: np.ndarray,
        landmarks_3d: np.ndarray,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        vec = hand_to_feature_vector(landmarks_2d, landmarks_3d)
        return self.predict_topk(vec, k=k)