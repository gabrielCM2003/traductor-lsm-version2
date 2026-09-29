"""Esqueleto del cuerpo (MediaPipe Pose Landmarker) para ubicar las manos
respecto a la persona.

El vector de 126 de las manos (normalize_keypoints en sign_classifier.py)
centra cada mano en su muneca y la escala por su tamano: describe la FORMA de
la mano, pero borra a proposito DONDE esta. Para el alfabeto eso esta bien,
pero para las palabras no alcanza (la misma forma de mano en la frente o en el
pecho son senas distintas). Este modulo agrega esa informacion como un bloque
APARTE de N_BODY_FEATURES valores (ver body_location_features), sin tocar el
vector de 126: el alfabeto estatico, el dinamico (DTW) y lsm_words.onnx
siguen funcionando igual.

Se usa MediaPipe Pose y no YOLOv8-pose porque viene en el mismo paquete
mediapipe que ya usa el proyecto (sin torch ni ultralytics, que ademas exige
opencv-python y lo reinstala encima del opencv-contrib-python de mediapipe),
y porque tiene puntos de la boca (9, 10) que el esqueleto COCO de YOLO no
tiene.

Ojo con el espejo: CameraThread (senas.py) y los recolectores voltean el
frame con cv2.flip ANTES de inferir. HandLandmarker asume imagen en espejo,
pero PoseLandmarker etiqueta izquierda/derecha como si NO lo estuviera
(medido: su "hombro izquierdo" cae del mismo lado de la imagen con y sin
espejo). O sea, los left/right de la pose y los "Left"/"Right" de las manos
no se corresponden. Por eso aqui solo se usan referencias SIMETRICAS (centro
y ancho de hombros, centro de la boca), y para unir cada brazo con su mano se
empareja por cercania en la imagen, nunca por etiqueta.
"""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import cv2
import numpy as np
import mediapipe as mp

log = logging.getLogger("body_tracker")

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{model}/float16/1/pose_landmarker_{model}.task"
)
POSE_MODELS = ("lite", "full")

# "full" por defecto: medido contra "lite" con la persona quieta, tiembla
# menos (0.34 px contra 0.88 px de desviacion) a cambio de ~2.6 ms mas por
# frame (8.8 contra 6.2 ms en una laptop). En la Raspberry Pi, si hace falta
# el tiempo de CPU, se puede bajar a "lite" (pose_model en la config de
# senas.py); conviene grabar y reconocer con el mismo modelo.
DEFAULT_POSE_MODEL = "full"
DEFAULT_MODELS_DIR = Path.home() / ".sign_translator" / "models"

# Indices del esqueleto de 33 puntos de MediaPipe Pose (solo los que se usan).
N_POSE_LANDMARKS = 33
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# Un punto cuenta como detectado (BodyDetection.visible) si sus coordenadas
# caen DENTRO de la imagen y visibility >= MIN_VISIBILITY. No se usa presence,
# aunque MediaPipe la da: medido con una persona real y los codos en cuadro
# cerca del borde, en modo VIDEO presence oscila frame a frame entre 0.21 y
# 0.76 con la imagen quieta (visibility estable en 0.94-0.99), asi que
# cualquier umbral hacia parpadear los brazos o los borraba (con 0.8 no se
# dibujaban nunca). Lo que si separa "en cuadro" de "fuera": recortando la
# imagen para dejar los codos fuera, MediaPipe los ubica fuera del rango
# [0, 1] (y = 1.08-1.87, x = -0.03 o 1.06).
MIN_VISIBILITY = 0.5

# Si los hombros miden menos que esto en pixeles, la persona esta demasiado
# lejos o de perfil y dividir entre ese ancho daria valores disparados.
MIN_SHOULDER_WIDTH_PX = 20.0

# Para unir un brazo con la mano detectada: la muneca de la mano debe estar a
# menos de esta fraccion del ancho de hombros de la muneca que estima la pose.
HAND_MATCH_MAX_DIST = 0.6

# Bloque de ubicacion: 4 por mano (slots Left, Right) + 1 bandera de cuerpo.
N_BODY_FEATURES_PER_HAND = 4
N_BODY_FEATURES = 2 * N_BODY_FEATURES_PER_HAND + 1

# Solo hombros y brazos (mas cuello y cara, abajo): es lo que importa para
# senar. Las caderas no se dibujan: casi nunca salen en cuadro frente a la
# camara y, cuando quedan justo fuera, MediaPipe las ubica pegadas al borde
# DENTRO de la imagen (medido: y = 0.97-0.996 con presence 0.52-0.69), asi que
# ni las coordenadas ni presence permiten descartarlas. Los antebrazos
# (codo -> muneca) se dibujan aparte, porque pueden terminar en la muneca de
# la mano detectada en vez de en la de la pose.
BODY_CONNECTIONS: list[tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
]
FOREARMS = ((LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST))
FACE_CONNECTIONS: list[tuple[int, int]] = [
    (NOSE, LEFT_EYE), (LEFT_EYE, LEFT_EAR),
    (NOSE, RIGHT_EYE), (RIGHT_EYE, RIGHT_EAR),
    (MOUTH_LEFT, MOUTH_RIGHT),
]

BODY_COLOR = (0, 215, 255)        # amarillo (BGR), distinto de los colores de los dedos
REFERENCE_COLOR = (255, 0, 255)   # magenta: centro de hombros y de la boca


def ensure_pose_model(model: str = DEFAULT_POSE_MODEL, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Igual que ensure_hand_model de senas.py, pero para el modelo de pose.
    Se descarga una sola vez (lite ~5.5 MB, full ~9 MB) junto a hand_landmarker.task."""
    if model not in POSE_MODELS:
        raise ValueError(f"pose_model debe ser uno de {POSE_MODELS}, recibido {model!r}")
    url = _POSE_MODEL_URL.format(model=model)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / f"pose_landmarker_{model}.task"
    if target.exists() and target.stat().st_size > 1000:
        return target

    log.info("Descargando modelo de MediaPipe Pose Landmarker (%s) desde Google...", model)
    log.info("URL: %s", url)

    tmp = target.with_suffix(".task.partial")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.rename(target)
        log.info("Modelo descargado: %s (%.1f MB)", target, len(data) / 1024 / 1024)
        return target
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise OSError(
            f"No se pudo descargar el modelo de MediaPipe Pose.\n"
            f"Verifica tu conexión a internet o descárgalo manualmente:\n"
            f"  curl -o {target} {url}\n"
            f"Error original: {e}"
        ) from e


@dataclass
class BodyDetection:
    image_xyz: np.ndarray    # (33, 3): x,y normalizados a la imagen [0,1], z relativa, tal como los da MediaPipe
    visibility: np.ndarray   # (33,): 0..1, que tan probable es que el punto no este tapado
    presence: np.ndarray     # (33,): 0..1, dentro de la imagen; solo para los datos crudos (oscila, ver MIN_VISIBILITY)
    world_xyz: np.ndarray    # (33, 3): metros aprox., origen en el centro de la cadera

    def visible(self, *indices: int) -> bool:
        return all(
            0.0 <= self.image_xyz[i, 0] <= 1.0
            and 0.0 <= self.image_xyz[i, 1] <= 1.0
            and self.visibility[i] >= MIN_VISIBILITY
            for i in indices
        )


def parse_pose(results) -> Optional[BodyDetection]:
    """Resultado de PoseLandmarker -> BodyDetection, o None si no hay nadie en cuadro."""
    if not results.pose_landmarks:
        return None
    lms = results.pose_landmarks[0]
    image_xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    # visibility/presence son Optional en la API de Tasks; sin dato cuenta como no visible.
    visibility = np.array(
        [lm.visibility if lm.visibility is not None else 0.0 for lm in lms],
        dtype=np.float32,
    )
    presence = np.array(
        [lm.presence if lm.presence is not None else 0.0 for lm in lms],
        dtype=np.float32,
    )
    world_xyz = np.zeros((N_POSE_LANDMARKS, 3), dtype=np.float32)
    if results.pose_world_landmarks:
        world_xyz = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks[0]],
            dtype=np.float32,
        )
    return BodyDetection(image_xyz=image_xyz, visibility=visibility, presence=presence, world_xyz=world_xyz)


class BodyTracker:
    """PoseLandmarker en modo VIDEO con timestamps REALES (una sola persona).

    En modo VIDEO, MediaPipe suaviza los puntos del cuerpo con un filtro que
    depende del tiempo entre frames. senas.py y los recolectores le pasan al
    HandLandmarker timestamps falsos (+1 ms por frame). A las manos no les
    afecta (medido: mismo error con +1 ms que con +33 ms), pero a la pose si:
    el filtro cree que cada frame dura 33 veces menos y suaviza de mas. Medido
    con movimiento sintetico a 180 px/s: con +1 ms, 24 px de error medio y 4
    frames (~130 ms) de retraso; con tiempos reales, 1.6 px y 0 frames.

    Por eso esta clase lleva su propio reloj (time.monotonic) y detect() no
    recibe timestamp: asi nadie puede volver a pasarle el contador de frames.
    """

    def __init__(
        self,
        model: str = DEFAULT_POSE_MODEL,
        models_dir: Path = DEFAULT_MODELS_DIR,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = ensure_pose_model(model, models_dir)
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def detect(self, mp_image) -> Optional[BodyDetection]:
        # MediaPipe exige timestamps estrictamente crecientes.
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        return parse_pose(self._landmarker.detect_for_video(mp_image, ts_ms))

    def close(self) -> None:
        self._landmarker.close()


def body_location_features(
    hands: Mapping[str, Any],
    body: Optional[BodyDetection],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Bloque de N_BODY_FEATURES (9) valores: donde esta cada mano respecto al cuerpo.

      [0:4]  mano "Left":  muneca (dx, dy) desde el centro de los hombros,
                           punta del indice (dx, dy) desde el centro de la boca
      [4:8]  mano "Right": igual
      [8]    1.0 si el cuerpo es utilizable (hombros y boca visibles), 0.0 si no

    Todo va en pixeles dividido entre el ancho de hombros en pixeles, asi no
    depende de que tan lejos este la persona ni de donde este parada en la
    imagen. dy positivo = hacia abajo (convencion de imagen). La muneca da la
    zona general (cara, pecho, espacio neutro) y la punta del indice contra la
    boca afina las zonas de la cara (frente, mejilla, boca, barbilla).

    Una mano ausente deja sus 4 valores en ceros (como en el vector de 126); si
    el cuerpo no es utilizable, TODO el bloque queda en ceros, incluida la
    bandera [8], para que el modelo distinga "sin cuerpo" de "mano en el origen".

    `hands` es el mismo dict {"Left": HandDetection, "Right": HandDetection}
    que reciben build_feature_vector (recolectores) y build_dynamic_feature_vector
    (senas.py), con los mismos slots. Las coordenadas se pasan a pixeles antes
    de medir: MediaPipe las normaliza dividiendo x entre el ancho e y entre el
    alto por separado, y medir distancias asi deformaria la geometria 4:3.
    """
    vec = np.zeros(N_BODY_FEATURES, dtype=np.float32)
    if body is None or not body.visible(LEFT_SHOULDER, RIGHT_SHOULDER, MOUTH_LEFT, MOUTH_RIGHT):
        return vec

    to_px = np.array([frame_w, frame_h], dtype=np.float32)
    pts = body.image_xyz[:, :2] * to_px
    shoulder_mid = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2.0
    shoulder_width = float(np.linalg.norm(pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER]))
    if shoulder_width < MIN_SHOULDER_WIDTH_PX:
        return vec
    mouth_mid = (pts[MOUTH_LEFT] + pts[MOUTH_RIGHT]) / 2.0

    for slot_idx, handedness in enumerate(("Left", "Right")):
        hand = hands.get(handedness)
        if hand is None:
            continue
        hand_px = hand.landmarks_2d[:, :2] * to_px
        offset = slot_idx * N_BODY_FEATURES_PER_HAND
        vec[offset:offset + 2] = (hand_px[0] - shoulder_mid) / shoulder_width
        vec[offset + 2:offset + 4] = (hand_px[8] - mouth_mid) / shoulder_width

    vec[-1] = 1.0
    return vec


def _match_hands_to_wrists(pts_px: np.ndarray, body: BodyDetection, hand_wrists_px: list[np.ndarray]) -> dict[int, np.ndarray]:
    """{indice de muneca de la pose: muneca de la mano detectada} por cercania.

    Se empareja por distancia en la imagen, nunca por etiqueta (ver el aviso
    del espejo al inicio del modulo). Primero el par mas cercano, y cada mano
    se usa una sola vez. La muneca de la pose se usa aunque tenga baja
    visibilidad: justo cuando la mano tapa la muneca, la pose la ve peor."""
    if not hand_wrists_px:
        return {}
    if body.visible(LEFT_SHOULDER, RIGHT_SHOULDER):
        max_dist = HAND_MATCH_MAX_DIST * float(np.linalg.norm(pts_px[LEFT_SHOULDER] - pts_px[RIGHT_SHOULDER]))
    else:
        max_dist = 80.0
    pairs = sorted(
        (float(np.linalg.norm(pts_px[w] - hw)), w, h)
        for w in (LEFT_WRIST, RIGHT_WRIST)
        for h, hw in enumerate(hand_wrists_px)
    )
    matched: dict[int, np.ndarray] = {}
    used_hands: set[int] = set()
    for dist, w, h in pairs:
        if dist > max_dist or w in matched or h in used_hands:
            continue
        matched[w] = hand_wrists_px[h]
        used_hands.add(h)
    return matched


def draw_body_skeleton(image: np.ndarray, body: BodyDetection, hands: Iterable[Any] = ()) -> None:
    """Dibuja torso, brazos y cara (solo puntos que pasan visible()), el cuello
    y, en magenta, los dos puntos de referencia que usa body_location_features:
    centro de los hombros y centro de la boca.

    `hands` (HandDetection del mismo frame, opcional): si una mano detectada
    queda cerca de una muneca de la pose, ese antebrazo termina en la muneca
    de la MANO. El detector de manos la ubica mejor que la pose, y asi el brazo
    se une con los dedos en vez de quedar desfasado. Se dibuja ANTES que las
    manos para que los dedos queden encima."""
    h, w = image.shape[:2]
    to_px = np.array([w, h], dtype=np.float32)
    pts_px = body.image_xyz[:, :2] * to_px
    hand_wrists_px = [hand.landmarks_2d[0, :2] * to_px for hand in hands]
    matched = _match_hands_to_wrists(pts_px, body, hand_wrists_px)

    def pt(p: np.ndarray) -> tuple[int, int]:
        return int(p[0]), int(p[1])

    for a, b in BODY_CONNECTIONS:
        if body.visible(a, b):
            cv2.line(image, pt(pts_px[a]), pt(pts_px[b]), BODY_COLOR, 3, cv2.LINE_AA)

    for elbow, wrist in FOREARMS:
        if not body.visible(elbow):
            continue
        if wrist in matched:
            cv2.line(image, pt(pts_px[elbow]), pt(matched[wrist]), BODY_COLOR, 3, cv2.LINE_AA)
        elif body.visible(wrist):
            cv2.line(image, pt(pts_px[elbow]), pt(pts_px[wrist]), BODY_COLOR, 3, cv2.LINE_AA)

    for a, b in FACE_CONNECTIONS:
        if body.visible(a, b):
            cv2.line(image, pt(pts_px[a]), pt(pts_px[b]), BODY_COLOR, 2, cv2.LINE_AA)

    shoulder_mid = mouth_mid = None
    if body.visible(LEFT_SHOULDER, RIGHT_SHOULDER):
        shoulder_mid = (pts_px[LEFT_SHOULDER] + pts_px[RIGHT_SHOULDER]) / 2.0
    if body.visible(MOUTH_LEFT, MOUTH_RIGHT):
        mouth_mid = (pts_px[MOUTH_LEFT] + pts_px[MOUTH_RIGHT]) / 2.0
    if shoulder_mid is not None and mouth_mid is not None:
        cv2.line(image, pt(shoulder_mid), pt(mouth_mid), BODY_COLOR, 3, cv2.LINE_AA)   # cuello

    # Puntos: las munecas unidas a una mano no se dibujan (ahi ya esta la mano).
    joints = (
        {i for pair in BODY_CONNECTIONS + FACE_CONNECTIONS for i in pair}
        | {i for pair in FOREARMS for i in pair}
    ) - set(matched)
    for i in sorted(joints):
        if body.visible(i):
            radius = 3 if i in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT) else 5
            cv2.circle(image, pt(pts_px[i]), radius, BODY_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, pt(pts_px[i]), radius, (255, 255, 255), 1, cv2.LINE_AA)

    for mid in (shoulder_mid, mouth_mid):
        if mid is not None:
            cv2.drawMarker(image, pt(mid), REFERENCE_COLOR, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


def draw_body_status(image: np.ndarray, body_ok: bool, origin: tuple[int, int] = (10, 90)) -> None:
    """Linea de estado para los recolectores: avisa ANTES de grabar si la
    ubicacion respecto al cuerpo se esta guardando o quedaria en ceros."""
    if body_ok:
        text, color = "Cuerpo: visible", (102, 255, 102)
    else:
        text, color = "Cuerpo: NO visible (que se vean hombros y boca)", (0, 165, 255)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
