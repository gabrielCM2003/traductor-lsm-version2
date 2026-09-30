from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import platform
import queue
import statistics
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, QTimer, QSize, QSettings, pyqtSignal,
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QSlider, QFrame, QComboBox,
    QFileDialog, QPlainTextEdit, QStatusBar, QToolBar, QSizePolicy,
    QCheckBox,
)

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: falta 'mediapipe'. Instala con: pip install mediapipe", file=sys.stderr)
    raise

from sign_classifier import SignClassifier, PredictionSmoother, normalize_keypoints, hand_to_feature_vector

try:
    from segmentador_automatico import AutoSegmenter, MIN_SEQUENCE_MS as DYN_STANDALONE_MIN_SEQUENCE_MS
    from dtw_recognizer import DTWRecognizer
except ImportError as e:
    # Alfabeto dinamico (J,K,Ñ,Q,X,Z): opcional. Si falta fastdtw/scipy o los
    # archivos aun no existen, el alfabeto estatico sigue funcionando igual
    # que antes; el modo dinamico simplemente queda deshabilitado.
    AutoSegmenter = None
    DTWRecognizer = None
    DYN_STANDALONE_MIN_SEQUENCE_MS = 170
    logging.getLogger("sign_translator").warning(
        "Alfabeto dinamico no disponible (%s). Instala fastdtw/scipy para habilitarlo.", e
    )

try:
    from body_tracker import BodyTracker, draw_body_skeleton, body_location_features
except ImportError as e:
    # Esqueleto de pose: puramente visual/diagnostico (checkbox "Dibujar
    # esqueleto (Pose)"), ver body_tracker.py. NO alimenta al clasificador ni
    # al segmentador de ninguna forma - si falta este modulo o su modelo, el
    # alfabeto estatico y dinamico siguen funcionando exactamente igual, el
    # checkbox simplemente no hace nada.
    BodyTracker = None
    draw_body_skeleton = None
    body_location_features = None
    logging.getLogger("sign_translator").warning(
        "Esqueleto de pose no disponible (%s). body_tracker.py es opcional.", e
    )

# --------------------------------------------------------------------------- #
# Parametros ajustables del alfabeto dinamico integrado en la GUI.
#
# Son deliberadamente independientes de las constantes de
# segmentador_automatico.py (modo consola, NO_HAND_MS_TO_END / MIN_SEQUENCE_MS):
# ese script no se toca. Valores de partida pensados para no cambiar el
# comportamiento actual hasta que se afinen con datos reales de evaluacion.
# --------------------------------------------------------------------------- #

# Confianza minima del candidato top-1 (softmax entre las N señas dinamicas
# disponibles) para comprometer la letra a la palabra (grupo NORMAL, via
# confianza). Ver DYN_NORMAL_MIN_MARGIN mas abajo para la via alternativa
# de margen del mismo grupo.
DYN_MIN_CONF = 0.55

# Mas tolerante que el modo consola (270ms): senas como la J son un trazo
# largo y a veces la mano se pierde un instante a mitad del gesto sin que
# eso signifique que ya termino.
DYN_NO_HAND_MS_TO_END = 700

# Tope duro: si el usuario no baja la mano, no seguir grabando para siempre.
DYN_MAX_SEQUENCE_MS = 5000

# ------------------------------------------------------------------------- #
# Dos grupos de letras, cada uno con su propia regla de commit (ver
# dynamic_commit_decision). Separados por letra (no una sola regla global)
# para poder recalibrar cada grupo por separado si hace falta.
# ------------------------------------------------------------------------- #

# Grupo NORMAL: confianza absoluta del top-1 suele ser un buen indicador.
# Datos reales (2026-09-29) confirman el mismo patron que ya se veia en
# K/Q/Z: un margen amplio sobre el 2.º lugar tambien es señal solida de
# acierto aunque la confianza absoluta no llegue a DYN_MIN_CONF (ej. J con
# margenes de 34.5pp, 24.6pp y 37pp -> deberian comprometer aunque la
# confianza este debajo de 0.55; margenes de 11.1pp y 2.1pp -> correctamente
# dudosos, no deben comprometer). Por eso el grupo NORMAL compromete por
# confianza >= DYN_MIN_CONF O por margen >= DYN_NORMAL_MIN_MARGIN, lo que se
# cumpla primero.
DYN_NORMAL_LETTERS = {"J", "Ñ", "X"}
DYN_NORMAL_MIN_MARGIN = 0.20

# Grupo EXPERIMENTAL: en evaluacion, K, Q y Z resultaron poco confiables en
# confianza absoluta (las 6 clases quedan muy juntas en distancia DTW:
# ninguna de esas tres cruzo nunca ~45%, aunque el top-1 fuera correcto), asi
# que DYN_MIN_CONF=0.55 las bloqueaba siempre, acertaran o no. En vez de
# bloquearlas siempre, se dejan comprometer si el MARGEN sobre el 2.º lugar
# es lo bastante grande, aunque la confianza absoluta del top-1 no alcance
# DYN_MIN_CONF. Es lo que de verdad distingue un acierto solido de una
# adivinanza para estas tres: en evaluacion real, K/Q/Z con margen >15pp
# resultaron ser el top-1 correcto, mientras que con margen <2pp era
# practicamente un empate entre las 6 clases (ej. K 29.8% vs Z 28.2%, margen
# 1.6pp, dudoso; K 38.9% vs Z 23.5%, margen 15.4pp, solido). DYN_EXPERIMENTAL_
# MIN_CONF es solo un piso de cordura (no aceptar un top-1 absurdamente bajo
# aunque el margen diera grande por casualidad), no el criterio principal.
DYN_EXPERIMENTAL_LETTERS = {"K", "Q", "Z"}
DYN_EXPERIMENTAL_MIN_MARGIN = 0.12
DYN_EXPERIMENTAL_MIN_CONF = 0.30

# Caso puntual, diagnostico 2026-09-29: Ñ nunca tuvo muestras propias reales
# (a diferencia de J/K/Q/Z), asi que en vivo (fuera de las condiciones de
# laboratorio de datos_dinamicas/) sus consultas se desvian mas de lo que
# LOSO sugiere (98.9%). El unico vecino con el que Ñ se confunde, incluso en
# LOSO puro, es Q (evaluar_dtw.py: 1/93). El problema es que Q esta en el
# grupo EXPERIMENTAL (compromete con solo 12pp de margen) mientras Ñ esta en
# el grupo NORMAL (necesita conf>=55% o 20pp de margen), asi que cuando el
# DTW en vivo empuja a una Ñ real hacia el territorio de Q, a Q le basta
# mucho menos margen del que le costaria a la propia Ñ para comprometerse
# primero. DYN_NQ_PAIR_MIN_MARGIN exige un margen reforzado SOLO cuando el
# top-1 es Q Y el 2.º lugar es especificamente Ñ; si no se alcanza, no se
# compromete ninguna de las dos letras por esa clasificacion. No afecta a Q
# contra K/X/Z (siguen con DYN_EXPERIMENTAL_MIN_MARGIN de siempre) ni a Ñ
# como top-1 (sigue con su regla NORMAL sin cambios, vea dynamic_commit_
# decision). Mitigacion temporal mientras se graban muestras propias reales
# de Ñ (que es la solucion de fondo); revisar si sigue haciendo falta una
# vez exista esa galeria.
DYN_NQ_PAIR_MIN_MARGIN = 0.22

# DTWRecognizer.try_load() tarda ~2s en parsear las plantillas de
# datos_dinamicas/ (cientos de JSON). HandTrackingThread se recrea cada vez
# que el watchdog reinicia la IA por inactividad, y eso pasaba en el hilo de
# GUI (bloqueando toda la ventana, no solo el video) porque __init__ volvia a
# cargarlo desde disco cada vez. Se cachea una sola vez por proceso.
_dtw_recognizer_singleton: Optional["DTWRecognizer"] = None
_dtw_recognizer_load_attempted = False


def _get_dtw_recognizer() -> Optional["DTWRecognizer"]:
    global _dtw_recognizer_singleton, _dtw_recognizer_load_attempted
    if not _dtw_recognizer_load_attempted:
        _dtw_recognizer_load_attempted = True
        if DTWRecognizer is not None:
            try:
                _dtw_recognizer_singleton = DTWRecognizer.try_load()
            except Exception:
                logging.getLogger("sign_translator").exception("Error cargando DTWRecognizer")
    return _dtw_recognizer_singleton


APP_NAME = "SignTranslator"
APP_ORG = "OpenLSM"
APP_VERSION = "3.3-lsm-alfabeto-dinamico"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_FILENAME = "hand_landmarker.task"

DEFAULT_CONFIG = {
    "camera_index": 0,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1,                 
    "smoothing_window": 7,
    "stable_frames_to_commit": 12,
    "no_hand_frames_for_space": 25,
    "keypoint_buffer_size": 30,           
    "queue_maxsize": 1,
    "watchdog_timeout_s": 5.0,
    "draw_landmarks": True,
    "draw_connections": True,
    # Puramente visual/diagnostico (ver body_tracker.py). Default False: es
    # opcional, no debe costarle CPU/descarga de modelo a quien no lo activa.
    "draw_body_skeleton": False,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sign_translator")



@dataclass
class AppConfig:
    camera_index: int = DEFAULT_CONFIG["camera_index"]
    max_num_hands: int = DEFAULT_CONFIG["max_num_hands"]
    min_detection_confidence: float = DEFAULT_CONFIG["min_detection_confidence"]
    min_tracking_confidence: float = DEFAULT_CONFIG["min_tracking_confidence"]
    model_complexity: int = DEFAULT_CONFIG["model_complexity"]
    smoothing_window: int = DEFAULT_CONFIG["smoothing_window"]
    stable_frames_to_commit: int = DEFAULT_CONFIG["stable_frames_to_commit"]
    no_hand_frames_for_space: int = DEFAULT_CONFIG["no_hand_frames_for_space"]
    keypoint_buffer_size: int = DEFAULT_CONFIG["keypoint_buffer_size"]
    queue_maxsize: int = DEFAULT_CONFIG["queue_maxsize"]
    watchdog_timeout_s: float = DEFAULT_CONFIG["watchdog_timeout_s"]
    draw_landmarks: bool = DEFAULT_CONFIG["draw_landmarks"]
    draw_connections: bool = DEFAULT_CONFIG["draw_connections"]
    draw_body_skeleton: bool = DEFAULT_CONFIG["draw_body_skeleton"]

    @classmethod
    def load(cls, json_path: Optional[Path] = None) -> "AppConfig":
        cfg = cls()
        if json_path and json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                log.info("Configuración cargada desde %s", json_path)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("No se pudo leer %s: %s. Usando defaults.", json_path, e)
        return cfg

    def save(self, json_path: Path) -> None:
        try:
            json_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("No se pudo guardar config: %s", e)



def ensure_hand_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / HAND_MODEL_FILENAME
    if target.exists() and target.stat().st_size > 1000:
        return target

    log.info("Descargando modelo de MediaPipe Hand Landmarker desde Google...")
    log.info("URL: %s", HAND_MODEL_URL)

    tmp = target.with_suffix(".task.partial")
    try:
        with urllib.request.urlopen(HAND_MODEL_URL, timeout=30) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.rename(target)
        log.info("Modelo descargado: %s (%.1f MB)", target, len(data) / 1024 / 1024)
        return target
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise OSError(
            f"No se pudo descargar el modelo de MediaPipe.\n"
            f"Verifica tu conexión a internet o descárgalo manualmente:\n"
            f"  curl -o {target} {HAND_MODEL_URL}\n"
            f"Error original: {e}"
        ) from e



def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/cpuinfo", "r") as f:
            if "raspberry pi" in f.read().lower():
                return True
    except OSError:
        pass
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "raspberry pi" in f.read().lower():
                return True
    except OSError:
        pass
    return (
        platform.machine().lower() in ("armv7l", "aarch64")
        and os.path.exists("/dev/vchiq")
    )


_PREFERRED_BACKEND = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY


# --------------------------------------------------------------------------- #
# Soporte de cámara CSI en Raspberry Pi 5 (libcamera / Picamera2)
# --------------------------------------------------------------------------- #
#
# En la Raspberry Pi 5 con Bookworm, las cámaras CSI (módulos oficiales:
# imx708 / imx219 / ov5647) NO se exponen de forma fiable como /dev/videoX
# para cv2.CAP_V4L2, porque se eliminó el firmware de cámara legacy. La forma
# correcta de acceder a ellas es a través de libcamera, mediante Picamera2.
#
# Esta clase envuelve Picamera2 con la misma interfaz mínima que
# cv2.VideoCapture (isOpened / read / release / set) para que el resto del
# código (CameraThread) no necesite cambios.

def _picamera2_available() -> bool:
    """True si Picamera2 puede importarse (solo disponible en Raspberry Pi OS)."""
    try:
        import picamera2  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


class PiCameraCapture:
    """Adaptador de Picamera2 con interfaz compatible con cv2.VideoCapture.

    Solo se usa en Raspberry Pi con cámara CSI. Entrega frames en formato BGR
    (igual que OpenCV) para que el pipeline existente no cambie.
    """

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30):
        self._picam = None
        self._opened = False
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"size": (int(width), int(height)), "format": "RGB888"},
            )
            self._picam.configure(config)
            # Fija el framerate vía duración de frame (en microsegundos).
            try:
                frame_us = int(1_000_000 / max(1, framerate))
                self._picam.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
            except Exception:
                pass  # no es crítico si el sensor no lo soporta
            self._picam.start()
            self._opened = True
            log.info("Picamera2 iniciada (%dx%d @ %dfps)", width, height, framerate)
        except Exception as e:
            log.error("No se pudo iniciar Picamera2: %s", e)
            self._opened = False
            if self._picam is not None:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        """Devuelve (ret, frame_bgr) imitando cv2.VideoCapture.read()."""
        if not self._opened or self._picam is None:
            return False, None
        try:
            frame = self._picam.capture_array()  # RGB888 -> array RGB
            # Picamera2 entrega RGB; el resto del código asume BGR (OpenCV).
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            log.warning("Fallo al capturar frame de Picamera2: %s", e)
            return False, None

    def set(self, *args, **kwargs) -> bool:
        """Acepta las llamadas cap.set(...) de CameraThread sin hacer nada.

        La resolución/fps ya se fijaron en __init__; las propiedades de
        cv2.CAP_PROP_* no aplican a Picamera2, así que se ignoran de forma
        segura para mantener compatibilidad de interfaz.
        """
        return True

    def release(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
            try:
                self._picam.close()
            except Exception:
                pass
        self._picam = None
        self._opened = False


def list_available_cameras(max_check: int = 5) -> list[int]:
    # En Raspberry Pi con cámara CSI, el escaneo de /dev/video* no es fiable
    # (los nodos suelen ser del ISP, no del sensor). Si Picamera2 está
    # disponible, exponemos la cámara CSI como índice 0.
    if _is_raspberry_pi() and _picamera2_available():
        return [0]

    available: list[int] = []

    if platform.system() == "Linux":
        video_nodes = sorted(
            int(p.replace("/dev/video", ""))
            for p in glob.glob("/dev/video*")
            if p.replace("/dev/video", "").isdigit()
        )
        candidates = video_nodes if video_nodes else list(range(max_check))
    else:
        candidates = list(range(max_check))

    for i in candidates:
        cap = cv2.VideoCapture(i, _PREFERRED_BACKEND)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
        else:
            cap.release()
    return available



@dataclass
class HandDetection:
    handedness: str                  
    confidence: float                
    landmarks_2d: np.ndarray        
    landmarks_3d: np.ndarray         


@dataclass
class FrameDetections:
    hands: list[HandDetection] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def num_hands(self) -> int:
        return len(self.hands)


HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Índice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Medio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Anular
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Meñique
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = {
    "thumb":  (255, 102, 102),   
    "index":  (102, 255, 102),   
    "middle": (255, 178, 102),  
    "ring":   (178, 102, 255),   
    "pinky":  (102, 178, 255),   
    "palm":   (200, 200, 200),   
}

LANDMARK_GROUP = {
    0: "palm",
    1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
    5: "index", 6: "index", 7: "index", 8: "index",
    9: "middle", 10: "middle", 11: "middle", 12: "middle",
    13: "ring", 14: "ring", 15: "ring", 16: "ring",
    17: "pinky", 18: "pinky", 19: "pinky", 20: "pinky",
}


def draw_hand_landmarks(
    image: np.ndarray,
    hand: HandDetection,
    draw_connections: bool = True,
    draw_points: bool = True,
) -> None:
    h, w = image.shape[:2]
    pts_px = np.zeros((21, 2), dtype=np.int32)
    for i in range(21):
        pts_px[i, 0] = int(hand.landmarks_2d[i, 0] * w)
        pts_px[i, 1] = int(hand.landmarks_2d[i, 1] * h)

    if draw_connections:
        for a, b in HAND_CONNECTIONS:
            color = FINGER_COLORS[LANDMARK_GROUP[b]]
            cv2.line(image, tuple(pts_px[a]), tuple(pts_px[b]), color, 2, cv2.LINE_AA)

    if draw_points:
        for i in range(21):
            color = FINGER_COLORS[LANDMARK_GROUP[i]]
            radius = 6 if i in (0, 5, 9, 13, 17) else 4
            cv2.circle(image, tuple(pts_px[i]), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(pts_px[i]), radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hand_label(image: np.ndarray, hand: HandDetection) -> None:
    h, w = image.shape[:2]
    wrist_px = (
        int(hand.landmarks_2d[0, 0] * w),
        int(hand.landmarks_2d[0, 1] * h),
    )
    label_text = "Derecha" if hand.handedness == "Left" else "Izquierda"
    text = f"{label_text} ({hand.confidence:.0%})"
    cv2.putText(
        image, text, (wrist_px[0] - 30, wrist_px[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (wrist_px[0] - 30, wrist_px[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


# =========================================================================== #
# Hilo 1: cámara con reconexión 
# =========================================================================== #

class CameraThread(QThread):
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    camera_lost_signal = pyqtSignal()
    camera_recovered_signal = pyqtSignal()

    def __init__(self, frame_queue: queue.Queue, camera_index: int):
        super().__init__()
        self._frame_queue = frame_queue
        self._camera_index = camera_index
        self._run_flag = True
        self._reconnect_delay_s = 1.0
        self._max_reconnect_delay_s = 8.0

    def _open_camera(self):
        # En Raspberry Pi con cámara CSI, usar Picamera2 (libcamera) en lugar
        # de V4L2, que no expone el sensor de forma fiable en la Pi 5.
        if _is_raspberry_pi() and _picamera2_available():
            cap = PiCameraCapture(width=640, height=480, framerate=30)
            if not cap.isOpened():
                cap.release()
                return None
            return cap

        cap = cv2.VideoCapture(self._camera_index, _PREFERRED_BACKEND)
        if not cap.isOpened():
            cap.release()
            # Reintento con backend automático por si no soporta V4L2.
            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                cap.release()
                return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        cap = self._open_camera()
        if cap is None:
            msg = f"No se pudo abrir la cámara (índice {self._camera_index})."
            log.error(msg)
            self.error_signal.emit(msg)
            return

        self.status_signal.emit("Cámara conectada")
        log.info("Cámara abierta en índice %d", self._camera_index)
        consecutive_failures = 0
        camera_was_lost = False
        delay = self._reconnect_delay_s

        try:
            while self._run_flag:
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        log.warning("Lectura fallida, intentando recuperar...")

                    if consecutive_failures >= 5:
                        if not camera_was_lost:
                            self.camera_lost_signal.emit()
                            camera_was_lost = True
                        cap.release()
                        time.sleep(delay)
                        delay = min(delay * 1.5, self._max_reconnect_delay_s)
                        cap = self._open_camera()
                        if cap is None:
                            self.status_signal.emit(f"Reintentando cámara en {delay:.1f}s...")
                            continue
                        consecutive_failures = 0
                        delay = self._reconnect_delay_s
                        log.info("Cámara reconectada")
                        if camera_was_lost:
                            self.camera_recovered_signal.emit()
                            camera_was_lost = False
                    else:
                        time.sleep(0.05)
                    continue

                consecutive_failures = 0
                frame = cv2.flip(frame, 1)  # efecto espejo

                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            if cap is not None:
                cap.release()
            log.info("Hilo de cámara terminado")

    def stop(self) -> None:
        self._run_flag = False
        self.wait(2000)


# =========================================================================== #
# Alfabeto dinamico: vector de 126 (dos manos) para DTWRecognizer
# =========================================================================== #

def build_dynamic_feature_vector(hands: dict[str, "HandDetection"]) -> np.ndarray:
    """Vector de 126 = [mano izquierda normalizada (63)] + [mano derecha normalizada (63)].

    Misma convencion de slots (Left, Right) y misma normalizacion por mano que
    recolector_dinamico.build_feature_vector / procesar_dataset_dinamico.py
    (que a su vez usan normalize_keypoints y hand_to_feature_vector de
    sign_classifier.py). Se reimplementa aqui en vez de importarse de
    recolector_dinamico.py porque ese modulo ya importa cosas de senas.py, y
    un import en sentido contrario crearia un ciclo.
    """
    vec = np.zeros(126, dtype=np.float32)
    for slot_idx, handedness in enumerate(("Left", "Right")):
        hand = hands.get(handedness)
        if hand is None:
            continue
        raw = hand_to_feature_vector(hand.landmarks_2d, hand.landmarks_3d)
        norm = normalize_keypoints(raw)
        offset = slot_idx * 63
        vec[offset:offset + 63] = norm
    return vec


def is_experimental_dynamic_letter(letter: str) -> bool:
    """True si `letter` esta en DYN_EXPERIMENTAL_LETTERS (K, Q, Z): usa la
    regla EXPERIMENTAL de margen (DYN_EXPERIMENTAL_MIN_MARGIN/MIN_CONF) en
    vez de la regla del grupo NORMAL (DYN_MIN_CONF/DYN_NORMAL_MIN_MARGIN)."""
    return letter in DYN_EXPERIMENTAL_LETTERS


def is_nq_blocking_pair(topk: list[tuple[str, float]]) -> bool:
    """True si el top-1 es Q y el 2.º lugar es especificamente Ñ.

    Unico caso donde aplica el margen reforzado DYN_NQ_PAIR_MIN_MARGIN (ver
    dynamic_commit_decision). Q contra cualquier otra letra (K, X, Z) sigue
    con DYN_EXPERIMENTAL_MIN_MARGIN de siempre, sin cambios.
    """
    if len(topk) < 2:
        return False
    return topk[0][0] == "Q" and topk[1][0] == "Ñ"


def dynamic_commit_decision(topk: list[tuple[str, float]]) -> tuple[bool, float, str]:
    """Decide si el top-1 de una clasificacion dinamica se compromete o no.

    Dos grupos de letras, cada uno con su propia regla (ver constantes al
    inicio del archivo):
      - Grupo NORMAL (DYN_NORMAL_LETTERS, hoy J, Ñ, X): compromete si la
        confianza del top-1 alcanza DYN_MIN_CONF, O si el margen sobre el
        top-2 (confianza_1 - confianza_2) alcanza DYN_NORMAL_MIN_MARGIN,
        lo que se cumpla primero. La via de margen existe porque, igual que
        con K/Q/Z, un margen amplio es señal solida de acierto aunque la
        confianza absoluta se quede corta. Esta regla se usa TAL CUAL cuando
        el top-1 es Ñ, sin importar quien quede en 2.º lugar (el ajuste de
        abajo es unidireccional: solo protege a Ñ de perder el commit contra
        Q, nunca al reves).
      - Grupo EXPERIMENTAL (DYN_EXPERIMENTAL_LETTERS, hoy K, Q, Z): su
        confianza absoluta nunca cruza ~45% aunque el top-1 sea correcto (las
        6 clases quedan muy juntas en distancia DTW), asi que DYN_MIN_CONF
        las bloquearia siempre. Se comprometen si el margen sobre el 2.º
        lugar supera DYN_EXPERIMENTAL_MIN_MARGIN, con DYN_EXPERIMENTAL_
        MIN_CONF como piso minimo de cordura (no la condicion principal).
        Caso especial dentro de este grupo (ver is_nq_blocking_pair y
        DYN_NQ_PAIR_MIN_MARGIN): si el top-1 es Q y el 2.º lugar es
        especificamente Ñ, se exige el margen reforzado DYN_NQ_PAIR_MIN_
        MARGIN en vez de DYN_EXPERIMENTAL_MIN_MARGIN. Si no se alcanza, no
        se compromete ni Q ni Ñ por esa clasificacion. Q contra K/X/Z no
        cambia.

    Se centraliza aqui para que _process_dynamic_frame (que decide si se
    agrega la letra) y _on_diagnostic_update en la GUI (que solo explica por
    que no se agrego) usen exactamente el mismo criterio.

    Devuelve (se_compromete, margen, regla), donde regla es una de:
      "confianza"       -> grupo normal, comprometio por DYN_MIN_CONF.
      "margen"          -> grupo normal, comprometio por DYN_NORMAL_MIN_MARGIN.
      "experimental"    -> grupo experimental, comprometio por margen amplio.
      "experimental_nq" -> Q comprometio contra Ñ en 2.º lugar, con el margen
                            reforzado DYN_NQ_PAIR_MIN_MARGIN.
      ""                -> no se comprometio.
    """
    if not topk:
        return False, 0.0, ""
    letra1 = topk[0][0]
    conf1 = topk[0][1]
    margin = conf1 - topk[1][1] if len(topk) > 1 else conf1

    if is_experimental_dynamic_letter(letra1):
        if is_nq_blocking_pair(topk):
            should_commit = conf1 >= DYN_EXPERIMENTAL_MIN_CONF and margin >= DYN_NQ_PAIR_MIN_MARGIN
            return should_commit, margin, "experimental_nq" if should_commit else ""
        should_commit = conf1 >= DYN_EXPERIMENTAL_MIN_CONF and margin >= DYN_EXPERIMENTAL_MIN_MARGIN
        return should_commit, margin, "experimental" if should_commit else ""

    if conf1 >= DYN_MIN_CONF:
        return True, margin, "confianza"
    if margin >= DYN_NORMAL_MIN_MARGIN:
        return True, margin, "margen"
    return False, margin, ""


# =========================================================================== #
# Hilo 2: MediaPipe Hands
# =========================================================================== #

@dataclass
class InferenceMetrics:
    fps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    num_hands_avg: float = 0.0
    last_inference_ts: float = field(default_factory=time.time)


class HandTrackingThread(QThread):
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    hands_detected_signal = pyqtSignal(object)        
    sign_detected_signal = pyqtSignal(str, float)    
    sign_diagnostic_signal = pyqtSignal(object)       
    letter_committed_signal = pyqtSignal(str)
    space_committed_signal = pyqtSignal()
    metrics_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    model_loaded_signal = pyqtSignal()
    heartbeat_signal = pyqtSignal()

    def __init__(self, frame_queue: queue.Queue, config: AppConfig):
        super().__init__()
        self._frame_queue = frame_queue
        self._cfg = config
        self._run_flag = True
        self._hands_solution = None

        # Esqueleto de pose: puramente visual/diagnostico, ver body_tracker.py
        # y _get_body_tracker(). No participa en self._classifier ni en
        # self._auto_segmenter/self._dtw_recognizer de ninguna forma.
        self._body_tracker: Optional["BodyTracker"] = None
        self._body_tracker_failed: bool = False

        self._keypoint_buffer: deque[np.ndarray] = deque(
            maxlen=config.keypoint_buffer_size
        )

        self._frames_without_hand = 0
        self._space_already_committed = False
        self._last_committed_label: Optional[str] = None

        self._classifier: Optional[SignClassifier] = SignClassifier.try_load()
        self._per_letter_confidence: dict[str, float] = {}
        self._smoother = PredictionSmoother(
            window_size=max(5, config.smoothing_window),
            min_confidence=0.55,
            min_margin=0.15,
            per_letter_confidence=self._per_letter_confidence,
        )
        self._diagnostic_mode: bool = False
        self._stable_letter: Optional[str] = None
        self._stable_frames: int = 0

        # Alfabeto dinamico (J,K,Ñ,Q,X,Z): opcional, requiere AutoSegmenter y
        # DTWRecognizer (ver import con try/except al inicio del archivo) y
        # plantillas en datos_dinamicas/.
        self._dynamic_mode: bool = False
        self._dtw_recognizer: Optional["DTWRecognizer"] = _get_dtw_recognizer()
        self._auto_segmenter: Optional["AutoSegmenter"] = self._new_auto_segmenter()

        # predict_topk() contra ~500+ plantillas puede tardar varios segundos
        # (medido: ~4.5s con 558 plantillas). Corriendolo en el propio hilo de
        # captura congelaba visiblemente el video (y en casos mas lentos podia
        # superar watchdog_timeout_s y disparar un reinicio del hilo a medio
        # reconocimiento). Se despacha a un hilo aparte que solo deja caer el
        # resultado en esta cola; el bucle principal la revisa sin bloquearse.
        self._dynamic_result_queue: "queue.Queue[tuple]" = queue.Queue()
        self._dynamic_classifying: bool = False
        self._dynamic_classify_start: float = 0.0

        # Texto que se muestra en el cuadro "Estado" mientras no hay nada
        # nuevo que reportar (ni grabando ni clasificando): al arrancar dice
        # "Esperando mano", y despues de cada clasificacion queda mostrando
        # esa ultima letra (comprometida o no) hasta que empiece una seña
        # nueva. Antes se volvia a "..." en el siguiente frame (menos de
        # 33ms), practicamente invisible para el usuario.
        self._dynamic_idle_text: str = "Esperando mano"

        # Instrumentacion por seña (duracion real, frames, fps efectivo,
        # hueco maximo sin mano dentro de la seña). Se reinicia en cada
        # "inicio" y se vuelca a consola cuando la seña termina.
        self._dynamic_stats: Optional[dict] = None

        # "Identidad de mano(s)" de la secuencia dinamica en curso: que
        # handedness ("Left"/"Right") estaban presentes en el primer frame
        # detectado de la seña. Mientras se graba, cualquier mano que NO
        # estaba en esta identidad se ignora (se deja en ceros) en vez de
        # agregarse al vector de 126 - evita que una mano que se asoma sin
        # intencion de señar (p.ej. de forma pasajera) convierta una seña de
        # una sola mano en un vector que no se parece a ninguna plantilla
        # (la enorme mayoria del dataset es de una sola mano activa). None
        # cuando no hay una secuencia en curso.
        self._dynamic_sequence_hand_identity: Optional[set[str]] = None

        self._latencies: deque[float] = deque(maxlen=100)
        self._frame_times: deque[float] = deque(maxlen=30)
        self._hand_counts: deque[int] = deque(maxlen=60)
        self._last_metrics_emit = 0.0


    def _new_auto_segmenter(self) -> Optional["AutoSegmenter"]:
        """Crea el segmentador del modo dinamico con los umbrales de senas.py
        (DYN_NO_HAND_MS_TO_END / DYN_MAX_SEQUENCE_MS), no los del modo consola
        de segmentador_automatico.py. Ver comentario junto a esas constantes."""
        if AutoSegmenter is None:
            return None
        return AutoSegmenter(
            no_hand_ms_to_end=DYN_NO_HAND_MS_TO_END,
            min_sequence_ms=DYN_STANDALONE_MIN_SEQUENCE_MS,
            max_duration_ms=DYN_MAX_SEQUENCE_MS,
        )

    def _reset_dynamic_state(self) -> None:
        """Descarta cualquier grabacion/clasificacion en curso del modo
        dinamico y vuelve al estado inicial. Se usa al togglear el modo, al
        pedir un espacio/borrar manualmente, y al reiniciar el hilo."""
        if self._auto_segmenter is not None:
            self._auto_segmenter = self._new_auto_segmenter()
        self._dynamic_classifying = False
        self._dynamic_idle_text = "Esperando mano"
        self._dynamic_stats = None
        self._dynamic_sequence_hand_identity = None
        while not self._dynamic_result_queue.empty():
            try:
                self._dynamic_result_queue.get_nowait()
            except queue.Empty:
                break

    def set_draw_landmarks(self, value: bool) -> None:
        self._cfg.draw_landmarks = value

    def set_draw_connections(self, value: bool) -> None:
        self._cfg.draw_connections = value

    def set_draw_body_skeleton(self, value: bool) -> None:
        self._cfg.draw_body_skeleton = value

    def _get_body_tracker(self) -> Optional["BodyTracker"]:
        """Crea BodyTracker (MediaPipe Pose) la primera vez que hace falta,
        no al iniciar el hilo: es un diagnostico opcional (checkbox "Dibujar
        esqueleto (Pose)"), no debe costarle descarga de modelo ni CPU a
        quien nunca lo activa. Puramente visual - su resultado no se usa en
        self._classifier ni en self._auto_segmenter/self._dtw_recognizer."""
        if BodyTracker is None:
            return None
        if self._body_tracker is None and not self._body_tracker_failed:
            try:
                self._body_tracker = BodyTracker()
                log.info("BodyTracker (MediaPipe Pose) listo para diagnostico visual.")
            except Exception as e:
                self._body_tracker_failed = True
                log.exception("No se pudo inicializar BodyTracker (Pose): %s", e)
        return self._body_tracker

    def reset_word_state(self) -> None:
        self._frames_without_hand = 0
        self._space_already_committed = False
        self._last_committed_label = None
        self._stable_letter = None
        self._stable_frames = 0
        self._smoother.reset()
        self._reset_dynamic_state()


    def set_stable_frames_to_commit(self, value: int) -> None:
        
        self._cfg.stable_frames_to_commit = max(1, int(value))

    def set_min_confidence(self, value: float) -> None:
      
        self._smoother.min_confidence = max(0.0, min(1.0, float(value)))

    def set_min_margin(self, value: float) -> None:
        
        self._smoother.min_margin = max(0.0, min(1.0, float(value)))

    def set_per_letter_confidence(self, letter: str, value: float) -> None:
        
        if not letter:
            return
        self._smoother.per_letter_confidence[letter.upper()] = float(value)

    def set_diagnostic_mode(self, enabled: bool) -> None:
        self._diagnostic_mode = bool(enabled)

    def set_dynamic_mode(self, enabled: bool) -> None:
        self._dynamic_mode = bool(enabled) and self._dtw_recognizer is not None
        self._reset_dynamic_state()

    @property
    def has_classifier(self) -> bool:
        return self._classifier is not None

    @property
    def has_dynamic_recognizer(self) -> bool:
        return self._dtw_recognizer is not None

    @property
    def dynamic_labels(self) -> list[str]:
        return self._dtw_recognizer.labels if self._dtw_recognizer is not None else []

    # ---- ciclo principal --------------------------------------------------

    def run(self) -> None:
        if not self._init_mediapipe():
            return

        timestamp_ms = 0

        while self._run_flag:
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                self.heartbeat_signal.emit()
                continue

            t0 = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 1
                results = self._hands_solution.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                log.exception("MediaPipe falló: %s", e)
                continue

            detections = self._parse_results(results)

            # Pose (esqueleto del cuerpo): en paralelo a las manos, puramente
            # visual/diagnostico (ver body_tracker.py). Solo corre si el
            # checkbox esta activado, y su resultado se usa UNICAMENTE dentro
            # de _render() para dibujar - jamas llega a self._classifier, a
            # self._smoother ni a self._auto_segmenter/_process_dynamic_frame.
            body_detection = None
            if self._cfg.draw_body_skeleton:
                tracker = self._get_body_tracker()
                if tracker is not None:
                    try:
                        body_detection = tracker.detect(mp_image)
                    except Exception as e:
                        log.exception("BodyTracker.detect() falló: %s", e)

            annotated = self._render(frame, detections, body_detection)

            self._update_keypoint_buffer(detections)
            self._update_word_state(detections)

            sign_text = "—"
            sign_conf = 0.0
            if self._dynamic_mode:
                sign_text, sign_conf = self._process_dynamic_frame(detections)
            elif self._classifier is not None and detections.num_hands > 0:
                hand = next(
                    (h for h in detections.hands if h.handedness == "Right"),
                    detections.hands[0],
                )
                try:
                    topk = self._classifier.predict_topk_from_hand(
                        hand.landmarks_2d, hand.landmarks_3d, k=3,
                    )
                    smoothed = self._smoother.push(topk)

                    if self._diagnostic_mode:
                        self.sign_diagnostic_signal.emit(topk)

                    if smoothed.letter is not None:
                        sign_text = smoothed.letter
                        sign_conf = smoothed.confidence
                        if smoothed.letter == self._stable_letter:
                            self._stable_frames += 1
                        else:
                            self._stable_letter = smoothed.letter
                            self._stable_frames = 1

                        if (
                            self._stable_frames >= self._cfg.stable_frames_to_commit
                            and smoothed.letter != self._last_committed_label
                        ):
                            self.letter_committed_signal.emit(smoothed.letter)
                            self._last_committed_label = smoothed.letter
                    else:
                        self._stable_frames = 0
                        if smoothed.raw_top1 and smoothed.raw_top1[1] > 0.35:
                            sign_text = f"?{smoothed.raw_top1[0]}"
                            sign_conf = smoothed.raw_top1[1]
                        else:
                            sign_text = "..."
                except Exception as e:
                    log.exception("Error en clasificador: %s", e)
                    sign_text = "—"
            elif self._classifier is None and detections.num_hands > 0:
                sign_text = f"{detections.num_hands} mano(s)"
                sign_conf = 1.0

            self.change_pixmap_signal.emit(annotated)
            self.hands_detected_signal.emit(detections)
            self.sign_detected_signal.emit(sign_text, sign_conf)
            self.heartbeat_signal.emit()

            dt = time.perf_counter() - t0
            self._update_metrics(dt, detections.num_hands)


    # ---- instrumentacion por seña (duracion, frames, fps, hueco maximo) ---

    def _dynamic_stats_reset(self, now: float) -> None:
        self._dynamic_stats = {
            "inicio": now,
            "frames": 0,
            "hueco_inicio": None,
            "hueco_max_ms": 0.0,
        }

    def _dynamic_stats_update(self, has_hand: bool, now: float) -> None:
        stats = self._dynamic_stats
        if stats is None:
            return
        stats["frames"] += 1
        if has_hand:
            # se cierra el hueco (tramo sin mano) que estuviera abierto
            if stats["hueco_inicio"] is not None:
                hueco_ms = (now - stats["hueco_inicio"]) * 1000.0
                stats["hueco_max_ms"] = max(stats["hueco_max_ms"], hueco_ms)
                stats["hueco_inicio"] = None
        elif stats["hueco_inicio"] is None:
            stats["hueco_inicio"] = now

    def _dynamic_stats_log(self, now: float, kind: str) -> None:
        stats = self._dynamic_stats
        if stats is None:
            return
        if stats["hueco_inicio"] is not None:
            hueco_ms = (now - stats["hueco_inicio"]) * 1000.0
            stats["hueco_max_ms"] = max(stats["hueco_max_ms"], hueco_ms)
        duracion_s = max(1e-6, now - stats["inicio"])
        fps_efectivo = stats["frames"] / duracion_s
        etiqueta = "descartada por corta" if kind == "fin_descartada" else "seña"
        log.info(
            "[dinamico] %s: duracion=%.2fs frames=%d fps_efectivo=%.1f hueco_max=%.0fms",
            etiqueta, duracion_s, stats["frames"], fps_efectivo, stats["hueco_max_ms"],
        )

    # ---- ciclo del modo dinamico -------------------------------------------

    def _process_dynamic_frame(self, detections: FrameDetections) -> tuple[str, float]:
        """Alfabeto dinamico (J,K,Ñ,Q,X,Z): segmenta con AutoSegmenter y
        clasifica con DTWRecognizer cuando detecta el fin de una seña.

        Misma convencion de slots que build_dynamic_feature_vector: la
        primera mano detectada por cada handedness ("Left"/"Right") gana,
        igual que _update_keypoint_buffer.

        predict_topk() contra cientos de plantillas tarda varios segundos
        (medido: ~4.5s con 558 plantillas) - demasiado para correrlo aqui
        mismo, en el hilo que tambien produce el video: lo congelaba de forma
        visible y, si la clasificacion se alargaba mas de watchdog_timeout_s,
        el watchdog reiniciaba este hilo a medio reconocimiento. Por eso se
        despacha a un hilo aparte (_classify_dynamic_sequence) que solo deja
        el resultado en _dynamic_result_queue; este metodo la revisa primero,
        sin bloquearse, antes de seguir con el frame actual.
        """
        assert self._auto_segmenter is not None and self._dtw_recognizer is not None

        now = time.perf_counter()

        # 1. Si una clasificacion en curso ya termino, se recoge aqui primero
        #    (sin bloquear: get_nowait). El top-3 se emite SIEMPRE en modo
        #    dinamico (no solo con el checkbox de diagnostico) y el texto de
        #    Estado se queda mostrando este resultado (comprometido o no)
        #    hasta que arranque una seña nueva, en vez de volver a "..." en
        #    el siguiente frame (antes duraba <33ms en pantalla).
        try:
            letter, conf, topk = self._dynamic_result_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._dynamic_classifying = False
            self.sign_diagnostic_signal.emit(topk)
            should_commit, margin, rule = dynamic_commit_decision(topk)
            top_str = "  ".join(f"{w} {c * 100:.1f}%" for w, c in topk)

            if rule == "experimental":
                # K, Q o Z comprometida por la regla EXPERIMENTAL de margen
                # (ver dynamic_commit_decision): se etiqueta distinto para que
                # quede claro que no vino de la regla normal de confianza.
                self.letter_committed_signal.emit(letter)
                self._last_committed_label = letter
                self._dynamic_idle_text = f"{letter} {conf * 100:.1f}% (margen alto)"
                log.info(
                    "[dinamico] top-3: %s -> agregada por regla EXPERIMENTAL (margen=%.1fpp >= %.0fpp)",
                    top_str, margin * 100, DYN_EXPERIMENTAL_MIN_MARGIN * 100,
                )
            elif rule == "experimental_nq":
                # Q comprometida contra Ñ en 2.º lugar, con el margen
                # reforzado DYN_NQ_PAIR_MIN_MARGIN (ver dynamic_commit_decision
                # e is_nq_blocking_pair): un margen tan amplio sobre Ñ
                # especificamente es señal solida incluso con el umbral mas
                # estricto de este par.
                self.letter_committed_signal.emit(letter)
                self._last_committed_label = letter
                self._dynamic_idle_text = f"{letter} {conf * 100:.1f}% (margen alto, par Ñ/Q)"
                log.info(
                    "[dinamico] top-3: %s -> agregada por regla EXPERIMENTAL reforzada, par Ñ/Q (margen=%.1fpp >= %.0fpp)",
                    top_str, margin * 100, DYN_NQ_PAIR_MIN_MARGIN * 100,
                )
            elif rule == "margen":
                # Grupo normal comprometido por margen amplio aunque la
                # confianza absoluta no llegara a DYN_MIN_CONF.
                self.letter_committed_signal.emit(letter)
                self._last_committed_label = letter
                self._dynamic_idle_text = f"{letter} {conf * 100:.1f}% (margen amplio)"
                log.info(
                    "[dinamico] top-3: %s -> agregada por margen amplio (margen=%.1fpp >= %.0fpp)",
                    top_str, margin * 100, DYN_NORMAL_MIN_MARGIN * 100,
                )
            elif rule == "confianza":
                self.letter_committed_signal.emit(letter)
                self._last_committed_label = letter
                self._dynamic_idle_text = f"{letter} ({conf * 100:.0f}%)"
                log.info("[dinamico] top-3: %s -> agregada (confianza=%.1f%%)", top_str, conf * 100)
            elif is_experimental_dynamic_letter(letter):
                # K, Q y Z (grupo experimental): no alcanzaron el margen de
                # la regla experimental (o, si es el par Ñ/Q, el margen
                # reforzado DYN_NQ_PAIR_MIN_MARGIN). Se siguen mostrando en
                # el top-3 para poder seguir evaluandolas.
                nq_pair = is_nq_blocking_pair(topk)
                sufijo = " - modo experimental, no se agregó (par Ñ/Q)" if nq_pair else " - modo experimental, no se agregó"
                self._dynamic_idle_text = f"¿{letter}? ({conf * 100:.0f}%){sufijo}"
                motivos = []
                if conf < DYN_EXPERIMENTAL_MIN_CONF:
                    motivos.append(f"confianza {conf * 100:.1f}% < {DYN_EXPERIMENTAL_MIN_CONF * 100:.0f}%")
                margen_requerido = DYN_NQ_PAIR_MIN_MARGIN if nq_pair else DYN_EXPERIMENTAL_MIN_MARGIN
                if margin < margen_requerido:
                    etiqueta_margen = "margen (par Ñ/Q, reforzado)" if nq_pair else "margen"
                    motivos.append(f"{etiqueta_margen} {margin * 100:.1f}pp < {margen_requerido * 100:.0f}pp")
                log.info(
                    "[dinamico] top-3: %s -> NO agregada (modo experimental, %s)",
                    top_str, "; ".join(motivos) or "umbral no alcanzado",
                )
            else:
                # Grupo normal: ni confianza ni margen alcanzaron su umbral.
                self._dynamic_idle_text = f"¿{letter}? ({conf * 100:.0f}%) - no agregada"
                log.info(
                    "[dinamico] top-3: %s -> NO agregada (confianza %.1f%% < %.0f%% y margen %.1fpp < %.0fpp)",
                    top_str, conf * 100, DYN_MIN_CONF * 100, margin * 100, DYN_NORMAL_MIN_MARGIN * 100,
                )
            return (letter, conf)

        # 2. Construir el vector de este frame y avanzar el segmentador.
        hands_by_side: dict[str, HandDetection] = {}
        for h in detections.hands:
            if h.handedness not in hands_by_side:
                hands_by_side[h.handedness] = h
        has_hand = bool(hands_by_side)

        if self._auto_segmenter.state == "esperando":
            # Este es el frame que, si has_hand, va a disparar "inicio" (ver
            # AutoSegmenter.push: la transicion es inmediata, sin espera de
            # varios frames). Fija la identidad de mano(s) de la secuencia
            # nueva ANTES de filtrar nada: aqui no hay nada que filtrar
            # todavia, este frame es el que define la identidad.
            self._dynamic_sequence_hand_identity = set(hands_by_side.keys()) if has_hand else None
        elif self._dynamic_sequence_hand_identity:
            # Seguimos dentro de la MISMA secuencia (state == "grabando"):
            # cualquier mano que aparezca y no estaba en la identidad
            # original se trata como ruido (se ignora, no se agrega al
            # vector) en vez de convertir esto en una postura de dos manos.
            for handedness in list(hands_by_side.keys()):
                if handedness not in self._dynamic_sequence_hand_identity:
                    del hands_by_side[handedness]

        vector = build_dynamic_feature_vector(hands_by_side)
        event = self._auto_segmenter.push(has_hand, vector, now)

        if event is None:
            if self._auto_segmenter.state == "grabando":
                self._dynamic_stats_update(has_hand, now)
                return ("Grabando...", 0.0)
            if self._dynamic_classifying:
                elapsed_s = now - self._dynamic_classify_start
                return (f"Clasificando... ({elapsed_s:.0f}s)", 0.0)
            return (self._dynamic_idle_text, 0.0)

        kind, sequence = event

        if kind == "inicio":
            self._dynamic_stats_reset(now)
            self._dynamic_stats_update(True, now)
            return ("Grabando...", 0.0)

        if kind == "fin_descartada":
            self._dynamic_stats_log(now, kind)
            self._dynamic_stats = None
            return (self._dynamic_idle_text, 0.0)

        # kind == "fin_valida"
        self._dynamic_stats_log(now, kind)
        self._dynamic_stats = None

        if self._dynamic_classifying:
            # Item 6: una seña nueva termino de grabarse mientras la anterior
            # todavia se estaba clasificando. Se descarta (no se encola) en
            # vez de lanzar una segunda clasificacion DTW en paralelo, para
            # mantener el orden de resultados simple y predecible; se avisa
            # tanto en consola como en el propio cuadro de Estado.
            log.warning(
                "[dinamico] seña descartada: la clasificacion anterior aun no termina (%d frames)",
                len(sequence),
            )
            self._dynamic_idle_text = "Seña descartada (clasificando la anterior)"
            return (self._dynamic_idle_text, 0.0)

        self._dynamic_classifying = True
        self._dynamic_classify_start = now
        threading.Thread(
            target=self._classify_dynamic_sequence, args=(sequence,), daemon=True
        ).start()
        return ("Clasificando... (0s)", 0.0)

    def _classify_dynamic_sequence(self, sequence: list[np.ndarray]) -> None:
        """Corre en un hilo aparte (no QThread): solo hace la clasificacion y
        deja el resultado en una queue.Queue, que es segura entre hilos y no
        involucra el mecanismo de senales/slots de Qt (HandTrackingThread.run
        es un bucle propio, no QThread.exec(), asi que una senal emitida
        desde otro hilo aqui no se entregaria de forma confiable)."""
        assert self._dtw_recognizer is not None
        try:
            topk = self._dtw_recognizer.predict_topk(sequence, k=3)
        except Exception as e:
            log.exception("Error en DTWRecognizer: %s", e)
            return
        letter, conf = topk[0]
        self._dynamic_result_queue.put((letter, conf, topk))


    def _init_mediapipe(self) -> bool:
        try:
            log.info("Inicializando MediaPipe Hand Landmarker (Tasks API)...")

            # Asegurar que el modelo esté descargado.
            models_dir = Path.home() / ".sign_translator" / "models"
            model_path = ensure_hand_model(models_dir)

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=self._cfg.max_num_hands,
                min_hand_detection_confidence=self._cfg.min_detection_confidence,
                min_hand_presence_confidence=self._cfg.min_detection_confidence,
                min_tracking_confidence=self._cfg.min_tracking_confidence,
            )
            self._hands_solution = HandLandmarker.create_from_options(options)
            log.info("MediaPipe Hand Landmarker listo.")
            self.model_loaded_signal.emit()
            return True
        except Exception as e:
            log.exception("Error inicializando MediaPipe")
            self.error_signal.emit(f"Error inicializando MediaPipe: {e}")
            return False

    def _parse_results(self, results) -> FrameDetections:
        det = FrameDetections()
        if not results.hand_landmarks:
            return det

        n_hands = len(results.hand_landmarks)
        for i in range(n_hands):
            hand_lms = results.hand_landmarks[i]

            handedness = "Right"
            confidence = 0.0
            if results.handedness and i < len(results.handedness):
                cat_list = results.handedness[i]
                if cat_list:
                    cat = cat_list[0]
                    handedness = cat.category_name  
                    confidence = cat.score

            lm_2d = np.array(
                [[lm.x, lm.y] for lm in hand_lms],
                dtype=np.float32,
            )

            lm_3d = np.zeros((21, 3), dtype=np.float32)
            if (
                results.hand_world_landmarks
                and i < len(results.hand_world_landmarks)
            ):
                world = results.hand_world_landmarks[i]
                lm_3d = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world],
                    dtype=np.float32,
                )

            det.hands.append(HandDetection(
                handedness=handedness,
                confidence=confidence,
                landmarks_2d=lm_2d,
                landmarks_3d=lm_3d,
            ))

        return det

    def _render(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        body: Optional["BodyDetection"] = None,
    ) -> np.ndarray:
        out = frame.copy()

        # Esqueleto de pose: se dibuja ANTES del "return out" de "sin manos"
        # de abajo, para que se vea aunque todavia no se levanten las manos.
        # Puramente visual: no cambia nada de lo que sigue (deteccion de
        # manos, clasificacion, ni el texto/banner de mas abajo).
        if self._cfg.draw_body_skeleton and body is not None and draw_body_skeleton is not None:
            try:
                draw_body_skeleton(out, body, detections.hands)
            except Exception:
                log.exception("Error dibujando esqueleto de pose")

            if self._diagnostic_mode and body_location_features is not None:
                hands_by_side: dict[str, HandDetection] = {}
                for h in detections.hands:
                    if h.handedness not in hands_by_side:
                        hands_by_side[h.handedness] = h
                h_frame, w_frame = out.shape[:2]
                vec = body_location_features(hands_by_side, body, w_frame, h_frame)
                texto = "Pose (b0..b8): " + " ".join(f"{v:+.2f}" for v in vec)
                y_texto = h_frame - 15
                cv2.putText(out, texto, (10, y_texto), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, texto, (10, y_texto), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 215, 255), 1, cv2.LINE_AA)

        if detections.num_hands == 0:
            cv2.putText(
                out, "Sin manos detectadas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                out, "Sin manos detectadas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
            )
            return out

        for hand in detections.hands:
            if self._cfg.draw_landmarks or self._cfg.draw_connections:
                draw_hand_landmarks(
                    out, hand,
                    draw_connections=self._cfg.draw_connections,
                    draw_points=self._cfg.draw_landmarks,
                )
            draw_hand_label(out, hand)

        banner = f"Manos: {detections.num_hands}"
        cv2.putText(
            out, banner, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            out, banner, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 102), 2, cv2.LINE_AA,
        )

        return out

    def _update_keypoint_buffer(self, detections: FrameDetections) -> None:
        max_h = self._cfg.max_num_hands
        vec = np.zeros((max_h * 21 * 3,), dtype=np.float32)

        slot_map: dict[str, HandDetection] = {}
        for h in detections.hands:
            if h.handedness not in slot_map:
                slot_map[h.handedness] = h

        for slot_idx, handedness in enumerate(("Left", "Right")):
            if slot_idx >= max_h:
                break
            if handedness in slot_map:
                hand = slot_map[handedness]
                offset = slot_idx * 21 * 3
                vec[offset:offset + 21 * 2] = hand.landmarks_2d.flatten()
                vec[offset + 21 * 2:offset + 21 * 3] = hand.landmarks_3d[:, 2]

        self._keypoint_buffer.append(vec)

    def _update_word_state(self, detections: FrameDetections) -> None:
        if detections.num_hands == 0:
            self._frames_without_hand += 1
            if (
                self._frames_without_hand >= self._cfg.no_hand_frames_for_space
                and not self._space_already_committed
                and self._last_committed_label is not None
            ):
                self.space_committed_signal.emit()
                self._space_already_committed = True
                self._last_committed_label = None
                self._stable_letter = None
                self._stable_frames = 0
                self._smoother.reset()
        else:
            self._frames_without_hand = 0
            self._space_already_committed = False

    def _update_metrics(self, dt: float, num_hands: int) -> None:
        if dt <= 0:
            return
        self._latencies.append(dt * 1000.0)
        self._frame_times.append(1.0 / dt)
        self._hand_counts.append(num_hands)
        now = time.perf_counter()
        if now - self._last_metrics_emit < 0.25:
            return
        self._last_metrics_emit = now
        try:
            m = InferenceMetrics(
                fps=statistics.mean(self._frame_times) if self._frame_times else 0.0,
                latency_p50_ms=statistics.median(self._latencies) if self._latencies else 0.0,
                latency_p95_ms=(
                    statistics.quantiles(self._latencies, n=20)[18]
                    if len(self._latencies) >= 20 else max(self._latencies, default=0.0)
                ),
                num_hands_avg=statistics.mean(self._hand_counts) if self._hand_counts else 0.0,
                last_inference_ts=time.time(),
            )
            self.metrics_signal.emit(m)
        except statistics.StatisticsError:
            pass

    def stop(self) -> None:
        self._run_flag = False
        self.wait(3000)
        if self._hands_solution is not None:
            try:
                self._hands_solution.close()
            except Exception:
                pass
        if self._body_tracker is not None:
            try:
                self._body_tracker.close()
            except Exception:
                pass


# =========================================================================== #
# Ventana principal
# =========================================================================== #

class SignLanguageApp(QMainWindow):
    def __init__(self, config: AppConfig, config_path: Optional[Path] = None):
        super().__init__()
        self.cfg = config
        self.config_path = config_path
        self.settings = QSettings(APP_ORG, APP_NAME)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.cfg.queue_maxsize)
        self.camera_thread: Optional[CameraThread] = None
        self.ai_thread: Optional[HandTrackingThread] = None

        self.current_word = ""
        self.history: list[str] = []
        self._last_annotated_frame: Optional[np.ndarray] = None

        self._last_heartbeat = time.time()
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(1000)
        self._watchdog.timeout.connect(self._check_watchdog)
        self._watchdog_active = False

        self._available_cameras: list[int] = []
        self._dynamic_mode_enabled = False

        self.setWindowTitle(f"Traductor LSM v{APP_VERSION}")
        self.setMinimumSize(QSize(1100, 720))

        self._build_ui()
        self._restore_window_state()

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_central_widget()
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Principal")
        toolbar.setObjectName("MainToolBar")  
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.action_start = QAction("▶ Iniciar", self)
        self.action_start.setShortcut(QKeySequence("Ctrl+R"))
        self.action_start.triggered.connect(self.start_system)
        toolbar.addAction(self.action_start)

        self.action_stop = QAction("■ Detener", self)
        self.action_stop.setShortcut(QKeySequence("Ctrl+T"))
        self.action_stop.triggered.connect(self.stop_system)
        self.action_stop.setEnabled(False)
        toolbar.addAction(self.action_stop)

        toolbar.addSeparator()

        self.action_dynamic_mode = QAction("🤟 Alfabeto dinámico (J K Ñ Q X Z)", self)
        self.action_dynamic_mode.setCheckable(True)
        self.action_dynamic_mode.setShortcut(QKeySequence("Ctrl+D"))
        self.action_dynamic_mode.setToolTip(
            "Alterna entre el alfabeto estático (A-Y, frame a frame) y el\n"
            "alfabeto dinámico (J,K,Ñ,Q,X,Z, señas con movimiento)."
        )
        self.action_dynamic_mode.toggled.connect(self._on_toggle_dynamic_mode)
        toolbar.addAction(self.action_dynamic_mode)

        toolbar.addSeparator()

        action_screenshot = QAction("📷 Captura", self)
        action_screenshot.setShortcut(QKeySequence("Ctrl+S"))
        action_screenshot.triggered.connect(self.save_screenshot)
        toolbar.addAction(action_screenshot)

        action_export = QAction("💾 Exportar historial", self)
        action_export.triggered.connect(self.export_history)
        toolbar.addAction(action_export)

        toolbar.addSeparator()

        action_clear = QAction("⌫ Borrar palabra", self)
        action_clear.setShortcut(QKeySequence("Ctrl+Backspace"))
        action_clear.triggered.connect(self.clear_current_word)
        toolbar.addAction(action_clear)

        action_space = QAction("␣ Espacio manual", self)
        action_space.setShortcut(QKeySequence("Ctrl+Space"))
        action_space.triggered.connect(self.insert_space)
        toolbar.addAction(action_space)

    def _build_central_widget(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Panel video.
        video_box = QVBoxLayout()
        self.image_label = QLabel("Pulsa Iniciar (Ctrl+R) para comenzar")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888; font-size: 14px; border-radius: 8px;"
        )
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_box.addWidget(self.image_label, stretch=1)

        word_frame = QFrame()
        word_frame.setFrameShape(QFrame.Shape.StyledPanel)
        word_layout = QVBoxLayout(word_frame)
        word_title = QLabel("PALABRA EN CONSTRUCCIÓN")
        word_title.setStyleSheet("font-size: 11px; color: #777; font-weight: bold;")
        self.word_label = QLabel("")
        self.word_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #1a5490; "
            "letter-spacing: 4px; padding: 8px;"
        )
        self.word_label.setMinimumHeight(60)
        word_layout.addWidget(word_title)
        word_layout.addWidget(self.word_label)
        video_box.addWidget(word_frame)

        # Panel lateral.
        side = QVBoxLayout()

        title = QLabel("ESTADO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #555;")
        side.addWidget(title)

        self.sign_label = QLabel("—")
        self.sign_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_label.setStyleSheet(
            "font-size: 32px; color: #2E86C1; font-weight: bold;"
            "border: 2px solid #d0d0d0; border-radius: 12px; padding: 18px;"
        )
        self.sign_label.setMinimumHeight(100)
        side.addWidget(self.sign_label)

        self.classifier_info_label = QLabel(self._classifier_status_text())
        self.classifier_info_label.setWordWrap(True)
        self.classifier_info_label.setStyleSheet("color: #777; font-size: 11px; font-style: italic;")
        self.classifier_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side.addWidget(self.classifier_info_label)

        side.addSpacing(8)
        clf_title = QLabel("AJUSTES DE RECONOCIMIENTO")
        clf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clf_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #555;")
        side.addWidget(clf_title)

        stab_row = QHBoxLayout()
        stab_label = QLabel("Estabilidad:")
        stab_label.setToolTip(
            "Cuántos frames seguidos debe mantenerse una letra para fijarla.\n"
            "Bajo = más rápido, más errores. Alto = más lento, más seguro."
        )
        stab_row.addWidget(stab_label)
        self.stable_frames_slider = QSlider(Qt.Orientation.Horizontal)
        self.stable_frames_slider.setRange(3, 25)
        self.stable_frames_slider.setValue(self.cfg.stable_frames_to_commit)
        self.stable_frames_slider.valueChanged.connect(self._on_stable_frames_changed)
        self.stable_frames_value_label = QLabel(str(self.cfg.stable_frames_to_commit))
        self.stable_frames_value_label.setMinimumWidth(28)
        stab_row.addWidget(self.stable_frames_slider, stretch=1)
        stab_row.addWidget(self.stable_frames_value_label)
        side.addLayout(stab_row)

        conf_row = QHBoxLayout()
        conf_label = QLabel("Conf. mín:")
        conf_label.setToolTip(
            "Qué tan seguro debe estar el modelo para que la letra cuente.\n"
            "Alto = solo letras muy claras. Bajo = acepta predicciones inseguras."
        )
        conf_row.addWidget(conf_label)
        self.min_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_confidence_slider.setRange(30, 90)
        self.min_confidence_slider.setValue(55)
        self.min_confidence_slider.valueChanged.connect(self._on_min_confidence_changed)
        self.min_confidence_value_label = QLabel("0.55")
        self.min_confidence_value_label.setMinimumWidth(40)
        conf_row.addWidget(self.min_confidence_slider, stretch=1)
        conf_row.addWidget(self.min_confidence_value_label)
        side.addLayout(conf_row)

        margin_row = QHBoxLayout()
        margin_label = QLabel("Margen:")
        margin_label.setToolTip(
            "Diferencia mínima entre la 1ª y 2ª letra más probables.\n"
            "Alto = bloquea cuando el modelo duda (M vs N, V vs W).\n"
            "Bajo = acepta predicciones aunque sean parejas."
        )
        margin_row.addWidget(margin_label)
        self.min_margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_margin_slider.setRange(0, 50)
        self.min_margin_slider.setValue(15)
        self.min_margin_slider.valueChanged.connect(self._on_min_margin_changed)
        self.min_margin_value_label = QLabel("0.15")
        self.min_margin_value_label.setMinimumWidth(40)
        margin_row.addWidget(self.min_margin_slider, stretch=1)
        margin_row.addWidget(self.min_margin_value_label)
        side.addLayout(margin_row)

        # Toggle: modo diagnóstico
        self.cb_diagnostic = QCheckBox("Modo diagnóstico (mostrar top-3)")
        self.cb_diagnostic.setToolTip(
            "Muestra las 3 letras más probables en cada frame.\n"
            "Útil para entender por qué una letra falla."
        )
        self.cb_diagnostic.toggled.connect(self._on_diagnostic_toggled)
        side.addWidget(self.cb_diagnostic)

        self.diagnostic_label = QLabel("")
        self.diagnostic_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #888;"
            "background: #f5f5f5; border-radius: 4px; padding: 6px;"
        )
        self.diagnostic_label.setMinimumHeight(48)
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.hide()
        side.addWidget(self.diagnostic_label)

        side.addSpacing(8)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Cámara:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(f"#{self.cfg.camera_index}", self.cfg.camera_index)
        self.refresh_cameras_btn = QPushButton("↻")
        self.refresh_cameras_btn.setFixedWidth(32)
        self.refresh_cameras_btn.setToolTip("Buscar cámaras conectadas")
        self.refresh_cameras_btn.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(self.camera_combo, stretch=1)
        cam_row.addWidget(self.refresh_cameras_btn)
        side.addLayout(cam_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Confianza:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(int(self.cfg.min_detection_confidence * 100))
        self.threshold_slider.setToolTip("Confianza mínima para detectar una mano")
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_value_label = QLabel(f"{int(self.cfg.min_detection_confidence * 100)}%")
        self.threshold_value_label.setMinimumWidth(40)
        thr_row.addWidget(self.threshold_slider, stretch=1)
        thr_row.addWidget(self.threshold_value_label)
        side.addLayout(thr_row)

        self.cb_landmarks = QCheckBox("Dibujar puntos (círculos)")
        self.cb_landmarks.setChecked(self.cfg.draw_landmarks)
        self.cb_landmarks.toggled.connect(self._on_draw_landmarks)
        side.addWidget(self.cb_landmarks)

        self.cb_connections = QCheckBox("Dibujar conexiones (estructura)")
        self.cb_connections.setChecked(self.cfg.draw_connections)
        self.cb_connections.toggled.connect(self._on_draw_connections)
        side.addWidget(self.cb_connections)

        self.cb_body_skeleton = QCheckBox("Dibujar esqueleto (Pose)")
        self.cb_body_skeleton.setToolTip(
            "Solo visual/diagnostico (MediaPipe Pose). No afecta el "
            "reconocimiento de letras estaticas ni dinamicas."
        )
        self.cb_body_skeleton.setChecked(self.cfg.draw_body_skeleton)
        self.cb_body_skeleton.toggled.connect(self._on_draw_body_skeleton)
        side.addWidget(self.cb_body_skeleton)

        side.addSpacing(8)

        side.addWidget(QLabel("HISTORIAL"))
        self.history_view = QPlainTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setMaximumHeight(150)
        self.history_view.setStyleSheet("font-family: monospace; font-size: 12px;")
        side.addWidget(self.history_view)

        side.addStretch()

        main.addLayout(video_box, stretch=7)
        main.addLayout(side, stretch=3)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.status_engine = QLabel("⚙ MediaPipe Hands")
        self.status_camera = QLabel("● Cámara: detenida")
        self.status_hands = QLabel("✋ Manos: 0")
        self.status_fps = QLabel("FPS: —")
        self.status_latency = QLabel("Latencia: —")

        bar.addPermanentWidget(self.status_engine)
        bar.addPermanentWidget(self._sep())
        bar.addPermanentWidget(self.status_camera)
        bar.addPermanentWidget(self._sep())
        bar.addPermanentWidget(self.status_hands)
        bar.addPermanentWidget(self._sep())
        bar.addPermanentWidget(self.status_fps)
        bar.addPermanentWidget(self._sep())
        bar.addPermanentWidget(self.status_latency)

    @staticmethod
    def _sep() -> QFrame:
        s = QFrame()
        s.setFrameShape(QFrame.Shape.VLine)
        s.setFrameShadow(QFrame.Shadow.Sunken)
        return s

    def _dynamic_recognizer_available(self) -> bool:
        models_dir = Path(__file__).resolve().parent
        dynamic_dir = models_dir / "datos_dinamicas"
        return (
            AutoSegmenter is not None and DTWRecognizer is not None
            and dynamic_dir.is_dir() and any(dynamic_dir.iterdir())
        )

    def _static_help_text(self) -> str:
        from sign_classifier import MODEL_FILENAME, LABELS_FILENAME
        models_dir = Path(__file__).resolve().parent
        onnx_path = models_dir / MODEL_FILENAME
        labels_path = models_dir / LABELS_FILENAME

        dynamic_line = (
            "Alfabeto dinámico (J,K,Ñ,Q,X,Z) disponible: Ctrl+D para activarlo."
            if self._dynamic_recognizer_available() else
            "Alfabeto dinámico no disponible (faltan plantillas o fastdtw/scipy)."
        )

        if onnx_path.exists() and labels_path.exists():
            return (
                "Modo ESTÁTICO activo (alfabeto, 21 letras A-Y).\n"
                "Mantén una seña ~12 frames para fijar la letra.\n"
                "Baja las manos ~25 frames para insertar un espacio.\n"
                f"{dynamic_line}"
            )
        return (
            "Sin modelo de clasificación cargado.\n"
            f"Coloca {MODEL_FILENAME} y {LABELS_FILENAME} en:\n{models_dir}\n"
            f"{dynamic_line}"
        )

    def _dynamic_help_text(self) -> str:
        labels = self.ai_thread.dynamic_labels if self.ai_thread is not None else []
        letras = ", ".join(labels) if labels else "J, K, Ñ, Q, X, Z"
        return (
            f"Modo DINÁMICO activo ({letras}).\n"
            "1) Levanta la mano y haz la seña completa.\n"
            "2) Bájala al terminar: se clasifica sola, sin presionar nada.\n"
            f"Fin de seña tras ~{DYN_NO_HAND_MS_TO_END}ms sin mano "
            f"(tope máx. {DYN_MAX_SEQUENCE_MS / 1000:.0f}s por seña).\n"
            "Ctrl+D para volver al alfabeto estático."
        )

    def _classifier_status_text(self) -> str:
        # El texto de ayuda cambia segun el modo activo (item 1 del pedido):
        # explica el flujo dinamico (levantar mano / señar / bajar mano)
        # cuando ese modo esta encendido, o el estatico en caso contrario.
        if self._dynamic_mode_enabled:
            return self._dynamic_help_text()
        return self._static_help_text()


    def _restore_window_state(self) -> None:
        geom = self.settings.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self.settings.value("window/state")
        if state:
            self.restoreState(state)

    def _save_window_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())


    def _wire_ai_thread(self) -> None:
        assert self.ai_thread is not None
        self.ai_thread.change_pixmap_signal.connect(self.update_image)
        self.ai_thread.sign_detected_signal.connect(self.update_sign)
        self.ai_thread.sign_diagnostic_signal.connect(self._on_diagnostic_update)
        self.ai_thread.hands_detected_signal.connect(self.update_hands)
        self.ai_thread.letter_committed_signal.connect(self.on_letter_committed)
        self.ai_thread.space_committed_signal.connect(self.on_space_committed)
        self.ai_thread.metrics_signal.connect(self.update_metrics)
        self.ai_thread.error_signal.connect(self._on_ai_error)
        self.ai_thread.model_loaded_signal.connect(self._on_model_loaded)
        self.ai_thread.heartbeat_signal.connect(self._on_heartbeat)

    def start_system(self) -> None:
        if self.camera_thread is not None or self.ai_thread is not None:
            return

        self.cfg.camera_index = int(self.camera_combo.currentData())

        self.action_start.setEnabled(False)
        self.status_camera.setText("● Conectando...")

        self.camera_thread = CameraThread(self.frame_queue, self.cfg.camera_index)
        self.ai_thread = HandTrackingThread(self.frame_queue, self.cfg)

        self.ai_thread.set_stable_frames_to_commit(self.stable_frames_slider.value())
        self.ai_thread.set_min_confidence(self.min_confidence_slider.value() / 100.0)
        self.ai_thread.set_min_margin(self.min_margin_slider.value() / 100.0)
        self.ai_thread.set_diagnostic_mode(self.cb_diagnostic.isChecked())
        self._apply_pending_dynamic_mode()

        self.camera_thread.error_signal.connect(self._on_camera_error)
        self.camera_thread.status_signal.connect(lambda s: self.status_camera.setText(f"● {s}"))
        self.camera_thread.camera_lost_signal.connect(
            lambda: self.status_camera.setText("● Cámara perdida, reintentando")
        )
        self.camera_thread.camera_recovered_signal.connect(
            lambda: self.status_camera.setText("● Cámara recuperada")
        )

        self._wire_ai_thread()

        self.ai_thread.start()
        self.camera_thread.start()

        self._last_heartbeat = time.time()
        self._watchdog_active = True
        self._watchdog.start()

        self.action_stop.setEnabled(True)

    def stop_system(self) -> None:
        self._watchdog_active = False
        self._watchdog.stop()

        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

        if self.ai_thread is not None:
            self.ai_thread.stop()
            self.ai_thread = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.image_label.setText("Cámara detenida")
        self.image_label.setPixmap(QPixmap())
        self.sign_label.setText("—")
        self.status_camera.setText("● Cámara: detenida")
        self.status_hands.setText("✋ Manos: 0")
        self.status_fps.setText("FPS: —")
        self.status_latency.setText("Latencia: —")

        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)

    def _restart_ai_thread(self) -> None:
        log.warning("Watchdog: reiniciando hilo de IA")
        if self.ai_thread is not None:
            self.ai_thread.stop()
        self.ai_thread = HandTrackingThread(self.frame_queue, self.cfg)
        self._wire_ai_thread()
        self.ai_thread.set_stable_frames_to_commit(self.stable_frames_slider.value())
        self.ai_thread.set_min_confidence(self.min_confidence_slider.value() / 100.0)
        self.ai_thread.set_min_margin(self.min_margin_slider.value() / 100.0)
        self.ai_thread.set_diagnostic_mode(self.cb_diagnostic.isChecked())
        self._apply_pending_dynamic_mode()
        self.ai_thread.start()
        self._last_heartbeat = time.time()
        self.statusBar().showMessage("IA reiniciada por inactividad", 3000)


    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_value_label.setText(f"{value}%")
        self.cfg.min_detection_confidence = value / 100.0

    def _warn_dynamic_unavailable(self) -> None:
        QMessageBox.warning(
            self, "Alfabeto dinámico no disponible",
            "No se encontraron plantillas dinámicas (datos_dinamicas/) o "
            "falta instalar fastdtw/scipy.\n\n"
            "Corre recolector_dinamico.py o procesar_dataset_dinamico.py, "
            "y revisa la consola al iniciar la app para más detalle.",
        )
        self._dynamic_mode_enabled = False
        self.action_dynamic_mode.blockSignals(True)
        self.action_dynamic_mode.setChecked(False)
        self.action_dynamic_mode.blockSignals(False)
        self.classifier_info_label.setText(self._classifier_status_text())

    def _on_toggle_dynamic_mode(self, checked: bool) -> None:
        # Solo podemos saber si el reconocedor dinamico esta disponible una
        # vez que existe ai_thread (se crea al Iniciar). Si todavia no existe,
        # guardamos la preferencia y se valida/aplica en start_system() via
        # _apply_pending_dynamic_mode().
        if checked and self.ai_thread is not None and not self.ai_thread.has_dynamic_recognizer:
            self._warn_dynamic_unavailable()
            return

        self._dynamic_mode_enabled = checked
        if self.ai_thread is not None:
            self.ai_thread.set_dynamic_mode(checked)

        # El texto de ayuda de abajo del Estado y el panel top-3 dependen del
        # modo activo (items 1 y 2 del pedido), no solo del checkbox de
        # diagnostico estatico.
        self.classifier_info_label.setText(self._classifier_status_text())
        self.sign_label.setText("Esperando mano" if checked else "—")

        if checked:
            self.diagnostic_label.show()
            self.diagnostic_label.setText("Esperando seña...")
            labels = self.ai_thread.dynamic_labels if self.ai_thread is not None else []
            self.statusBar().showMessage(f"Modo dinámico activo ({', '.join(labels)})", 4000)
        else:
            if not self.cb_diagnostic.isChecked():
                self.diagnostic_label.hide()
            self.statusBar().showMessage("Modo estático activo (alfabeto A-Y)", 3000)

    def _apply_pending_dynamic_mode(self) -> None:
        assert self.ai_thread is not None
        if self._dynamic_mode_enabled and not self.ai_thread.has_dynamic_recognizer:
            self._warn_dynamic_unavailable()
            return
        self.ai_thread.set_dynamic_mode(self._dynamic_mode_enabled)

    def _on_draw_landmarks(self, checked: bool) -> None:
        self.cfg.draw_landmarks = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_landmarks(checked)

    def _on_draw_connections(self, checked: bool) -> None:
        self.cfg.draw_connections = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_connections(checked)

    def _on_draw_body_skeleton(self, checked: bool) -> None:
        self.cfg.draw_body_skeleton = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_body_skeleton(checked)

    def _on_stable_frames_changed(self, value: int) -> None:
        self.stable_frames_value_label.setText(str(value))
        self.cfg.stable_frames_to_commit = value
        if self.ai_thread is not None:
            self.ai_thread.set_stable_frames_to_commit(value)

    def _on_min_confidence_changed(self, value: int) -> None:
        v = value / 100.0
        self.min_confidence_value_label.setText(f"{v:.2f}")
        if self.ai_thread is not None:
            self.ai_thread.set_min_confidence(v)

    def _on_min_margin_changed(self, value: int) -> None:
        v = value / 100.0
        self.min_margin_value_label.setText(f"{v:.2f}")
        if self.ai_thread is not None:
            self.ai_thread.set_min_margin(v)

    def _on_diagnostic_toggled(self, checked: bool) -> None:
        if checked:
            self.diagnostic_label.show()
            self.diagnostic_label.setText("Esperando seña...")
        else:
            self.diagnostic_label.hide()
        if self.ai_thread is not None:
            self.ai_thread.set_diagnostic_mode(checked)

    def _on_diagnostic_update(self, topk: list) -> None:
        if not topk:
            return
        lines = []
        for i, (letter, conf) in enumerate(topk[:3]):
            bar_len = int(conf * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"{i+1}. {letter}  {bar} {conf*100:5.1f}%")

        if self._dynamic_mode_enabled:
            # Item 2: top-3 siempre visible al terminar cada clasificacion
            # dinamica (no solo con el checkbox de diagnostico), marcando si
            # se agrego la letra o no y por que, con el mismo criterio
            # (dynamic_commit_decision) que usa HandTrackingThread para
            # decidir el commit real.
            should_commit, margin, rule = dynamic_commit_decision(topk)
            if rule == "experimental":
                lines.append(f"→ agregada por regla experimental (margen {margin * 100:.1f}pp)")
            elif rule == "experimental_nq":
                lines.append(f"→ agregada por regla experimental reforzada, par Ñ/Q (margen {margin * 100:.1f}pp)")
            elif rule == "margen":
                lines.append(f"→ agregada por margen amplio (margen {margin * 100:.1f}pp)")
            elif rule == "confianza":
                lines.append(f"→ agregada (confianza {topk[0][1] * 100:.1f}%)")
            elif is_experimental_dynamic_letter(topk[0][0]):
                if is_nq_blocking_pair(topk):
                    lines.append("→ par Ñ/Q: margen insuficiente, no se agregó (regla reforzada)")
                else:
                    lines.append("→ modo experimental, no se agregó")
            else:
                lines.append("→ baja confianza, no se agregó")
            self.diagnostic_label.show()

        self.diagnostic_label.setText("\n".join(lines))

    def _on_model_loaded(self) -> None:
        self.statusBar().showMessage("MediaPipe Hands listo", 3000)

    def _on_camera_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error de cámara", msg)
        self.stop_system()

    def _on_ai_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error del modelo", msg)
        self.stop_system()

    def _on_heartbeat(self) -> None:
        self._last_heartbeat = time.time()

    def _check_watchdog(self) -> None:
        if not self._watchdog_active:
            return
        if time.time() - self._last_heartbeat > self.cfg.watchdog_timeout_s:
            self._restart_ai_thread()

    def update_sign(self, text: str, conf: float) -> None:
        self.sign_label.setText(text)

    def update_hands(self, detections: FrameDetections) -> None:
        self.status_hands.setText(f"✋ Manos: {detections.num_hands}")

    def update_metrics(self, m: InferenceMetrics) -> None:
        self.status_fps.setText(f"FPS: {m.fps:.1f}")
        self.status_latency.setText(f"Latencia: {m.latency_p50_ms:.0f}/{m.latency_p95_ms:.0f}ms")

    def update_image(self, cv_img: np.ndarray) -> None:
        if cv_img is None or cv_img.size == 0:
            return
        self._last_annotated_frame = cv_img
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        scaled = qt_img.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))


    def on_letter_committed(self, letter: str) -> None:
        if len(letter) > 1 and self.current_word and not self.current_word.endswith(" "):
            self.current_word += " "
        self.current_word += letter
        self.word_label.setText(self.current_word)

    def on_space_committed(self) -> None:
        if not self.current_word.strip():
            return
        word = self.current_word.strip()
        self.history.append(word)
        self.history_view.appendPlainText(word)
        self.current_word = ""
        self.word_label.setText("")
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state()

    def clear_current_word(self) -> None:
        self.current_word = ""
        self.word_label.setText("")
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state()

    def insert_space(self) -> None:
        self.on_space_committed()


    def _refresh_cameras(self) -> None:
        self.refresh_cameras_btn.setEnabled(False)
        self.statusBar().showMessage("Buscando cámaras...", 2000)
        QApplication.processEvents()
        try:
            self._available_cameras = list_available_cameras(max_check=5)
        finally:
            self.refresh_cameras_btn.setEnabled(True)

        if not self._available_cameras:
            QMessageBox.warning(self, "Sin cámaras", "No se detectaron cámaras conectadas.")
            return

        current = self.camera_combo.currentData()
        self.camera_combo.clear()
        for idx in self._available_cameras:
            self.camera_combo.addItem(f"Cámara #{idx}", idx)
        if current in self._available_cameras:
            self.camera_combo.setCurrentIndex(self._available_cameras.index(current))


    def save_screenshot(self) -> None:
        if self._last_annotated_frame is None:
            QMessageBox.information(self, "Sin imagen", "Aún no hay frame disponible.")
            return
        default = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar captura", default, "Imagen PNG (*.png)"
        )
        if not path:
            return
        ok = cv2.imwrite(path, self._last_annotated_frame)
        if ok:
            self.statusBar().showMessage(f"Captura guardada: {path}", 3000)
        else:
            QMessageBox.warning(self, "Error", "No se pudo guardar la captura.")

    def export_history(self) -> None:
        if not self.history and not self.current_word:
            QMessageBox.information(self, "Sin historial", "Aún no hay palabras traducidas.")
            return
        default = f"traduccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar historial", default, "Texto (*.txt)"
        )
        if not path:
            return
        try:
            content = "\n".join(self.history)
            if self.current_word.strip():
                content += f"\n[en curso] {self.current_word.strip()}"
            Path(path).write_text(content + "\n", encoding="utf-8")
            self.statusBar().showMessage(f"Historial exportado: {path}", 3000)
        except OSError as e:
            QMessageBox.warning(self, "Error", f"No se pudo guardar: {e}")


    def closeEvent(self, event) -> None:
        log.info("Cerrando aplicación")
        self._save_window_state()
        if self.config_path is not None:
            self.cfg.save(self.config_path)
        self.stop_system()
        event.accept()


# =========================================================================== #
# CLI y main
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traductor de Lengua de Señas (MediaPipe)")
    p.add_argument("--camera", type=int, help="Índice de la cámara")
    p.add_argument("--config", type=Path, help="Archivo de configuración JSON")
    p.add_argument("--threshold", type=float, help="Confianza mínima de detección (0-1)")
    p.add_argument("--max-hands", type=int, help="Número máximo de manos a detectar")
    p.add_argument("-v", "--verbose", action="store_true", help="Logs detallados")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = args.config or Path.home() / ".sign_translator" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig.load(config_path)
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.threshold is not None:
        cfg.min_detection_confidence = args.threshold
    if args.max_hands is not None:
        cfg.max_num_hands = args.max_hands

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")

    window = SignLanguageApp(cfg, config_path=config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())