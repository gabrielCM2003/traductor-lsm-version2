from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import platform
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import deque
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
    QCheckBox, QProgressBar,
)

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: falta 'mediapipe'. Instala con: pip install mediapipe", file=sys.stderr)
    raise

from sign_classifier import (
    SignClassifier, PredictionSmoother, LetterCommitter,
    MODEL_FILENAME, LABELS_FILENAME,
)


APP_NAME = "SignTranslator"
APP_ORG = "OpenLSM"
APP_VERSION = "3.2-lsm-classifier"

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
    "smoothing_window": 7,
    "stable_frames_to_commit": 12,
    "no_hand_frames_for_space": 25,
    "repeat_release_frames": 8,        # pausa para repetir letra (LL, RR, EE)
    "hand_lost_reset_frames": 5,       # sin mano: se descarta el progreso
    "min_letter_confidence": 0.55,
    "min_letter_margin": 0.15,
    "dominant_hand": "Right",          # mano que deletrea: "Right" | "Left"
    "speak_words": False,
    "queue_maxsize": 1,
    "watchdog_timeout_s": 5.0,
    "draw_landmarks": True,
    "draw_connections": True,
}

# Rangos válidos al cargar config.json / CLI. Los que tienen slider usan su
# mismo rango. queue_maxsize >= 1: con 0 la cola sería infinita.
CONFIG_RANGES: dict[str, tuple[float, float]] = {
    "camera_index": (0, 63),
    "max_num_hands": (1, 4),
    "min_detection_confidence": (0.10, 0.95),
    "min_tracking_confidence": (0.0, 1.0),
    "smoothing_window": (5, 30),
    "stable_frames_to_commit": (3, 25),
    "no_hand_frames_for_space": (5, 300),
    "repeat_release_frames": (2, 60),
    "hand_lost_reset_frames": (1, 60),
    "min_letter_confidence": (0.30, 0.90),
    "min_letter_margin": (0.0, 0.50),
    "queue_maxsize": (1, 10),
    "watchdog_timeout_s": (2.0, 60.0),
}
CONFIG_CHOICES: dict[str, tuple[str, ...]] = {
    "dominant_hand": ("Right", "Left"),
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
    smoothing_window: int = DEFAULT_CONFIG["smoothing_window"]
    stable_frames_to_commit: int = DEFAULT_CONFIG["stable_frames_to_commit"]
    no_hand_frames_for_space: int = DEFAULT_CONFIG["no_hand_frames_for_space"]
    repeat_release_frames: int = DEFAULT_CONFIG["repeat_release_frames"]
    hand_lost_reset_frames: int = DEFAULT_CONFIG["hand_lost_reset_frames"]
    min_letter_confidence: float = DEFAULT_CONFIG["min_letter_confidence"]
    min_letter_margin: float = DEFAULT_CONFIG["min_letter_margin"]
    dominant_hand: str = DEFAULT_CONFIG["dominant_hand"]
    speak_words: bool = DEFAULT_CONFIG["speak_words"]
    queue_maxsize: int = DEFAULT_CONFIG["queue_maxsize"]
    watchdog_timeout_s: float = DEFAULT_CONFIG["watchdog_timeout_s"]
    draw_landmarks: bool = DEFAULT_CONFIG["draw_landmarks"]
    draw_connections: bool = DEFAULT_CONFIG["draw_connections"]

    @classmethod
    def load(cls, json_path: Optional[Path] = None) -> "AppConfig":
        cfg = cls()
        if not (json_path and json_path.exists()):
            return cfg
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("No se pudo leer %s: %s. Usando defaults.", json_path, e)
            return cfg
        if not isinstance(data, dict):
            log.warning("%s no contiene un objeto JSON. Usando defaults.", json_path)
            return cfg
        for k, v in data.items():
            if k in DEFAULT_CONFIG:
                cfg.set_validated(k, v)
            else:
                log.debug("Config: clave desconocida ignorada: %s", k)
        log.info("Configuración cargada desde %s", json_path)
        return cfg

    def set_validated(self, name: str, value) -> bool:
        """Asigna `value` si tiene el tipo correcto; lo recorta a su rango.

        Un valor inválido se ignora (con aviso) y se conserva el actual.
        """
        default = DEFAULT_CONFIG[name]
        # bool va primero: en Python bool es subclase de int.
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            if ok:
                value = float(value)
        else:
            ok = isinstance(value, str)
            if ok and name in CONFIG_CHOICES:
                value = value.capitalize()
                ok = value in CONFIG_CHOICES[name]

        if not ok:
            log.warning(
                "Config: valor inválido para %s: %r (se mantiene %r)",
                name, value, getattr(self, name),
            )
            return False

        if name in CONFIG_RANGES:
            lo, hi = CONFIG_RANGES[name]
            clamped = type(default)(min(max(value, lo), hi))
            if clamped != value:
                log.warning(
                    "Config: %s=%r fuera de rango [%s, %s]; se usa %r",
                    name, value, lo, hi, clamped,
                )
            value = clamped

        setattr(self, name, value)
        return True

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

def _picamera2_has_camera() -> bool:
    """True si Picamera2 está instalado Y detecta al menos una cámara CSI.

    Raspberry Pi OS trae Picamera2 preinstalado aunque solo haya una webcam
    USB, así que importar el módulo no basta para decidir usar libcamera.
    """
    try:
        from picamera2 import Picamera2  # type: ignore
        return len(Picamera2.global_camera_info()) > 0
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
            # Ojo: en Picamera2 el formato "RGB888" se guarda en memoria como
            # [B, G, R] (nomenclatura de libcamera/DRM), es decir, ya es BGR
            # como espera OpenCV. Convertirlo intercambiaría rojo y azul.
            frame = self._picam.capture_array()
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
    # (los nodos suelen ser del ISP, no del sensor). Si Picamera2 detecta una
    # cámara CSI, la exponemos como índice 0.
    if _is_raspberry_pi() and _picamera2_has_camera():
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


def draw_hand_label(image: np.ndarray, hand: HandDetection, active: bool = False) -> None:
    """Etiqueta la mano; `active` marca (en amarillo) la que se está clasificando."""
    h, w = image.shape[:2]
    wrist_px = (
        int(hand.landmarks_2d[0, 0] * w),
        int(hand.landmarks_2d[0, 1] * h),
    )
    # hand.handedness ya es la mano real del usuario (ver _parse_results).
    label_text = "Derecha" if hand.handedness == "Right" else "Izquierda"
    text = f"{label_text} ({hand.confidence:.0%})"
    if active:
        text = f"> {text}"
    color = (0, 255, 255) if active else (255, 255, 255)
    cv2.putText(
        image, text, (wrist_px[0] - 30, wrist_px[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (wrist_px[0] - 30, wrist_px[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
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

    def _sleep(self, seconds: float) -> None:
        """time.sleep interrumpible: stop() no debe esperar a que acabe."""
        end = time.monotonic() + seconds
        while self._run_flag and time.monotonic() < end:
            time.sleep(0.05)

    def _open_camera(self):
        # En Raspberry Pi con cámara CSI, usar Picamera2 (libcamera) en lugar
        # de V4L2, que no expone el sensor de forma fiable en la Pi 5.
        if _is_raspberry_pi() and _picamera2_has_camera():
            cap = PiCameraCapture(width=640, height=480, framerate=30)
            if cap.isOpened():
                return cap
            cap.release()
            log.info("Picamera2 no pudo abrir la cámara; probando con OpenCV")

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
                # cap es None mientras la cámara está perdida y reconectando.
                if cap is None:
                    self._sleep(delay)
                    if not self._run_flag:
                        break
                    cap = self._open_camera()
                    if cap is None:
                        delay = min(delay * 1.5, self._max_reconnect_delay_s)
                        self.status_signal.emit(f"Reintentando cámara en {delay:.1f}s...")
                        continue
                    consecutive_failures = 0
                    delay = self._reconnect_delay_s
                    log.info("Cámara reconectada")
                    if camera_was_lost:
                        self.camera_recovered_signal.emit()
                        camera_was_lost = False
                    continue

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
                        cap = None
                    else:
                        self._sleep(0.05)
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

    def stop(self, timeout_ms: int = 3000) -> bool:
        """Pide parar y espera. Devuelve False si el hilo sigue vivo."""
        self._run_flag = False
        return self.wait(timeout_ms)


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

    def __init__(
        self,
        frame_queue: queue.Queue,
        config: AppConfig,
        classifier: Optional[SignClassifier] = None,
    ):
        super().__init__()
        self._frame_queue = frame_queue
        self._cfg = config
        self._run_flag = True
        self._hands_solution = None
        # Peticiones desde el hilo de la UI; se aplican dentro de run().
        self._pending_reset: Optional[bool] = None   # has_letters del reset pedido
        self._reinit_requested = False

        self._classifier: Optional[SignClassifier] = classifier
        self._smoother = PredictionSmoother(
            window_size=config.smoothing_window,
            min_confidence=config.min_letter_confidence,
            min_margin=config.min_letter_margin,
        )
        self._committer = LetterCommitter(
            stable_frames=config.stable_frames_to_commit,
            release_frames=config.repeat_release_frames,
            space_frames=config.no_hand_frames_for_space,
            hand_lost_reset_frames=config.hand_lost_reset_frames,
        )
        self._diagnostic_mode: bool = False

        self._latencies: deque[float] = deque(maxlen=100)
        self._frame_stamps: deque[float] = deque(maxlen=30)
        self._hand_counts: deque[int] = deque(maxlen=60)
        self._last_metrics_emit = 0.0


    def set_draw_landmarks(self, value: bool) -> None:
        self._cfg.draw_landmarks = value

    def set_draw_connections(self, value: bool) -> None:
        self._cfg.draw_connections = value

    def reset_word_state(self, has_letters: bool = False) -> None:
        """Reinicia el estado de deletreo. `has_letters`: si la palabra en
        curso aún tiene letras (para que el espacio automático siga activo).

        Se llama desde el hilo de la UI. Limpiar aquí el deque del smoother
        mientras run() lo recorre lanza "deque mutated during iteration",
        así que solo se marca y el hilo de IA lo aplica en su bucle.
        """
        self._pending_reset = has_letters

    def request_reinit(self) -> None:
        """Recrea el HandLandmarker para aplicar cambios de confianza de detección."""
        self._reinit_requested = True

    def _apply_pending_reset(self) -> None:
        has_letters = self._pending_reset
        if has_letters is None:
            return
        self._pending_reset = None
        self._committer.reset(has_letters)
        self._smoother.reset()


    def set_stable_frames_to_commit(self, value: int) -> None:
        self._committer.stable_frames = max(1, int(value))

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

    @property
    def has_classifier(self) -> bool:
        return self._classifier is not None

    # ---- ciclo principal --------------------------------------------------

    def run(self) -> None:
        if not self._init_mediapipe():
            return
        try:
            self._loop()
        finally:
            # Se cierra aquí (en el hilo que lo usa) y no desde stop(): si
            # stop() agota su espera, cerrarlo desde fuera mientras
            # detect_for_video() sigue en curso puede tumbar el proceso.
            self._close_landmarker()

    def _loop(self) -> None:
        timestamp_ms = 0

        while self._run_flag:
            if self._reinit_requested:
                self._reinit_requested = False
                self._close_landmarker()
                if not self._init_mediapipe():
                    return

            self._apply_pending_reset()

            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                self.heartbeat_signal.emit()
                continue

            t0 = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # El modo VIDEO usa el timestamp para el tracking entre frames:
                # debe ser tiempo real y estrictamente creciente.
                timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
                results = self._hands_solution.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                log.exception("MediaPipe falló: %s", e)
                continue

            detections = self._parse_results(results)
            hand = self._select_hand(detections)
            annotated = self._render(
                frame, detections, hand if self._classifier is not None else None,
            )

            sign_text = "—"
            sign_conf = 0.0
            letter: Optional[str] = None
            if self._classifier is not None and hand is not None:
                try:
                    topk = self._classify(hand)
                    smoothed = self._smoother.push(topk)

                    if self._diagnostic_mode:
                        self.sign_diagnostic_signal.emit(topk)

                    if smoothed.letter is not None:
                        letter = smoothed.letter
                        sign_text = letter
                        sign_conf = smoothed.confidence
                    elif smoothed.raw_top1 and smoothed.raw_top1[1] > 0.35:
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

            committed = self._committer.update(letter, hand_present=hand is not None)
            if self._committer.hand_just_lost:
                # Sin esto, los votos de antes de bajar la mano seguían
                # contando al volver a levantarla.
                self._smoother.reset()
            if committed == LetterCommitter.SPACE:
                self.space_committed_signal.emit()
            elif committed is not None:
                self.letter_committed_signal.emit(committed)

            self.change_pixmap_signal.emit(annotated)
            self.hands_detected_signal.emit(detections)
            self.sign_detected_signal.emit(sign_text, sign_conf)
            self.heartbeat_signal.emit()

            dt = time.perf_counter() - t0
            self._update_metrics(dt, detections.num_hands)

    def _select_hand(self, detections: FrameDetections) -> Optional[HandDetection]:
        """Elige la mano que deletrea.

        Con una sola mano se usa esa aunque MediaPipe dude de su lateralidad.
        Con varias, la de la mano dominante configurada (si ninguna lo es,
        la de mayor confianza) para no mezclar ambas en el suavizado.
        """
        if not detections.hands:
            return None
        if len(detections.hands) == 1:
            return detections.hands[0]
        dominant = [h for h in detections.hands if h.handedness == self._cfg.dominant_hand]
        return max(dominant or detections.hands, key=lambda h: h.confidence)

    def _classify(self, hand: HandDetection) -> list[tuple[str, float]]:
        landmarks_2d = hand.landmarks_2d
        if self._cfg.dominant_hand == "Left":
            # Reflejo horizontal: la mano izquierda se ve como una derecha,
            # que es la que espera el modelo. La z no cambia con el reflejo.
            landmarks_2d = landmarks_2d.copy()
            landmarks_2d[:, 0] = 1.0 - landmarks_2d[:, 0]
        return self._classifier.predict_topk_from_hand(
            landmarks_2d, hand.landmarks_3d, k=3,
        )


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

    def _close_landmarker(self) -> None:
        if self._hands_solution is not None:
            try:
                self._hands_solution.close()
            except Exception:
                pass
            self._hands_solution = None

    def _parse_results(self, results) -> FrameDetections:
        det = FrameDetections()
        if not results.hand_landmarks:
            return det

        n_hands = len(results.hand_landmarks)
        for i in range(n_hands):
            hand_lms = results.hand_landmarks[i]

            # MediaPipe asume imagen en espejo (tipo selfie) al decidir la
            # lateralidad. CameraThread ya voltea el frame, así que su
            # etiqueta es la mano real del usuario: no hay que invertirla.
            # Si alguna cámara la diera al revés, este es el único sitio
            # que hay que tocar.
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
        active_hand: Optional[HandDetection] = None,
    ) -> np.ndarray:
        out = frame.copy()

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
            draw_hand_label(out, hand, active=hand is active_hand)

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

    def _update_metrics(self, dt: float, num_hands: int) -> None:
        if dt <= 0:
            return
        now = time.perf_counter()
        self._latencies.append(dt * 1000.0)
        self._frame_stamps.append(now)
        self._hand_counts.append(num_hands)
        if now - self._last_metrics_emit < 0.25:
            return
        self._last_metrics_emit = now
        # FPS reales = frames procesados / tiempo transcurrido. (1/latencia
        # daba el rendimiento teórico del modelo, p. ej. 100 FPS con una
        # cámara de 30.)
        span = self._frame_stamps[-1] - self._frame_stamps[0]
        fps = (len(self._frame_stamps) - 1) / span if span > 0 else 0.0
        try:
            m = InferenceMetrics(
                fps=fps,
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

    def stop(self, timeout_ms: int = 3000) -> bool:
        """Pide parar y espera. Devuelve False si el hilo sigue vivo."""
        self._run_flag = False
        return self.wait(timeout_ms)


# =========================================================================== #
# Voz (texto a voz del sistema)
# =========================================================================== #

# Orden de preferencia de voces en macOS. Las "Eloquence" (Eddy, Flo,
# Grandma...) existen para cada idioma pero suenan robóticas.
_MAC_LOCALE_ORDER = ("es_MX", "es_US", "es_419", "es_ES")
_MAC_PREFERRED_VOICES = ("Paulina", "Juan", "Mónica", "Jorge")


def _pick_mac_spanish_voice() -> Optional[str]:
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # Formato: "Paulina (Español (México)) es_MX    # ¡Hola! Me llamo Paulina."
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z0-9]+)\s+#", line)
        if m and m.group(2).startswith("es_"):
            voices.append((m.group(1).strip(), m.group(2)))
    if not voices:
        return None

    def rank(voice: tuple[str, str]) -> tuple[int, int]:
        name, locale = voice
        loc = _MAC_LOCALE_ORDER.index(locale) if locale in _MAC_LOCALE_ORDER else len(_MAC_LOCALE_ORDER)
        preferred = 0 if name.split(" (")[0] in _MAC_PREFERRED_VOICES else 1
        return (loc, preferred)

    return min(voices, key=rank)[0]


class Speaker:
    """Lee palabras en voz alta con el TTS del sistema sin bloquear la UI.

    macOS: `say`. Linux / Raspberry Pi: `espeak-ng` o `espeak`.
    """

    def __init__(self):
        self._base_cmd: Optional[list[str]] = None
        if shutil.which("say"):
            self._base_cmd = ["say"]
        elif shutil.which("espeak-ng"):
            self._base_cmd = ["espeak-ng", "-v", "es-419"]
        elif shutil.which("espeak"):
            self._base_cmd = ["espeak", "-v", "es-la"]
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return self._base_cmd is not None

    def say(self, text: str) -> None:
        if not self.available or not text.strip():
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="tts", daemon=True)
            self._worker.start()
        self._queue.put(text)

    def _run(self) -> None:
        # Las palabras se leen en orden, sin solaparse.
        cmd = list(self._base_cmd)
        if cmd[0] == "say":
            voice = _pick_mac_spanish_voice()   # tarda ~0.2 s: fuera de la UI
            if voice:
                cmd += ["-v", voice]
        while True:
            text = self._queue.get()
            try:
                subprocess.run(
                    cmd + [text], timeout=30, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError) as e:
                log.warning("No se pudo leer en voz alta: %s", e)


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
        # Hilos que no terminaron a tiempo al pararlos. Hay que conservar la
        # referencia hasta que acaben: si Python destruye un QThread en
        # marcha, Qt aborta el proceso ("Destroyed while thread is still running").
        self._retiring_threads: list[QThread] = []

        # Se carga una vez (y no en cada Iniciar / reinicio del watchdog).
        self.classifier: Optional[SignClassifier] = SignClassifier.try_load()
        self.speaker = Speaker()
        self._confidence_state: Optional[str] = None

        # Recrear el HandLandmarker es caro: se espera a que el slider se detenga.
        self._threshold_apply_timer = QTimer(self)
        self._threshold_apply_timer.setSingleShot(True)
        self._threshold_apply_timer.setInterval(400)
        self._threshold_apply_timer.timeout.connect(self._apply_threshold)

        self.current_word = ""
        self.history: list[str] = []
        self._last_annotated_frame: Optional[np.ndarray] = None

        self._last_heartbeat = time.time()
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(1000)
        self._watchdog.timeout.connect(self._check_watchdog)
        self._watchdog_active = False

        self._available_cameras: list[int] = []

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

        action_screenshot = QAction("📷 Captura", self)
        action_screenshot.setShortcut(QKeySequence("Ctrl+S"))
        action_screenshot.triggered.connect(self.save_screenshot)
        toolbar.addAction(action_screenshot)

        action_export = QAction("💾 Exportar historial", self)
        action_export.triggered.connect(self.export_history)
        toolbar.addAction(action_export)

        toolbar.addSeparator()

        action_backspace = QAction("⌫ Borrar letra", self)
        action_backspace.setShortcut(QKeySequence(Qt.Key.Key_Backspace))
        action_backspace.setToolTip("Borra la última letra (Retroceso)")
        action_backspace.triggered.connect(self.delete_last_letter)
        toolbar.addAction(action_backspace)

        action_clear = QAction("✕ Borrar palabra", self)
        action_clear.setShortcut(QKeySequence("Ctrl+Backspace"))
        action_clear.triggered.connect(self.clear_current_word)
        toolbar.addAction(action_clear)

        # Enter y no Ctrl+Espacio: en macOS Ctrl se traduce a Cmd y
        # Cmd+Espacio lo captura Spotlight antes que la app.
        action_space = QAction("␣ Espacio (Enter)", self)
        action_space.setShortcuts([
            QKeySequence(Qt.Key.Key_Return), QKeySequence(Qt.Key.Key_Enter),
        ])
        action_space.setToolTip("Termina la palabra actual (Enter)")
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

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("Confianza: %p%")
        self.confidence_bar.setMaximumHeight(18)
        self.confidence_bar.setToolTip(
            "Verde: letra reconocida con seguridad.\n"
            "Naranja: el modelo duda (se muestra como ?X y no se fija)."
        )
        self.confidence_bar.setVisible(self.classifier is not None)
        side.addWidget(self.confidence_bar)
        self._set_confidence(0.0, "none")

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
        self.min_confidence_slider.setValue(round(self.cfg.min_letter_confidence * 100))
        self.min_confidence_slider.valueChanged.connect(self._on_min_confidence_changed)
        self.min_confidence_value_label = QLabel(f"{self.cfg.min_letter_confidence:.2f}")
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
        self.min_margin_slider.setValue(round(self.cfg.min_letter_margin * 100))
        self.min_margin_slider.valueChanged.connect(self._on_min_margin_changed)
        self.min_margin_value_label = QLabel(f"{self.cfg.min_letter_margin:.2f}")
        self.min_margin_value_label.setMinimumWidth(40)
        margin_row.addWidget(self.min_margin_slider, stretch=1)
        margin_row.addWidget(self.min_margin_value_label)
        side.addLayout(margin_row)

        hand_row = QHBoxLayout()
        hand_label = QLabel("Mano que deletrea:")
        hand_label.setToolTip(
            "Con las dos manos en cámara se lee esta.\n"
            "Con «Izquierda» la mano se refleja para verse como una derecha."
        )
        hand_row.addWidget(hand_label)
        self.hand_combo = QComboBox()
        self.hand_combo.addItem("Derecha", "Right")
        self.hand_combo.addItem("Izquierda (zurdos)", "Left")
        self.hand_combo.setCurrentIndex(self.hand_combo.findData(self.cfg.dominant_hand))
        self.hand_combo.currentIndexChanged.connect(self._on_dominant_hand_changed)
        hand_row.addWidget(self.hand_combo, stretch=1)
        side.addLayout(hand_row)

        self.cb_speak = QCheckBox("Leer palabras en voz alta")
        if self.speaker.available:
            self.cb_speak.setChecked(self.cfg.speak_words)
            self.cb_speak.setToolTip("Lee cada palabra al terminarla (espacio o Enter).")
        else:
            self.cb_speak.setEnabled(False)
            self.cb_speak.setToolTip(
                "No se encontró un motor de voz.\n"
                "En Linux / Raspberry Pi: sudo apt install espeak-ng"
            )
        self.cb_speak.toggled.connect(self._on_speak_toggled)
        side.addWidget(self.cb_speak)

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
        thr_row.addWidget(QLabel("Detección de mano:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(round(self.cfg.min_detection_confidence * 100))
        self.threshold_slider.setToolTip("Confianza mínima para detectar una mano")
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_value_label = QLabel(f"{round(self.cfg.min_detection_confidence * 100)}%")
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

    def _classifier_status_text(self) -> str:
        # Se basa en si el modelo cargó de verdad, no en si existen los
        # archivos (p. ej. falta onnxruntime o el .onnx está corrupto).
        if self.classifier is not None:
            return (
                f"Clasificador LSM activo (alfabeto, {len(self.classifier.labels)} letras estáticas).\n"
                f"Mantén una seña ~{self.cfg.stable_frames_to_commit} frames para fijar la letra.\n"
                "Para repetir una letra (LL, RR...), relaja la mano un instante.\n"
                f"Baja las manos ~{self.cfg.no_hand_frames_for_space} frames para insertar un espacio."
            )
        models_dir = Path(__file__).resolve().parent
        return (
            "Sin modelo de clasificación cargado.\n"
            f"Coloca {MODEL_FILENAME} y {LABELS_FILENAME} en:\n{models_dir}\n"
            "y verifica que onnxruntime esté instalado (ver log)."
        )


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

    def _create_ai_thread(self) -> None:
        self.ai_thread = HandTrackingThread(self.frame_queue, self.cfg, self.classifier)
        self._wire_ai_thread()
        self.ai_thread.set_diagnostic_mode(self.cb_diagnostic.isChecked())

    def _retire_thread(self, thread: QThread) -> None:
        if thread.stop():
            return
        log.warning("%s no terminó a tiempo; se liberará al acabar", type(thread).__name__)
        self._retiring_threads.append(thread)
        thread.finished.connect(lambda t=thread: self._forget_thread(t))
        if thread.isFinished():
            self._forget_thread(thread)

    def _forget_thread(self, thread: QThread) -> None:
        if thread in self._retiring_threads:
            self._retiring_threads.remove(thread)

    def start_system(self) -> None:
        if self.camera_thread is not None or self.ai_thread is not None:
            return

        self.cfg.camera_index = int(self.camera_combo.currentData())

        self.action_start.setEnabled(False)
        # Escanear cámaras con la cámara en uso la interfiere (y la hace
        # desaparecer de la lista), así que se bloquea mientras corre.
        self.camera_combo.setEnabled(False)
        self.refresh_cameras_btn.setEnabled(False)
        self.status_camera.setText("● Conectando...")

        self.camera_thread = CameraThread(self.frame_queue, self.cfg.camera_index)
        self._create_ai_thread()

        self.camera_thread.error_signal.connect(self._on_camera_error)
        self.camera_thread.status_signal.connect(lambda s: self.status_camera.setText(f"● {s}"))
        self.camera_thread.camera_lost_signal.connect(
            lambda: self.status_camera.setText("● Cámara perdida, reintentando")
        )
        self.camera_thread.camera_recovered_signal.connect(
            lambda: self.status_camera.setText("● Cámara recuperada")
        )

        self.ai_thread.start()
        self.camera_thread.start()

        # El watchdog se arma en _on_model_loaded: la primera vez el hilo de
        # IA descarga el modelo (puede tardar >5 s sin emitir heartbeats) y
        # el watchdog lo reiniciaba en bucle a mitad de la descarga.
        self._last_heartbeat = time.time()
        self._watchdog_active = False
        self._watchdog.start()

        self.action_stop.setEnabled(True)

    def stop_system(self) -> None:
        self._watchdog_active = False
        self._watchdog.stop()
        self._threshold_apply_timer.stop()

        if self.camera_thread is not None:
            self._retire_thread(self.camera_thread)
            self.camera_thread = None

        if self.ai_thread is not None:
            self._retire_thread(self.ai_thread)
            self.ai_thread = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # setText ya quita el pixmap; el setPixmap(QPixmap()) que venía
        # detrás borraba a su vez este texto y dejaba el panel vacío.
        self.image_label.setText("Cámara detenida")
        self.sign_label.setText("—")
        self._set_confidence(0.0, "none")
        self.status_camera.setText("● Cámara: detenida")
        self.status_hands.setText("✋ Manos: 0")
        self.status_fps.setText("FPS: —")
        self.status_latency.setText("Latencia: —")

        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.refresh_cameras_btn.setEnabled(True)

    def _restart_ai_thread(self) -> None:
        log.warning("Watchdog: reiniciando hilo de IA")
        self._watchdog_active = False  # se rearma cuando el nuevo hilo cargue
        if self.ai_thread is not None:
            self._retire_thread(self.ai_thread)
        self._create_ai_thread()
        self.ai_thread.start()
        self._last_heartbeat = time.time()
        self.statusBar().showMessage("IA reiniciada por inactividad", 3000)


    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_value_label.setText(f"{value}%")
        self.cfg.min_detection_confidence = value / 100.0
        # El HandLandmarker fija la confianza al crearse: sin recrearlo, el
        # slider no tenía efecto hasta el siguiente Iniciar.
        if self.ai_thread is not None:
            self._threshold_apply_timer.start()

    def _apply_threshold(self) -> None:
        if self.ai_thread is not None:
            self.ai_thread.request_reinit()

    def _on_draw_landmarks(self, checked: bool) -> None:
        self.cfg.draw_landmarks = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_landmarks(checked)

    def _on_draw_connections(self, checked: bool) -> None:
        self.cfg.draw_connections = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_connections(checked)

    def _on_stable_frames_changed(self, value: int) -> None:
        self.stable_frames_value_label.setText(str(value))
        self.cfg.stable_frames_to_commit = value
        self.classifier_info_label.setText(self._classifier_status_text())
        if self.ai_thread is not None:
            self.ai_thread.set_stable_frames_to_commit(value)

    def _on_min_confidence_changed(self, value: int) -> None:
        v = value / 100.0
        self.min_confidence_value_label.setText(f"{v:.2f}")
        self.cfg.min_letter_confidence = v
        if self.ai_thread is not None:
            self.ai_thread.set_min_confidence(v)

    def _on_min_margin_changed(self, value: int) -> None:
        v = value / 100.0
        self.min_margin_value_label.setText(f"{v:.2f}")
        self.cfg.min_letter_margin = v
        if self.ai_thread is not None:
            self.ai_thread.set_min_margin(v)

    def _on_dominant_hand_changed(self, _index: int) -> None:
        self.cfg.dominant_hand = self.hand_combo.currentData()
        # Lo acumulado con la otra mano no sirve para la nueva.
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state(has_letters=bool(self.current_word))

    def _on_speak_toggled(self, checked: bool) -> None:
        self.cfg.speak_words = checked

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
        self.diagnostic_label.setText("\n".join(lines))

    def _on_model_loaded(self) -> None:
        self.statusBar().showMessage("MediaPipe Hands listo", 3000)
        if self.ai_thread is not None:
            self._last_heartbeat = time.time()
            self._watchdog_active = True

    # Se para ANTES de mostrar el diálogo: el QMessageBox modal sigue
    # procesando eventos, y con el watchdog activo se reiniciaba el hilo de
    # IA (que volvía a fallar) apilando un diálogo nuevo cada 5 s.
    def _on_camera_error(self, msg: str) -> None:
        self.stop_system()
        QMessageBox.critical(self, "Error de cámara", msg)

    def _on_ai_error(self, msg: str) -> None:
        self.stop_system()
        QMessageBox.critical(self, "Error del modelo", msg)

    def _on_heartbeat(self) -> None:
        self._last_heartbeat = time.time()

    def _check_watchdog(self) -> None:
        if not self._watchdog_active:
            return
        if time.time() - self._last_heartbeat > self.cfg.watchdog_timeout_s:
            self._restart_ai_thread()

    # Las señales encoladas justo antes de Detener llegan después de él; sin
    # estas guardas repintaban el último frame sobre "Cámara detenida".
    def update_sign(self, text: str, conf: float) -> None:
        if self.ai_thread is None:
            return
        self.sign_label.setText(text)
        if text.startswith("?"):
            self._set_confidence(conf, "doubt")
        elif len(text) == 1 and text != "—":
            self._set_confidence(conf, "ok")
        else:
            self._set_confidence(0.0, "none")

    def _set_confidence(self, conf: float, state: str) -> None:
        self.confidence_bar.setValue(round(conf * 100))
        # Cambiar la hoja de estilo cuesta: solo cuando cambia el estado.
        if state == self._confidence_state:
            return
        self._confidence_state = state
        color = {"ok": "#27ae60", "doubt": "#e67e22"}.get(state, "#bbbbbb")
        self.confidence_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )

    def update_hands(self, detections: FrameDetections) -> None:
        if self.ai_thread is None:
            return
        self.status_hands.setText(f"✋ Manos: {detections.num_hands}")

    def update_metrics(self, m: InferenceMetrics) -> None:
        if self.ai_thread is None:
            return
        self.status_fps.setText(f"FPS: {m.fps:.1f}")
        self.status_latency.setText(f"Latencia: {m.latency_p50_ms:.0f}/{m.latency_p95_ms:.0f}ms")

    def update_image(self, cv_img: np.ndarray) -> None:
        if self.ai_thread is None or cv_img is None or cv_img.size == 0:
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
        if self.cfg.speak_words:
            # En minúsculas: algunos motores deletrean las palabras en mayúsculas.
            self.speaker.say(word.lower())

    def delete_last_letter(self) -> None:
        if not self.current_word:
            return
        self.current_word = self.current_word[:-1]
        self.word_label.setText(self.current_word)
        # Tras corregir, la letra se puede volver a signar de inmediato.
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state(has_letters=bool(self.current_word))

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
        for thread in list(self._retiring_threads):
            thread.wait(5000)
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
    # Los argumentos pasan por la misma validación que config.json.
    if args.camera is not None:
        cfg.set_validated("camera_index", args.camera)
    if args.threshold is not None:
        cfg.set_validated("min_detection_confidence", args.threshold)
    if args.max_hands is not None:
        cfg.set_validated("max_num_hands", args.max_hands)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")

    window = SignLanguageApp(cfg, config_path=config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())