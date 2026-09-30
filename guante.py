"""Guante con ESP32: recepcion de datos por WiFi (UDP) y reconocimiento de
senas contra el dataset grabado con grabar_guante.py.

La ESP32 es punto de acceso (red GUANTE_LSM, IP 192.168.4.1) y manda un JSON
por lectura al ultimo equipo que le dijo "hola" al puerto 4210:

    {"pulgar": [ax, ay, az, gx, gy, gz, pitch, roll], "indice": [...],
     "medio": [...], "anular": [...], "menique": [...], "mano": [...], "err": 0}

6 sensores x 8 valores = 48 por lectura, ~21 lecturas por segundo. Una seña
es una ventana de 2 s (lo que dura cada muestra del dataset); se resume en un
vector de 192 valores (frames_to_features) y se compara con las muestras por
vecinos mas cercanos.

Sin Qt: la ventana (senas.py) llama GloveSession.tick() con un QTimer, y
grabar_guante.py / las pruebas usan las mismas clases.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import threading
import unicodedata
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("guante")

GLOVE_IP = "192.168.4.1"        # IP fija de la ESP32 (punto de acceso)
GLOVE_PORT = 4210               # igual a PUERTO_UDP del firmware
SENSOR_NAMES = ("pulgar", "indice", "medio", "anular", "menique", "mano")
VALUES_PER_SENSOR = 8           # ax, ay, az (g), gx, gy, gz (°/s), pitch, roll (°)
N_VALUES = len(SENSOR_NAMES) * VALUES_PER_SENSOR

WINDOW_S = 2.0                  # duracion de cada muestra del dataset
# Ventana del modo automatico / respuesta en vivo: 1 s, para seguir el cambio
# de letra casi al paso de la camara (con 2 s el guante iba 1-2 s atrasado).
# Clasificando cada mitad de cada muestra contra las demas: 28/30 y 30/30.
LIVE_WINDOW_S = 1.0
MIN_FRAMES = 15                 # lecturas minimas para aceptar una ventana
HELLO_EVERY_S = 1.0             # cada cuanto se le recuerda a la ESP a quien mandar
CONNECTED_TIMEOUT_S = 1.0       # sin paquetes en este tiempo = desconectado
BUFFER_S = 10.0                 # lecturas que se guardan (para capturar y la ventana)

# Escala de cada valor para que aceleracion (g), giroscopio (°/s) y angulos (°)
# pesen parecido al comparar.
SCALE = np.tile([0.5, 0.5, 0.5, 50, 50, 50, 30, 30], len(SENSOR_NAMES))
TEMPERATURE = 0.1               # sube = porcentajes mas repartidos

# Vecinos por clase: con 6 muestras por seña, promediar 3 vecinos castiga a las
# señas con muestras variadas (A, Y): dejando una fuera, k=1 acierta 30/30 y
# k=3 26/30. Se usa ~1 vecino por cada 4 muestras (maximo 3).
MAX_NEIGHBORS = 3

# Umbral de "no se parece a nada": se calibra con el dataset (distancia de
# cada muestra a su vecino mas cercano de la misma seña, dejandola fuera) y se
# multiplica por este margen, porque en vivo la mano nunca queda igual.
DISTANCE_MARGIN = 1.5
MIN_MAX_DISTANCE = 0.3
MIN_MARGIN = 0.20               # 1.a vs 2.a seña (en probabilidad)

# Modo automatico: la misma seña aceptada en N evaluaciones seguidas (se
# evalua ~4 veces por segundo) se escribe; para repetirla hay que cambiar de
# postura RELEASE_TICKS evaluaciones.
STABLE_TICKS = 4
RELEASE_TICKS = 3
# Palabras (etiquetas de mas de una letra, HOLA, MAMÁ): llevan movimiento y la
# ventana de 2 s solo coincide con la muestra grabada un momento, asi que
# basta con menos evaluaciones seguidas.
STABLE_TICKS_WORD = 2

DEFAULT_DATASET = Path(__file__).resolve().parent / "datos_guante" / "dataset_guante.jsonl"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z~]|\x1b.")


def clean_label(label: str) -> str:
    """Etiqueta limpia: sin secuencias de teclas (una flecha en input() deja
    '\\x1b[D' pegado: '\\x1b[DL' -> 'L'), en mayusculas y sin espacios."""
    label = _ANSI.sub("", str(label))
    label = "".join(ch for ch in label if ch.isprintable())
    return "_".join(label.upper().split())


def plain_label(label: str) -> str:
    """Etiqueta sin acentos, para comparar MAMA (tecleada) con MAMÁ (camara)."""
    return "".join(ch for ch in unicodedata.normalize("NFD", label) if unicodedata.category(ch) != "Mn")


def parse_packet(raw: bytes) -> Optional[list[float]]:
    """JSON de la ESP -> 48 valores, o None si esta corrupto o algun sensor
    fallo (err != 0)."""
    try:
        d = json.loads(raw.decode("utf-8", errors="ignore"))
        if not isinstance(d, dict) or d.get("err", 0) != 0:
            return None
        vec: list[float] = []
        for name in SENSOR_NAMES:
            values = d[name]
            if len(values) != VALUES_PER_SENSOR:
                return None
            vec.extend(float(v) for v in values)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(vec)):
        return None
    return vec


SENSOR_TITLES = ("Pulgar", "Índice", "Medio", "Anular", "Meñique", "Mano")


def format_reading(vec: Optional[list[float]]) -> str:
    """Tabla de texto (monoespaciada) con la ultima lectura de cada sensor:
    aceleracion (g), giroscopio (°/s) y angulos pitch/roll (°)."""
    header = f"{'':8}{'ax':>6}{'ay':>6}{'az':>6}{'gx':>6}{'gy':>6}{'gz':>6}{'pitch':>7}{'roll':>6}"
    if vec is None:
        return header + "\n(sin datos)"
    rows = [header]
    for title, v in zip(SENSOR_TITLES, np.asarray(vec, dtype=float).reshape(-1, VALUES_PER_SENSOR)):
        rows.append(f"{title:8} {v[0]:5.2f} {v[1]:5.2f} {v[2]:5.2f}"
                    f"{v[3]:6.0f}{v[4]:6.0f}{v[5]:6.0f}{v[6]:7.0f}{v[7]:6.0f}")
    return "\n".join(rows)


def frames_to_features(frames: np.ndarray) -> np.ndarray:
    """Captura (n lecturas x 48) -> 192 valores: promedio de cada tercio
    (postura y como cambia) y desviacion estandar (cuanto se movio), todo
    dividido entre SCALE."""
    frames = np.asarray(frames, dtype=float)
    thirds = np.array_split(frames, 3)
    means = [t.mean(axis=0) for t in thirds]
    return np.concatenate(means + [frames.std(axis=0)]) / np.tile(SCALE, 4)


# --------------------------------------------------------------------------- #
# Recepcion
# --------------------------------------------------------------------------- #


class GloveReceiver:
    """Recibe los JSON del guante en un hilo aparte y guarda las ultimas
    lecturas (hora, vector de 48). Solo un programa a la vez: la ESP le manda
    los datos al ultimo que le dijo "hola"."""

    def __init__(self, ip: str = GLOVE_IP, port: int = GLOVE_PORT):
        self.address = (ip, port)
        self._frames: deque[tuple[float, list[float]]] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self.last_rx = 0.0
        self.bad_packets = 0

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="guante", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _loop(self) -> None:
        sock = self._sock
        last_hello = 0.0
        while self._running:
            if time.time() - last_hello >= HELLO_EVERY_S:
                try:
                    sock.sendto(b"hola", self.address)
                    last_hello = time.time()
                except OSError:
                    # Sin ruta a la ESP (no estamos en la red GUANTE_LSM).
                    last_hello = time.time()
                    time.sleep(0.2)
                    continue
            try:
                raw, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    break
                time.sleep(0.2)
                continue
            vec = parse_packet(raw)
            if vec is None:
                self.bad_packets += 1
                continue
            self.push(vec)

    def push(self, vec: list[float], now: Optional[float] = None) -> None:
        """Agrega una lectura (el hilo de red; las pruebas la llaman directo)."""
        now = time.time() if now is None else now
        with self._lock:
            self._frames.append((now, vec))
            while self._frames and now - self._frames[0][0] > BUFFER_S:
                self._frames.popleft()
        self.last_rx = now

    def connected(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return now - self.last_rx < CONNECTED_TIMEOUT_S

    def latest(self) -> Optional[list[float]]:
        """Ultima lectura recibida (48 valores), o None si no ha llegado nada."""
        with self._lock:
            return list(self._frames[-1][1]) if self._frames else None

    def frames_since(self, t0: float) -> np.ndarray:
        with self._lock:
            sel = [v for t, v in self._frames if t >= t0]
        return np.array(sel, dtype=float).reshape(-1, N_VALUES)

    def window(self, seconds: float = WINDOW_S, now: Optional[float] = None) -> np.ndarray:
        now = time.time() if now is None else now
        return self.frames_since(now - seconds)

    def rate_hz(self, now: Optional[float] = None) -> float:
        """Lecturas por segundo en los ultimos 2 s."""
        now = time.time() if now is None else now
        with self._lock:
            n = sum(1 for t, _ in self._frames if now - t <= 2.0)
        return n / 2.0


# --------------------------------------------------------------------------- #
# Dataset y clasificacion
# --------------------------------------------------------------------------- #


def load_dataset(path: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Lee el .jsonl de grabar_guante.py -> (capturas, etiquetas, personas).
    Salta lineas rotas o capturas con otra forma, con aviso."""
    frames, labels, people = [], [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                m = json.loads(line)
                arr = np.asarray(m["frames"], dtype=float)
                label = clean_label(m["etiqueta"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                log.warning("%s línea %d ignorada: %s", path.name, n, e)
                continue
            if arr.ndim != 2 or arr.shape[1] != N_VALUES or len(arr) < MIN_FRAMES or not label:
                log.warning("%s línea %d ignorada: captura %s", path.name, n, arr.shape)
                continue
            frames.append(arr)
            labels.append(label)
            people.append(str(m.get("persona", "")))
    return frames, labels, people


@dataclass
class GloveResult:
    topk: list[tuple[str, float]]       # [(seña, probabilidad 0-1)], mejor primero
    distance: float                     # distancia a la mejor seña
    max_distance: float                 # umbral de "no se parece a nada"
    accepted: bool
    reason: str = ""                    # por que no se acepto
    n_frames: int = 0

    @property
    def label(self) -> Optional[str]:
        return self.topk[0][0] if self.topk else None


class GloveClassifier:
    """Vecinos mas cercanos contra las muestras del dataset."""

    def __init__(self, frames: list[np.ndarray], labels: list[str],
                 max_distance: Optional[float] = None):
        if not labels:
            raise ValueError("el dataset del guante no tiene muestras")
        self.X = np.array([frames_to_features(f) for f in frames])
        self.y = np.array(labels)
        self.counts = Counter(labels)
        self.labels = sorted(self.counts)
        self.max_distance = max_distance if max_distance else self._calibrate()

    @classmethod
    def from_file(cls, path: Path, max_distance: Optional[float] = None) -> "GloveClassifier":
        frames, labels, _ = load_dataset(Path(path))
        return cls(frames, labels, max_distance)

    @staticmethod
    def _neighbors(n: int) -> int:
        return max(1, min(MAX_NEIGHBORS, n // 4))

    def _class_distances(self, d: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        y = self.y if mask is None else self.y[mask]
        d = d if mask is None else d[mask]
        out = np.full(len(self.labels), np.inf)
        for i, c in enumerate(self.labels):
            dc = np.sort(d[y == c])
            if len(dc):
                out[i] = dc[: self._neighbors(len(dc))].mean()
        return out

    def _distances(self, feat: np.ndarray) -> np.ndarray:
        return np.sqrt(((self.X - feat) ** 2).mean(axis=1))

    def _calibrate(self) -> float:
        """Distancia de cada muestra a su seña (dejandola fuera); el umbral es
        la mayor por DISTANCE_MARGIN."""
        same = []
        for i in range(len(self.y)):
            others = np.arange(len(self.y)) != i
            if not np.any(others & (self.y == self.y[i])):
                continue           # seña con una sola muestra
            d = self._distances(self.X[i])
            same.append(self._class_distances(d, others)[self.labels.index(self.y[i])])
        if not same:
            return 1.0
        return max(MIN_MAX_DISTANCE, float(max(same)) * DISTANCE_MARGIN)

    def leave_one_out(self) -> tuple[int, int]:
        """(aciertos, total) clasificando cada muestra con las demas."""
        ok = 0
        for i in range(len(self.y)):
            others = np.arange(len(self.y)) != i
            dc = self._class_distances(self._distances(self.X[i]), others)
            ok += self.labels[int(np.argmin(dc))] == self.y[i]
        return ok, len(self.y)

    def classify(self, frames: np.ndarray) -> GloveResult:
        frames = np.asarray(frames, dtype=float)
        if len(frames) < MIN_FRAMES:
            return GloveResult([], float("inf"), self.max_distance, False,
                               f"solo {len(frames)} lecturas (mínimo {MIN_FRAMES})", len(frames))
        dc = self._class_distances(self._distances(frames_to_features(frames)))
        p = np.exp(-(dc - dc.min()) / TEMPERATURE)
        p /= p.sum()
        order = np.argsort(-p)[:3]
        topk = [(self.labels[i], float(p[i])) for i in order]
        best = float(dc[order[0]])
        margin = topk[0][1] - (topk[1][1] if len(topk) > 1 else 0.0)
        reason = ""
        if best > self.max_distance:
            reason = f"no se parece a ninguna seña grabada (distancia {best:.2f} > {self.max_distance:.2f})"
        elif margin < MIN_MARGIN:
            reason = f"duda entre {topk[0][0]} y {topk[1][0]}"
        return GloveResult(topk, best, self.max_distance, not reason, reason, len(frames))


# --------------------------------------------------------------------------- #
# Modo automatico y captura con cuenta atras
# --------------------------------------------------------------------------- #


class GloveSpotter:
    """Decide cuando escribir una seña en modo automatico: la misma seña
    aceptada STABLE_TICKS veces seguidas. Despues, esa seña no se repite hasta
    que la postura cambie RELEASE_TICKS evaluaciones (como relajar la mano
    para LL o RR con la camara)."""

    def __init__(self, stable_ticks: int = STABLE_TICKS, release_ticks: int = RELEASE_TICKS,
                 stable_ticks_word: int = STABLE_TICKS_WORD):
        self.stable_ticks = stable_ticks
        self.stable_ticks_word = stable_ticks_word
        self.release_ticks = release_ticks
        self.reset()

    def _needed(self, label: str) -> int:
        return self.stable_ticks_word if len(label) > 1 else self.stable_ticks

    def reset(self) -> None:
        self._candidate: Optional[str] = None
        self._count = 0
        self._locked: Optional[str] = None
        self._away = 0

    def update(self, result: Optional[GloveResult]) -> Optional[str]:
        label = result.label if result is not None and result.accepted else None
        if self._locked is not None:
            if label == self._locked:
                self._away = 0
                return None
            self._away += 1
            if self._away >= self.release_ticks:
                self._locked = None
        if label is None or label != self._candidate:
            self._candidate = label
            self._count = 1 if label else 0
            if label is None or self._needed(label) > 1:
                return None
        else:
            self._count += 1
        if self._count >= self._needed(label) and label != self._locked:
            self._locked = label
            self._away = 0
            self._candidate = None
            self._count = 0
            return label
        return None


@dataclass
class GloveEvent:
    kind: str       # "estado" | "cuenta" | "capturando" | "resultado"
    data: dict = field(default_factory=dict)


class GloveSession:
    """Une receptor, clasificador y modo automatico. La ventana llama tick()
    unas 4 veces por segundo y reacciona a los eventos:

    - estado:     {"connected", "hz"} (cuando cambia la conexion, y cada tick)
    - cuenta:     {"n"} cuenta atras de una captura pedida (3, 2, 1)
    - capturando: {} empieza la captura de WINDOW_S
    - vivo:       {"result": GloveResult} cada evaluacion del modo automatico
    - resultado:  {"result": GloveResult, "commit": seña o None, "manual": bool}
    """

    COUNTDOWN_S = 2

    def __init__(self, receiver: GloveReceiver, classifier: GloveClassifier, auto: bool = True):
        self.receiver = receiver
        self.classifier = classifier
        self.auto = auto
        self.spotter = GloveSpotter()
        self._capture_at: Optional[float] = None     # hora en que empieza a capturar
        self._last_count: Optional[int] = None
        self._capturing = False

    @property
    def busy(self) -> bool:
        return self._capture_at is not None

    def request_capture(self, now: Optional[float] = None) -> None:
        """Captura con cuenta atras (como leer_guante.py): 2, 1, ¡ya! y 2 s."""
        now = time.time() if now is None else now
        self._capture_at = now + self.COUNTDOWN_S
        self._last_count = None
        self._capturing = False

    def cancel_capture(self) -> None:
        self._capture_at = None
        self._capturing = False

    def tick(self, now: Optional[float] = None) -> list[GloveEvent]:
        now = time.time() if now is None else now
        events = [GloveEvent("estado", {"connected": self.receiver.connected(now),
                                        "hz": self.receiver.rate_hz(now)})]
        if self._capture_at is not None:
            if now < self._capture_at:
                n = int(np.ceil(self._capture_at - now))
                if n != self._last_count:
                    self._last_count = n
                    events.append(GloveEvent("cuenta", {"n": n}))
            elif now < self._capture_at + WINDOW_S:
                if not self._capturing:
                    self._capturing = True
                    events.append(GloveEvent("capturando"))
            else:
                frames = self.receiver.frames_since(self._capture_at)
                self.cancel_capture()
                result = self.classifier.classify(frames)
                self.spotter.reset()
                # Que el automatico no vuelva a escribir la misma postura.
                if result.accepted:
                    self.spotter._locked = result.label
                events.append(GloveEvent("resultado", {
                    "result": result, "commit": result.label if result.accepted else None, "manual": True}))
            return events

        if self.auto and self.receiver.connected(now):
            frames = self.receiver.window(LIVE_WINDOW_S, now)
            if len(frames) >= MIN_FRAMES:
                result = self.classifier.classify(frames)
                events.append(GloveEvent("vivo", {"result": result}))
                label = self.spotter.update(result)
                if label is not None:
                    events.append(GloveEvent("resultado", {"result": result, "commit": label, "manual": False}))
        return events
