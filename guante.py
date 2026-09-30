"""Guante con ESP32: recepcion de datos por WiFi (UDP) y reconocimiento de
senas contra el dataset grabado con grabar_guante.py.

La Raspberry es el punto de acceso (red GUANTE_LSM, 2.4 GHz, IP 10.42.0.1) y
la ESP32 de cada mano se conecta como cliente y manda, sin handshake ni
respuesta, un JSON por datagrama UDP a 10.42.0.1:4210:

    {"n": 12, "t": 3400, "h": "D", "err": 0,
     "pulgar": [ax, ay, az, gx, gy, gz, pitch, roll], "indice": [...],
     "medio": [...], "anular": [...], "menique": [...], "mano": [...]}

"h" es la mano ("D" derecha, "I" izquierda; tambien se aceptan "R"/"L" y
minusculas). Las dos llegan al mismo puerto; el receptor las separa y cada
mano tiene su propio dataset (el izquierdo lleva los sensores en espejo, asi
que no se compara con las muestras del derecho).

Frases con las dos manos ("DI"): GloveBoth junta cada lectura del derecho con
la del izquierdo mas cercana en el tiempo (96 valores) y se comparan contra
un tercer dataset, grabado con los dos guantes puestos (grabar_guante.py
--mano DI). Cada frase dura lo que se grabo (--segundos, 3 por defecto).
6 sensores x 8 valores = 48 por lectura, ~25 lecturas por segundo. Una seña
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

GLOVE_BIND_IP = "0.0.0.0"       # se escucha en todas las interfaces (hotspot 10.42.0.1)
GLOVE_PORT = 4210               # igual a PUERTO_UDP del firmware
GLOVE_HAND = "D"                # mano por defecto ("D" derecha)
GLOVE_HANDS = ("D", "I")        # manos que se aceptan: derecha e izquierda
GLOVE_BOTH = "DI"               # los dos guantes juntos (frases de dos manos)
HAND_NAMES = {"D": "derecho", "I": "izquierdo", GLOVE_BOTH: "de las dos manos"}
# "h" que manda cada firmware -> mano. Un firmware copiado del derecho y
# cambiado a mano puede mandar "L", "i", "izq"...; se aceptan todas.
_HAND_CODES = {"D": "D", "R": "D", "DER": "D", "DERECHA": "D", "DERECHO": "D", "RIGHT": "D",
               "I": "I", "L": "I", "IZQ": "I", "IZQUIERDA": "I", "IZQUIERDO": "I", "LEFT": "I"}
SENSOR_NAMES = ("pulgar", "indice", "medio", "anular", "menique", "mano")
VALUES_PER_SENSOR = 8           # ax, ay, az (g), gx, gy, gz (°/s), pitch, roll (°)
N_VALUES = len(SENSOR_NAMES) * VALUES_PER_SENSOR

WINDOW_S = 2.0                  # duracion de cada muestra del dataset
# Ventana del modo automatico / respuesta en vivo: 1 s, para seguir el cambio
# de letra casi al paso de la camara (con 2 s el guante iba 1-2 s atrasado).
# Clasificando cada mitad de cada muestra contra las demas: 28/30 y 30/30.
LIVE_WINDOW_S = 1.0
MIN_FRAMES = 15                 # lecturas minimas para aceptar una ventana
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

# Dos manos: una lectura del izquierdo se junta con la del derecho si llegaron
# con menos de esto de diferencia (a ~25 Hz cada una van a 40 ms).
PAIR_MAX_DT = 0.15
PHRASE_WINDOW_S = 3.0           # duracion por defecto de una frase grabada
# Dos ESP32 mandando la misma "h" desde IPs distintas en este tiempo: el
# firmware del izquierdo quedo con "h":"D" (o al reves) y las lecturas se mezclan.
SAME_HAND_CLASH_S = 2.0

# Modo automatico: la misma seña aceptada en N evaluaciones seguidas (se
# evalua ~4 veces por segundo) se escribe; para repetirla hay que cambiar de
# postura RELEASE_TICKS evaluaciones.
STABLE_TICKS = 4
RELEASE_TICKS = 3
# Palabras (etiquetas de mas de una letra, HOLA, MAMÁ): llevan movimiento y la
# ventana de 2 s solo coincide con la muestra grabada un momento, asi que
# basta con menos evaluaciones seguidas.
STABLE_TICKS_WORD = 2
# Etiquetas de "ninguna seña" (manos en reposo, moviendose entre señas...):
# se graban como cualquier otra y nunca se escriben. Con pocas señas grabadas,
# sobre todo frases, todo se parece a algo; grabar NADA le da al clasificador
# con que comparar lo que no es seña.
NO_SIGN_LABELS = frozenset({"NADA", "REPOSO", "_"})

DEFAULT_DATASET = Path(__file__).resolve().parent / "datos_guante" / "dataset_guante.jsonl"
DEFAULT_DATASET_LEFT = DEFAULT_DATASET.with_name("dataset_guante_izquierdo.jsonl")
DEFAULT_DATASET_BOTH = DEFAULT_DATASET.with_name("dataset_guante_ambas.jsonl")

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z~]|\x1b.")


def clean_label(label: str) -> str:
    """Etiqueta limpia: sin secuencias de teclas (una flecha en input() deja
    '\\x1b[D' pegado: '\\x1b[DL' -> 'L'), en mayusculas y sin espacios."""
    label = _ANSI.sub("", str(label))
    label = "".join(ch for ch in label if ch.isprintable())
    return "_".join(label.upper().split())


def parse_hand(text: str) -> str:
    """"D", "der", "derecha", "I", "izq", "izquierda"... -> "D" o "I"."""
    t = plain_label(str(text).strip().upper())
    if t in ("D", "R") or t.startswith("DER") or t.startswith("RIGHT"):
        return "D"
    if t in ("I", "L") or t.startswith("IZQ") or t.startswith("LEFT"):
        return "I"
    raise ValueError(f"mano desconocida: {text!r} (usa D o I)")


def parse_glove(text: str) -> str:
    """Como parse_hand, y ademas "DI", "ambas", "ambos", "2" -> "DI" (los dos
    guantes juntos, para frases)."""
    t = plain_label(str(text).strip().upper())
    if t in ("DI", "ID", "2", "AMBAS", "AMBOS", "LAS DOS", "BOTH"):
        return GLOVE_BOTH
    return parse_hand(text)


def hand_code(h) -> Optional[str]:
    """"h" del paquete -> "D", "I" o None si no se reconoce."""
    if not isinstance(h, str):
        return None
    return _HAND_CODES.get(h.strip().upper())


def default_dataset(hand: str = GLOVE_HAND) -> Path:
    """Dataset de cada mano: el derecho es el original (dataset_guante.jsonl)."""
    return {"I": DEFAULT_DATASET_LEFT, GLOVE_BOTH: DEFAULT_DATASET_BOTH}.get(hand, DEFAULT_DATASET)


def plain_label(label: str) -> str:
    """Etiqueta sin acentos, para comparar MAMA (tecleada) con MAMÁ (camara)."""
    return "".join(ch for ch in unicodedata.normalize("NFD", label) if unicodedata.category(ch) != "Mn")


def _packet_values(d: dict) -> Optional[list[float]]:
    vec: list[float] = []
    try:
        for name in SENSOR_NAMES:
            values = d[name]
            if len(values) != VALUES_PER_SENSOR:
                return None
            vec.extend(float(v) for v in values)
    except (KeyError, TypeError, ValueError):
        return None
    return vec if all(np.isfinite(vec)) else None


def decode_packet_full(raw: bytes, hands=GLOVE_HANDS) -> tuple[Optional[list[float]], Optional[str], str, dict]:
    """Como decode_packet, y ademas {"h": lo que mando, "err": codigo,
    "values": lecturas de un paquete con err != 0, si se pudieron leer}. Las
    lecturas con err no se reconocen, pero se muestran para ver que ese guante
    si llega y que sensor falla."""
    hands = (hands,) if isinstance(hands, str) else tuple(hands)
    try:
        d = json.loads(raw.decode("utf-8", errors="ignore"))
    except (json.JSONDecodeError, ValueError):
        return None, None, "json", {}
    if not isinstance(d, dict):
        return None, None, "json", {}
    info = {"h": d.get("h"), "err": d.get("err")}
    hand = hand_code(d.get("h"))
    if hand not in hands:
        return None, None, "mano", info
    vec = _packet_values(d)
    if d.get("err") != 0:
        info["values"] = vec
        return None, hand, "err", info
    if vec is None:
        return None, hand, "forma", info
    return vec, hand, "", info


def decode_packet(raw: bytes, hands=GLOVE_HANDS) -> tuple[Optional[list[float]], Optional[str], str]:
    """JSON de la ESP -> (48 valores, mano, "") o (None, mano o None, motivo
    del descarte): "json" (corrupto), "mano" (una mano que no esta en hands,
    o sin "h"), "err" (algun sensor fallo, err != 0) o "forma" (faltan
    sensores o valores)."""
    return decode_packet_full(raw, hands)[:3]


def parse_packet(raw: bytes, hand: str = GLOVE_HAND) -> Optional[list[float]]:
    """JSON de la ESP -> 48 valores de esa mano, o None si se descarta (ver
    decode_packet)."""
    return decode_packet(raw, (hand,))[0]


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
    """Captura (n lecturas x 48, o x 96 con los dos guantes) -> 4 veces ese
    ancho: promedio de cada tercio (postura y como cambia) y desviacion
    estandar (cuanto se movio), todo dividido entre SCALE."""
    frames = np.asarray(frames, dtype=float)
    thirds = np.array_split(frames, 3)
    means = [t.mean(axis=0) for t in thirds]
    scale = np.tile(SCALE, frames.shape[1] // N_VALUES)
    return np.concatenate(means + [frames.std(axis=0)]) / np.tile(scale, 4)


# --------------------------------------------------------------------------- #
# Recepcion
# --------------------------------------------------------------------------- #


class GloveReceiver:
    """Recibe los JSON de los guantes en un hilo aparte (daemon, no bloquea el
    video) y guarda las ultimas lecturas de cada mano (hora, vector de 48).

    Es el UNICO lugar que abre el puerto 4210: las dos ESP32 mandan al mismo
    puerto y aqui se separan por "h". Sin SO_REUSEADDR a proposito: si otro
    programa ya lo tiene, start() falla con "Address already in use" en vez
    de quedarse callado sin recibir nada.

    Los metodos de lecturas (connected, latest, window...) son de la mano
    `hand` (por defecto la primera de hands, la derecha); hand("I") devuelve
    una vista de una sola mano con esos mismos metodos, para GloveSession."""

    def __init__(self, port: int = GLOVE_PORT, bind_ip: str = GLOVE_BIND_IP, hands=GLOVE_HANDS):
        self.address = (bind_ip, port)
        self.hands = (hands,) if isinstance(hands, str) else tuple(hands)
        self._frames: dict[str, deque[tuple[float, list[float]]]] = {h: deque() for h in self.hands}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._last_rx = {h: 0.0 for h in self.hands}
        self.bad_packets = 0
        # Paquetes descartados por motivo ("json", "mano", "err", "forma"), y
        # los err != 0 de cada mano (para su linea de estado).
        self.dropped: Counter[str] = Counter()
        self.err_by_hand: Counter[str] = Counter()
        self.last_sender: dict[str, tuple[str, int]] = {}
        # Diagnostico (para que un guante que llega mal se vea en pantalla y
        # no parezca apagado): "h" desconocidas por IP, ultimo paquete con
        # err de cada mano (hora, err, lecturas) e IPs que mandan cada mano.
        self.unknown_hands: dict[str, tuple[float, object]] = {}
        self._err_latest: dict[str, tuple[float, object, Optional[list[float]]]] = {}
        self._senders: dict[str, dict[str, float]] = {h: {} for h in self.hands}

    @property
    def hand_default(self) -> str:
        return self.hands[0]

    def _h(self, hand: Optional[str]) -> str:
        return self.hand_default if hand is None else hand

    @property
    def last_rx(self) -> float:
        """Ultimo paquete valido de cualquier mano."""
        return max(self._last_rx.values())

    def hand(self, hand: str):
        """Vista de una mano (GloveHand) o de las dos juntas ("DI", GloveBoth)."""
        if hand == GLOVE_BOTH and all(h in self.hands for h in GLOVE_HANDS):
            return GloveBoth(self)
        if hand not in self.hands:
            raise ValueError(f"este receptor no escucha la mano {hand!r}")
        return GloveHand(self, hand)

    def start(self) -> None:
        """Abre el puerto; lanza OSError si esta ocupado (sin reintentos)."""
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(self.address)
        except OSError as e:
            sock.close()
            raise OSError(e.errno, f"no se pudo abrir UDP {self.address[0]}:{self.address[1]} "
                                   f"({e.strerror}); ¿otro programa lo usa? ss -lunp | grep {self.address[1]}") from e
        sock.settimeout(0.2)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="guante", daemon=True)
        self._thread.start()
        log.info("Guante: escuchando UDP %s:%d (manos %s)", self.address[0], self.address[1],
                 ", ".join(self.hands))

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
        while self._running:
            try:
                raw, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    break
                time.sleep(0.2)
                continue
            self.handle_packet(raw, addr)

    def handle_packet(self, raw: bytes, addr: tuple[str, int], now: Optional[float] = None) -> None:
        """Un datagrama (el hilo de red; las pruebas lo llaman directo con
        otras IPs). Nada de lo que llegue puede tumbar el hilo: lo raro se
        descarta y se anota para el diagnostico."""
        now = time.time() if now is None else now
        vec, hand, reason, info = decode_packet_full(raw, self.hands)
        if vec is None:
            self.bad_packets += 1
            self.dropped[reason] += 1
            if reason == "err":
                self.err_by_hand[hand] += 1
                self._err_latest[hand] = (now, info.get("err"), info.get("values"))
                self._senders[hand][addr[0]] = now
            elif reason == "mano":
                if addr[0] not in self.unknown_hands:
                    log.warning("Guante: paquetes con \"h\"=%r desde %s (se esperaba %s)",
                                info.get("h"), addr[0], " o ".join(self.hands))
                self.unknown_hands[addr[0]] = (now, info.get("h"))
            return
        self.last_sender[hand] = addr
        self._senders[hand][addr[0]] = now
        self.push(vec, now, hand=hand)

    def push(self, vec: list[float], now: Optional[float] = None, hand: Optional[str] = None) -> None:
        """Agrega una lectura (el hilo de red; las pruebas la llaman directo)."""
        now = time.time() if now is None else now
        hand = self._h(hand)
        with self._lock:
            frames = self._frames[hand]
            frames.append((now, vec))
            while frames and now - frames[0][0] > BUFFER_S:
                frames.popleft()
        self._last_rx[hand] = now

    def connected(self, now: Optional[float] = None, hand: Optional[str] = None) -> bool:
        now = time.time() if now is None else now
        return now - self._last_rx[self._h(hand)] < CONNECTED_TIMEOUT_S

    def connected_hands(self, now: Optional[float] = None) -> list[str]:
        return [h for h in self.hands if self.connected(now, h)]

    def latest(self, hand: Optional[str] = None) -> Optional[list[float]]:
        """Ultima lectura recibida (48 valores), o None si no ha llegado nada."""
        with self._lock:
            frames = self._frames[self._h(hand)]
            return list(frames[-1][1]) if frames else None

    def frames_since(self, t0: float, hand: Optional[str] = None) -> np.ndarray:
        with self._lock:
            sel = [v for t, v in self._frames[self._h(hand)] if t >= t0]
        return np.array(sel, dtype=float).reshape(-1, N_VALUES)

    def window(self, seconds: float = WINDOW_S, now: Optional[float] = None,
               hand: Optional[str] = None) -> np.ndarray:
        now = time.time() if now is None else now
        return self.frames_since(now - seconds, hand)

    def timed_frames_since(self, t0: float, hand: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """(horas, lecturas) desde t0; para juntar las dos manos (GloveBoth)."""
        with self._lock:
            sel = [(t, v) for t, v in self._frames[self._h(hand)] if t >= t0]
        return (np.array([t for t, _ in sel], dtype=float),
                np.array([v for _, v in sel], dtype=float).reshape(-1, N_VALUES))

    def err_reading(self, now: Optional[float] = None,
                    hand: Optional[str] = None) -> Optional[tuple[object, Optional[list[float]]]]:
        """(err, lecturas) si esa mano solo esta mandando paquetes con err != 0
        (ninguno valido en CONNECTED_TIMEOUT_S), o None."""
        now = time.time() if now is None else now
        hand = self._h(hand)
        last = self._err_latest.get(hand)
        if last is None or now - last[0] >= CONNECTED_TIMEOUT_S or self.connected(now, hand):
            return None
        return last[1], last[2]

    def problems(self, now: Optional[float] = None) -> list[str]:
        """Avisos cortos de lo que esta llegando mal (para la barra de estado):
        guante con "h" desconocida, dos ESP con la misma mano o una mano
        que solo manda err."""
        now = time.time() if now is None else now
        out = []
        for ip, (t, h) in self.unknown_hands.items():
            if now - t < SAME_HAND_CLASH_S:
                out.append(f"{ip} manda \"h\":{json.dumps(h)} (usa \"D\" o \"I\")")
        for hand in self.hands:
            ips = sorted(ip for ip, t in self._senders[hand].items() if now - t < SAME_HAND_CLASH_S)
            if len(ips) > 1:
                out.append(f"{' y '.join(ips)} mandan los dos \"h\":\"{hand}\"")
            err = self.err_reading(now, hand)
            if err is not None:
                out.append(f"{hand}: solo llegan paquetes con err={err[0]}")
        return out

    def rate_hz(self, now: Optional[float] = None, hand: Optional[str] = None) -> float:
        """Lecturas por segundo en los ultimos 2 s."""
        now = time.time() if now is None else now
        with self._lock:
            n = sum(1 for t, _ in self._frames[self._h(hand)] if now - t <= 2.0)
        return n / 2.0

    def status_line(self, now: Optional[float] = None, hand: Optional[str] = None) -> str:
        """Linea corta de estado de una mano, ej. "D 25.0 Hz, err 0" (err =
        paquetes de esa mano descartados por err != 0 desde que arranco)."""
        hand = self._h(hand)
        line = f"{hand} {self.rate_hz(now, hand):.1f} Hz, err {self.err_by_hand[hand]}"
        other = sum(n for k, n in self.dropped.items() if k != "err")
        return line + (f", descartados {other}" if other else "")


class GloveHand:
    """Vista de una sola mano de un GloveReceiver, con sus mismos metodos de
    lecturas. start/stop no: el puerto es del receptor compartido."""

    def __init__(self, receiver: GloveReceiver, hand: str):
        self.receiver = receiver
        self.hand = hand

    def connected(self, now: Optional[float] = None) -> bool:
        return self.receiver.connected(now, self.hand)

    def latest(self) -> Optional[list[float]]:
        return self.receiver.latest(self.hand)

    def frames_since(self, t0: float) -> np.ndarray:
        return self.receiver.frames_since(t0, self.hand)

    def window(self, seconds: float = WINDOW_S, now: Optional[float] = None) -> np.ndarray:
        return self.receiver.window(seconds, now, self.hand)

    def rate_hz(self, now: Optional[float] = None) -> float:
        return self.receiver.rate_hz(now, self.hand)

    def status_line(self, now: Optional[float] = None) -> str:
        return self.receiver.status_line(now, self.hand)

    def push(self, vec: list[float], now: Optional[float] = None) -> None:
        self.receiver.push(vec, now, self.hand)


def pair_frames(t_d: np.ndarray, f_d: np.ndarray, t_i: np.ndarray, f_i: np.ndarray,
                max_dt: float = PAIR_MAX_DT) -> np.ndarray:
    """Junta cada lectura del derecho con la del izquierdo mas cercana en el
    tiempo -> (n x 96: derecho y luego izquierdo). Las ESP no van
    sincronizadas; se sueltan las del derecho sin pareja a menos de max_dt."""
    if not len(t_d) or not len(t_i):
        return np.zeros((0, 2 * N_VALUES))
    j = np.clip(np.searchsorted(t_i, t_d), 1, len(t_i) - 1) if len(t_i) > 1 else np.zeros(len(t_d), int)
    if len(t_i) > 1:
        prev = j - 1
        j = np.where(np.abs(t_i[prev] - t_d) <= np.abs(t_i[j] - t_d), prev, j)
    keep = np.abs(t_i[j] - t_d) <= max_dt
    return np.hstack([f_d[keep], f_i[j[keep]]])


class GloveBoth:
    """Vista de los dos guantes juntos, con los mismos metodos que GloveHand:
    cada lectura son 96 valores (derecho + izquierdo). Esta conectada solo si
    llegan las dos manos."""

    hand = GLOVE_BOTH

    def __init__(self, receiver: GloveReceiver):
        self.receiver = receiver

    def connected(self, now: Optional[float] = None) -> bool:
        return all(self.receiver.connected(now, h) for h in GLOVE_HANDS)

    def latest(self) -> Optional[list[float]]:
        d, i = (self.receiver.latest(h) for h in GLOVE_HANDS)
        return d + i if d is not None and i is not None else None

    def frames_since(self, t0: float) -> np.ndarray:
        return pair_frames(*self.receiver.timed_frames_since(t0, "D"),
                           *self.receiver.timed_frames_since(t0, "I"))

    def window(self, seconds: float = WINDOW_S, now: Optional[float] = None) -> np.ndarray:
        now = time.time() if now is None else now
        return self.frames_since(now - seconds)

    def rate_hz(self, now: Optional[float] = None) -> float:
        return min(self.receiver.rate_hz(now, h) for h in GLOVE_HANDS)

    def status_line(self, now: Optional[float] = None) -> str:
        return " · ".join(self.receiver.status_line(now, h) for h in GLOVE_HANDS)


# --------------------------------------------------------------------------- #
# Dataset y clasificacion
# --------------------------------------------------------------------------- #


def dataset_width(path: Path) -> int:
    """Valores por lectura de un dataset: 48 (un guante) o 96 (los dos)."""
    return 2 * N_VALUES if Path(path).name == DEFAULT_DATASET_BOTH.name else N_VALUES


def load_dataset(path: Path, n_values: Optional[int] = None) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Lee el .jsonl de grabar_guante.py -> (capturas, etiquetas, personas).
    Salta lineas rotas o capturas con otra forma, con aviso. n_values: 48 (un
    guante) o 96 (los dos; por defecto segun el nombre del archivo)."""
    n_values = n_values or dataset_width(path)
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
            if arr.ndim != 2 or arr.shape[1] != n_values or len(arr) < MIN_FRAMES or not label:
                log.warning("%s línea %d ignorada: captura %s", path.name, n, arr.shape)
                continue
            frames.append(arr)
            labels.append(label)
            people.append(str(m.get("persona", "")))
    return frames, labels, people


def sample_seconds(path: Path) -> float:
    """Duracion de las muestras de un dataset (mediana de "segundos"; las
    grabadas antes de guardarla duran WINDOW_S)."""
    secs = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    secs.append(float(json.loads(line).get("segundos", WINDOW_S)))
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    continue
    except OSError:
        pass
    return float(np.median(secs)) if secs else WINDOW_S


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
                 max_distance: Optional[float] = None, window_s: float = WINDOW_S):
        if not labels:
            raise ValueError("el dataset del guante no tiene muestras")
        self.window_s = window_s        # lo que dura cada muestra (frases: lo grabado)
        self.X = np.array([frames_to_features(f) for f in frames])
        # Mano de cada columna (0 derecho, 1 izquierdo con los dos guantes).
        n_hands = max(1, self.X.shape[1] // (4 * N_VALUES))
        self._hand_of_col = (np.arange(self.X.shape[1]) % (n_hands * N_VALUES)) // N_VALUES
        self.n_hands = n_hands
        self.y = np.array(labels)
        self.counts = Counter(labels)
        self.labels = sorted(self.counts)
        self.max_distance = max_distance if max_distance else self._calibrate()

    @property
    def sign_labels(self) -> list[str]:
        """Las señas que se pueden escribir (sin NADA)."""
        return [l for l in self.labels if l not in NO_SIGN_LABELS]

    @classmethod
    def from_file(cls, path: Path, max_distance: Optional[float] = None,
                  n_values: Optional[int] = None) -> "GloveClassifier":
        frames, labels, _ = load_dataset(Path(path), n_values)
        return cls(frames, labels, max_distance, sample_seconds(Path(path)))

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
        """Distancia a cada muestra. Con los dos guantes, la peor de las dos
        manos: cada una tiene que parecerse (promediando, una mano igual
        tapaba a la otra distinta: L + A salia como la frase de C + C)."""
        sq = (self.X - feat) ** 2
        if self.n_hands == 1:
            return np.sqrt(sq.mean(axis=1))
        return np.max([np.sqrt(sq[:, self._hand_of_col == h].mean(axis=1)) for h in range(self.n_hands)], axis=0)

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
        if topk[0][0] in NO_SIGN_LABELS:
            reason = f"sin seña ({topk[0][0]})"
        elif best > self.max_distance:
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

    def __init__(self, receiver, classifier: GloveClassifier, auto: bool = True,
                 live_window_s: float = LIVE_WINDOW_S, min_gap_s: float = 0.0):
        # receiver: GloveReceiver (su mano por defecto), GloveHand (una mano)
        # o GloveBoth (las dos, frases). live_window_s: ventana del
        # automatico; una frase se evalua con lo que dura (classifier.window_s).
        # min_gap_s: tras escribir, nada mas en ese tiempo (una frase no se
        # puede hacer mas rapido de lo que dura; sin esto, al sostenerla la
        # ventana pasa por la frase parecida y se escriben las dos).
        self.receiver = receiver
        self.classifier = classifier
        self.auto = auto
        self.live_window_s = live_window_s
        self.min_gap_s = min_gap_s
        self._last_auto_commit = -np.inf
        self._last_auto_label: Optional[str] = None
        self.spotter = GloveSpotter()
        self._capture_at: Optional[float] = None     # hora en que empieza a capturar
        self._last_count: Optional[int] = None
        self._capturing = False

    @property
    def busy(self) -> bool:
        return self._capture_at is not None

    def request_capture(self, now: Optional[float] = None) -> None:
        """Captura con cuenta atras (como grabar_guante.py): 2, 1, ¡ya! y lo
        que dura cada muestra (2 s; una frase, lo grabado)."""
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
            elif now < self._capture_at + self.classifier.window_s:
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
            frames = self.receiver.window(self.live_window_s, now)
            if len(frames) >= MIN_FRAMES:
                result = self.classifier.classify(frames)
                events.append(GloveEvent("vivo", {"result": result}))
                label = self.spotter.update(result)
                if label is not None and now - self._last_auto_commit < self.min_gap_s:
                    # No se escribe; sigue bloqueada la que si se escribio.
                    self.spotter._locked = self._last_auto_label
                    label = None
                if label is not None:
                    self._last_auto_commit = now
                    self._last_auto_label = label
                    events.append(GloveEvent("resultado", {"result": result, "commit": label, "manual": False}))
        return events
