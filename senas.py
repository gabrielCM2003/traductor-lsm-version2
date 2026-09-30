from __future__ import annotations

import argparse
import errno
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
    QLabel, QPushButton, QMessageBox, QSlider, QComboBox,
    QFileDialog, QStatusBar, QSizePolicy, QCheckBox, QStackedWidget,
)

from interfaz_lsm import (
    COLORS, PHASE_STYLE, STYLESHEET, Card, CandidateBars, Feedback, FeedbackPanel, ManualPage, ManualWindow, Pill,
    SettingsDialog, StartPage, big_button, feedback_for_result, guidance_feedback, how_to_sign, sentence_html,
)

from guante import (
    DEFAULT_DATASET as GLOVE_DEFAULT_DATASET, DEFAULT_DATASET_BOTH as GLOVE_DEFAULT_DATASET_BOTH,
    DEFAULT_DATASET_LEFT as GLOVE_DEFAULT_DATASET_LEFT, GLOVE_BOTH, GLOVE_HANDS, GLOVE_PORT, N_VALUES as GLOVE_N_VALUES,
    HAND_NAMES as GLOVE_HAND_NAMES, GloveClassifier, GloveReceiver, GloveSession, format_reading, plain_label,
)

# Camara + guante. Si la camara vio una mano hace menos de esto (o la sena
# esta en curso), el guante no escribe solo: su respuesta se combina con la de
# la camara (fuse_topk) y tienen que coincidir. Si la camara no ve la mano
# (guante oscuro, fuera de cuadro, camara apagada), el guante escribe por su
# cuenta.
GLOVE_CAMERA_HAND_S = 1.0
# Los dos guantes escriben por su cuenta (sin camara): una palabra de dos
# manos grabada en ambos (AYUDA) llegaria dos veces. La misma seña de la otra
# mano dentro de este tiempo no se vuelve a escribir.
GLOVE_BOTH_HANDS_DEDUP_S = 1.5
# Guante de cada mano de la persona (Ajustes -> Mano que deletrea).
GLOVE_HAND_OF_DOMINANT = {"Right": "D", "Left": "I"}
# Frases de dos manos (los dos guantes juntos): las letras que un guante solo
# escribio durante la frase (su postura paso por una letra) se borran cuando
# llega la frase, como la camara con las palabras. Solo si son pocas.
GLOVE_PHRASE_MAX_RETRACTED = 3

# Ilustraciones del manual de senas (las genera generar_manual.py).
MANUAL_DIR = Path(__file__).resolve().parent / "manual"

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: falta 'mediapipe'. Instala con: pip install mediapipe", file=sys.stderr)
    raise

from sign_classifier import SignClassifier, PredictionSmoother, normalize_keypoints, hand_to_feature_vector
from body_tracker import (
    BodyDetection, BodyTracker, DEFAULT_POSE_MODEL, POSE_MODELS, body_location_features,
    draw_body_skeleton, draw_rest_line, hands_in_signing_space, rest_line_y,
)

try:
    from segmentador_automatico import (
        AutoSegmenter, MIN_SEQUENCE_MS as DYN_STANDALONE_MIN_SEQUENCE_MS,
        PALABRAS_MAX_SEQUENCE_MS, PALABRAS_MIN_SEQUENCE_MS, PALABRAS_REST_MS_TO_END,
    )
    from dtw_recognizer import (
        DEFAULT_WORDS_DIR, WORD_BODY_WEIGHT, WORD_TEMPERATURE, DTWRecognizer, distances_to_topk,
        mirror_and_swap_hands,
    )
except ImportError as e:
    # Alfabeto dinamico (J,K,Ñ,Q,X,Z): opcional. Si falta fastdtw/scipy o los
    # archivos aun no existen, el alfabeto estatico sigue funcionando igual
    # que antes; el modo dinamico simplemente queda deshabilitado.
    AutoSegmenter = None
    DTWRecognizer = None
    mirror_and_swap_hands = None
    distances_to_topk = None
    DYN_STANDALONE_MIN_SEQUENCE_MS = 170
    PALABRAS_REST_MS_TO_END, PALABRAS_MIN_SEQUENCE_MS, PALABRAS_MAX_SEQUENCE_MS = 400, 300, 8000
    DEFAULT_WORDS_DIR, WORD_BODY_WEIGHT, WORD_TEMPERATURE = None, 4.0, 0.5
    logging.getLogger("sign_translator").warning(
        "Alfabeto dinamico no disponible (%s). Instala fastdtw/scipy para habilitarlo.", e
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
# La ventana precarga los reconocedores en un hilo al abrir (preload_models),
# para que Iniciar no congele la ventana; este candado evita cargarlos dos veces.
_models_lock = threading.RLock()

# Plantillas de letras dinamicas a ~15 fps (1 de cada 2 frames de las de 30
# fps): el DTW contra las 689 plantillas bajo de 94 a 27 ms en la laptop.
# Medido con 150 plantillas, cada una contra las demas: letra correcta 146 ->
# 145, bien escritas 142 -> 140, mal escritas 1 -> 1. La consulta se toma a
# la misma frecuencia (LETTER_DTW_FPS), que es mas o menos lo que procesa una
# Raspberry Pi. Las palabras se quedan a 30 fps: su DTW ya es barato (12 ms)
# y a 15 fps bajaban de 56/59 a 54/59 con personas nuevas.
LETTER_TEMPLATE_STEP = 2
TEMPLATE_FPS = 30.0
LETTER_DTW_FPS = TEMPLATE_FPS / LETTER_TEMPLATE_STEP


def _get_dtw_recognizer() -> Optional["DTWRecognizer"]:
    global _dtw_recognizer_singleton, _dtw_recognizer_load_attempted
    with _models_lock:
        if not _dtw_recognizer_load_attempted:
            _dtw_recognizer_load_attempted = True
            if DTWRecognizer is not None:
                try:
                    _dtw_recognizer_singleton = DTWRecognizer.try_load(template_step=LETTER_TEMPLATE_STEP)
                except Exception:
                    logging.getLogger("sign_translator").exception("Error cargando DTWRecognizer")
    return _dtw_recognizer_singleton


def preload_models() -> None:
    """Carga las plantillas (letras y palabras) y los perfiles de palabras.
    La ventana lo llama en un hilo al abrir."""
    _get_dtw_recognizer()
    word_profiles()


# --------------------------------------------------------------------------- #
# Modo automatico: letras estaticas, letras dinamicas y palabras completas a
# la vez, sin botones para cambiar de modo (es el modo de la ventana).
#
# Una ACTIVIDAD dura desde que una mano sube sobre la linea de reposo (pose,
# ver body_tracker.rest_line_y) hasta que baja; sin hombros en cuadro, desde
# que aparece la mano hasta que sale. Es el mismo corte que
# segmentador_automatico.py --modo palabras (constantes PALABRAS_*).
#   - Letras estaticas: el clasificador de siempre corre en cada frame, pero
#     solo fija la letra si la mano esta QUIETA (AUTO_STATIC_MAX_SPEED), arriba
#     de la linea de reposo y es la UNICA mano arriba (el alfabeto es de una
#     mano; POR FAVOR y AYUDA son de dos).
#   - Al bajar las manos, la actividad completa se compara con DTW contra las
#     letras dinamicas (datos_dinamicas/) y las palabras
#     (datos_palabras_dinamicas/).
#     La categoria se decide comparando SOLO las manos (el mismo vector de
#     126 en las dos): es palabra si su plantilla mas cercana esta a menos de
#     AUTO_WORD_PREFERENCE veces la distancia de la letra mas cercana.
#   - Letra: solo si en la actividad no se fijo ninguna letra estatica (si
#     se fijo, fue deletreo); se aplica dynamic_commit_decision. Las letras
#     con movimiento (J, K, Ñ, Q, X, Z) se hacen solas: subir la mano, hacer
#     la letra y bajarla.
#   - Palabra: el DTW con la ubicacion respecto al cuerpo decide cual y
#     word_commit_decision si se escribe. Muchas palabras tienen una pausa con
#     la mano quieta (HOLA en la frente, medio segundo) en la que el
#     clasificador estatico alcanza a fijar una letra: si en la actividad se
#     fijaron a lo mas AUTO_MAX_RETRACTED_LETTERS, se borran y se escribe la
#     palabra. Con mas letras fue deletreo y se respeta.
# --------------------------------------------------------------------------- #

# Mano quieta para fijar una letra estatica: velocidad media de la muneca en
# los ultimos AUTO_SPEED_WINDOW frames, en tamanos de mano (muneca -> base del
# dedo medio) por segundo, para que no dependa de la distancia a la camara.
AUTO_STATIC_MAX_SPEED = 1.5
AUTO_SPEED_WINDOW = 6
# Palabra si d_palabra < AUTO_WORD_PREFERENCE * d_letra (solo manos). Con
# personas que no estan en las plantillas (cada persona de los videos contra
# las otras dos), la distancia a las palabras crece: con 1.2, 55/59 palabras
# quedaban en su categoria; con 1.4, 58/59. Las letras (cada plantilla contra
# las demas): 687/689 con 1.2, ~680/689 con 1.4. Se prefiere 1.4 porque las
# palabras fallaban justo con gente nueva.
AUTO_WORD_PREFERENCE = 1.4
# Letras estaticas que una palabra o letra con movimiento puede reemplazar
# (ver arriba). Medido pasando los videos por la app: HOLA fijaba una "R" en
# la pausa de la frente.
AUTO_MAX_RETRACTED_LETTERS = 2

# Misma prioridad para los tres tipos: una letra fija de la sena en curso solo
# se reemplaza si la sena con movimiento lo demuestra, no por ser de otro tipo.
# Pasando las plantillas por el clasificador estatico, casi todas las letras
# con movimiento fijan antes su letra de partida (J->I, K->P, N->Ñ, Z->D,
# X->G/L, Q->L; cota superior: sin la regla de mano quieta). Antes eso
# bloqueaba siempre la letra con movimiento. Pero una letra fija SOSTENIDA
# tambien se parece a una con movimiento (I->J, D->Z, N->Ñ pasan la regla de
# escritura), asi que se exige distancia DTW <= DYN_REPLACE_MAX_DISTANCE:
# ninguna de las 19 letras fijas sostenidas (poses de manual/letras) quedo por
# debajo de 1.15, y 64% de las letras con movimiento bien escritas (cada
# plantilla contra las demas) quedan por debajo de 1.0.
DYN_REPLACE_MAX_DISTANCE = 1.0
# Palabras sin cuerpo visible: comparando solo las manos, una letra fija
# sostenida queda tan cerca de una palabra como una palabra real (letras
# desde 0.65; palabras, mediana 1.42), asi que sin la ubicacion la palabra
# solo reemplaza letras si esta muy cerca. Con cuerpo visible se reemplazan
# como siempre (medido con los videos).
WORD_NO_BODY_REPLACE_MAX = 0.6
# Palabras con cuerpo en menos de esta fraccion de frames se comparan SOLO con
# las manos. Con el bloque de cuerpo en ceros, el DTW con cuerpo se iba a
# MAMA (su mano esta junto a la boca: su bloque es casi cero): sin cuerpo, las
# 12 plantillas de HOLA salian MAMA y 39/59 palabras bien; solo con las manos,
# 53/59 (cada plantilla contra las demas). Si falta el cuerpo en pocos frames,
# se rellenan con el frame con cuerpo mas cercano.
WORD_MIN_BODY_FRACTION = 0.5


def fill_missing_body(seq: np.ndarray) -> np.ndarray:
    """Frames sin cuerpo (bandera en 0) toman el bloque de cuerpo del frame
    con cuerpo mas cercano. Sin ningun frame con cuerpo, la deja igual."""
    ok = seq[:, 134] > 0
    if ok.all() or not ok.any():
        return seq
    idx = np.flatnonzero(ok)
    nearest = idx[np.abs(idx[None, :] - np.arange(len(seq))[:, None]).argmin(axis=1)]
    out = seq.copy()
    out[:, 126:135] = seq[nearest, 126:135]
    return out

# Palabras (DTW con cuerpo, peso WORD_BODY_WEIGHT; confianza con
# WORD_TEMPERATURE). Medido reconociendo a cada persona de los videos solo con
# las plantillas de las otras dos: 56/59 bien. Con margen >= 0.15 se escriben
# 54 de esas 56 y no se agrega ningun error (los 3 errores, AYUDA<->GRACIAS,
# tienen margen alto y ningun umbral los separa). La regla anterior (margen
# 0.30 sin calibrar y distancia <= 10) solo escribia 49: era el "titubeo".
WORD_MIN_MARGIN = 0.15
# Por encima de esta distancia DTW el movimiento no se parece a ninguna
# palabra. Con personas nuevas la mayor distancia de una palabra bien
# reconocida fue 11.1 (con las mismas personas, 9.4).
WORD_MAX_DISTANCE = 16.0

# HOLA y MAMA se confunden en vivo (una mano, a la altura de la cara). Se
# distinguen por donde queda la punta del indice: MAMA en la boca, HOLA en la
# frente. Fraccion de frames con la punta a menos de 0.35 anchos de hombro de
# la boca, en las 24 plantillas: HOLA 0.00-0.20, MAMA 0.71-0.98. Si HOLA y
# MAMA son las 2 primeras, esa fraccion decide (<= HOLA_MAX_NEAR -> HOLA,
# >= MAMA_MIN_NEAR -> MAMA; en medio no se toca). Hace falta ver el cuerpo.
NEAR_MOUTH = 0.35
HOLA_MAX_NEAR = 0.35
MAMA_MIN_NEAR = 0.55
HOLA_MAMA = ("HOLA", "MAMÁ")

_word_recognizers: Optional[tuple["DTWRecognizer", "DTWRecognizer"]] = None
_word_recognizers_load_attempted = False


def _get_word_recognizers() -> Optional[tuple["DTWRecognizer", "DTWRecognizer"]]:
    """(solo manos, manos + cuerpo) sobre las mismas plantillas de palabras,
    o None si no hay. El de solo manos sirve para comparar contra las letras
    en la misma escala; el de cuerpo, para decidir cual palabra."""
    global _word_recognizers, _word_recognizers_load_attempted
    with _models_lock:
        if not _word_recognizers_load_attempted:
            _word_recognizers_load_attempted = True
            if DTWRecognizer is not None and DEFAULT_WORDS_DIR is not None and DEFAULT_WORDS_DIR.is_dir():
                try:
                    hands_only = DTWRecognizer(data_dir=DEFAULT_WORDS_DIR, auto_save_labels=False)
                    with_body = DTWRecognizer(
                        data_dir=DEFAULT_WORDS_DIR, auto_save_labels=False, body_weight=WORD_BODY_WEIGHT
                    )
                    if with_body.labels:
                        _word_recognizers = (hands_only, with_body)
                except Exception:
                    logging.getLogger("sign_translator").exception("Error cargando las plantillas de palabras")
    return _word_recognizers


@dataclass
class SignStats:
    """Donde y como se hizo una sena con movimiento, en anchos de hombro (ver
    body_location_features): muneca respecto al centro de los hombros (dy > 0
    es hacia abajo), distancia de la punta del indice a la boca, fraccion de
    frames con dos manos y duracion. La interfaz compara esto con el perfil de
    cada palabra para decir que corregir (interfaz_lsm.compare_to)."""
    wrist_dx: float = 0.0
    wrist_dy: float = 0.0
    tip_mouth: float = 9.0
    two_hands: float = 0.0
    duration_s: float = 0.0
    has_body: bool = False
    # Fraccion de frames con la punta del indice a menos de NEAR_MOUTH de la
    # boca (desempata HOLA / MAMA, ver hola_mama_rule).
    near_mouth: float = 0.0


def sign_stats(sequence: np.ndarray, duration_s: float) -> SignStats:
    """SignStats de una secuencia cruda (T, 126 + 9). La mano que se mide es
    la que aparece en mas frames con cuerpo visible."""
    seq = np.asarray(sequence, dtype=np.float64)
    hands, body = seq[:, :126], seq[:, 126:135]
    left = np.abs(hands[:, :63]).sum(axis=1) > 0
    right = np.abs(hands[:, 63:126]).sum(axis=1) > 0
    two = float(np.mean(left & right)) if len(seq) else 0.0
    with_body = body[:, 8] > 0
    use_left = (left & with_body).sum() >= (right & with_body).sum()
    mask, off = ((left & with_body), 0) if use_left else ((right & with_body), 4)
    if not mask.any():
        return SignStats(two_hands=two, duration_s=duration_s)
    wrist = body[mask, off:off + 2]
    tip = body[mask, off + 2:off + 4]
    tip_dist = np.linalg.norm(tip, axis=1)
    return SignStats(
        wrist_dx=float(np.median(wrist[:, 0])),
        wrist_dy=float(np.median(wrist[:, 1])),
        tip_mouth=float(np.median(tip_dist)),
        two_hands=two,
        duration_s=duration_s,
        has_body=True,
        near_mouth=float(np.mean(tip_dist < NEAR_MOUTH)),
    )


_word_profiles: Optional[dict[str, SignStats]] = None


def word_profiles() -> dict[str, SignStats]:
    """{PALABRA como se muestra: perfil mediano de sus plantillas}. Las
    plantillas vienen de videos a ~30 fps (duracion = frames / 30)."""
    with _models_lock:
        return _compute_word_profiles()


def _compute_word_profiles() -> dict[str, SignStats]:
    global _word_profiles
    if _word_profiles is None:
        _word_profiles = {}
        recognizers = _get_word_recognizers()
        if recognizers is not None:
            with_body = recognizers[1]
            for label, templates in with_body._templates.items():
                per = []
                for tmpl in templates:
                    raw = np.array(tmpl, dtype=np.float64)
                    raw[:, 126:] /= WORD_BODY_WEIGHT     # las plantillas guardan el cuerpo ya ponderado
                    per.append(sign_stats(raw, len(raw) / 30.0))
                with_b = [s for s in per if s.has_body] or per
                _word_profiles[word_display(label)] = SignStats(
                    wrist_dx=float(np.median([s.wrist_dx for s in with_b])),
                    wrist_dy=float(np.median([s.wrist_dy for s in with_b])),
                    tip_mouth=float(np.median([s.tip_mouth for s in with_b])),
                    two_hands=float(np.median([s.two_hands for s in per])),
                    duration_s=float(np.median([s.duration_s for s in per])),
                    has_body=any(s.has_body for s in per),
                )
    return _word_profiles


def word_display(label: str) -> str:
    """Etiqueta de carpeta -> texto: POR_FAVOR -> POR FAVOR."""
    return label.replace("_", " ")


def hola_mama_rule(topk: list[tuple[str, float]], stats: Optional["SignStats"]) -> tuple[list[tuple[str, float]], str]:
    """Desempate HOLA / MAMA por la distancia de la punta del indice a la
    boca. Si decide, la ganadora queda primera con margen suficiente para
    escribirse. Devuelve (topk, nota para el log; "" si no cambio nada)."""
    if len(topk) < 2 or {topk[0][0], topk[1][0]} != set(HOLA_MAMA) or stats is None or not stats.has_body:
        return topk, ""
    if stats.near_mouth <= HOLA_MAX_NEAR:
        winner = "HOLA"
    elif stats.near_mouth >= MAMA_MIN_NEAR:
        winner = "MAMÁ"
    else:
        return topk, ""
    loser = HOLA_MAMA[1] if winner == "HOLA" else HOLA_MAMA[0]
    total = topk[0][1] + topk[1][1]
    p_win = max(topk[0][1] if topk[0][0] == winner else topk[1][1], total * 0.8)
    new = [(winner, p_win), (loser, total - p_win)] + list(topk[2:])
    if new[0][0] == topk[0][0] and abs(new[0][1] - topk[0][1]) < 1e-9:
        return topk, ""
    return new, f"desempate {winner} (punta del índice cerca de la boca {stats.near_mouth * 100:.0f}% del tiempo)"


def word_commit_decision(topk: list[tuple[str, float]], best_distance: float) -> tuple[bool, float, str]:
    """Si se escribe la palabra top-1 del DTW con cuerpo. Devuelve
    (se_escribe, margen sobre la 2.a, motivo si no se escribe)."""
    if not topk:
        return False, 0.0, "sin candidatos"
    margin = topk[0][1] - topk[1][1] if len(topk) > 1 else topk[0][1]
    if best_distance > WORD_MAX_DISTANCE:
        return False, margin, f"no se parece a ninguna palabra (distancia {best_distance:.1f})"
    if margin < WORD_MIN_MARGIN:
        return False, margin, f"margen {margin * 100:.0f}pp < {WORD_MIN_MARGIN * 100:.0f}pp"
    return True, margin, ""


def _is_raspberry_pi() -> bool:
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:
        return False


IS_RASPBERRY_PI = _is_raspberry_pi()

APP_NAME = "SignTranslator"
APP_ORG = "OpenLSM"
APP_VERSION = "4.0-lsm-automatico"

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
    # Alfabeto estatico: frames con otra letra (o sin mano) para poder volver
    # a confirmar la misma letra (LL, RR, EE): "relajar la mano un instante".
    "repeat_release_frames": 8,
    # Sliders de confianza minima y margen del alfabeto estatico (se guardan).
    "min_letter_confidence": 0.55,
    "min_letter_margin": 0.15,
    # Mano que deletrea, la de la PERSONA: "Right" | "Left" (ver
    # MEDIAPIPE_LABEL_OF_HAND para como se traduce a la etiqueta de MediaPipe).
    "dominant_hand": "Right",
    "speak_words": False,
    "keypoint_buffer_size": 30,
    "queue_maxsize": 1,
    "watchdog_timeout_s": 5.0,
    "draw_landmarks": True,
    "draw_connections": True,
    # Esqueleto del cuerpo (MediaPipe Pose, ver body_tracker.py). En False no
    # se carga el modelo de pose: util si en la Raspberry Pi hace falta el
    # tiempo de CPU y solo se usa el alfabeto.
    "body_tracking": True,
    "draw_body": True,
    # "full" (mas estable) o "lite" (mas rapido). En la Raspberry Pi va
    # "lite" por defecto. Ver DEFAULT_POSE_MODEL en body_tracker.py.
    "pose_model": "lite" if IS_RASPBERRY_PI else DEFAULT_POSE_MODEL,
    # La pose corre en su propio hilo, en paralelo a las manos, y cada frame
    # usa el ultimo cuerpo disponible (a lo mas un frame atras; hombros y
    # boca casi no se mueven en ese tiempo). Medido en la laptop: manos +
    # pose en serie 16.4 ms por frame, en paralelo ~8 ms (lo de las manos).
    "pose_async": True,
    # Guante con ESP32 (guante.py). La Raspberry es el punto de acceso (red
    # GUANTE_LSM, 10.42.0.1) y la ESP le manda los datos por UDP a este
    # puerto; aqui solo se escucha (las dos manos, "h" D e I). glove_dataset
    # y glove_dataset_left vacios = los de datos_guante/ (los graba
    # grabar_guante.py --mano D / I); glove_dataset_both, las frases con los
    # dos guantes (--mano DI). glove_auto: escribe la sena en cuanto se
    # sostiene; si no, solo con la captura (Ctrl+G, Ctrl+Shift+G la frase).
    "glove_port": GLOVE_PORT,
    "glove_dataset": "",
    "glove_dataset_left": "",
    "glove_dataset_both": "",
    "glove_auto": True,
}

# Rangos validos al cargar config.json / CLI. Los que tienen slider usan su
# mismo rango. queue_maxsize >= 1: con 0 la cola seria infinita.
CONFIG_RANGES: dict[str, tuple[float, float]] = {
    "camera_index": (0, 63),
    "max_num_hands": (1, 4),
    "min_detection_confidence": (0.10, 0.95),
    "min_tracking_confidence": (0.0, 1.0),
    "smoothing_window": (5, 30),
    "stable_frames_to_commit": (3, 25),
    "no_hand_frames_for_space": (5, 300),
    "repeat_release_frames": (2, 60),
    "min_letter_confidence": (0.30, 0.90),
    "min_letter_margin": (0.0, 0.50),
    "keypoint_buffer_size": (5, 300),
    "queue_maxsize": (1, 10),
    "watchdog_timeout_s": (2.0, 60.0),
    "glove_port": (1, 65535),
}
CONFIG_CHOICES: dict[str, tuple[str, ...]] = {
    "dominant_hand": ("Right", "Left"),
    "pose_model": POSE_MODELS,
}

# Etiqueta que MediaPipe le pone a cada mano de la persona en ESTE programa
# (frame volteado en espejo antes de detectar): la contraria. Medido: 585 de
# las 586 plantillas de datos_dinamicas/ (CICESE + grabaciones del equipo,
# personas que deletrean con la derecha) tienen la mano en el slot "Left", y
# draw_hand_label ya muestra "Derecha" para "Left". No confiar en la
# documentacion de MediaPipe, que dice lo contrario para imagenes en espejo.
MEDIAPIPE_LABEL_OF_HAND = {"Right": "Left", "Left": "Right"}


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
    min_letter_confidence: float = DEFAULT_CONFIG["min_letter_confidence"]
    min_letter_margin: float = DEFAULT_CONFIG["min_letter_margin"]
    dominant_hand: str = DEFAULT_CONFIG["dominant_hand"]
    speak_words: bool = DEFAULT_CONFIG["speak_words"]
    keypoint_buffer_size: int = DEFAULT_CONFIG["keypoint_buffer_size"]
    queue_maxsize: int = DEFAULT_CONFIG["queue_maxsize"]
    watchdog_timeout_s: float = DEFAULT_CONFIG["watchdog_timeout_s"]
    draw_landmarks: bool = DEFAULT_CONFIG["draw_landmarks"]
    draw_connections: bool = DEFAULT_CONFIG["draw_connections"]
    body_tracking: bool = DEFAULT_CONFIG["body_tracking"]
    draw_body: bool = DEFAULT_CONFIG["draw_body"]
    pose_model: str = DEFAULT_CONFIG["pose_model"]
    pose_async: bool = DEFAULT_CONFIG["pose_async"]
    glove_port: int = DEFAULT_CONFIG["glove_port"]
    glove_dataset: str = DEFAULT_CONFIG["glove_dataset"]
    glove_dataset_left: str = DEFAULT_CONFIG["glove_dataset_left"]
    glove_dataset_both: str = DEFAULT_CONFIG["glove_dataset_both"]
    glove_auto: bool = DEFAULT_CONFIG["glove_auto"]

    @classmethod
    def load(cls, json_path: Optional[Path] = None) -> "AppConfig":
        cfg = cls()
        if not (json_path and json_path.exists()):
            return cfg
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            log.warning("No se pudo leer %s: %s. Usando defaults.", json_path, e)
            return cfg
        # Antes se hacia setattr de lo que viniera: un config.json que no
        # fuera un objeto tronaba al arrancar, y un valor de otro tipo o fuera
        # de rango (ej. "camera_index": "0", "max_num_hands": 0) fallaba
        # despues, lejos de aqui.
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

        Un valor invalido se ignora (con aviso) y se conserva el actual.
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
                match = [c for c in CONFIG_CHOICES[name] if c.lower() == value.strip().lower()]
                ok = bool(match)
                if ok:
                    value = match[0]

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
    body: Optional[BodyDetection] = None    # None si body_tracking esta apagado o no hay nadie en cuadro
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


# Mano con el guante puesto: el guante es oscuro y se pierde en el video.
# Mientras el guante esta conectado, la zona de esa mano se aclara (antes de
# detectar, para que MediaPipe distinga mejor los dedos, y asi se ve en
# pantalla) y la mano se dibuja en colores claros.
# Aclarado: lo mas claro de la zona (percentil 99) se lleva a blanco y luego
# gamma 0.5, que levanta sobre todo lo oscuro. Nunca oscurece (estirar
# tambien el percentil 1 a negro dejaba NEGRO un guante que fuera lo mas
# oscuro de la zona). Con un guante oscuro simulado (mano al 22-35% de
# brillo) en las 6 fotos de inspeccion_pose/, MediaPipe encontraba la mano en
# 6/12; aclarando, en 8/12 (zona o frame completo).
GLOVE_GAMMA = 0.5
_GLOVE_GAMMA_LUT = np.array([255 * (i / 255) ** GLOVE_GAMMA for i in range(256)], dtype=np.float32)
GLOVE_BOX_MARGIN = 0.35         # la zona crece 35% por lado (la mano se mueve entre frames)
GLOVE_BOX_HOLD_S = 0.6          # si se pierde la mano, se sigue aclarando donde estaba
# Colores de los dedos aclarados (mezcla con blanco) para la mano con guante.
GLOVE_FINGER_COLORS = {
    k: tuple(int(c + (255 - c) * 0.6) for c in v) for k, v in {
        "thumb": (255, 102, 102), "index": (102, 255, 102), "middle": (255, 178, 102),
        "ring": (178, 102, 255), "pinky": (102, 178, 255), "palm": (200, 200, 200),
    }.items()
}


def hand_box(hand: HandDetection, w: int, h: int, margin: float = GLOVE_BOX_MARGIN) -> tuple[int, int, int, int]:
    """Rectangulo (x0, y0, x1, y1) en pixeles alrededor de la mano, con margen."""
    xs = hand.landmarks_2d[:, 0] * w
    ys = hand.landmarks_2d[:, 1] * h
    mx = (xs.max() - xs.min()) * margin + 10
    my = (ys.max() - ys.min()) * margin + 10
    x0, x1 = int(max(0, xs.min() - mx)), int(min(w, xs.max() + mx))
    y0, y1 = int(max(0, ys.min() - my)), int(min(h, ys.max() + my))
    return x0, y0, x1, y1


def _glove_lut(image: np.ndarray) -> np.ndarray:
    """LUT de aclarado para esta imagen: su percentil 99 pasa a 255 y luego
    gamma. El percentil sale de 1 de cada 16 pixeles (rapido en la Raspberry
    Pi)."""
    gray = cv2.cvtColor(np.ascontiguousarray(image[::4, ::4]), cv2.COLOR_BGR2GRAY)
    hi = max(1.0, float(np.percentile(gray, 99)))
    idx = np.clip(np.arange(256) * 255.0 / hi, 0, 255).astype(np.uint8)
    return _GLOVE_GAMMA_LUT[idx].astype(np.uint8)


def _feather_mask(size: int = 64) -> np.ndarray:
    """Elipse con borde difuminado (0-1), se escala al tamano de cada zona."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 2 - 1, size // 2 - 1), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (size // 4 + 1, size // 4 + 1), 0).astype(np.float32) / 255.0


_GLOVE_MASK = _feather_mask()


def brighten_regions(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Aclara las zonas dadas (borde difuminado, sin cortes visibles). Sin
    zonas deja el frame igual: antes se aclaraba todo el frame para encontrar
    la mano con guante, y la pantalla se ponia blanca al conectar el guante."""
    if not boxes:
        return frame
    out = frame.copy()
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 4:
            continue
        region = out[y0:y1, x0:x1]
        alpha = cv2.resize(_GLOVE_MASK, (bw, bh), interpolation=cv2.INTER_LINEAR)
        # Solo lo oscuro (el guante): el fondo claro alrededor casi no cambia.
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = (alpha * np.clip((170.0 - gray) / 110.0, 0.0, 1.0))[..., None]
        light = cv2.LUT(region, _glove_lut(region)).astype(np.float32)
        out[y0:y1, x0:x1] = (region + (light - region) * alpha).astype(np.uint8)
    return out


# Flechas de los sensores del guante sobre la mano: una por sensor, desde el
# nudillo de su dedo (el de la mano, desde la muneca). La direccion es el roll
# del sensor (0 = hacia arriba) y el largo baja con el pitch (a 90° queda corta:
# el sensor apunta hacia la camara o en contra).
GLOVE_SENSOR_ANCHORS = {"pulgar": 2, "indice": 5, "medio": 9, "anular": 13, "menique": 17, "mano": 0}
GLOVE_SENSOR_COLORS = {"pulgar": "thumb", "indice": "index", "medio": "middle",
                       "anular": "ring", "menique": "pinky", "mano": "palm"}


def draw_glove_vectors(image: np.ndarray, reading, anchors_px: Optional[np.ndarray] = None,
                       hand_size_px: float = 60.0) -> None:
    """Dibuja las flechas de los 6 sensores. anchors_px: los 21 puntos de la
    mano en pixeles (si la camara la ve); sin ellos, un recuadro abajo a la
    izquierda con las 6 flechas, para ver el guante aunque la camara no
    encuentre la mano."""
    from guante import SENSOR_NAMES, VALUES_PER_SENSOR
    vals = np.asarray(reading, dtype=float).reshape(len(SENSOR_NAMES), VALUES_PER_SENSOR)
    h, w = image.shape[:2]
    if anchors_px is None:
        box_w, box_h = 220, 90
        x0, y0 = 10, h - box_h - 10
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
        cv2.putText(image, "Guante", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        step = box_w // len(SENSOR_NAMES)
        origins = [(x0 + step // 2 + i * step, y0 + box_h - 18) for i in range(len(SENSOR_NAMES))]
        length = 32.0
    else:
        origins = [tuple(int(v) for v in anchors_px[GLOVE_SENSOR_ANCHORS[n]]) for n in SENSOR_NAMES]
        length = max(20.0, hand_size_px * 0.6)
    for name, (ox, oy), v in zip(SENSOR_NAMES, origins, vals):
        pitch, roll = np.radians(v[6]), np.radians(v[7])
        size = length * (0.3 + 0.7 * abs(np.cos(pitch)))
        tip = (int(ox + size * np.sin(roll)), int(oy - size * np.cos(roll)))
        color = GLOVE_FINGER_COLORS[GLOVE_SENSOR_COLORS[name]]
        cv2.arrowedLine(image, (ox, oy), tip, (30, 30, 30), 4, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(image, (ox, oy), tip, color, 2, cv2.LINE_AA, tipLength=0.3)


# Camara + guante: una sola respuesta, en la que los dos coinciden.
#   - El guante solo opina de las señas que tiene grabadas (vocab) y solo
#     reordena las candidatas de la camara (nunca agrega otra).
#   - Coinciden (la 1.a del guante es la 1.a de la camara, o una de sus
#     candidatas y al combinarlas queda primera): se escribe con la confianza
#     combinada. Asi el guante desempata a la camara (A 50% / B 40% y el guante
#     dice B -> B).
#   - No coinciden (el guante conoce la seña de la camara y dice otra, o la
#     del guante esta entre las candidatas de la camara pero no gana): no se
#     escribe nada, hay que repetir.
#   - El guante no conoce ninguna candidata de la camara (una letra que no se
#     ha grabado con el guante): decide la camara sola.
# Combinacion: p_camara * (n * p_guante) ** GLOVE_FUSION_WEIGHT para las
# señas del guante (n = cuantas tiene; en promedio el factor es 1), con un
# piso para que el guante solo nunca borre una seña.
GLOVE_FUSION_WEIGHT = 0.5
GLOVE_FUSION_FLOOR = 0.02
# Respuesta del guante que todavia vale para las letras fijas (cada frame).
GLOVE_OPINION_FRESH_S = 0.75
# Camara sin respuesta (la sena no se parece a nada) y guante muy seguro: se
# escribe lo del guante (la camara vio algo, pero no lo reconocio).
GLOVE_ALONE_MIN_PROB = 0.8


def fuse_topk(camera: list[tuple[str, float]], glove: Optional[list[tuple[str, float]]],
              vocab: set[str]) -> tuple[list[tuple[str, float]], str]:
    """(top-k combinado, estado). Estado: "solo_camara" (el guante no opina
    de estas candidatas), "coinciden" o "no_coinciden". Si no coinciden, el
    top-k queda empatado entre la de la camara y la del guante, para que
    ninguna regla de escritura lo acepte."""
    if not camera or not glove or not vocab:
        return camera, "solo_camara"
    labels = [l for l, _ in camera]
    ctop, gtop = labels[0], glove[0][0]
    if not any(l in vocab for l in labels):
        return camera, "solo_camara"
    if gtop not in labels and ctop not in vocab:
        return camera, "solo_camara"
    g = dict(glove)
    n = len(vocab)
    scores = {}
    for l, p in camera:
        factor = (n * max(g.get(l, 0.0), GLOVE_FUSION_FLOOR)) ** GLOVE_FUSION_WEIGHT if l in vocab else 1.0
        scores[l] = max(p, GLOVE_FUSION_FLOOR) * factor
    total = sum(scores.values())
    fused = sorted(((l, v / total) for l, v in scores.items()), key=lambda x: -x[1])
    if fused[0][0] == gtop:
        return fused, "coinciden"
    tie = [(ctop, 0.45), (gtop, 0.45)] + [(l, 0.1 / max(1, len(labels) - 1)) for l in labels if l not in (ctop, gtop)]
    return tie[: len(camera) if len(camera) > 1 else 2], "no_coinciden"


def draw_hand_landmarks(
    image: np.ndarray,
    hand: HandDetection,
    draw_connections: bool = True,
    draw_points: bool = True,
    light: bool = False,
) -> None:
    h, w = image.shape[:2]
    pts_px = np.zeros((21, 2), dtype=np.int32)
    for i in range(21):
        pts_px[i, 0] = int(hand.landmarks_2d[i, 0] * w)
        pts_px[i, 1] = int(hand.landmarks_2d[i, 1] * h)

    colors = GLOVE_FINGER_COLORS if light else FINGER_COLORS
    if draw_connections:
        for a, b in HAND_CONNECTIONS:
            if light:   # contorno oscuro para que la linea clara resalte sobre el guante aclarado
                cv2.line(image, tuple(pts_px[a]), tuple(pts_px[b]), (40, 40, 40), 4, cv2.LINE_AA)
            color = colors[LANDMARK_GROUP[b]]
            cv2.line(image, tuple(pts_px[a]), tuple(pts_px[b]), color, 2, cv2.LINE_AA)

    if draw_points:
        outline = (40, 40, 40) if light else (255, 255, 255)
        for i in range(21):
            color = colors[LANDMARK_GROUP[i]]
            radius = 6 if i in (0, 5, 9, 13, 17) else 4
            cv2.circle(image, tuple(pts_px[i]), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(pts_px[i]), radius, outline, 1, cv2.LINE_AA)


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


# Cerca de la camara, MediaPipe SIGUE la mano mientras no la pierda (medido:
# la siguio aun llenando toda la imagen), pero si la pierde ya no la puede
# volver a DETECTAR: el detector inicial no encuentra manos que ocupen ~80% o
# mas de la imagen (medido, deteccion desde cero: 70% 5/5, 80% 2/5, 90% 0/5).
# Basta un movimiento rapido o la camara desenfocada de cerca para perderla, y
# en el modo dinamico eso corta la sena a la mitad. Se avisa desde antes.
HAND_TOO_CLOSE_FRAC = 0.6
# Si la mano desaparece poco despues de haber estado muy cerca, se avisa que
# se perdio por eso (no se sabe donde quedo, asi que el aviso caduca).
HAND_LOST_WARNING_S = 3.0


def hand_size_fraction(hand: HandDetection, frame_w: int, frame_h: int) -> float:
    """Lado mayor del recuadro de la mano entre el lado mayor de la imagen
    (el detector de MediaPipe trabaja sobre la imagen hecha cuadrada)."""
    pts = hand.landmarks_2d[:, :2] * np.array([frame_w, frame_h], dtype=np.float32)
    return float(np.ptp(pts, axis=0).max()) / max(frame_w, frame_h)


class HandDistanceWarning:
    """Decide que aviso mostrar, frame a frame: mano demasiado cerca, o mano
    perdida justo despues de haber estado demasiado cerca."""

    def __init__(self) -> None:
        self._last_close = float("-inf")

    def update(self, hands, frame_w: int, frame_h: int, now: float) -> Optional[str]:
        hands = list(hands)
        if any(hand_size_fraction(h, frame_w, frame_h) >= HAND_TOO_CLOSE_FRAC for h in hands):
            self._last_close = now
            return "Mano muy cerca de la camara: alejala un poco"
        if not hands and now - self._last_close < HAND_LOST_WARNING_S:
            return "Se perdio la mano por estar muy cerca: alejala"
        return None


def draw_warning(image: np.ndarray, text: str) -> None:
    """Aviso en naranja en la parte de abajo del video."""
    y = image.shape[0] - 15
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)


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

    def stop(self, timeout_ms: int = 2000) -> bool:
        """Pide terminar y espera. False si sigue corriendo (p. ej. una camara
        trabada en cap.read()): ver SignLanguageApp._retire_thread."""
        self._run_flag = False
        return self.wait(timeout_ms)


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


class AsyncBodyTracker:
    """BodyTracker en su propio hilo. detect() le deja el frame y devuelve
    sin esperar el ultimo cuerpo que ya termino (de uno o dos frames atras),
    asi la pose corre en paralelo a las manos en otro nucleo: MediaPipe
    suelta el GIL mientras infiere. Medido en la laptop: manos + pose en serie
    16.4 ms por frame; con la pose aparte, lo de las manos (~8 ms). BodyTracker
    sigue usando su reloj real (ver su docstring)."""

    def __init__(self, tracker: BodyTracker):
        self._tracker = tracker
        self._cond = threading.Condition()
        self._pending = None
        self._latest: Optional[BodyDetection] = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="pose", daemon=True)
        self._thread.start()

    def detect(self, mp_image) -> Optional[BodyDetection]:
        with self._cond:
            self._pending = mp_image      # si habia uno sin procesar, se reemplaza
            self._cond.notify()
            return self._latest

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                image, self._pending = self._pending, None
            try:
                body = self._tracker.detect(image)
            except Exception:
                log.exception("MediaPipe Pose falló")
                body = None
            with self._cond:
                self._latest = body

    def close(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self._thread.join(timeout=2.0)
        self._tracker.close()


class HandTrackingThread(QThread):
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    hands_detected_signal = pyqtSignal(object)        
    sign_detected_signal = pyqtSignal(str, float)    
    sign_diagnostic_signal = pyqtSignal(object)       
    letter_committed_signal = pyqtSignal(str)
    space_committed_signal = pyqtSignal()
    auto_result_signal = pyqtSignal(object)           # modo automatico: {"topk", "detail"}
    letters_retracted_signal = pyqtSignal(object)     # letras (list[str]) que una palabra reemplaza
    phase_signal = pyqtSignal(str)                    # "reposo" | "seña" | "clasificando"
    guidance_signal = pyqtSignal(object)              # estado para consejos en vivo (~4 por segundo)
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
        # La GUI lo pide al mover el slider de deteccion; lo atiende run() en
        # este mismo hilo (el HandLandmarker no se debe tocar desde otro).
        self._reinit_hands_requested = False
        self._body_tracker: Optional[BodyTracker] = None
        self._distance_warning = HandDistanceWarning()

        self._keypoint_buffer: deque[np.ndarray] = deque(
            maxlen=config.keypoint_buffer_size
        )

        self._frames_without_hand = 0
        self._space_already_committed = False
        # Ultima letra confirmada: no se vuelve a confirmar hasta que la sena
        # se interrumpe repeat_release_frames frames (_release_frames).
        self._last_committed_label: Optional[str] = None
        self._release_frames = 0
        # Si la palabra en curso tiene letras (para el espacio automatico).
        # Antes se usaba _last_committed_label para esto, y por eso liberar la
        # letra para repetirla (LL, RR) hubiera impedido cerrar la palabra.
        self._word_has_letters = False

        self._classifier: Optional[SignClassifier] = SignClassifier.try_load()
        self._per_letter_confidence: dict[str, float] = {}
        self._smoother = PredictionSmoother(
            window_size=max(5, config.smoothing_window),
            min_confidence=config.min_letter_confidence,
            min_margin=config.min_letter_margin,
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

        # Modo automatico (ver AUTO_* al inicio del archivo): el que usa la
        # ventana. Tiene su propio corte de actividades y su propia cola, y
        # no toca el estado del modo dinamico de solo letras (set_dynamic_mode),
        # que se conserva para probar_modo_dinamico_senas.py.
        self._word_recognizers = _get_word_recognizers()
        self._activity_segmenter: Optional["AutoSegmenter"] = self._new_activity_segmenter()
        self._auto_mode: bool = self._activity_segmenter is not None and (
            self._dtw_recognizer is not None or self._word_recognizers is not None
        )
        # Letras estaticas fijadas en la actividad en curso (ver AUTO_MAX_RETRACTED_LETTERS).
        self._activity_letters: list[str] = []
        self._auto_result_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._auto_classifying = False
        self._auto_classify_start = 0.0
        self._auto_idle_text = "Esperando mano"
        self._wrist_track: deque[tuple[float, float, float, float]] = deque(maxlen=AUTO_SPEED_WINDOW)
        self._activity_start = 0.0
        self._phase = ""
        self._last_guidance = 0.0
        self._last_static_topk: list[tuple[str, float]] = []
        self._last_distance_warning = None
        # Guante conectado (lo pone la ventana): aclarar y dibujar en claro la
        # mano que deletrea. _glove_box: donde estaba esa mano y cuando.
        self._glove_on_hand = False
        self._glove_box: Optional[tuple[int, int, int, int]] = None
        # Guante conectado: receptor (flechas de los sensores), señas que
        # conoce (con los nombres de la camara) y sus respuestas recientes
        # (hora perf_counter, top-k) para combinarlas con la camara.
        self._glove_receiver = None
        self._glove_vocab: set[str] = set()
        self._glove_history: deque = deque(maxlen=64)
        self._last_glove_status = "solo_camara"
        self._last_glove_conflict: Optional[tuple[str, str]] = None
        self._pending_activity_glove: Optional[list[tuple[str, float]]] = None
        self._glove_box_t = 0.0

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

    def set_glove(self, receiver, vocab: set[str]) -> None:
        """Guante conectado (receptor y señas que conoce) o None si no."""
        self._glove_receiver = receiver
        self._glove_vocab = set(vocab) if receiver is not None else set()
        self._glove_on_hand = receiver is not None
        if receiver is None:
            self._glove_box = None
            self._glove_history.clear()

    def set_glove_opinion(self, topk: Optional[list[tuple[str, float]]]) -> None:
        """Ultima respuesta del guante (la ventana la manda ~4 veces por
        segundo); None si no reconocio nada."""
        if topk:
            self._glove_history.append((time.perf_counter(), list(topk)))

    def _fresh_glove_opinion(self, now: float) -> Optional[list[tuple[str, float]]]:
        if self._glove_history and now - self._glove_history[-1][0] <= GLOVE_OPINION_FRESH_S:
            return self._glove_history[-1][1]
        return None

    def _glove_opinion_since(self, t0: float) -> Optional[list[tuple[str, float]]]:
        """Promedio de las respuestas del guante desde t0 (toda la sena)."""
        sums: dict[str, float] = {}
        n = 0
        for t, topk in self._glove_history:
            if t >= t0:
                n += 1
                for label, p in topk:
                    sums[label] = sums.get(label, 0.0) + p
        if not n:
            return None
        return sorted(((l, v / n) for l, v in sums.items()), key=lambda x: -x[1])[:3]

    def set_draw_body(self, value: bool) -> None:
        self._cfg.draw_body = value

    def reset_word_state(self, has_letters: bool = False) -> None:
        """Palabra nueva (has_letters=False) o corregida con Retroceso
        (has_letters=True si le quedan letras: el espacio automatico debe
        seguir cerrandola). En ambos casos la ultima letra se puede volver a
        signar de inmediato."""
        self._frames_without_hand = 0
        self._space_already_committed = False
        self._last_committed_label = None
        self._release_frames = 0
        self._word_has_letters = has_letters
        self._stable_letter = None
        self._stable_frames = 0
        self._smoother.reset()
        self._reset_dynamic_state()

    def request_hands_reinit(self) -> None:
        """Pide recrear el HandLandmarker con los umbrales actuales de la
        config (lo hace run(), en este hilo, antes del siguiente frame)."""
        self._reinit_hands_requested = True

    def _commit_letter(self, letter: str) -> None:
        self.letter_committed_signal.emit(letter)
        self._last_committed_label = letter
        self._release_frames = 0
        self._word_has_letters = True

    def _update_release(self, letter: Optional[str]) -> None:
        """Cuenta los frames en que la sena NO es la ultima letra confirmada
        (otra letra, ninguna o sin mano). Al llegar a repeat_release_frames,
        esa letra se puede volver a confirmar: asi se escriben LL, RR, EE."""
        if self._last_committed_label is None:
            return
        if letter == self._last_committed_label:
            self._release_frames = 0
            return
        self._release_frames += 1
        if self._release_frames >= self._cfg.repeat_release_frames:
            released = self._last_committed_label
            self._last_committed_label = None
            self._release_frames = 0
            # La repeticion debe sostenerse otros stable_frames_to_commit
            # frames: si la mano solo se ausento, el conteo de esa misma letra
            # seguia alto y se hubiera repetido en cuanto volviera. (Si ya se
            # esta haciendo otra letra, su conteo no se toca.)
            if self._stable_letter == released:
                self._stable_letter = None
                self._stable_frames = 0

    def _select_hand(self, detections: FrameDetections) -> Optional[HandDetection]:
        """Mano que deletrea para el alfabeto estatico. Con una sola, esa. Con
        varias, la de la mano dominante (traducida a la etiqueta de MediaPipe,
        ver MEDIAPIPE_LABEL_OF_HAND). Antes se elegia la etiqueta "Right", que
        en este programa es la mano IZQUIERDA de la persona: con las dos manos
        en cuadro se clasificaba la que no hacia la sena."""
        if not detections.hands:
            return None
        if len(detections.hands) == 1:
            return detections.hands[0]
        label = MEDIAPIPE_LABEL_OF_HAND[self._cfg.dominant_hand]
        dominant = [h for h in detections.hands if h.handedness == label]
        return max(dominant or detections.hands, key=lambda h: h.confidence)

    def _classify_static(self, hand: HandDetection) -> list[tuple[str, float]]:
        landmarks_2d = hand.landmarks_2d
        if self._cfg.dominant_hand == "Left":
            # Reflejo horizontal: la mano izquierda se ve como una derecha,
            # que es la que espera el modelo. La z no cambia con el reflejo.
            landmarks_2d = landmarks_2d.copy()
            landmarks_2d[:, 0] = 1.0 - landmarks_2d[:, 0]
        return self._classifier.predict_topk_from_hand(landmarks_2d, hand.landmarks_3d, k=3)


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
        # Los modelos se cierran aqui, al salir, en ESTE hilo. Antes los
        # cerraba stop() desde el hilo de la GUI aunque el hilo siguiera
        # corriendo (si no terminaba en 3 s), con MediaPipe todavia en uso.
        try:
            if self._init_mediapipe():
                self._loop()
        finally:
            self._close_models()

    def _close_models(self) -> None:
        for solution in (self._hands_solution, self._body_tracker):
            if solution is not None:
                try:
                    solution.close()
                except Exception:
                    pass
        self._hands_solution = None
        self._body_tracker = None

    def _loop(self) -> None:
        timestamp_ms = 0

        while self._run_flag:
            if self._reinit_hands_requested:
                self._reinit_hands_requested = False
                try:
                    new_solution = self._create_hand_landmarker()
                except Exception:
                    log.exception("No se pudo aplicar el nuevo umbral de deteccion")
                else:
                    old, self._hands_solution = self._hands_solution, new_solution
                    try:
                        old.close()
                    except Exception:
                        pass
                    log.info(
                        "Umbral de deteccion de manos aplicado: %.2f",
                        self._cfg.min_detection_confidence,
                    )

            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                self.heartbeat_signal.emit()
                continue

            t0 = time.perf_counter()

            if self._glove_on_hand:
                recent = self._glove_box is not None and time.monotonic() - self._glove_box_t < GLOVE_BOX_HOLD_S
                frame = brighten_regions(frame, [self._glove_box] if recent else [])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 1
                results = self._hands_solution.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                log.exception("MediaPipe falló: %s", e)
                continue

            detections = self._parse_results(results)
            gloved = self._select_hand(detections) if self._glove_on_hand else None
            if gloved is not None:
                self._glove_box = hand_box(gloved, frame.shape[1], frame.shape[0])
                self._glove_box_t = time.monotonic()
            if self._body_tracker is not None:
                # Sin timestamp_ms a proposito: BodyTracker usa tiempo real
                # (con el +1 de las manos la pose se retrasa, ver body_tracker.py).
                try:
                    detections.body = self._body_tracker.detect(mp_image)
                except Exception as e:
                    log.exception("MediaPipe Pose falló: %s", e)
            annotated = self._render(frame, detections)

            self._update_keypoint_buffer(detections)
            frame_h, frame_w = frame.shape[:2]
            auto = self._auto_mode and not self._dynamic_mode
            # En modo automatico, manos bajo la linea de reposo cuentan como
            # "sin mano" para el espacio automatico (bajar las manos sin
            # sacarlas de cuadro cierra la palabra). None = sin hombros.
            in_space = (
                hands_in_signing_space(detections.hands, detections.body, frame_w, frame_h)
                if auto else None
            )
            self._update_word_state(detections, resting=in_space is False)

            if self._dynamic_mode:
                sign_text, sign_conf = self._process_dynamic_frame(detections)
            elif auto:
                sign_text, sign_conf = self._process_auto_frame(detections, frame_w, frame_h, in_space)
            else:
                sign_text, sign_conf, _ = self._process_static_frame(detections)

            self.change_pixmap_signal.emit(annotated)
            self.hands_detected_signal.emit(detections)
            self.sign_detected_signal.emit(sign_text, sign_conf)
            self.heartbeat_signal.emit()

            dt = time.perf_counter() - t0
            self._update_metrics(dt, detections.num_hands)


    # ---- alfabeto estatico ---------------------------------------------------

    def _process_static_frame(
        self, detections: FrameDetections, allow_commit: bool = True
    ) -> tuple[str, float, bool]:
        """Alfabeto estatico: clasifica la mano del frame, suaviza y fija la
        letra tras stable_frames_to_commit frames estables. Devuelve (texto
        de Estado, confianza, si se fijo una letra en este frame).

        allow_commit=False (modo automatico con la mano en movimiento o en
        reposo) sigue mostrando la letra, pero no cuenta frames estables: la
        cuenta empieza de nuevo cuando la mano se queda quieta."""
        sign_text, sign_conf, committed = "—", 0.0, False
        if self._classifier is not None and detections.num_hands > 0:
            hand = self._select_hand(detections)
            try:
                topk = self._classify_static(hand)
                if self._glove_vocab:
                    glove = self._fresh_glove_opinion(time.perf_counter())
                    camera_top = topk[0][0] if topk else None
                    topk, self._last_glove_status = fuse_topk(topk, glove, self._glove_vocab)
                    self._last_glove_conflict = (
                        (camera_top, glove[0][0]) if self._last_glove_status == "no_coinciden" else None)
                self._last_static_topk = topk
                smoothed = self._smoother.push(topk)
                self._update_release(smoothed.letter)

                if self._diagnostic_mode:
                    self.sign_diagnostic_signal.emit(topk)

                if smoothed.letter is not None:
                    sign_text = smoothed.letter
                    sign_conf = smoothed.confidence
                    if not allow_commit:
                        self._stable_letter = None
                        self._stable_frames = 0
                    elif smoothed.letter == self._stable_letter:
                        self._stable_frames += 1
                    else:
                        self._stable_letter = smoothed.letter
                        self._stable_frames = 1

                    if (
                        allow_commit
                        and self._stable_frames >= self._cfg.stable_frames_to_commit
                        and smoothed.letter != self._last_committed_label
                    ):
                        self._commit_letter(smoothed.letter)
                        committed = True
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
        elif self._classifier is not None:
            # Sin mano tambien cuenta como "interrumpir la sena" para
            # poder repetir la letra (bajar la mano un instante).
            self._update_release(None)
        return sign_text, sign_conf, committed

    # ---- modo automatico (estatico + dinamico + palabras) ---------------------

    def _new_activity_segmenter(self) -> Optional["AutoSegmenter"]:
        """Corte de actividades del modo automatico: el del modo palabras de
        segmentador_automatico.py (linea de reposo, PALABRAS_*)."""
        if AutoSegmenter is None:
            return None
        return AutoSegmenter(
            no_hand_ms_to_end=PALABRAS_REST_MS_TO_END,
            min_sequence_ms=PALABRAS_MIN_SEQUENCE_MS,
            max_duration_ms=PALABRAS_MAX_SEQUENCE_MS,
        )

    def set_auto_mode(self, enabled: bool) -> None:
        self._auto_mode = bool(enabled) and self._activity_segmenter is not None
        self._activity_segmenter = self._new_activity_segmenter()
        self._activity_letters = []
        self._wrist_track.clear()

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def word_labels(self) -> list[str]:
        if self._word_recognizers is None:
            return []
        return [word_display(label) for label in self._word_recognizers[1].labels]

    @staticmethod
    def _raised_hands(hands: dict, body: Optional[BodyDetection], frame_w: int, frame_h: int) -> int:
        """Cuantas manos tienen la muneca sobre la linea de reposo (sin
        hombros en cuadro: cuantas manos hay)."""
        line_y = rest_line_y(body, frame_w, frame_h)
        if line_y is None:
            return len(hands)
        return sum(1 for h in hands.values() if h.landmarks_2d[0, 1] * frame_h < line_y)

    def _hand_is_still(
        self, hand: Optional[HandDetection], frame_w: int, frame_h: int, now: float
    ) -> bool:
        """True si la muneca de `hand` casi no se movio en los ultimos
        AUTO_SPEED_WINDOW frames seguidos con mano: velocidad entre el
        promedio de los 3 primeros y el de los 3 ultimos, en tamanos de mano
        por segundo, <= AUTO_STATIC_MAX_SPEED. Comparar promedios, y no
        sumar el recorrido frame a frame, ignora el temblor de MediaPipe."""
        if hand is None:
            self._wrist_track.clear()
            return False
        px = hand.landmarks_2d[:, :2] * np.array([frame_w, frame_h], dtype=np.float32)
        self._wrist_track.append((now, float(px[0, 0]), float(px[0, 1]), float(np.linalg.norm(px[9] - px[0]))))
        if len(self._wrist_track) < AUTO_SPEED_WINDOW:
            return False
        track = np.array(self._wrist_track, dtype=np.float64)
        first, last = track[:3].mean(axis=0), track[-3:].mean(axis=0)
        elapsed = last[0] - first[0]
        hand_size = float(np.median(track[:, 3]))
        if elapsed <= 0 or hand_size < 1.0:
            return False
        speed = float(np.linalg.norm(last[1:3] - first[1:3])) / hand_size / elapsed
        return speed <= AUTO_STATIC_MAX_SPEED

    def _process_auto_frame(
        self,
        detections: FrameDetections,
        frame_w: int,
        frame_h: int,
        in_space: Optional[bool],
    ) -> tuple[str, float]:
        """Un frame del modo automatico (ver el comentario de AUTO_* al inicio
        del archivo). in_space: hands_in_signing_space de este frame (None si
        no se ven los hombros)."""
        assert self._activity_segmenter is not None
        now = time.perf_counter()

        # 1. Resultado de una clasificacion DTW que termino en su hilo.
        try:
            result = self._auto_result_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._auto_classifying = False
            self._apply_auto_result(result)

        # 2. Letras estaticas: solo con UNA mano arriba, quieta.
        hands_by_side: dict[str, HandDetection] = {}
        for h in detections.hands:
            if h.handedness not in hands_by_side:
                hands_by_side[h.handedness] = h
        raised = self._raised_hands(hands_by_side, detections.body, frame_w, frame_h)
        hand = self._select_hand(detections) if detections.num_hands > 0 else None
        still = self._hand_is_still(hand, frame_w, frame_h, now)
        sign_text, sign_conf, committed = self._process_static_frame(
            detections, allow_commit=still and in_space is not False and raised <= 1
        )

        # 3. Actividad (subir la mano, senar, bajarla): letras dinamicas y palabras.
        active = in_space if in_space is not None else bool(hands_by_side)
        vector = np.concatenate([
            build_dynamic_feature_vector(hands_by_side),
            body_location_features(hands_by_side, detections.body, frame_w, frame_h),
        ])
        event = self._activity_segmenter.push(active, vector, now)
        if event is not None and event[0] == "inicio":
            self._activity_letters = []
            self._activity_start = now
        if committed:
            self._activity_letters.append(self._last_committed_label)
        if event is not None and event[0] != "inicio":
            kind, sequence = event
            letters, self._activity_letters = self._activity_letters, []
            # El segmentador corta PALABRAS_REST_MS_TO_END despues del ultimo
            # frame activo; eso no es parte de la sena.
            duration_s = max(0.0, now - self._activity_start - PALABRAS_REST_MS_TO_END / 1000.0)
            if kind == "fin_valida":
                self._pending_activity_glove = (
                    self._glove_opinion_since(self._activity_start) if self._glove_vocab else None)
                self._start_auto_classification(sequence, letters, now, duration_s)
            elif len(sequence) >= 6 and not letters:
                self.auto_result_signal.emit({"kind": "corta", "code": "corta", "topk": [],
                                              "detail": "seña muy corta"})

        # 4. Fase y consejos para la interfaz.
        if self._auto_classifying:
            self._set_phase("clasificando")
        else:
            self._set_phase("seña" if self._activity_segmenter.state == "grabando" else "reposo")
        if now - self._last_guidance >= 0.25:
            self._last_guidance = now
            unsure = None
            if sign_text.startswith("?") and len(self._last_static_topk) >= 2:
                unsure = self._last_static_topk[:2]
            self.guidance_signal.emit({
                "hands": detections.num_hands,
                "raised": raised,
                "body_tracking": self._body_tracker is not None,
                "body_visible": rest_line_y(detections.body, frame_w, frame_h) is not None,
                "too_close": self._last_distance_warning is not None,
                "two_raised_still": raised >= 2 and still,
                "static_unsure": unsure if still else None,
                "moving_letter": (not still and raised == 1 and detections.num_hands > 0
                                  and len(sign_text) == 1 and sign_text.isalpha()),
                "glove_conflict": self._last_glove_conflict if detections.num_hands else None,
            })

        # 5. Texto de Estado.
        if self._auto_classifying:
            return (f"Clasificando... ({now - self._auto_classify_start:.0f}s)", 0.0)
        if detections.num_hands > 0 and sign_text != "—":
            return (sign_text, sign_conf)
        return (self._auto_idle_text, 0.0)

    def _set_phase(self, phase: str) -> None:
        if phase != self._phase:
            self._phase = phase
            self.phase_signal.emit(phase)

    def _start_auto_classification(
        self, sequence: list[np.ndarray], letters: list[str], now: float, duration_s: float = 0.0
    ) -> None:
        if self._auto_classifying:
            # Igual que el modo dinamico: no se encolan dos clasificaciones.
            log.warning("[auto] seña descartada: la clasificacion anterior aun no termina (%d frames)", len(sequence))
            self._auto_idle_text = "Seña descartada (clasificando la anterior)"
            return
        if self._dtw_recognizer is None and self._word_recognizers is None:
            return
        self._auto_classifying = True
        self._auto_classify_start = now
        threading.Thread(
            target=self._classify_auto_sequence, args=(sequence, letters, duration_s), daemon=True
        ).start()

    def _classify_auto_sequence(
        self, sequence: list[np.ndarray], letters: Optional[list[str]] = None, duration_s: float = 0.0
    ) -> None:
        """En un hilo aparte (como _classify_dynamic_sequence): decide si la
        actividad fue una letra dinamica o una palabra y deja en la cola un
        dict (kind, topk, best_dist, d_word, d_letter, letters, stats,
        duration_s), o None si fallo."""
        result: Optional[dict] = None
        try:
            seq = np.asarray(sequence, dtype=np.float64)
            if self._cfg.dominant_hand == "Left":
                # Plantillas de mano derecha (ver _classify_dynamic_sequence);
                # tambien refleja el bloque de cuerpo.
                seq = mirror_and_swap_hands(seq)
            hands = seq[:, :126]
            # Letras: consulta a ~LETTER_DTW_FPS, como sus plantillas. En una
            # laptop a 30 fps se toma 1 de cada 2 frames; en la Pi, que
            # procesa ~15 fps, la secuencia completa.
            fps = len(hands) / duration_s if duration_s > 0.2 else TEMPLATE_FPS
            letter_step = max(1, int(round(fps / LETTER_DTW_FPS)))
            letter_query = hands[::letter_step] if len(hands) >= 2 * letter_step else hands
            letter_dist = self._dtw_recognizer.compute_distances(letter_query) if self._dtw_recognizer else {}
            best_letter = min(letter_dist.values(), default=float("inf"))
            best_word = float("inf")
            hands_dist: dict[str, float] = {}
            if self._word_recognizers is not None:
                hands_only, with_body = self._word_recognizers
                hands_dist = hands_only.compute_distances(hands)
                best_word = min(hands_dist.values(), default=float("inf"))
            body_frac = float(np.mean(seq[:, 134] > 0)) if len(seq) else 0.0
            base = {"d_word": best_word, "d_letter": best_letter, "letters": list(letters or []),
                    "stats": sign_stats(seq, duration_s), "duration_s": duration_s, "body_frac": body_frac}
            if best_word < AUTO_WORD_PREFERENCE * best_letter:
                if body_frac >= WORD_MIN_BODY_FRACTION:
                    word_dist = with_body.compute_distances(fill_missing_body(seq))
                    topk, note = hola_mama_rule(distances_to_topk(word_dist, 3, WORD_TEMPERATURE), base["stats"])
                else:
                    word_dist = hands_dist
                    topk, note = distances_to_topk(word_dist, 3, WORD_TEMPERATURE), "sin cuerpo: solo manos"
                result = dict(base, kind="palabra", topk=topk, best_dist=min(word_dist.values()), note=note)
            elif letter_dist:
                result = dict(base, kind="letra", topk=distances_to_topk(letter_dist, 3), best_dist=best_letter)
        except Exception as e:
            log.exception("Error en la clasificacion automatica: %s", e)
        self._auto_result_queue.put(result)

    def _apply_auto_result(self, result: Optional[dict]) -> None:
        """Escribe (o no) la letra o palabra de una actividad y avisa a la
        interfaz con auto_result_signal: {kind, code, label, topk, detail,
        stats, letters, too_long}. code: "ok", "ambigua", "desconocida",
        "deletreo_largo", "deletreo", "letra_dudosa" o "error"."""
        if result is None:
            self._auto_idle_text = "Error al clasificar la seña (ver consola)"
            self.auto_result_signal.emit({"kind": "error", "code": "error", "topk": [], "detail": "error"})
            return
        kind, topk, letters = result["kind"], result["topk"], result["letters"]
        # Camara + guante (ver fuse_topk): la respuesta del guante durante
        # toda la sena reordena las candidatas de la camara.
        glove, self._pending_activity_glove = self._pending_activity_glove, None
        camera_top = topk[0][0] if topk else None
        topk, glove_status = fuse_topk(topk, glove, self._glove_vocab)
        result = dict(result, topk=topk)
        label, conf = topk[0]
        too_long = result["duration_s"] >= PALABRAS_MAX_SEQUENCE_MS / 1000.0 - 0.5
        if kind == "palabra":
            committed, margin, reason = word_commit_decision(topk, result["best_dist"])
            code = "ok" if committed else ("desconocida" if result["best_dist"] > WORD_MAX_DISTANCE else "ambigua")
            shown = [(word_display(w), c) for w, c in topk]
            if committed and len(letters) > AUTO_MAX_RETRACTED_LETTERS:
                committed, code = False, "deletreo_largo"
                reason = f"se fijaron {len(letters)} letras: fue deletreo"
            elif (committed and letters and result.get("body_frac", 1.0) < WORD_MIN_BODY_FRACTION
                  and result["best_dist"] > WORD_NO_BODY_REPLACE_MAX):
                # Sin la ubicacion no hay como distinguirla de la letra sostenida.
                committed, code = False, "deletreo"
                reason = f"sin cuerpo visible, se conserva {''.join(letters)}"
            if committed:
                if letters:
                    self.letters_retracted_signal.emit(letters)
                self._commit_word(label)
                self._auto_idle_text = f"{word_display(label)} (palabra, {conf * 100:.0f}%)"
                detail = f"palabra agregada (margen {margin * 100:.0f}pp)"
                if result.get("note"):
                    detail += f", {result['note']}"
                if letters:
                    detail += f", reemplaza {''.join(letters)}"
            else:
                self._auto_idle_text = f"¿{word_display(label)}? - no agregada"
                detail = f"palabra no agregada: {reason}"
            label = word_display(label)
        elif letters and not (
            len(letters) <= AUTO_MAX_RETRACTED_LETTERS
            and result["best_dist"] <= DYN_REPLACE_MAX_DISTANCE
            and dynamic_commit_decision(topk)[0]
        ):
            # Las letras fijas se quedan: la letra con movimiento no se
            # distingue de ellas sostenidas (ver DYN_REPLACE_MAX_DISTANCE).
            committed, shown, code = False, topk, "deletreo"
            detail = (f"deletreo ({''.join(letters)}): se conservan las letras "
                      f"({label} a distancia {result['best_dist']:.2f})")
        else:
            committed, margin, rule = dynamic_commit_decision(topk)
            shown = topk
            code = "ok" if committed else "letra_dudosa"
            if committed:
                if letters:
                    # La letra de partida (I de la J, N de la Ñ...) la reemplaza.
                    self.letters_retracted_signal.emit(letters)
                self._commit_letter(label)
                self._auto_idle_text = f"{label} ({conf * 100:.0f}%)"
                detail = f"letra agregada (regla {rule}, margen {margin * 100:.0f}pp)"
                if letters:
                    detail += f", reemplaza {''.join(letters)}"
            else:
                self._auto_idle_text = f"¿{label}? ({conf * 100:.0f}%) - no agregada"
                detail = f"letra no agregada (margen {margin * 100:.0f}pp)"
        # La camara no reconocio la sena (no se parece a nada) pero el guante
        # si, y muy seguro: la camara vio algo que no sabe leer (p. ej. el
        # guante oscuro); se escribe lo del guante.
        if (code == "desconocida" and glove and glove[0][1] >= GLOVE_ALONE_MIN_PROB
                and glove[0][0] in self._glove_vocab and len(letters) <= AUTO_MAX_RETRACTED_LETTERS):
            g_label = glove[0][0]
            if letters:
                self.letters_retracted_signal.emit(letters)
            if len(g_label) > 1:
                self._commit_word(g_label)
                kind = "palabra"
            else:
                self._commit_letter(g_label)
                kind = "letra"
            label, code, glove_status = word_display(g_label), "ok", "guante_solo"
            shown = [(label, glove[0][1])]
            detail = f"la cámara no la reconoció; se escribe {label} del guante ({glove[0][1] * 100:.0f}%)"
        if glove:
            detail += f" | guante: {glove[0][0]} {glove[0][1] * 100:.0f}% ({glove_status})"
        log.info(
            "[auto] %s: %s | distancia solo manos: palabra %.2f, letra %.2f -> %s",
            kind, "  ".join(f"{w} {c * 100:.1f}%" for w, c in shown), result["d_word"], result["d_letter"], detail,
        )
        self.auto_result_signal.emit({
            "kind": kind, "code": code, "label": label, "topk": shown, "detail": detail,
            "stats": result["stats"], "letters": letters, "too_long": too_long,
            "glove_status": glove_status, "camera_top": camera_top,
            "glove_top": glove[0][0] if glove else None,
        })

    def _commit_word(self, label: str) -> None:
        """Escribe la palabra y la cierra, como si se hubieran bajado las
        manos el tiempo del espacio automatico."""
        self._commit_letter(word_display(label))
        self.space_committed_signal.emit()
        self._space_already_committed = True
        self._word_has_letters = False
        self._last_committed_label = None

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
            result = self._dynamic_result_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._dynamic_classifying = False
            if result is None:
                # La clasificacion fallo (ver _classify_dynamic_sequence):
                # antes no llegaba nada y _dynamic_classifying se quedaba en
                # True para siempre, descartando todas las senas siguientes.
                self._dynamic_idle_text = "Error al clasificar la seña (ver consola)"
                return (self._dynamic_idle_text, 0.0)
            letter, conf, topk = result
            self.sign_diagnostic_signal.emit(topk)
            should_commit, margin, rule = dynamic_commit_decision(topk)
            top_str = "  ".join(f"{w} {c * 100:.1f}%" for w, c in topk)

            if rule == "experimental":
                # K, Q o Z comprometida por la regla EXPERIMENTAL de margen
                # (ver dynamic_commit_decision): se etiqueta distinto para que
                # quede claro que no vino de la regla normal de confianza.
                self._commit_letter(letter)
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
                self._commit_letter(letter)
                self._dynamic_idle_text = f"{letter} {conf * 100:.1f}% (margen alto, par Ñ/Q)"
                log.info(
                    "[dinamico] top-3: %s -> agregada por regla EXPERIMENTAL reforzada, par Ñ/Q (margen=%.1fpp >= %.0fpp)",
                    top_str, margin * 100, DYN_NQ_PAIR_MIN_MARGIN * 100,
                )
            elif rule == "margen":
                # Grupo normal comprometido por margen amplio aunque la
                # confianza absoluta no llegara a DYN_MIN_CONF.
                self._commit_letter(letter)
                self._dynamic_idle_text = f"{letter} {conf * 100:.1f}% (margen amplio)"
                log.info(
                    "[dinamico] top-3: %s -> agregada por margen amplio (margen=%.1fpp >= %.0fpp)",
                    top_str, margin * 100, DYN_NORMAL_MIN_MARGIN * 100,
                )
            elif rule == "confianza":
                self._commit_letter(letter)
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
            if self._cfg.dominant_hand == "Left":
                # Las plantillas son de la mano derecha (585 de 586, ver
                # MEDIAPIPE_LABEL_OF_HAND): reflejar la sena e intercambiar
                # los slots de mano la vuelve comparable con ellas.
                sequence = list(mirror_and_swap_hands(np.asarray(sequence, dtype=np.float64)))
            topk = self._dtw_recognizer.predict_topk(sequence, k=3)
        except Exception as e:
            log.exception("Error en DTWRecognizer: %s", e)
            # Sin esto _dynamic_classifying se quedaba en True para siempre.
            self._dynamic_result_queue.put(None)
            return
        letter, conf = topk[0]
        self._dynamic_result_queue.put((letter, conf, topk))


    def _create_hand_landmarker(self):
        # Asegurar que el modelo esté descargado.
        model_path = ensure_hand_model(Path.home() / ".sign_translator" / "models")

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
        return HandLandmarker.create_from_options(options)

    def _init_mediapipe(self) -> bool:
        try:
            log.info("Inicializando MediaPipe Hand Landmarker (Tasks API)...")
            models_dir = Path.home() / ".sign_translator" / "models"
            self._hands_solution = self._create_hand_landmarker()
            log.info("MediaPipe Hand Landmarker listo.")

            # El cuerpo es opcional: si el modelo de pose no se puede cargar
            # (p. ej. sin internet la primera vez), las manos y el alfabeto
            # siguen funcionando igual, solo sin esqueleto.
            if self._cfg.body_tracking:
                try:
                    tracker = BodyTracker(self._cfg.pose_model, models_dir)
                    self._body_tracker = AsyncBodyTracker(tracker) if self._cfg.pose_async else tracker
                    log.info("MediaPipe Pose Landmarker (%s%s) listo.", self._cfg.pose_model,
                             ", en paralelo" if self._cfg.pose_async else "")
                except Exception:
                    log.exception("No se pudo iniciar MediaPipe Pose; se sigue sin esqueleto del cuerpo")
                    self._body_tracker = None

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

    def _render(self, frame: np.ndarray, detections: FrameDetections) -> np.ndarray:
        out = frame.copy()

        # Antes que las manos (y antes del return de "sin manos"): el
        # esqueleto se ve aunque no haya ninguna mano en cuadro.
        if detections.body is not None and self._cfg.draw_body:
            draw_body_skeleton(out, detections.body, detections.hands)
            if self._auto_mode and not self._dynamic_mode:
                # Arriba de la linea empieza la sena; bajar las manos la termina.
                draw_rest_line(out, detections.body)

        warning = self._distance_warning.update(
            detections.hands, out.shape[1], out.shape[0], time.monotonic(),
        )
        self._last_distance_warning = warning
        if warning is not None:
            draw_warning(out, warning)

        # Lectura del guante (flechas de los sensores), si esta conectado.
        reading = None
        if self._glove_receiver is not None and self._glove_receiver.connected():
            reading = self._glove_receiver.latest()

        if detections.num_hands == 0:
            if reading is not None:
                # La camara no encuentra la mano: el guante se ve en el recuadro.
                draw_glove_vectors(out, reading)
            cv2.putText(
                out, "Sin manos detectadas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                out, "Sin manos detectadas", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
            )
            return out

        gloved = self._select_hand(detections) if self._glove_on_hand else None
        for hand in detections.hands:
            if hand is gloved:
                # La mano con guante siempre muestra su vector (los 21 puntos
                # que usa la camara), aunque el dibujo este apagado en Ajustes.
                draw_hand_landmarks(out, hand, draw_connections=True, draw_points=True, light=True)
            elif self._cfg.draw_landmarks or self._cfg.draw_connections:
                draw_hand_landmarks(
                    out, hand,
                    draw_connections=self._cfg.draw_connections,
                    draw_points=self._cfg.draw_landmarks,
                )
            draw_hand_label(out, hand)
        if gloved is not None and reading is not None:
            h, w = out.shape[:2]
            pts = gloved.landmarks_2d[:, :2] * np.array([w, h], dtype=np.float32)
            draw_glove_vectors(out, reading, pts, float(np.linalg.norm(pts[9] - pts[0])))
        elif reading is not None:
            draw_glove_vectors(out, reading)

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

    def _update_word_state(self, detections: FrameDetections, resting: bool = False) -> None:
        """Espacio automatico tras no_hand_frames_for_space frames sin mano
        (o, en modo automatico, con las manos en reposo: resting)."""
        if detections.num_hands == 0 or resting:
            self._frames_without_hand += 1
            if (
                self._frames_without_hand >= self._cfg.no_hand_frames_for_space
                and not self._space_already_committed
                and self._word_has_letters
            ):
                self.space_committed_signal.emit()
                self._space_already_committed = True
                self._word_has_letters = False
                self._last_committed_label = None
                self._release_frames = 0
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

    def stop(self, timeout_ms: int = 3000) -> bool:
        """Pide terminar y espera. Devuelve False si el hilo sigue corriendo:
        el llamador debe conservar la referencia hasta que termine (ver
        SignLanguageApp._retire_thread). Los modelos los cierra run() al salir."""
        self._run_flag = False
        return self.wait(timeout_ms)


# =========================================================================== #
# Voz (texto a voz del sistema)
# =========================================================================== #

# Orden de preferencia de voces en macOS. Las "Eloquence" (Eddy, Flo,
# Grandma...) existen para cada idioma pero suenan robóticas.
_MAC_LOCALE_ORDER = ("es_MX", "es_US", "es_419", "es_ES")
_MAC_PREFERRED_VOICES = ("Paulina", "Juan", "Mónica", "Jorge")

# Windows: la voz del sistema (System.Speech) por PowerShell. El texto va por
# variable de entorno, no dentro del comando, para no tener que escaparlo.
_WINDOWS_TTS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'es-*' } "
    "| Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
    "$s.Speak($env:LSM_TTS_TEXT)"
)


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

    macOS: `say`. Linux / Raspberry Pi: `espeak-ng` o `espeak`. Windows:
    System.Speech por PowerShell.
    """

    def __init__(self):
        self._base_cmd: Optional[list[str]] = None
        if shutil.which("say"):
            self._base_cmd = ["say"]
        elif shutil.which("espeak-ng"):
            self._base_cmd = ["espeak-ng", "-v", "es-419"]
        elif shutil.which("espeak"):
            self._base_cmd = ["espeak", "-v", "es-la"]
        elif sys.platform == "win32" and shutil.which("powershell"):
            self._base_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_TTS_SCRIPT]
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
        powershell = cmd[0] == "powershell"
        while True:
            text = self._queue.get()
            try:
                subprocess.run(
                    cmd if powershell else cmd + [text],
                    env={**os.environ, "LSM_TTS_TEXT": text} if powershell else None,
                    timeout=30, check=False,
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

        self.current_word = ""
        self.history: list[str] = []
        self._last_annotated_frame: Optional[np.ndarray] = None

        self._last_heartbeat = time.time()
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(1000)
        self._watchdog.timeout.connect(self._check_watchdog)
        self._watchdog_active = False

        self._available_cameras: list[int] = []

        # Hilos que no terminaron a tiempo al pararlos. Hay que conservar la
        # referencia hasta que acaben: si Python destruye un QThread en
        # marcha, Qt aborta el proceso ("Destroyed while thread is still
        # running"). Pasaba si el watchdog reiniciaba un hilo trabado.
        self._retiring_threads: list[QThread] = []

        self.speaker = Speaker()

        # Guante (ESP32): independiente de la camara. El QTimer evalua la
        # ventana de lecturas ~4 veces por segundo (el clasificador tarda
        # menos de 1 ms, no hace falta otro hilo; la red ya va en el suyo).
        # Un receptor (un solo puerto) para los dos guantes y una sesion por
        # cada mano que tenga dataset ("D", "I").
        self.glove_receiver: Optional[GloveReceiver] = None
        self.gloves: dict[str, GloveSession] = {}
        self._glove_connected: dict[str, bool] = {}
        self._glove_last_commit: tuple[str, str, float] = ("", "", 0.0)   # (mano, seña, hora)
        # Letras que escribio un guante solo (hora, letra), por si una frase
        # de dos manos llega despues y las reemplaza.
        self._glove_letters: deque = deque(maxlen=16)
        self._camera_hand_at = 0.0
        self._glove_timer = QTimer(self)
        self._glove_timer.setInterval(250)
        self._glove_timer.timeout.connect(self._glove_tick)

        # Recrear el HandLandmarker es caro: se espera a que el slider de
        # confianza de deteccion se detenga antes de aplicarlo.
        self._threshold_apply_timer = QTimer(self)
        self._threshold_apply_timer.setSingleShot(True)
        self._threshold_apply_timer.setInterval(400)
        self._threshold_apply_timer.timeout.connect(self._apply_threshold)

        # Estado de la interfaz: fase de la sena, letras pendientes (de la
        # sena en curso, que una palabra todavia puede reemplazar), hasta
        # cuando se muestra el ultimo resultado y desde cuando se cumple cada
        # condicion de los consejos en vivo.
        self._phase = "detenido"
        self._pending = 0
        # Primera letra fija de la sena en curso, retenida (no escrita): en
        # la pausa de una palabra (HOLA en la frente) el clasificador estatico
        # alcanza a fijar una letra; si se escribiera de inmediato, se veria
        # aparecer y borrarse cuando llega la palabra. Se escribe al terminar
        # la sena o en cuanto llega una segunda letra (_spelling: es deletreo).
        self._held: list[str] = []
        self._spelling = False
        self._result_hold_until = 0.0
        self._guidance_since: dict[str, float] = {}
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(lambda: self._apply_phase_style(self._phase))
        self._settings_dialog: Optional[SettingsDialog] = None
        self._manual_window: Optional[ManualWindow] = None
        self._manual_page: Optional[ManualPage] = None

        self.setWindowTitle("Traductor LSM")
        self.setMinimumSize(QSize(1180, 760))
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._restore_window_state()
        # El guante siempre esta activo: se conecta solo al abrir (sin boton).
        self.start_glove(quiet=True)
        # Las plantillas se cargan en segundo plano mientras la persona se
        # acomoda, para que Iniciar no congele la ventana (en la Raspberry Pi,
        # la primera vez, varios segundos).
        threading.Thread(target=preload_models, name="precarga", daemon=True).start()

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_actions()
        self._build_controls()
        self._build_central_widget()
        self._build_status_bar()
        self._render_sentence()
        self._apply_phase_style("detenido")
        self.statusBar().setVisible(False)       # se abre en el menu inicial

    def _build_actions(self) -> None:
        """Atajos de teclado (sin barra de herramientas: los botones estan en
        la ventana)."""
        def action(text: str, keys: list, slot) -> QAction:
            act = QAction(text, self)
            act.setShortcuts([QKeySequence(k) for k in keys])
            act.triggered.connect(slot)
            self.addAction(act)
            return act

        self.action_start = action("Iniciar", ["Ctrl+R"], self.start_system)
        self.action_stop = action("Detener", ["Ctrl+T"], self.stop_system)
        self.action_stop.setEnabled(False)
        action("Captura", ["Ctrl+S"], self.save_screenshot)
        action("Borrar letra", [Qt.Key.Key_Backspace], self.delete_last_letter)
        action("Borrar palabra", ["Ctrl+Backspace"], self.clear_current_word)
        action("Terminar palabra", [Qt.Key.Key_Return, Qt.Key.Key_Enter, "Ctrl+Space"], self.insert_space)
        action("Ajustes", ["Ctrl+,"], self.open_settings)
        action("Manual de señas", ["F1"], self.open_manual)
        action("Capturar seña del guante", ["Ctrl+G"], self.glove_capture)
        action("Capturar frase con los dos guantes", ["Ctrl+Shift+G"], lambda: self.glove_capture(phrase=True))

    def _build_controls(self) -> None:
        """Controles de ajustes tecnicos. Viven en la ventana de Ajustes, pero
        se crean aqui porque start_system y el watchdog leen sus valores."""
        def slider(lo: int, hi: int, value: int, slot, tip: str) -> QSlider:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(value)
            s.setToolTip(tip)
            s.valueChanged.connect(slot)
            return s

        self.stable_frames_slider = slider(
            3, 25, self.cfg.stable_frames_to_commit, self._on_stable_frames_changed,
            "Cuántos frames seguidos debe mantenerse una letra para fijarla.\n"
            "Bajo = más rápido, más errores. Alto = más lento, más seguro.")
        self.stable_frames_value_label = QLabel(str(self.cfg.stable_frames_to_commit))
        self.min_confidence_slider = slider(
            30, 90, round(self.cfg.min_letter_confidence * 100), self._on_min_confidence_changed,
            "Qué tan seguro debe estar el modelo para que una letra fija cuente.")
        self.min_confidence_value_label = QLabel(f"{self.cfg.min_letter_confidence:.2f}")
        self.min_margin_slider = slider(
            0, 50, round(self.cfg.min_letter_margin * 100), self._on_min_margin_changed,
            "Diferencia mínima entre la 1.ª y la 2.ª letra más probables.")
        self.min_margin_value_label = QLabel(f"{self.cfg.min_letter_margin:.2f}")
        self.threshold_slider = slider(
            10, 95, int(self.cfg.min_detection_confidence * 100), self._on_threshold_changed,
            "Confianza mínima para detectar una mano.")
        self.threshold_value_label = QLabel(f"{int(self.cfg.min_detection_confidence * 100)}%")

        self.hand_combo = QComboBox()
        self.hand_combo.addItem("Derecha", "Right")
        self.hand_combo.addItem("Izquierda", "Left")
        self.hand_combo.setCurrentIndex(self.hand_combo.findData(self.cfg.dominant_hand))
        self.hand_combo.setToolTip(
            "Con la izquierda, las señas se reflejan para compararlas con\n"
            "el modelo y las plantillas, que son de la mano derecha.")
        self.hand_combo.currentIndexChanged.connect(self._on_dominant_hand_changed)

        self.cb_speak = QCheckBox("Leer cada palabra en voz alta al terminarla")
        if self.speaker.available:
            self.cb_speak.setChecked(self.cfg.speak_words)
        else:
            self.cb_speak.setEnabled(False)
            self.cb_speak.setToolTip("No se encontró un motor de voz.\n"
                                     "Linux / Raspberry Pi: sudo apt install espeak-ng")
        self.cb_speak.toggled.connect(self._on_speak_toggled)

        self.cb_landmarks = QCheckBox("Puntos de las manos")
        self.cb_landmarks.setChecked(self.cfg.draw_landmarks)
        self.cb_landmarks.toggled.connect(self._on_draw_landmarks)
        self.cb_connections = QCheckBox("Conexiones de las manos")
        self.cb_connections.setChecked(self.cfg.draw_connections)
        self.cb_connections.toggled.connect(self._on_draw_connections)
        self.cb_body = QCheckBox("Esqueleto del cuerpo y línea de reposo")
        self.cb_body.setChecked(self.cfg.draw_body)
        self.cb_body.setEnabled(self.cfg.body_tracking)
        self.cb_body.toggled.connect(self._on_draw_body)

        self.cb_diagnostic = QCheckBox("Mostrar las 3 letras más probables en cada frame")
        self.cb_diagnostic.toggled.connect(self._on_diagnostic_toggled)
        self.diagnostic_label = QLabel("")
        self.diagnostic_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.hide()

        self.cb_glove_auto = QCheckBox("Escribir la seña del guante en cuanto se sostiene")
        self.cb_glove_auto.setChecked(self.cfg.glove_auto)
        self.cb_glove_auto.setToolTip(
            "Apagado: el guante solo escribe con la captura con cuenta atrás (Ctrl+G).")
        self.cb_glove_auto.toggled.connect(self._on_glove_auto_toggled)
        self.glove_info_label = QLabel("")
        self.glove_info_label.setObjectName("Muted")
        self.glove_info_label.setWordWrap(True)

    @staticmethod
    def _labeled_row(text: str, widget: QWidget, value: Optional[QLabel] = None) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(text)
        label.setMinimumWidth(150)
        row.addWidget(label)
        row.addWidget(widget, stretch=1)
        if value is not None:
            value.setMinimumWidth(44)
            row.addWidget(value)
        return row

    def _build_central_widget(self) -> None:
        # Tres pantallas: menu inicial -> manual de senas -> traductor.
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.start_page = StartPage()
        self.start_page.start_requested.connect(self.show_manual_page)
        self.start_page.manual_requested.connect(self.show_manual_page)
        self.start_page.quit_requested.connect(self.close)
        self.stack.addWidget(self.start_page)

        central = QWidget()
        self.app_page = central
        self.stack.addWidget(central)
        # La barra de estado (camara, FPS...) solo tiene sentido en el traductor.
        self.stack.currentChanged.connect(
            lambda _i: self.statusBar().setVisible(self.stack.currentWidget() is self.app_page))
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(14)

        # Encabezado: titulo, camara, guia, ajustes e iniciar/detener.
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Traductor LSM")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Lengua de Señas Mexicana a texto y voz")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(f"Cámara #{self.cfg.camera_index}", self.cfg.camera_index)
        self.camera_combo.setMinimumWidth(130)
        self.refresh_cameras_btn = QPushButton("↻")
        self.refresh_cameras_btn.setToolTip("Buscar cámaras conectadas")
        self.refresh_cameras_btn.clicked.connect(self._refresh_cameras)
        guide_btn = QPushButton("📖 Manual")
        guide_btn.setToolTip("Cómo se hace cada seña; se puede consultar con la cámara encendida (F1)")
        guide_btn.clicked.connect(self.open_manual)
        settings_btn = QPushButton("⚙ Ajustes")
        settings_btn.setToolTip("Ajustes de cámara, reconocimiento y voz (Ctrl+,)")
        settings_btn.clicked.connect(self.open_settings)
        self.start_button = QPushButton("▶  Iniciar")
        self.start_button.setObjectName("Primary")
        self.start_button.setMinimumWidth(130)
        self.start_button.setToolTip("Iniciar o detener la cámara (Ctrl+R / Ctrl+T)")
        self.start_button.clicked.connect(self._toggle_system)
        for w in (self.camera_combo, self.refresh_cameras_btn, guide_btn, settings_btn, self.start_button):
            w.setCursor(Qt.CursorShape.PointingHandCursor)
            header.addWidget(w)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        # Columna izquierda: video y texto traducido.
        left = QVBoxLayout()
        left.setSpacing(14)
        video_card = Card()
        self.image_label = QLabel("Presiona  ▶ Iniciar  para encender la cámara")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_card.body.addWidget(self.image_label, stretch=1)
        video_footer = QHBoxLayout()
        self.phase_pill = Pill()
        video_footer.addWidget(self.phase_pill)
        self.hint_label = QLabel("Sube la mano sobre la línea punteada para empezar una seña.")
        self.hint_label.setObjectName("Muted")
        video_footer.addWidget(self.hint_label, stretch=1)
        video_card.body.addLayout(video_footer)
        left.addWidget(video_card, stretch=1)

        text_card = Card("Texto traducido")
        self.sentence_label = QLabel()
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setTextFormat(Qt.TextFormat.RichText)
        self.sentence_label.setMinimumHeight(70)
        self.sentence_label.setStyleSheet("font-size: 30px; font-weight: 600; letter-spacing: 1px;")
        text_card.body.addWidget(self.sentence_label)
        buttons = QHBoxLayout()
        for text, tip, slot in (
            ("⌫  Borrar letra", "Borra la última letra (Retroceso)", self.delete_last_letter),
            ("✕  Borrar palabra", "Borra la palabra en curso (Ctrl+Retroceso)", self.clear_current_word),
            ("␣  Terminar palabra", "Cierra la palabra en curso (Enter)", self.insert_space),
            ("🔊  Leer", "Lee en voz alta la última palabra", self.speak_last),
            ("💾  Guardar texto", "Guarda todo el texto en un archivo", self.export_history),
            ("🗑  Limpiar", "Borra todo el texto", self.clear_all),
        ):
            b = big_button(text, tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        text_card.body.addLayout(buttons)
        left.addWidget(text_card)
        body.addLayout(left, stretch=62)

        # Columna derecha: sena actual y retroalimentacion.
        right = QVBoxLayout()
        right.setSpacing(14)
        sign_card = Card("Seña")
        self.sign_label = QLabel("—")
        self.sign_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_label.setMinimumHeight(110)
        self.sign_kind_label = QLabel("Esperando")
        self.sign_kind_label.setObjectName("Muted")
        self.sign_kind_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sign_card.body.addWidget(self.sign_label)
        sign_card.body.addWidget(self.sign_kind_label)
        self.candidates = CandidateBars()
        sign_card.body.addWidget(self.candidates)
        self._set_sign("—", COLORS["muted"], "Esperando")
        right.addWidget(sign_card)

        # Lectura en vivo de los sensores de cada guante (solo con el guante
        # encendido): derecho ("h": "D") e izquierdo ("h": "I").
        def sensors_card(title: str) -> tuple[Card, QLabel]:
            card = Card(title)
            label = QLabel(format_reading(None))
            label.setStyleSheet(f"font-family: monospace; font-size: 11px; color: {COLORS['muted']};")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card.body.addWidget(label)
            return card, label

        self.glove_card, self.glove_sensors_label = sensors_card("Guante derecho")
        right.addWidget(self.glove_card)
        self.glove_left_card, self.glove_left_sensors_label = sensors_card("Guante izquierdo")
        right.addWidget(self.glove_left_card)

        fb_card = Card("Retroalimentación")
        self.feedback = FeedbackPanel()
        fb_card.body.addWidget(self.feedback, stretch=1)
        right.addWidget(fb_card, stretch=1)
        body.addLayout(right, stretch=38)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_camera = QLabel("● Cámara detenida")
        self.status_body = QLabel("Cuerpo: —")
        self.status_hands = QLabel("✋ Manos: 0")
        self.status_fps = QLabel("FPS: —")
        self.status_latency = QLabel("Latencia: —")
        self.status_glove = QLabel("🧤 Guante apagado")
        for w in (self.status_camera, self.status_body, self.status_hands, self.status_fps, self.status_latency,
                  self.status_glove):
            bar.addPermanentWidget(w)

    def open_settings(self) -> None:
        if self._settings_dialog is None:
            cam_row = QHBoxLayout()
            cam_row.addWidget(QLabel("La cámara se elige arriba, junto a Iniciar."))
            self._settings_dialog = SettingsDialog([
                ("Letras fijas (A-Y)", [
                    self._labeled_row("Estabilidad (frames)", self.stable_frames_slider, self.stable_frames_value_label),
                    self._labeled_row("Confianza mínima", self.min_confidence_slider, self.min_confidence_value_label),
                    self._labeled_row("Margen mínimo", self.min_margin_slider, self.min_margin_value_label),
                ]),
                ("Cámara y detección", [
                    cam_row,
                    self._labeled_row("Detección de manos", self.threshold_slider, self.threshold_value_label),
                ]),
                ("Persona", [self._labeled_row("Mano que deletrea", self.hand_combo), self.cb_speak]),
                ("Dibujo sobre el video", [self.cb_landmarks, self.cb_connections, self.cb_body]),
                ("Guante (ESP32)", [self.cb_glove_auto, self.glove_info_label]),
                ("Diagnóstico", [self.cb_diagnostic, self.diagnostic_label]),
            ], self)
        self._refresh_glove_info()
        self._settings_dialog.show()
        self._settings_dialog.raise_()

    def _word_descriptions(self) -> dict[str, str]:
        """{PALABRA: 'con una mano, a la altura de la cabeza (~2 s).'}"""
        return {label: how_to_sign(label, prof).split(": ", 1)[-1]
                for label, prof in word_profiles().items()}

    def show_manual_page(self) -> None:
        """Despues del menu: el manual a pantalla completa, con el boton para
        seguir al traductor. Se construye la primera vez que se pide."""
        if self._manual_page is None:
            self._manual_page = ManualPage(MANUAL_DIR, self._word_descriptions())
            self._manual_page.back_requested.connect(lambda: self.stack.setCurrentWidget(self.start_page))
            self._manual_page.continue_requested.connect(self.enter_translator)
            self.stack.addWidget(self._manual_page)
        self.stack.setCurrentWidget(self._manual_page)

    def enter_translator(self) -> None:
        self.stack.setCurrentWidget(self.app_page)
        if self.camera_thread is None and self.ai_thread is None:
            self.start_system()

    def open_manual(self) -> None:
        """El manual en su propia ventana, para consultarlo sin detener la camara."""
        if self._manual_window is None:
            self._manual_window = ManualWindow(MANUAL_DIR, self._word_descriptions(), self)
        self._manual_window.show()
        self._manual_window.raise_()
        self._manual_window.activateWindow()

    # ---- estado visual ----------------------------------------------------

    def _set_sign(self, text: str, color: str, kind: str) -> None:
        # Se llama en cada frame: setStyleSheet obliga a Qt a recalcular el
        # estilo, asi que solo se toca si algo cambio.
        if (text, color, kind) == getattr(self, "_sign_state", None):
            return
        self._sign_state = (text, color, kind)
        size = 64 if len(text) <= 2 else (40 if len(text) <= 9 else 30)
        self.sign_label.setText(text)
        self.sign_label.setStyleSheet(f"font-size: {size}px; font-weight: 800; color: {color};")
        self.sign_kind_label.setText(kind)

    def _apply_phase_style(self, phase: str) -> None:
        text, color = PHASE_STYLE.get(phase, PHASE_STYLE["reposo"])
        # En reposo el marco queda discreto, pero la etiqueta se tiene que leer.
        self.phase_pill.set(text, COLORS["muted"] if phase in ("reposo", "detenido") else color)
        self.image_label.setStyleSheet(
            f"background: #0b1220; color: {COLORS['muted']}; font-size: 16px;"
            f"border: 3px solid {color}; border-radius: 12px;"
        )

    def _flash(self, phase: str, ms: int = 1400) -> None:
        """Marco verde (reconocida) o ambar (repetir) por un momento."""
        text, color = PHASE_STYLE[phase]
        self.phase_pill.set(text, color)
        self.image_label.setStyleSheet(
            f"background: #0b1220; color: {COLORS['muted']}; font-size: 16px;"
            f"border: 3px solid {color}; border-radius: 12px;"
        )
        self._flash_timer.start(ms)

    def _render_sentence(self) -> None:
        self.sentence_label.setText(sentence_html(self.history, self.current_word, self._pending))

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
        self.ai_thread.auto_result_signal.connect(self._on_auto_result)
        self.ai_thread.letters_retracted_signal.connect(self.on_letters_retracted)
        self.ai_thread.phase_signal.connect(self._on_phase)
        self.ai_thread.guidance_signal.connect(self._on_guidance)
        self.ai_thread.metrics_signal.connect(self.update_metrics)
        self.ai_thread.error_signal.connect(self._on_ai_error)
        self.ai_thread.model_loaded_signal.connect(self._on_model_loaded)
        self.ai_thread.heartbeat_signal.connect(self._on_heartbeat)

    def start_system(self) -> None:
        if self.camera_thread is not None or self.ai_thread is not None:
            return

        self.cfg.camera_index = int(self.camera_combo.currentData())
        self.stack.setCurrentWidget(self.app_page)   # p. ej. Ctrl+R desde el menu

        self.action_start.setEnabled(False)
        self.status_camera.setText("● Conectando...")
        self.start_button.setText("■  Detener")
        self.start_button.setObjectName("Danger")
        self.start_button.setStyleSheet("")    # que tome el estilo de #Danger
        self.image_label.setText("Encendiendo la cámara…")
        self._on_phase("reposo")

        self.camera_thread = CameraThread(self.frame_queue, self.cfg.camera_index)
        self.ai_thread = HandTrackingThread(self.frame_queue, self.cfg)

        self.ai_thread.set_stable_frames_to_commit(self.stable_frames_slider.value())
        self.ai_thread.set_min_confidence(self.min_confidence_slider.value() / 100.0)
        self.ai_thread.set_min_margin(self.min_margin_slider.value() / 100.0)
        self.ai_thread.set_diagnostic_mode(self.cb_diagnostic.isChecked())
        self._sync_glove_to_thread()

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

    def _retire_thread(self, thread: QThread) -> None:
        """Para el hilo; si no termina a tiempo, guarda la referencia hasta
        que termine (ver _retiring_threads) en vez de soltarla."""
        if thread is self.ai_thread:
            # Que un hilo viejo que todavia no sale no siga mandando video o
            # letras a la ventana mientras corre el nuevo.
            for signal in (
                thread.change_pixmap_signal, thread.sign_detected_signal,
                thread.sign_diagnostic_signal, thread.hands_detected_signal,
                thread.letter_committed_signal, thread.space_committed_signal,
                thread.metrics_signal, thread.error_signal,
                thread.model_loaded_signal, thread.heartbeat_signal,
                thread.auto_result_signal, thread.letters_retracted_signal,
                thread.phase_signal, thread.guidance_signal,
            ):
                try:
                    signal.disconnect()
                except TypeError:
                    pass
        if thread.stop():
            return
        log.warning("%s no terminó a tiempo; se espera en segundo plano", type(thread).__name__)
        self._retiring_threads.append(thread)
        thread.finished.connect(lambda t=thread: self._forget_thread(t))
        if thread.isFinished():   # termino justo entre stop() y el connect
            self._forget_thread(thread)

    def _forget_thread(self, thread: QThread) -> None:
        if thread in self._retiring_threads:
            self._retiring_threads.remove(thread)

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

        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Cámara detenida. Presiona  ▶ Iniciar  para continuar")
        self._set_sign("—", COLORS["muted"], "Esperando")
        self.candidates.set_candidates([])
        self.feedback.set_live(None)
        self._guidance_since.clear()
        self._pending = 0
        self._render_sentence()
        self._flash_timer.stop()
        self._on_phase("detenido")
        self.status_camera.setText("● Cámara detenida")
        self.status_body.setText("Cuerpo: —")
        self.status_hands.setText("✋ Manos: 0")
        self.status_fps.setText("FPS: —")
        self.status_latency.setText("Latencia: —")

        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.start_button.setText("▶  Iniciar")
        self.start_button.setObjectName("Primary")
        self.start_button.setStyleSheet("")

    def _toggle_system(self) -> None:
        if self.camera_thread is None and self.ai_thread is None:
            self.start_system()
        else:
            self.stop_system()

    def _restart_ai_thread(self) -> None:
        log.warning("Watchdog: reiniciando hilo de IA")
        if self.ai_thread is not None:
            self._retire_thread(self.ai_thread)
        self.ai_thread = HandTrackingThread(self.frame_queue, self.cfg)
        self._wire_ai_thread()
        self.ai_thread.set_stable_frames_to_commit(self.stable_frames_slider.value())
        self.ai_thread.set_min_confidence(self.min_confidence_slider.value() / 100.0)
        self.ai_thread.set_min_margin(self.min_margin_slider.value() / 100.0)
        self.ai_thread.set_diagnostic_mode(self.cb_diagnostic.isChecked())
        self._sync_glove_to_thread()
        self.ai_thread.start()
        self._last_heartbeat = time.time()
        self.statusBar().showMessage("IA reiniciada por inactividad", 3000)


    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_value_label.setText(f"{value}%")
        self.cfg.min_detection_confidence = value / 100.0
        # Antes solo se guardaba en la config y no tenia efecto hasta el
        # siguiente Iniciar: el detector se crea con el umbral una sola vez.
        self._threshold_apply_timer.start()

    def _apply_threshold(self) -> None:
        if self.ai_thread is not None:
            self.ai_thread.request_hands_reinit()

    def _on_draw_landmarks(self, checked: bool) -> None:
        self.cfg.draw_landmarks = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_landmarks(checked)

    def _on_draw_connections(self, checked: bool) -> None:
        self.cfg.draw_connections = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_connections(checked)

    def _on_draw_body(self, checked: bool) -> None:
        self.cfg.draw_body = checked
        if self.ai_thread is not None:
            self.ai_thread.set_draw_body(checked)

    def _on_stable_frames_changed(self, value: int) -> None:
        self.stable_frames_value_label.setText(str(value))
        self.cfg.stable_frames_to_commit = value
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
        # HandTrackingThread lee self.cfg (el mismo objeto) en cada frame.
        self.cfg.dominant_hand = self.hand_combo.currentData()
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state(has_letters=bool(self.current_word.strip()))
        # La camara ahora combina con el guante de la otra mano.
        self._sync_glove_to_thread()

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

    def _on_auto_result(self, info: dict) -> None:
        """Resultado de una sena con movimiento: la tarjeta de la sena, el
        top-3, el marco del video y el mensaje de retroalimentacion con lo
        que hay que corregir (interfaz_lsm.feedback_for_result)."""
        code = info.get("code", "")
        kind = info.get("kind", "")
        fb = feedback_for_result(info, word_profiles())
        topk = info.get("topk") or []
        glove_status = info.get("glove_status")
        if glove_status == "no_coinciden":
            self.feedback.add(Feedback(
                "warn", "La cámara y el guante no coinciden",
                f"Cámara: {word_display(info.get('camera_top') or '?')} · Guante: "
                f"{word_display(info.get('glove_top') or '?')}. Repite la seña."))
        if code == "ok":
            source = {"coinciden": " · cámara + guante", "guante_solo": " · guante"}.get(glove_status, "")
            self._set_sign(info.get("label", ""), COLORS["ok"],
                           ("Palabra" if kind == "palabra" else "Letra") + source)
            self.candidates.set_candidates(topk, COLORS["ok"])
            self._flash("ok")
        elif code in ("deletreo", "deletreo_largo"):
            return
        else:
            shown = f"¿{topk[0][0]}?" if topk else "?"
            self._set_sign(shown, COLORS["warn"], "Repite la seña")
            self.candidates.set_candidates(topk, COLORS["warn"])
            self._flash("repetir", 2200)
        self._result_hold_until = time.time() + 2.5
        if fb is not None:
            self.feedback.add(fb)

    def _on_phase(self, phase: str) -> None:
        self._phase = phase
        if phase in ("reposo", "detenido"):
            # La sena termino y ya se resolvio: la letra retenida se escribe
            # (si una palabra la reemplazo, ya se quito) y las demas dejan de
            # estar pendientes.
            self._flush_held()
            self._spelling = False
            if self._pending:
                self._pending = 0
                self._render_sentence()
        if not self._flash_timer.isActive():
            self._apply_phase_style(phase)
        hints = {
            "detenido": "Presiona Iniciar para encender la cámara.",
            "reposo": "Sube la mano sobre la línea punteada para empezar una seña.",
            "seña": "Haz la seña completa y baja las manos al terminar.",
            "clasificando": "Reconociendo la seña…",
        }
        self.hint_label.setText(hints.get(phase, ""))

    def _on_guidance(self, state: dict) -> None:
        """Estado del frame (~4 por segundo): barra de estado y consejo en
        vivo, que solo aparece si la condicion dura un rato."""
        now = time.time()
        conditions = {
            "sin_cuerpo": state.get("hands", 0) > 0 and state.get("body_tracking") and not state.get("body_visible"),
            "dos_manos": bool(state.get("two_raised_still")),
            "duda": bool(state.get("static_unsure")),
            "moviendo": bool(state.get("moving_letter")),
        }
        for key, active in conditions.items():
            if active:
                self._guidance_since.setdefault(key, now)
            else:
                self._guidance_since.pop(key, None)
        held = {key: now - since for key, since in self._guidance_since.items()}
        conflict = state.get("glove_conflict")
        if conflict:
            self._guidance_since.setdefault("guante", now)
        else:
            self._guidance_since.pop("guante", None)
        if conflict and now - self._guidance_since["guante"] >= 0.6:
            self.feedback.set_live(Feedback(
                "tip", f"¿{conflict[0]} o {conflict[1]}?",
                f"La cámara ve {conflict[0]} y el guante siente {conflict[1]}. Ajusta la forma de la mano."))
        else:
            self.feedback.set_live(guidance_feedback(state, held))
        if state.get("body_tracking"):
            self.status_body.setText("Cuerpo: visible" if state.get("body_visible") else "Cuerpo: no se ve")

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
        """Texto de Estado del hilo en cada frame. En la tarjeta solo se
        muestra la letra fija que se esta formando (la fase y los resultados
        de las senas con movimiento llegan por sus propias senales)."""
        letter = text[1:] if text.startswith("?") else text
        is_letter = len(letter) == 1 and letter.isalpha()
        if time.time() < self._result_hold_until and not (is_letter and self._phase == "seña"):
            return
        if is_letter and not text.startswith("?"):
            self._set_sign(letter, COLORS["text"], f"Letra · {conf * 100:.0f}% · mantén la mano quieta")
        elif is_letter:
            self._set_sign(letter, COLORS["muted"], "¿Letra? Ajusta la forma de la mano")
        elif self._phase == "seña":
            self._set_sign("…", COLORS["accent"], "Haciendo seña")
        elif self._phase == "clasificando":
            self._set_sign("…", COLORS["info"], "Reconociendo")
        elif time.time() >= self._result_hold_until:
            self._set_sign("—", COLORS["muted"], "Esperando")

    def update_hands(self, detections: FrameDetections) -> None:
        if detections.num_hands:
            self._camera_hand_at = time.time()
        text = f"✋ Manos: {detections.num_hands}"
        if self.status_hands.text() != text:
            self.status_hands.setText(text)

    def update_metrics(self, m: InferenceMetrics) -> None:
        self.status_fps.setText(f"FPS: {m.fps:.1f}")
        self.status_latency.setText(f"Latencia: {m.latency_p50_ms:.0f}/{m.latency_p95_ms:.0f}ms")

    def update_image(self, cv_img: np.ndarray) -> None:
        if cv_img is None or cv_img.size == 0:
            return
        self._last_annotated_frame = cv_img
        # Escalado con OpenCV (vectorizado, rapido en ARM) y QImage en BGR
        # directo: sin la conversion a RGB ni el escalado suave de Qt, que en
        # la Raspberry Pi se comian tiempo del hilo de la ventana en cada frame.
        h, w = cv_img.shape[:2]
        box_w = max(1, self.image_label.width() - 8)
        box_h = max(1, self.image_label.height() - 8)
        scale = min(box_w / w, box_h / h)
        if abs(scale - 1.0) > 0.01:
            cv_img = cv2.resize(cv_img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(cv_img)
        qt_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))


    def on_letter_committed(self, letter: str) -> None:
        in_sign = len(letter) == 1 and self._phase in ("seña", "clasificando")
        if in_sign and not self._spelling and not self._held:
            self._held.append(letter)
            self._set_sign(letter, COLORS["accent"], "Letra · se escribe al bajar la mano")
            self._result_hold_until = time.time() + 1.0
            return
        if in_sign and self._held:
            self._spelling = True    # segunda letra en la misma sena: es deletreo
            self._flush_held(pending=True)
        if len(letter) > 1 and self.current_word and not self.current_word.endswith(" "):
            self.current_word += " "
        self.current_word += letter
        if in_sign:
            self._pending += 1       # una palabra de esta misma sena todavia puede reemplazarla
            self._set_sign(letter, COLORS["ok"], "Letra agregada")
            self._result_hold_until = time.time() + 1.0
        self._render_sentence()

    def _flush_held(self, pending: bool = False) -> None:
        if not self._held:
            return
        self.current_word += "".join(self._held)
        if pending:
            self._pending += len(self._held)
        self._held.clear()
        self._render_sentence()

    def on_letters_retracted(self, letters: list) -> None:
        """Borra del final de la palabra en curso las letras estaticas que
        una palabra completa reemplaza (modo automatico). Si ya no estan al
        final (Enter o Retroceso de por medio), no se toca nada."""
        if self._held and list(letters) == self._held:
            self._held.clear()       # nunca se escribio: basta con olvidarla
            return
        tail = "".join(letters)
        current = self.current_word.rstrip(" ")
        if tail and current.endswith(tail):
            self.current_word = current[: -len(tail)].rstrip(" ")
            self._pending = max(0, self._pending - len(tail))
            self._render_sentence()

    def on_space_committed(self) -> None:
        self._flush_held()
        if not self.current_word.strip():
            return
        word = self.current_word.strip()
        self.history.append(word)
        self.current_word = ""
        self._pending = 0
        self._render_sentence()
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state()
        if self.cfg.speak_words:
            self.speaker.say(word.lower())

    def delete_last_letter(self) -> None:
        if self._held:
            self._held.pop()
            self._set_sign("—", COLORS["muted"], "Letra borrada")
            return
        if not self.current_word:
            return
        self.current_word = self.current_word[:-1].rstrip(" ")
        self._pending = min(self._pending, len(self.current_word))
        self._render_sentence()
        # Tras corregir, la letra se puede volver a signar de inmediato, y si
        # a la palabra le quedan letras el espacio automatico la sigue cerrando.
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state(has_letters=bool(self.current_word.strip()))

    def clear_current_word(self) -> None:
        self.current_word = ""
        self._held.clear()
        self._pending = 0
        self._render_sentence()
        if self.ai_thread is not None:
            self.ai_thread.reset_word_state()

    def clear_all(self) -> None:
        self.history.clear()
        self.clear_current_word()
        self.feedback.clear()

    def speak_last(self) -> None:
        text = self.current_word.strip() or (self.history[-1] if self.history else "")
        if not text:
            return
        if not self.speaker.available:
            QMessageBox.information(self, "Sin voz", "No se encontró un motor de voz en este equipo.")
            return
        self.speaker.say(text.lower())

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
        # cv2.imwrite no acepta rutas con caracteres no ASCII en Windows (la
        # carpeta del equipo es "...\Lenguaje de señas\..."): se codifica el
        # PNG en memoria y se escribe con Python.
        ok, png = cv2.imencode(".png", self._last_annotated_frame)
        if ok:
            try:
                Path(path).write_bytes(png.tobytes())
            except OSError:
                ok = False
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


    # ---- guante (ESP32) ---------------------------------------------------

    def _glove_dataset_path(self, hand: str = "D") -> Path:
        custom, default = {
            "I": (self.cfg.glove_dataset_left, GLOVE_DEFAULT_DATASET_LEFT),
            GLOVE_BOTH: (self.cfg.glove_dataset_both, GLOVE_DEFAULT_DATASET_BOTH),
        }.get(hand, (self.cfg.glove_dataset, GLOVE_DEFAULT_DATASET))
        return Path(custom).expanduser() if custom else default

    @staticmethod
    def _glove_source(hand: str) -> str:
        return "Dos guantes" if hand == GLOVE_BOTH else f"Guante {GLOVE_HAND_NAMES[hand]}"

    def _glove_sensor_labels(self) -> dict[str, QLabel]:
        return {"D": self.glove_sensors_label, "I": self.glove_left_sensors_label}

    def _spelling_glove(self) -> Optional[GloveSession]:
        """Sesion del guante de la mano que deletrea: la unica que se combina
        con la camara (la camara clasifica esa mano). None si no tiene dataset."""
        return self.gloves.get(GLOVE_HAND_OF_DOMINANT.get(self.cfg.dominant_hand, "D"))

    def start_glove(self, quiet: bool = False) -> None:
        """Carga el dataset de cada guante y empieza a escuchar a las ESP32 (un
        solo puerto para las dos manos). No depende de la camara: las senas
        del guante se escriben en el mismo texto. Arranca con al menos un
        dataset; la mano sin dataset solo muestra sus sensores. quiet: al
        abrir el programa, sin ventanas de error ni mensajes (la pantalla no
        cambia; el estado se ve en la barra y en la caja de sensores)."""
        if self.glove_receiver is not None:
            return
        classifiers: dict[str, GloveClassifier] = {}
        problems: list[str] = []
        for hand in GLOVE_HANDS + (GLOVE_BOTH,):
            path = self._glove_dataset_path(hand)
            try:
                classifiers[hand] = GloveClassifier.from_file(
                    path, n_values=2 * GLOVE_N_VALUES if hand == GLOVE_BOTH else GLOVE_N_VALUES)
            except (OSError, ValueError) as e:
                problems.append(f"{self._glove_source(hand)} ({path}): {e}")
                # Sin dataset del izquierdo o de frases es lo normal hasta grabarlo: aviso, no error.
                (log.info if hand != "D" and isinstance(e, FileNotFoundError) else log.warning)(
                    "%s sin dataset (%s): %s", self._glove_source(hand), path, e)
        receiver = None
        if classifiers:
            try:
                receiver = GloveReceiver(self.cfg.glove_port, hands=GLOVE_HANDS)
                receiver.start()
            except OSError as e:
                self.status_glove.setText(f"🧤 Guante: puerto {self.cfg.glove_port} ocupado"
                                          if e.errno == errno.EADDRINUSE else "🧤 Guante: error")
                # Error (no aviso): un puerto ocupado debe verse, no quedarse sin datos en silencio.
                log.error("Guante no iniciado: %s", e)
                problems.append(str(e))
                receiver = None
        else:
            self.status_glove.setText("🧤 Guante: sin muestras")
        if receiver is None:
            if not quiet:
                QMessageBox.warning(
                    self, "Guante",
                    "No se pudo iniciar el guante.\n\n" + "\n".join(problems) +
                    "\n\nGraba muestras con:  python grabar_guante.py --mano D  (o --mano I, o --mano DI para frases)")
            return
        self.glove_receiver = receiver
        # Frases: la ventana del automatico es lo que dura cada frase grabada.
        self.gloves = {hand: GloveSession(receiver.hand(hand), clf, auto=self.cfg.glove_auto,
                                          **({"live_window_s": clf.window_s, "min_gap_s": clf.window_s}
                                             if hand == GLOVE_BOTH else {}))
                       for hand, clf in classifiers.items()}
        self._glove_letters.clear()
        self._glove_connected = {}
        self._glove_timer.start()
        self.status_glove.setText("🧤 Buscando guante…")
        for label in self._glove_sensor_labels().values():
            label.setText(format_reading(None))
        for hand, clf in classifiers.items():
            log.info("%s: %d muestras (%s), umbral %.2f, %.1f s", self._glove_source(hand),
                     len(clf.y), ", ".join(clf.labels), clf.max_distance, clf.window_s)
        log.info("Guante: escuchando UDP :%d (manos %s)", self.cfg.glove_port, ", ".join(GLOVE_HANDS))
        self._refresh_glove_info()

    def stop_glove(self) -> None:
        self._glove_timer.stop()
        if self.glove_receiver is None:
            return
        self.glove_receiver.stop()
        self.glove_receiver = None
        self.gloves = {}
        self._glove_connected = {}
        self._sync_glove_to_thread()
        self.status_glove.setText("🧤 Guante apagado")
        for label in self._glove_sensor_labels().values():
            label.setText(format_reading(None) + "\n(guante apagado)")
        self._refresh_glove_info()

    def glove_capture(self, phrase: bool = False) -> None:
        """Ctrl+G: captura con cuenta atras (2, 1, ¡ya!, 2 s), como
        grabar_guante.py, con el guante de la mano que deletrea (o el que
        este mandando datos, si solo llega uno). Ctrl+Shift+G (phrase): una
        frase con los dos guantes, lo que dura cada frase grabada."""
        if self.glove_receiver is None:
            self.start_glove()
            if self.glove_receiver is None:
                return
        if phrase:
            session = self.gloves.get(GLOVE_BOTH)
            if session is None:
                self.feedback.add(Feedback(
                    "warn", "No hay frases grabadas",
                    "Grábalas con los dos guantes puestos:  python grabar_guante.py --mano DI"))
                return
            if not session.receiver.connected():
                self.feedback.add(Feedback(
                    "warn", "Faltan datos de un guante",
                    "Las frases necesitan los dos guantes: " + self._glove_problem_text()))
                return
            session.request_capture()
            return
        connected = [s for h, s in self.gloves.items() if h != GLOVE_BOTH and s.receiver.connected()]
        spelling = self._spelling_glove()
        session = spelling if spelling in connected else next(iter(connected), None)
        if session is None:
            self.feedback.add(Feedback(
                "warn", "El guante no manda datos",
                "Revisa que la ESP32 esté encendida y la Raspberry en la red GUANTE_LSM."))
            return
        session.request_capture()

    def _glove_problem_text(self) -> str:
        """Que guante no llega y por que (para la retroalimentacion)."""
        receiver = self.glove_receiver
        if receiver is None:
            return "el guante está apagado."
        missing = [GLOVE_HAND_NAMES[h] for h in GLOVE_HANDS if not receiver.connected(hand=h)]
        text = f"no llegan datos del {' ni del '.join(missing)}." if missing else "llegan los dos."
        problems = receiver.problems()
        return text + (" " + "; ".join(problems) + "." if problems else "")

    def _on_glove_auto_toggled(self, checked: bool) -> None:
        self.cfg.glove_auto = checked
        for session in self.gloves.values():
            session.auto = checked
            session.spotter.reset()

    def _refresh_glove_info(self) -> None:
        lines = [f"Escuchando UDP :{self.cfg.glove_port} (manos D e I)."]
        for hand in GLOVE_HANDS + (GLOVE_BOTH,):
            path = self._glove_dataset_path(hand)
            session = self.gloves.get(hand)
            if session is not None:
                clf = session.classifier
                counts = ", ".join(f"{word_display(l)} ({n})" for l, n in sorted(clf.counts.items()))
                lines.append(f"{self._glove_source(hand)} · {path.name}: {counts}. "
                             f"Umbral: {clf.max_distance:.2f}.")
            else:
                lines.append(f"{self._glove_source(hand)}: sin muestras "
                             f"(python grabar_guante.py --mano {hand}).")
        lines.append("Ctrl+G: capturar una seña con cuenta atrás. Ctrl+Shift+G: una frase con los dos guantes.")
        self.glove_info_label.setText("\n".join(lines))

    def _glove_tick(self) -> None:
        receiver = self.glove_receiver
        if receiver is None:
            return
        now = time.time()
        # Sensores y barra de estado de las dos manos (tambien la que no
        # tiene dataset: sirve para revisar el guante antes de grabarlo). Un
        # guante que llega pero con err != 0 se muestra igual, con el aviso:
        # si no, parece apagado.
        for hand, label in self._glove_sensor_labels().items():
            if receiver.connected(now, hand):
                label.setText(format_reading(receiver.latest(hand)))
                continue
            err = receiver.err_reading(now, hand)
            if err is not None:
                label.setText(format_reading(err[1]) + f"\n⚠ err={err[0]}: un sensor falla, no se reconoce")
            else:
                label.setText(format_reading(None))
        live = receiver.connected_hands(now)
        status = " · ".join(receiver.status_line(now, h) for h in live) if live else "Guante sin datos"
        problems = receiver.problems(now)
        self.status_glove.setText("🧤 " + status + ("  ⚠ " + "; ".join(problems) if problems else ""))
        changed = False
        for hand in GLOVE_HANDS:
            connected = receiver.connected(now, hand)
            if connected != self._glove_connected.get(hand):
                if hand in self._glove_connected or connected:
                    log.info("Guante %s %s", GLOVE_HAND_NAMES[hand], "conectado" if connected else "sin datos")
                self._glove_connected[hand] = connected
                changed = True
        if changed:
            # Solo la barra de estado y las cajas de sensores cambian:
            # conectar un guante no mueve nada mas en la pantalla.
            self._sync_glove_to_thread()

        fused = self._spelling_glove()
        # Primero la frase de dos manos: mientras los dos guantes la
        # reconocen, los guantes solos no escriben letras (su postura pasa
        # por letras a media frase).
        phrase_live = False
        order = sorted(self.gloves.items(), key=lambda kv: kv[0] != GLOVE_BOTH)
        for hand, session in order:
            for ev in session.tick(now):
                if ev.kind == "cuenta":
                    self._set_sign(str(ev.data["n"]), COLORS["info"], "Guante · prepara la seña")
                    self._result_hold_until = now + 1.5
                elif ev.kind == "capturando":
                    self._set_sign("…", COLORS["accent"], "Guante · sostén la seña")
                    self._result_hold_until = now + 2.5
                elif ev.kind == "vivo":
                    # A la camara (solo el guante de la mano que deletrea): se
                    # combina con lo que ella ve (fuse_topk).
                    res = ev.data["result"]
                    if hand == GLOVE_BOTH:
                        phrase_live = res.accepted
                    if session is fused and self.ai_thread is not None:
                        self.ai_thread.set_glove_opinion(
                            [(self._glove_camera_label(l), p) for l, p in res.topk] if res.accepted else None)
                elif ev.kind == "resultado":
                    auto = not ev.data.get("manual")
                    if hand == GLOVE_BOTH:
                        # La camara no conoce las frases: el par de guantes
                        # escribe aunque ella vea las manos.
                        if auto and self._glove_duplicate(hand, ev.data["commit"], now):
                            continue
                        if ev.data["commit"]:
                            self._retract_glove_letters(now - session.classifier.window_s)
                        self._on_glove_result(ev.data, hand)
                        continue
                    if auto and phrase_live:
                        continue     # es parte de una frase de los dos guantes
                    if auto and self._camera_sees_hand(now):
                        continue     # la camara ve la mano: la respuesta sale de los dos juntos
                    if auto and self._glove_duplicate(hand, ev.data["commit"], now):
                        continue     # la otra mano acaba de escribir la misma seña
                    self._on_glove_result(ev.data, hand)

    def _retract_glove_letters(self, since: float) -> None:
        """Borra las letras que los guantes solos escribieron desde since (a
        media frase de dos manos), si siguen al final de la palabra en curso."""
        letters = [text for t, text in self._glove_letters if t >= since]
        self._glove_letters.clear()
        if letters and len(letters) <= GLOVE_PHRASE_MAX_RETRACTED:
            log.info("Frase de dos guantes: se borran %s", "".join(letters))
            self.on_letters_retracted(letters)

    def _glove_duplicate(self, hand: str, label: Optional[str], now: float) -> bool:
        """La misma seña que el otro guante acaba de escribir (palabra de dos
        manos grabada en los dos). Registra esta como la ultima."""
        if not label:
            return False
        last_hand, last_label, last_t = self._glove_last_commit
        dup = (last_hand != hand and plain_label(last_label) == plain_label(label)
               and now - last_t < GLOVE_BOTH_HANDS_DEDUP_S)
        if not dup:
            self._glove_last_commit = (hand, label, now)
        return dup

    def _camera_sees_hand(self, now: float) -> bool:
        if self.ai_thread is None:
            return False
        return (now - self._camera_hand_at < GLOVE_CAMERA_HAND_S
                or self._phase in ("seña", "clasificando"))

    def _glove_text(self, label: str) -> str:
        """Etiqueta del guante como la escribe la camara: MAMA -> MAMÁ,
        POR_FAVOR -> POR FAVOR."""
        for word in word_profiles():
            if plain_label(word) == plain_label(label):
                return word_display(word)
        return word_display(label)

    def _glove_camera_label(self, label: str) -> str:
        """Etiqueta del guante con el nombre que usa la camara (MAMA -> MAMÁ)."""
        for word in word_profiles():
            if plain_label(word) == plain_label(label):
                return word
        return label

    def _sync_glove_to_thread(self) -> None:
        """Le pasa al hilo de la camara el guante conectado (para dibujar sus
        sensores y combinar respuestas) o nada si no hay guante."""
        if self.ai_thread is None:
            return
        session = self._spelling_glove()
        if session is not None and session.receiver.connected():
            vocab = {self._glove_camera_label(l) for l in session.classifier.sign_labels}
            self.ai_thread.set_glove(session.receiver, vocab)
        else:
            self.ai_thread.set_glove(None, set())

    def _on_glove_result(self, data: dict, hand: str = "D") -> None:
        result = data["result"]
        label = data["commit"]
        topk = [(self._glove_text(l), p) for l, p in result.topk]
        source = self._glove_source(hand)
        if label:
            text = self._glove_text(label)
            kind = "Frase" if hand == GLOVE_BOTH else "Palabra" if len(text) > 1 else "Letra"
            self._set_sign(text, COLORS["ok"], f"{source} · {kind}")
            self.candidates.set_candidates(topk, COLORS["ok"])
            self._commit_glove_label(text)
            if len(text) == 1:
                self._glove_letters.append((time.time(), text))
            if data.get("manual"):
                self.feedback.add(Feedback(
                    "ok", f"{source}: {text}", f"{topk[0][1] * 100:.0f}% · distancia {result.distance:.2f}"))
        else:
            self._set_sign(f"¿{topk[0][0]}?" if topk else "?", COLORS["warn"], f"{source} · repite la seña")
            self.candidates.set_candidates(topk, COLORS["warn"])
            self.feedback.add(Feedback("warn", f"{source}: no se reconoció la seña", result.reason.capitalize() + "."))
        self._result_hold_until = time.time() + 2.5

    def _commit_glove_label(self, text: str) -> None:
        """Letra: se agrega a la palabra en curso. Palabra (HOLA, POR FAVOR):
        se escribe y se cierra, como las palabras de la camara."""
        self._flush_held()
        if len(text) > 1:
            if self.current_word and not self.current_word.endswith(" "):
                self.current_word += " "
            self.current_word += text
            self.on_space_committed()
        else:
            self.current_word += text
            self._render_sentence()

    def closeEvent(self, event) -> None:
        log.info("Cerrando aplicación")
        self._save_window_state()
        if self.config_path is not None:
            self.cfg.save(self.config_path)
        self.stop_system()
        self.stop_glove()
        # Un hilo que no termino a tiempo tiene que acabar antes de que el
        # proceso salga, o Qt aborta al destruirlo.
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
    p.add_argument("--glove-dataset", help="Archivo .jsonl de muestras del guante derecho (grabar_guante.py)")
    p.add_argument("--glove-dataset-left", help="Archivo .jsonl del guante izquierdo (grabar_guante.py --mano I)")
    p.add_argument("--glove-dataset-both", help="Archivo .jsonl de frases con los dos guantes (grabar_guante.py --mano DI)")
    p.add_argument("-v", "--verbose", action="store_true", help="Logs detallados")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = args.config or Path.home() / ".sign_translator" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig.load(config_path)
    # Mismas validaciones que config.json (ej. --threshold 7 o --max-hands 0
    # hacian fallar a MediaPipe al iniciar).
    if args.camera is not None:
        cfg.set_validated("camera_index", args.camera)
    if args.threshold is not None:
        cfg.set_validated("min_detection_confidence", args.threshold)
    if args.max_hands is not None:
        cfg.set_validated("max_num_hands", args.max_hands)
    if args.glove_dataset is not None:
        cfg.set_validated("glove_dataset", args.glove_dataset)
    if args.glove_dataset_left is not None:
        cfg.set_validated("glove_dataset_left", args.glove_dataset_left)
    if args.glove_dataset_both is not None:
        cfg.set_validated("glove_dataset_both", args.glove_dataset_both)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")

    window = SignLanguageApp(cfg, config_path=config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())