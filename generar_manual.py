"""Genera las ilustraciones del manual de senas (carpeta manual/) a partir de
DATOS REALES, no dibujadas de memoria: una forma de mano inventada ensenaria
la sena mal.

  - Letras fijas (A-Y): de un video deletreando el abecedario en orden. Cada
    letra se toma de una pausa de la mano en la que el clasificador estatico
    del programa la reconoce con confianza Y que respeta el orden alfabetico
    (se elige la subsecuencia creciente mas larga), asi que dos fuentes
    independientes coinciden. Tambien se puede agregar una letra suelta desde
    un video donde se sostenga (--letra E --video e.mp4).
  - Letras con movimiento (J, K, Ñ, Q, X, Z): de las plantillas de
    datos_dinamicas/ (CICESE), la mas representativa de cada letra (la de
    menor distancia DTW al resto), animada. Esas plantillas estan centradas
    en la muneca: muestran la forma y el giro de la mano, no el recorrido en
    el aire.
  - Palabras: del video mas representativo de cada palabra (misma idea, con
    las plantillas de datos_palabras_dinamicas/), una figura dibujada con el
    esqueleto del cuerpo y las manos, SIN cara, con la trayectoria de la mano.

Todo queda en espejo, como la persona se ve en la pantalla del traductor.
Cada ilustracion guarda sus puntos en un .json junto al .png: con
--redibujar se vuelven a dibujar sin los videos.

Uso:
    python generar_manual.py --abecedario ~/Downloads/Entrenamiento/abecedario.mp4
    python generar_manual.py --letra E --video e.mp4
    python generar_manual.py --dinamicas
    python generar_manual.py --palabras ~/Downloads/Entrenamiento
    python generar_manual.py --redibujar
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MANUAL_DIR = PROJECT_DIR / "manual"
LETTERS_DIR = MANUAL_DIR / "letras"
WORDS_DIR = MANUAL_DIR / "palabras"

STATIC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N",
                  "O", "P", "R", "S", "T", "U", "V", "W", "Y"]
DYNAMIC_LETTERS = ["J", "K", "Ñ", "Q", "X", "Z"]
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Estilo de las ilustraciones.
SKIN = (160, 201, 242)        # BGR de #f2c9a0
SKIN_DARK = (69, 107, 154)    # contorno
BODY = (139, 116, 100)        # figura del cuerpo (gris azulado)
BODY_DARK = (94, 76, 62)
PATH_COLOR = (248, 189, 56)   # trayectoria (#38bdf8)
SS = 2                        # supermuestreo para bordes suaves

FINGERS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
PALM = [0, 1, 5, 9, 13, 17]


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #

def _canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h * SS, w * SS, 4), dtype=np.uint8)


def _finish(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w // SS, h // SS), interpolation=cv2.INTER_AREA)


def _line(img, a, b, color, thick):
    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (*color, 255), max(1, int(thick)), cv2.LINE_AA)


def draw_hand(img: np.ndarray, pts: np.ndarray, z: Optional[np.ndarray] = None, alpha: float = 1.0) -> None:
    """Mano (21 puntos en pixeles del lienzo) con palma rellena y dedos
    gruesos; los dedos mas lejanos (z mayor en MediaPipe) se dibujan primero."""
    layer = np.zeros_like(img)
    palm_len = float(np.linalg.norm(pts[9] - pts[0])) or 1.0
    thick = max(2.0, 0.22 * palm_len)
    poly = pts[PALM].astype(np.int32)
    cv2.fillPoly(layer, [poly], (*SKIN, 255), cv2.LINE_AA)
    cv2.polylines(layer, [poly], True, (*SKIN_DARK, 255), max(1, int(thick * 0.18)), cv2.LINE_AA)
    order = range(5)
    if z is not None:
        order = sorted(range(5), key=lambda f: -float(np.mean(z[list(FINGERS[f][1:])])))
    for f in order:
        chain = FINGERS[f]
        start = 1 if f == 0 else 1
        for a, b in zip(chain[start:-1], chain[start + 1:]):
            _line(layer, pts[a], pts[b], SKIN_DARK, thick + thick * 0.35)
        for a, b in zip(chain[start:-1], chain[start + 1:]):
            _line(layer, pts[a], pts[b], SKIN, thick)
        tip = pts[chain[-1]]
        cv2.circle(layer, tuple(int(v) for v in tip), max(1, int(thick * 0.5)), (*SKIN, 255), -1, cv2.LINE_AA)
    if alpha < 1.0:
        layer[..., 3] = (layer[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    _over(img, layer)


def _over(dst: np.ndarray, src: np.ndarray) -> None:
    """Compone src sobre dst (ambos BGRA, fondo transparente): operador
    "over" con alfa, para que lo tenue no se oscurezca contra el vacio."""
    a_s = src[..., 3:4].astype(np.float32) / 255.0
    a_d = dst[..., 3:4].astype(np.float32) / 255.0
    a_o = a_s + a_d * (1 - a_s)
    rgb = (src[..., :3] * a_s + dst[..., :3] * a_d * (1 - a_s)) / np.maximum(a_o, 1e-6)
    dst[..., :3] = rgb.astype(np.uint8)
    dst[..., 3] = (a_o[..., 0] * 255).astype(np.uint8)


def _fit(points_list: list[np.ndarray], w: int, h: int, margin: float = 0.12):
    """Escala y desplazamiento comunes para que todos los puntos quepan."""
    allp = np.vstack(points_list)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((w * SS * (1 - 2 * margin)) / span[0], (h * SS * (1 - 2 * margin)) / span[1])
    offset = np.array([w * SS, h * SS]) / 2 - (lo + hi) / 2 * scale
    return scale, offset


def render_hand(pts: np.ndarray, z: Optional[np.ndarray], size: int = 260) -> np.ndarray:
    img = _canvas(size, size)
    scale, offset = _fit([pts], size, size)
    draw_hand(img, pts * scale + offset, z)
    return _finish(img)


def render_hand_sequence(frames: list[np.ndarray], zs: list[Optional[np.ndarray]], size: int = 220):
    """(tira de cuadros para animar, imagen resumen). El resumen muestra el
    inicio tenue, el final solido y el recorrido de la punta del indice."""
    scale, offset = _fit(frames, size, size)
    strip = []
    for pts, z in zip(frames, zs):
        img = _canvas(size, size)
        draw_hand(img, pts * scale + offset, z)
        strip.append(_finish(img))
    # Resumen: la forma de inicio tenue detras de la forma a mitad de la sena.
    # Sin recorrido: las plantillas estan centradas en la muneca y el trazo de
    # la punta del dedo solo reflejaba el giro de la mano (se veia confuso).
    summary = _canvas(size, size)
    draw_hand(summary, frames[0] * scale + offset, zs[0], alpha=0.35)
    mid = len(frames) // 2
    draw_hand(summary, frames[mid] * scale + offset, zs[mid])
    return np.hstack(strip), _finish(summary)


def _arrow_head(img: np.ndarray, path: np.ndarray) -> None:
    if len(path) < 3:
        return
    end = path[-1]
    back = path[max(0, len(path) - 4)]
    d = end - back
    n = float(np.linalg.norm(d))
    if n < 1:
        return
    d /= n
    perp = np.array([-d[1], d[0]])
    size = 12 * SS
    tri = np.array([end, end - d * size + perp * size * 0.6, end - d * size - perp * size * 0.6]).astype(np.int32)
    cv2.fillPoly(img, [tri], (*PATH_COLOR, 255), cv2.LINE_AA)


# ---- figura del cuerpo (palabras) -------------------------------------------

NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24


def draw_body(img: np.ndarray, pose: np.ndarray, hands: dict, alpha: float = 1.0) -> None:
    """Figura sin rasgos: cabeza, cuello, torso y brazos del esqueleto; las
    manos con sus 21 puntos. pose (33, 2) y hands en pixeles del lienzo."""
    layer = np.zeros_like(img)
    sw = float(np.linalg.norm(pose[L_SH] - pose[R_SH])) or 1.0
    mid_sh = (pose[L_SH] + pose[R_SH]) / 2
    hips = np.array([pose[L_HIP], pose[R_HIP]])
    hip_mid = hips.mean(axis=0)
    if not np.all(np.isfinite(hip_mid)) or hip_mid[1] - mid_sh[1] < 0.8 * sw:
        hip_mid = mid_sh + np.array([0, 1.7 * sw])
        hips = np.array([hip_mid + [0.35 * sw, 0], hip_mid - [0.35 * sw, 0]])
    torso = np.array([pose[L_SH], pose[R_SH], hips[1] + (hips[1] - hip_mid) * 0.3,
                      hips[0] + (hips[0] - hip_mid) * 0.3]).astype(np.int32)
    cv2.fillPoly(layer, [torso], (*BODY, 255), cv2.LINE_AA)
    head_c = (pose[NOSE] + (pose[L_EAR] + pose[R_EAR]) / 2) / 2
    r = max(0.32 * sw, 0.55 * float(np.linalg.norm(pose[L_EAR] - pose[R_EAR])))
    neck_top = head_c + np.array([0, r * 0.8])
    _line(layer, mid_sh, neck_top, BODY, 0.28 * sw)
    cv2.ellipse(layer, tuple(int(v) for v in head_c), (int(r * 0.85), int(r)), 0, 0, 360, (*BODY, 255), -1, cv2.LINE_AA)
    for sh, el, wr in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        hand_wrist = _nearest_hand_wrist(pose[wr], hands, 0.8 * sw)
        end = hand_wrist if hand_wrist is not None else pose[wr]
        _line(layer, pose[sh], pose[el], BODY_DARK, 0.2 * sw)
        _line(layer, pose[el], end, BODY_DARK, 0.17 * sw)
    if alpha < 1.0:
        layer[..., 3] = (layer[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    _over(img, layer)
    for pts in hands.values():
        draw_hand(img, pts, None, alpha)


def _nearest_hand_wrist(pose_wrist: np.ndarray, hands: dict, max_dist: float) -> Optional[np.ndarray]:
    best, best_d = None, max_dist
    for pts in hands.values():
        d = float(np.linalg.norm(pts[0] - pose_wrist))
        if d < best_d:
            best, best_d = pts[0], d
    return best


def _trim_to_sign(frames: list[dict]) -> list[dict]:
    """Quita el subir y bajar la mano: la sena grabada empieza cuando la
    muneca cruza la linea de reposo, asi que los primeros y ultimos cuadros
    son solo el traslado desde la cintura. Se deja el tramo en que la mano
    que mas se mueve esta a menos de 1 ancho de hombros bajo los hombros, y
    se descartan los cuadros sin manos."""
    frames = [f for f in frames if f["hands"]]
    slot = _moving_slot(frames)
    if slot is None:
        return frames
    keep = []
    for i, f in enumerate(frames):
        if slot not in f["hands"]:
            continue
        sw = float(np.linalg.norm(f["pose"][L_SH] - f["pose"][R_SH]))
        sh_y = float(f["pose"][L_SH][1] + f["pose"][R_SH][1]) / 2
        if f["hands"][slot][0][1] <= sh_y + 1.0 * sw:
            keep.append(i)
    if len(keep) < 3:
        return frames
    return frames[keep[0]:keep[-1] + 1]


def render_word(frames: list[dict], w: int = 300, h: int = 340):
    """(tira de cuadros, resumen) de una palabra. frames: [{"pose": (33,2),
    "hands": {slot: (21,2)}}] en pixeles del video."""
    frames = _trim_to_sign(frames)
    keypts = []
    for f in frames:
        keypts.append(f["pose"][[NOSE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL]])
        keypts.extend(f["hands"].values())
        sw = float(np.linalg.norm(f["pose"][L_SH] - f["pose"][R_SH]))
        mid = (f["pose"][L_SH] + f["pose"][R_SH]) / 2
        keypts.append(np.array([mid + [0, 1.4 * sw], mid - [0, 0.9 * sw]]))
    scale, offset = _fit(keypts, w, h, margin=0.06)

    def tr(f):
        return f["pose"] * scale + offset, {k: v * scale + offset for k, v in f["hands"].items()}

    strip = []
    for f in frames:
        img = _canvas(w, h)
        pose, hands = tr(f)
        draw_body(img, pose, hands)
        strip.append(_finish(img))
    # resumen: cuerpo en el cuadro del medio, manos del inicio tenues y la
    # trayectoria de la muneca de la mano que mas se mueve.
    summary = _canvas(w, h)
    mid = frames[len(frames) // 2]
    pose, hands = tr(mid)
    start_pose, start_hands = tr(frames[0])
    for pts in start_hands.values():
        draw_hand(summary, pts, None, alpha=0.3)
    slot = _moving_slot(frames)
    if slot is not None:
        path = np.array([f["hands"][slot][0] for f in frames if slot in f["hands"]]) * scale + offset
        if len(path) >= 2:
            path = _smooth(path)
            for a, b in zip(path[:-1], path[1:]):
                _line(summary, a, b, PATH_COLOR, 3 * SS)
    draw_body(summary, pose, hands)
    if slot is not None and len(path) >= 3:
        _arrow_head(summary, path)
    return np.hstack(strip), _finish(summary)


def _moving_slot(frames: list[dict]) -> Optional[str]:
    best, best_len = None, 0.0
    for slot in ("Left", "Right"):
        pts = [f["hands"][slot][0] for f in frames if slot in f["hands"]]
        if len(pts) >= 3:
            length = float(np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)))
            if length > best_len:
                best, best_len = slot, length
    return best


def _smooth(path: np.ndarray, k: int = 3) -> np.ndarray:
    if len(path) <= k:
        return path
    kernel = np.ones(k) / k
    return np.column_stack([np.convolve(path[:, i], kernel, mode="valid") for i in range(2)])


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"no se pudo codificar {path}")
    path.write_bytes(data.tobytes())


# --------------------------------------------------------------------------- #
# Letras fijas desde video
# --------------------------------------------------------------------------- #

def _hand_frames(video: Path, crop_to_signer: bool = True):
    """[(idx, landmarks (21,2) en pixeles del recorte 640x480, z (21,), top-3
    del clasificador estatico)] por frame con mano. Mismo procesamiento que
    la camara: espejo, recorte 4:3 en quien sena, mismo clasificador."""
    import queue
    import mediapipe as mp
    import extraer_palabras_videos as ex
    from PyQt6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import senas

    th = senas.HandTrackingThread(queue.Queue(), senas.AppConfig())
    crop = None
    if crop_to_signer:
        _, pose = ex.create_detectors("full")
        bodies, prev = [], None
        for idx, bgr in ex.read_frames(video):
            h, w = bgr.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            b = ex.pick_signer(pose.detect_for_video(img, idx * 33 + 1), w, h, prev)
            if b is not None and ex._shoulders_px(b, w, h):
                prev = ex._shoulders_px(b, w, h)[0]
            bodies.append(b)
        pose.close()
        crop = ex.signer_crop(bodies, w, h)
    hands = th._create_hand_landmarker()
    out = []
    for idx, bgr in ex.read_frames(video):
        if crop is not None:
            x0, y0, cw, ch = crop
            bgr = np.ascontiguousarray(bgr[y0:y0 + ch, x0:x0 + cw])
        roi = cv2.resize(bgr, (640, 480))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        det = th._parse_results(hands.detect_for_video(img, idx * 33 + 1))
        if det.num_hands == 0:
            out.append(None)
            continue
        hd = th._select_hand(det)
        topk = th._classify_static(hd)
        out.append((idx, hd.landmarks_2d * [640, 480], hd.landmarks_3d[:, 2].copy(), topk[:3]))
    hands.close()
    return out


def _holds(frames: list, min_len: int = 7) -> list[tuple[int, int]]:
    """Tramos con la mano quieta y la forma estable (indices de frames)."""
    from sign_classifier import normalize_keypoints
    holds, cur = [], None
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        ok = a is not None and b is not None
        if ok:
            fa = normalize_keypoints(np.column_stack([a[1] / [640, 480], a[2]]).reshape(63)).reshape(21, 3)[:, :2]
            fb = normalize_keypoints(np.column_stack([b[1] / [640, 480], b[2]]).reshape(63)).reshape(21, 3)[:, :2]
            size = max(1.0, float(np.linalg.norm(b[1][9] - b[1][0])))
            ok = np.abs(fa - fb).mean() < 0.05 and np.linalg.norm(a[1][0] - b[1][0]) / size < 0.08
        if ok:
            cur = [cur[0], i] if cur else [i - 1, i]
        else:
            if cur and cur[1] - cur[0] >= min_len:
                holds.append(tuple(cur))
            cur = None
    if cur and cur[1] - cur[0] >= min_len:
        holds.append(tuple(cur))
    return holds


def _best_frame(frames: list, s: int, e: int, label: str):
    cands = [frames[k] for k in range(s, e + 1) if frames[k] is not None and frames[k][3][0][0] == label]
    return max(cands, key=lambda f: f[3][0][1]) if cands else None


def letters_from_alphabet(video: Path, min_conf: float = 0.8) -> dict[str, dict]:
    frames = _hand_frames(video)
    holds = []
    for s, e in _holds(frames):
        labels = [frames[k][3][0][0] for k in range(s, e + 1) if frames[k] is not None]
        label = max(set(labels), key=labels.count)
        best = _best_frame(frames, s, e, label)
        if label in STATIC_LETTERS and best is not None and best[3][0][1] >= min_conf:
            holds.append((s, e, label, best))
    # subsecuencia estrictamente creciente mas larga en el orden del abecedario
    idx = [STATIC_LETTERS.index(h[2]) for h in holds]
    n = len(idx)
    best_len, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if idx[j] < idx[i] and best_len[j] + 1 > best_len[i]:
                best_len[i], prev[i] = best_len[j] + 1, j
    chosen, i = [], int(np.argmax(best_len)) if n else -1
    while i >= 0:
        chosen.append(holds[i])
        i = prev[i]
    out = {}
    for s, e, label, best in reversed(chosen):
        out[label] = {"letra": label, "tipo": "fija", "puntos": best[1].tolist(), "z": best[2].tolist(),
                      "confianza": round(float(best[3][0][1]), 3), "fuente": f"{video.name} frames {s}-{e}"}
    ignored = [h[2] for h in holds if h not in chosen]
    print(f"Letras fijas encontradas: {' '.join(out)}"
          + (f" | pausas fuera de orden ignoradas: {' '.join(ignored)}" if ignored else ""))
    missing = [L for L in STATIC_LETTERS if L not in out]
    if missing:
        print(f"Sin ejemplo confiable: {' '.join(missing)} (agregalas con --letra X --video archivo)")
    return out


def letter_from_video(label: str, video: Path) -> Optional[dict]:
    frames = _hand_frames(video, crop_to_signer=False)
    holds = sorted(_holds(frames), key=lambda se: se[1] - se[0], reverse=True)
    for s, e in holds:
        best = _best_frame(frames, s, e, label) or max(
            (frames[k] for k in range(s, e + 1) if frames[k] is not None), key=lambda f: f[3][0][1])
        return {"letra": label, "tipo": "fija", "puntos": best[1].tolist(), "z": best[2].tolist(),
                "confianza": round(float(best[3][0][1]), 3), "clasificador": best[3][0][0],
                "fuente": f"{video.name} frames {s}-{e}"}
    return None


def draw_static_letter(data: dict) -> None:
    img = render_hand(np.array(data["puntos"]), np.array(data["z"]))
    save_png(LETTERS_DIR / f"{data['letra']}.png", img)
    (LETTERS_DIR / f"{data['letra']}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Letras con movimiento desde las plantillas
# --------------------------------------------------------------------------- #

# Las plantillas se normalizaron dividiendo x entre el ancho e y entre el alto
# del video, por separado. Se estimo el factor de x que deja constante el
# largo de cada hueso de la mano (los huesos no cambian de largo) con 150
# plantillas de las 6 letras: el mejor fue 1.0 (variacion 0.310, contra
# 0.318 con 4:3 y 0.338 con 16:9). Estimarlo por plantilla daba de 0.55 a 1.5:
# ruido de los giros de la mano. Por eso va fijo.
CICESE_X_SCALE = 1.0


def dynamic_letters(n_frames: int = 16) -> dict[str, dict]:
    import logging
    logging.disable(logging.WARNING)
    from dtw_recognizer import DTWRecognizer
    rec = DTWRecognizer(data_dir=PROJECT_DIR / "datos_dinamicas", auto_save_labels=False)
    out = {}
    for label in DYNAMIC_LETTERS:
        templates = rec._templates.get(label, [])[:40]
        if not templates:
            continue
        # la mas representativa: menor suma de distancias DTW a las demas
        sums = []
        for i, t in enumerate(templates):
            rec_t = {label: [x for j, x in enumerate(templates) if j != i]}
            saved, rec._templates = rec._templates, rec_t
            sums.append(rec._distances(t)[label])
            rec._templates = saved
        best = templates[int(np.argmin(sums))]
        hand = best[:, :63]
        present = np.abs(hand).sum(axis=1) > 0
        seq = hand[present].reshape(-1, 21, 3)
        picks = np.linspace(0, len(seq) - 1, min(n_frames, len(seq))).round().astype(int)
        sx = CICESE_X_SCALE
        pts = [seq[k, :, :2] * [sx, 1.0] for k in picks]
        out[label] = {"letra": label, "tipo": "movimiento", "cuadros": [p.tolist() for p in pts],
                      "z": [seq[k, :, 2].tolist() for k in picks], "escala_x": round(sx, 3),
                      "fuente": "datos_dinamicas (CICESE, CC BY 4.0), plantilla más representativa"}
        print(f"{label}: plantilla {int(np.argmin(sums))} ({len(seq)} frames)")
    return out


def draw_dynamic_letter(data: dict) -> None:
    frames = [np.array(f) for f in data["cuadros"]]
    zs = [np.array(z) for z in data["z"]]
    strip, summary = render_hand_sequence(frames, zs)
    save_png(LETTERS_DIR / f"{data['letra']}_anim.png", strip)
    save_png(LETTERS_DIR / f"{data['letra']}.png", summary)
    (LETTERS_DIR / f"{data['letra']}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Palabras desde los videos
# --------------------------------------------------------------------------- #

def words_from_videos(dataset: Path, n_frames: int = 18) -> dict[str, dict]:
    import logging
    logging.disable(logging.WARNING)
    import extraer_palabras_videos as ex
    from dtw_recognizer import DEFAULT_WORDS_DIR, WORD_BODY_WEIGHT, DTWRecognizer
    rec = DTWRecognizer(data_dir=DEFAULT_WORDS_DIR, auto_save_labels=False, body_weight=WORD_BODY_WEIGHT)
    hand_det, pose_det = ex.create_detectors("full")
    out, ts = {}, 0
    for label_dir in sorted(d for d in DEFAULT_WORDS_DIR.iterdir() if d.is_dir()):
        label = unicodedata.normalize("NFC", label_dir.name)
        files = sorted(label_dir.glob("*.json"))
        templates = rec._templates.get(label, [])
        if not templates or len(templates) != len(files):
            continue
        sums = []
        for i, t in enumerate(templates):
            saved, rec._templates = rec._templates, {label: [x for j, x in enumerate(templates) if j != i]}
            sums.append(rec._distances(t)[label])
            rec._templates = saved
        source = json.loads(files[int(np.argmin(sums))].read_text(encoding="utf-8"))["fuente"]
        video = dataset / source
        if not video.exists():
            print(f"{label}: no encuentro {video}")
            continue
        frames, signs, ts, fps, crop = ex.extract_video(video, hand_det, pose_det, ts + 1000)
        if not signs:
            print(f"{label}: no se detecto la sena en {source}")
            continue
        sign = max(signs, key=lambda s: s[-1].t_ms - s[0].t_ms)
        h, w = 720, 1280
        picks = np.linspace(0, len(sign) - 1, min(n_frames, len(sign))).round().astype(int)
        seq = []
        last_pose = None
        for k in picks:
            f = sign[k]
            body = f.body
            if body is None:
                if last_pose is None:
                    continue
                pose = last_pose
            else:
                pose = body.image_xyz[:, :2] * [w, h]
                last_pose = pose
            seq.append({"pose": pose.tolist(),
                        "hands": {s: (hd.landmarks_2d * [w, h]).tolist() for s, hd in f.hands.items()}})
        out[label] = {"palabra": label, "cuadros": seq, "fuente": source}
        print(f"{label}: {source} ({len(seq)} cuadros)")
    hand_det.close()
    pose_det.close()
    return out


def draw_word(data: dict) -> None:
    frames = [{"pose": np.array(f["pose"]), "hands": {k: np.array(v) for k, v in f["hands"].items()}}
              for f in data["cuadros"]]
    strip, summary = render_word(frames)
    name = data["palabra"]
    save_png(WORDS_DIR / f"{name}_anim.png", strip)
    save_png(WORDS_DIR / f"{name}.png", summary)
    (WORDS_DIR / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def redraw_all() -> None:
    for path in sorted(LETTERS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        (draw_static_letter if data["tipo"] == "fija" else draw_dynamic_letter)(data)
    for path in sorted(WORDS_DIR.glob("*.json")):
        draw_word(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    p = argparse.ArgumentParser(description="Genera las ilustraciones del manual de señas (manual/)")
    p.add_argument("--abecedario", type=Path, help="video deletreando el abecedario en orden")
    p.add_argument("--letra", help="agregar o reemplazar una letra fija (con --video)")
    p.add_argument("--video", type=Path, help="video donde se sostiene la letra de --letra")
    p.add_argument("--dinamicas", action="store_true", help="letras con movimiento desde datos_dinamicas/")
    p.add_argument("--palabras", type=Path, metavar="DATASET", help="carpeta con los videos de las palabras")
    p.add_argument("--redibujar", action="store_true", help="volver a dibujar todo desde manual/*.json")
    args = p.parse_args()
    if not any((args.abecedario, args.letra, args.dinamicas, args.palabras, args.redibujar)):
        p.error("indica que generar (ver --help)")
    if args.abecedario:
        for data in letters_from_alphabet(args.abecedario).values():
            draw_static_letter(data)
    if args.letra:
        if not args.video:
            p.error("--letra necesita --video")
        data = letter_from_video(args.letra.upper(), args.video)
        if data is None:
            print("No encontre una pausa con la mano en ese video.", file=sys.stderr)
            return 1
        draw_static_letter(data)
        print(f"{args.letra.upper()}: listo (el clasificador dijo {data['clasificador']}, {data['confianza']:.0%})")
    if args.dinamicas:
        for data in dynamic_letters().values():
            draw_dynamic_letter(data)
    if args.palabras:
        for data in words_from_videos(args.palabras).values():
            draw_word(data)
    if args.redibujar:
        redraw_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
