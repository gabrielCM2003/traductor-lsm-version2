"""Extrae plantillas de PALABRAS (JSON) a partir de videos, para el modo
palabras de senas.py y de segmentador_automatico.py.

Entrada: una carpeta con una subcarpeta por palabra y los videos dentro:

    Entrenamiento/
        HOLA/      video1.mp4, video2.mp4, ...
        GRACIAS/   ...
        PORFAVOR/  ...

Salida: datos_palabras_dinamicas/<PALABRA>/muestra_N.json, el MISMO formato
que graba segmentador_automatico.py --modo palabras --grabar: 126 valores de
las manos por frame en "frames" y 9 de ubicacion respecto al cuerpo en
"body_frames". Cada JSON guarda tambien el video de origen en "fuente": si se
vuelve a correr, los videos ya extraidos se saltan (con --sobrescribir se
rehacen en el mismo archivo).

Cada video se procesa igual que la camara en vivo, para que las plantillas se
parezcan a lo que la app ve al reconocer:
  - el frame se voltea en espejo antes de detectar (como CameraThread);
  - mismas caracteristicas de mano (sign_classifier.normalize_keypoints) y de
    cuerpo (body_tracker.body_location_features);
  - la sena se recorta con el mismo corte del modo palabras: empieza cuando una
    mano sube sobre la linea de reposo y termina al bajarla (AutoSegmenter con
    los umbrales PALABRAS_*). Si nunca se ven los hombros, vale "hay mano".

A diferencia de la camara en vivo, en los videos puede haber mas gente en
cuadro. Por eso la pose se detecta con hasta MAX_POSES personas y se sigue a
quien sena (la persona mas centrada al inicio), y solo se usan las manos que
caen junto a esa persona.

Solo depende de numpy, opencv y mediapipe (no de PyQt6), para poder correrlo
tambien en Google Colab: basta con subir este archivo junto con
body_tracker.py, sign_classifier.py y segmentador_automatico.py.

Uso:
    python extraer_palabras_videos.py RUTA_DATASET
    python extraer_palabras_videos.py RUTA_DATASET --revision revision_palabras
    python extraer_palabras_videos.py --solo-evaluar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from body_tracker import (
    DEFAULT_MODELS_DIR, DEFAULT_POSE_MODEL, LEFT_SHOULDER, LEFT_WRIST, MIN_SHOULDER_WIDTH_PX,
    MOUTH_LEFT, MOUTH_RIGHT, N_BODY_FEATURES, POSE_MODELS, RIGHT_SHOULDER, RIGHT_WRIST,
    BodyDetection, body_location_features, draw_body_skeleton, ensure_pose_model,
    hands_in_signing_space, parse_pose,
)
from segmentador_automatico import (
    PALABRAS_DIR, PALABRAS_MAX_SEQUENCE_MS, PALABRAS_MIN_SEQUENCE_MS, PALABRAS_REST_MS_TO_END,
    AutoSegmenter,
)
from sign_classifier import hand_to_feature_vector, normalize_keypoints

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# El nombre de la carpeta es la etiqueta. Alias para nombres escritos sin
# espacio; el guion bajo se muestra como espacio en senas.py ("POR FAVOR").
LABEL_ALIASES = {"PORFAVOR": "POR_FAVOR"}

# Mismo modelo de manos que senas.py (HAND_MODEL_URL); se repite aqui para no
# importar senas.py, que trae PyQt6.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# En vivo hay una sola persona y senas.py detecta 2 manos y 1 pose. En los
# videos se detecta de mas para que la mano o el cuerpo de alguien que pasa
# no le quiten el lugar a los de quien sena; luego se filtran.
MAX_HANDS = 4
MAX_POSES = 3
# Umbral de las manos mas bajo que en vivo (0.5): con la persona lejos, 0.5
# perdia la mano por ratos y cortaba la sena en pedazos (medido: con 0.3 se
# recuperan senas enteras). Las manos de mas se filtran con signer_hands.
HAND_MIN_CONFIDENCE = 0.3

# Una mano es de quien sena si su muneca cae a menos de esto (en anchos de
# hombro) de una muneca de su pose, o dentro de su "caja": hasta
# SIGNER_BOX_HALF_WIDTH anchos a cada lado del centro de los hombros y entre
# SIGNER_BOX_TOP arriba y SIGNER_BOX_BOTTOM abajo de ellos.
HAND_TO_POSE_WRIST_MAX = 0.8
SIGNER_BOX_HALF_WIDTH = 1.3
SIGNER_BOX_TOP = 1.8
SIGNER_BOX_BOTTOM = 2.6

# Una palabra de menos frames que esto no se guarda (a 30 fps son ~0.3 s,
# el minimo de PALABRAS_MIN_SEQUENCE_MS).
MIN_FRAMES = 10

# Las manos se detectan en un recorte 4:3 alrededor de quien sena, por dos
# razones:
#  - Proporcion: la app abre la camara a 640x480 (CameraThread). Las
#    coordenadas de la mano que van al vector de 126 estan normalizadas por
#    ancho y alto por separado, asi que dependen de la proporcion de la imagen:
#    con un video 16:9 la mano quedaria estirada en x respecto a lo que ve la
#    app. Un recorte 4:3 da la misma geometria que en vivo.
#  - Tamano: con la persona lejos, en la imagen completa las manos son tan
#    chicas que MediaPipe no las encuentra; en el recorte se ven mas grandes.
# El recorte es fijo en todo el video (mediana de los hombros), mide
# CROP_WIDTH_SW anchos de hombro y deja CROP_TOP_FRACTION de su alto por
# encima del centro de los hombros (cabeza y manos arriba de ella).
LIVE_ASPECT = 640 / 480
CROP_WIDTH_SW = 5.4
CROP_TOP_FRACTION = 0.52


@dataclass
class Hand:
    """Lo mismo que senas.HandDetection (lo que usan las funciones de
    caracteristicas), sin importar senas.py."""
    handedness: str
    confidence: float
    landmarks_2d: np.ndarray   # (21, 2) normalizados a la imagen COMPLETA (cuerpo, dibujo)
    landmarks_3d: np.ndarray   # (21, 3) world
    crop_2d: np.ndarray        # (21, 2) normalizados al recorte 4:3 (vector de 126)


@dataclass
class Frame:
    t_ms: float
    hands_vec: np.ndarray      # 126
    body_vec: np.ndarray       # 9
    hands: dict
    body: Optional[BodyDetection]


def normalize_label(folder_name: str) -> str:
    label = unicodedata.normalize("NFC", folder_name).strip().upper().replace(" ", "_")
    return LABEL_ALIASES.get(label, label)


def ensure_hand_model(models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "hand_landmarker.task"
    if not (target.exists() and target.stat().st_size > 1000):
        print(f"Descargando el modelo de manos a {target} ...")
        tmp = target.with_suffix(".task.partial")
        with urllib.request.urlopen(HAND_MODEL_URL, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(target)
    return target


def create_detectors(pose_model: str):
    vision = mp.tasks.vision
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=HAND_MIN_CONFIDENCE,
        min_hand_presence_confidence=HAND_MIN_CONFIDENCE,
        min_tracking_confidence=HAND_MIN_CONFIDENCE,
    ))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_pose_model(pose_model))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    return hands, pose


def parse_hands(results, crop: tuple[int, int, int, int], w: int, h: int) -> list[Hand]:
    """Manos detectadas en el recorte (x0, y0, ancho, alto), con sus puntos
    tambien pasados a coordenadas de la imagen completa."""
    x0, y0, cw, ch = crop
    out = []
    for i, lms in enumerate(results.hand_landmarks or []):
        handedness, confidence = "Right", 0.0
        if results.handedness and i < len(results.handedness) and results.handedness[i]:
            handedness = results.handedness[i][0].category_name
            confidence = results.handedness[i][0].score
        lm_3d = np.zeros((21, 3), dtype=np.float32)
        if results.hand_world_landmarks and i < len(results.hand_world_landmarks):
            lm_3d = np.array([[p.x, p.y, p.z] for p in results.hand_world_landmarks[i]], dtype=np.float32)
        crop_2d = np.array([[p.x, p.y] for p in lms], dtype=np.float32)
        full_2d = np.column_stack([(x0 + crop_2d[:, 0] * cw) / w, (y0 + crop_2d[:, 1] * ch) / h])
        out.append(Hand(
            handedness=handedness, confidence=confidence,
            landmarks_2d=full_2d.astype(np.float32), landmarks_3d=lm_3d, crop_2d=crop_2d,
        ))
    return out


def hands_feature_vector(hands: dict) -> np.ndarray:
    """126 = mano "Left" (63) + mano "Right" (63), igual que
    senas.build_dynamic_feature_vector y recolector_dinamico.build_feature_vector."""
    vec = np.zeros(126, dtype=np.float32)
    for slot, handedness in enumerate(("Left", "Right")):
        hand = hands.get(handedness)
        if hand is not None:
            vec[slot * 63:(slot + 1) * 63] = normalize_keypoints(
                hand_to_feature_vector(hand.crop_2d, hand.landmarks_3d)
            )
    return vec


def _shoulders_px(body: BodyDetection, w: int, h: int) -> Optional[tuple[np.ndarray, float]]:
    if not body.visible(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None
    pts = body.image_xyz[:, :2] * np.array([w, h], dtype=np.float32)
    width = float(np.linalg.norm(pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER]))
    if width < MIN_SHOULDER_WIDTH_PX:
        return None
    return (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2.0, width


def pick_signer(results, w: int, h: int, previous_mid: Optional[np.ndarray]) -> Optional[BodyDetection]:
    """La persona que sena: con hombros y boca en cuadro, la mas cercana a la
    de el frame anterior (para no saltar a otra persona) o, al inicio, la mas
    centrada en la imagen."""
    best, best_score = None, float("inf")
    for i in range(len(results.pose_landmarks or [])):
        body = parse_pose(results, i)
        if body is None or not body.visible(MOUTH_LEFT, MOUTH_RIGHT):
            continue
        shoulders = _shoulders_px(body, w, h)
        if shoulders is None:
            continue
        mid, _ = shoulders
        target = previous_mid if previous_mid is not None else np.array([w / 2.0, mid[1]])
        score = float(np.linalg.norm(mid - target))
        if score < best_score:
            best, best_score = body, score
    return best


def signer_hands(hands: list[Hand], body: Optional[BodyDetection], w: int, h: int) -> dict:
    """{"Left"/"Right": mano} solo con las manos de quien sena. Como en vivo,
    si dos manos traen la misma etiqueta gana la primera."""
    shoulders = _shoulders_px(body, w, h) if body is not None else None
    out: dict = {}
    for hand in hands:
        if shoulders is not None:
            mid, width = shoulders
            wrist = hand.landmarks_2d[0] * np.array([w, h], dtype=np.float32)
            pose_pts = body.image_xyz[:, :2] * np.array([w, h], dtype=np.float32)
            near_wrist = min(
                np.linalg.norm(wrist - pose_pts[LEFT_WRIST]),
                np.linalg.norm(wrist - pose_pts[RIGHT_WRIST]),
            ) <= HAND_TO_POSE_WRIST_MAX * width
            in_box = (
                abs(wrist[0] - mid[0]) <= SIGNER_BOX_HALF_WIDTH * width
                and mid[1] - SIGNER_BOX_TOP * width <= wrist[1] <= mid[1] + SIGNER_BOX_BOTTOM * width
            )
            if not (near_wrist or in_box):
                continue
        if hand.handedness not in out:
            out[hand.handedness] = hand
    return out


def signer_crop(bodies: list[Optional[BodyDetection]], w: int, h: int) -> tuple[int, int, int, int]:
    """Recorte 4:3 fijo (x0, y0, ancho, alto) alrededor de quien sena, a partir
    de la mediana de sus hombros en el video. Sin cuerpo, el 4:3 central."""
    mids, widths = [], []
    for body in bodies:
        shoulders = _shoulders_px(body, w, h) if body is not None else None
        if shoulders is not None:
            mids.append(shoulders[0])
            widths.append(shoulders[1])
    if mids:
        mid = np.median(np.array(mids), axis=0)
        cw = CROP_WIDTH_SW * float(np.median(widths))
    else:
        mid = np.array([w / 2.0, h / 2.0])
        cw = float(w)
    ch = cw / LIVE_ASPECT
    scale = min(1.0, w / cw, h / ch)   # que quepa en la imagen sin perder el 4:3
    cw, ch = cw * scale, ch * scale
    x0 = min(max(mid[0] - cw / 2.0, 0.0), w - cw)
    y0 = min(max(mid[1] - CROP_TOP_FRACTION * ch, 0.0), h - ch) if mids else (h - ch) / 2.0
    return int(round(x0)), int(round(y0)), int(round(cw)), int(round(ch))


def read_frames(path: Path):
    """(indice, frame BGR ya en espejo, como la camara en vivo) de cada frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("no se pudo abrir el video")
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield idx, cv2.flip(bgr, 1)
            idx += 1
    finally:
        cap.release()


def extract_video(path: Path, hand_det, pose_det, ts_offset_ms: int
                  ) -> tuple[list[Frame], list[list[Frame]], int, float, tuple[int, int, int, int]]:
    """Procesa el video en dos pasadas: 1) pose en la imagen completa, para
    seguir a quien sena y ubicar el recorte; 2) manos en el recorte, cuerpo y
    corte de la sena. Devuelve (frames, senas_detectadas, ultimo_timestamp_ms,
    fps, recorte). Los timestamps siguen creciendo entre videos (ts_offset_ms)
    porque los detectores en modo VIDEO los exigen crecientes; son de tiempo
    real del video, no +1 por frame, porque la pose suaviza segun el tiempo
    entre frames (ver BodyTracker)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    def timestamp(idx: int) -> int:
        return ts_offset_ms + int(round(idx * 1000.0 / fps))

    bodies: list[Optional[BodyDetection]] = []
    previous_mid = None
    w = h = 0
    for idx, bgr in read_frames(path):
        h, w = bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        body = pick_signer(pose_det.detect_for_video(mp_image, timestamp(idx)), w, h, previous_mid)
        if body is not None:
            shoulders = _shoulders_px(body, w, h)
            if shoulders is not None:
                previous_mid = shoulders[0]
        bodies.append(body)
    if not bodies:
        raise RuntimeError("el video no tiene frames")

    crop = signer_crop(bodies, w, h)
    x0, y0, cw, ch = crop
    segmenter = AutoSegmenter(
        no_hand_ms_to_end=PALABRAS_REST_MS_TO_END,
        min_sequence_ms=PALABRAS_MIN_SEQUENCE_MS,
        max_duration_ms=PALABRAS_MAX_SEQUENCE_MS,
    )
    frames: list[Frame] = []
    signs: list[list[Frame]] = []
    last_body = None
    for idx, bgr in read_frames(path):
        if idx >= len(bodies):
            break
        t_ms = idx * 1000.0 / fps
        body = bodies[idx]
        last_body = body or last_body
        roi = np.ascontiguousarray(bgr[y0:y0 + ch, x0:x0 + cw])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        found = parse_hands(hand_det.detect_for_video(mp_image, timestamp(idx)), crop, w, h)
        # Sin pose en este frame, las manos se filtran con la ultima pose vista.
        hands = signer_hands(found, last_body, w, h)

        frame = Frame(
            t_ms=t_ms,
            hands_vec=hands_feature_vector(hands),
            body_vec=body_location_features(hands, body, w, h),
            hands=hands,
            body=body,
        )
        frames.append(frame)

        active = bool(hands)
        in_space = hands_in_signing_space(hands.values(), body, w, h)
        if in_space is not None:
            active = in_space
        event = segmenter.push(active, frame, t_ms / 1000.0)
        if event is not None and event[0] == "fin_valida":
            signs.append(event[1])

    # El video puede terminar con la mano todavia arriba: se cierra la sena
    # igual que si la hubiera bajado (hasta el ultimo frame activo).
    if segmenter.state == "grabando" and segmenter.last_hand_idx >= 0:
        pending = segmenter.buffer[: segmenter.last_hand_idx + 1]
        if (pending[-1].t_ms - pending[0].t_ms) >= PALABRAS_MIN_SEQUENCE_MS:
            signs.append(pending)
    return frames, signs, timestamp(len(bodies)), fps, crop


def existing_sources(label_dir: Path) -> dict[str, Path]:
    out = {}
    for path in label_dir.glob("muestra_*.json"):
        try:
            src = json.loads(path.read_text(encoding="utf-8")).get("fuente")
        except (OSError, json.JSONDecodeError):
            continue
        if src:
            out[src] = path
    return out


def next_index(label_dir: Path) -> int:
    nums = []
    for path in label_dir.glob("muestra_*.json"):
        try:
            nums.append(int(path.stem.split("_", 1)[1]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def save_sample(path: Path, sign: list[Frame], source: str, fps: float) -> None:
    payload = {
        "n_frames": len(sign),
        "n_features": 126,
        "frames": [f.hands_vec.tolist() for f in sign],
        "n_body_features": N_BODY_FEATURES,
        "body_frames": [f.body_vec.tolist() for f in sign],
        "fuente": source,
        "fps_video": round(fps, 2),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def draw_hand(image: np.ndarray, hand: Hand) -> None:
    h, w = image.shape[:2]
    for x, y in hand.landmarks_2d:
        cv2.circle(image, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)


def save_review(path: Path, sign: list[Frame], video: Path, label: str,
                crop: tuple[int, int, int, int]) -> None:
    """Hoja de revision: 4 cuadros de la sena recortada, con el esqueleto de
    quien sena, para comprobar a ojo que se tomo a la persona correcta."""
    picks = [sign[int(round(k * (len(sign) - 1) / 3))] for k in range(4)]
    wanted = {round(f.t_ms) for f in picks}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles, idx = [], 0
    while len(tiles) < 4:
        ok, bgr = cap.read()
        if not ok:
            break
        t = round(idx * 1000.0 / fps)
        idx += 1
        if t not in wanted:
            continue
        frame = next(f for f in picks if round(f.t_ms) == t)
        bgr = cv2.flip(bgr, 1)
        if frame.body is not None:
            draw_body_skeleton(bgr, frame.body, frame.hands.values())
        for hand in frame.hands.values():
            draw_hand(bgr, hand)
        x0, y0, cw, ch = crop
        cv2.rectangle(bgr, (x0, y0), (x0 + cw, y0 + ch), (255, 200, 0), 2)
        tiles.append(cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1]))))
    cap.release()
    if tiles:
        sheet = np.hstack(tiles)
        cv2.putText(sheet, f"{label}  {video.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet)


def extract_dataset(dataset: Path, out_dir: Path, pose_model: str, overwrite: bool,
                    review_dir: Optional[Path]) -> int:
    label_dirs = sorted(d for d in dataset.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not label_dirs:
        print(f"ERROR: {dataset} no tiene subcarpetas (una por palabra).", file=sys.stderr)
        return 1
    loose = [p.name for p in dataset.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS]
    if loose:
        print(f"Aviso: se ignoran {len(loose)} videos sueltos fuera de las carpetas de palabra: {', '.join(loose)}")

    hand_det, pose_det = create_detectors(pose_model)
    ts = 0
    saved, skipped, failed = 0, 0, []
    t_start = time.perf_counter()
    try:
        for label_dir in label_dirs:
            label = normalize_label(label_dir.name)
            videos = sorted(p for p in label_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
            if not videos:
                continue
            target_dir = out_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            done = existing_sources(target_dir)
            print(f"\n== {label}: {len(videos)} videos -> {target_dir}")
            for video in videos:
                source = f"{label_dir.name}/{video.name}"
                if source in done and not overwrite:
                    skipped += 1
                    print(f"  {video.name}: ya extraido ({done[source].name}), se salta")
                    continue
                try:
                    frames, signs, ts, fps, crop = extract_video(video, hand_det, pose_det, ts + 1000)
                except Exception as e:
                    failed.append((source, str(e)))
                    print(f"  {video.name}: ERROR {e}")
                    continue
                signs = [s for s in signs if len(s) >= MIN_FRAMES]
                if not signs:
                    with_body = sum(f.body is not None for f in frames)
                    with_hand = sum(bool(f.hands) for f in frames)
                    reason = (f"no se detecto la sena ({len(frames)} frames, cuerpo en {with_body}, "
                              f"manos en {with_hand})")
                    failed.append((source, reason))
                    print(f"  {video.name}: {reason}")
                    continue
                # Un video = una sena: si el corte encontro varias, la mas larga.
                sign = max(signs, key=lambda s: s[-1].t_ms - s[0].t_ms)
                path = done.get(source) or target_dir / f"muestra_{next_index(target_dir)}.json"
                save_sample(path, sign, source, fps)
                saved += 1
                body_ok = sum(bool(f.body_vec[-1]) for f in sign)
                extra = f", {len(signs)} tramos (se tomo el mas largo)" if len(signs) > 1 else ""
                print(f"  {video.name}: {path.name}  {len(sign)} frames "
                      f"({(sign[-1].t_ms - sign[0].t_ms) / 1000:.1f}s), cuerpo en {body_ok}/{len(sign)}{extra}")
                if review_dir is not None:
                    save_review(review_dir / label / f"{path.stem}.jpg", sign, video, label, crop)
    finally:
        hand_det.close()
        pose_det.close()

    print(f"\nListo en {time.perf_counter() - t_start:.0f}s: {saved} muestras guardadas, "
          f"{skipped} ya estaban, {len(failed)} sin muestra.")
    for source, reason in failed:
        print(f"  - {source}: {reason}")
    return 0


def evaluate(out_dir: Path, weights: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)) -> None:
    """Deja-uno-fuera: cada muestra se reconoce contra todas las demas. Mide
    que tanto ayuda la ubicacion respecto al cuerpo (peso 0 = solo manos).
    Ojo: si cada palabra la grabo una sola persona, este acierto es optimista
    (la muestra y sus vecinas son de la misma persona)."""
    from dtw_recognizer import DTWRecognizer

    print(f"\nEvaluacion deja-uno-fuera de {out_dir}:")
    for weight in weights:
        rec = DTWRecognizer(data_dir=out_dir, auto_save_labels=False,
                            body_weight=weight if weight > 0 else None)
        if len(rec.labels) < 2:
            print("  Hacen falta al menos 2 palabras con muestras para evaluar.")
            return
        pairs = rec.leave_one_out()
        hits = sum(real == pred for real, pred in pairs)
        confusion: dict[tuple[str, str], int] = {}
        for real, pred in pairs:
            if real != pred:
                confusion[(real, pred)] = confusion.get((real, pred), 0) + 1
        name = "solo manos" if weight == 0 else f"manos + cuerpo (peso {weight:g})"
        errors = ", ".join(f"{a}->{b} x{n}" for (a, b), n in sorted(confusion.items())) or "sin errores"
        print(f"  {name:28s} {hits}/{len(pairs)} = {hits / len(pairs) * 100:5.1f}%   ({errors})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrae plantillas JSON de palabras LSM desde videos")
    parser.add_argument("dataset", nargs="?", type=Path,
                        help="carpeta con una subcarpeta de videos por palabra")
    parser.add_argument("--salida", type=Path, default=PALABRAS_DIR,
                        help=f"donde guardar los JSON (por defecto {PALABRAS_DIR.name}/ del programa)")
    parser.add_argument("--pose", choices=POSE_MODELS, default=DEFAULT_POSE_MODEL,
                        help="modelo de pose (conviene el mismo que usa la app al reconocer)")
    parser.add_argument("--sobrescribir", action="store_true",
                        help="rehacer los videos que ya se habian extraido")
    parser.add_argument("--revision", type=Path, metavar="CARPETA",
                        help="guardar una imagen por muestra con el esqueleto, para revisarlas")
    parser.add_argument("--solo-evaluar", action="store_true",
                        help="no extraer: solo evaluar las muestras que ya estan en --salida")
    args = parser.parse_args()

    if not args.solo_evaluar:
        if args.dataset is None or not args.dataset.is_dir():
            parser.error("indica la carpeta del dataset (o usa --solo-evaluar)")
        rc = extract_dataset(args.dataset, args.salida, args.pose, args.sobrescribir, args.revision)
        if rc:
            return rc
    if args.salida.is_dir():
        evaluate(args.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
