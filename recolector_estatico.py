"""Recolector de datos para señas ESTATICAS (postura fija, 1 o 2 manos).

Reutiliza la deteccion de manos (MediaPipe HandLandmarker) y la normalizacion
de landmarks ya definidas en senas.py / sign_classifier.py, para que los
vectores generados aqui sean compatibles con ese mismo esquema de features.
No reimplementa esa logica: la importa directamente.

Ademas de las 126 columnas de las manos (v0..v125), cada fila guarda la
ubicacion de las manos respecto al cuerpo en 9 columnas b0..b8 (MediaPipe
Pose, ver body_location_features en body_tracker.py). entrenar_palabras.py
sigue leyendo solo v0..v125 hasta que se entrene un modelo que las use.

Uso:
    python recolector_estatico.py [--camera 0]

Controles:
    ESPACIO  -> captura y guarda una muestra de la palabra actual
    n        -> termina la palabra actual y pide una nueva
    ESC      -> sale del programa (desde la ventana de la camara)
    /salir   -> sale del programa (escrito en el prompt de texto)

Nota: el comando para salir del prompt de palabra es "/salir", con diagonal,
a proposito. Las etiquetas de este recolector son palabras libres (no solo
letras sueltas), asi que un comando corto como "q" o una palabra comun como
"salir" podrian coincidir con una palabra real que alguien quiera grabar
(por ejemplo, "salir" es una palabra perfectamente valida en LSM). La
diagonal inicial nunca puede aparecer en una etiqueta normalizada, asi que
no hay forma de que choquen.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from sign_classifier import normalize_keypoints, hand_to_feature_vector
from senas import ensure_hand_model, HandDetection, draw_hand_landmarks
from body_tracker import (
    N_BODY_FEATURES, BodyTracker, body_location_features,
    draw_body_skeleton, draw_body_status,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "datos_palabras"
OUTPUT_CSV = OUTPUT_DIR / "dataset_palabras.csv"
N_FEATURES_PER_HAND = 63
N_FEATURES = N_FEATURES_PER_HAND * 2
FIELDNAMES = (
    [f"v{i}" for i in range(N_FEATURES)]
    + [f"b{i}" for i in range(N_BODY_FEATURES)]
    + ["etiqueta", "quien_grabo"]
)
TARGET_SAMPLES = (150, 200)

# Comando para terminar el programa desde el prompt de texto. Con diagonal a
# proposito: nunca puede coincidir con una etiqueta real (una palabra o letra
# que alguien quiera grabar), a diferencia de "q" o "salir" a secas.
EXIT_COMMAND = "/salir"


def init_hand_landmarker(max_num_hands: int = 2):
    # Mismas opciones que HandTrackingThread._init_mediapipe en senas.py.
    models_dir = Path.home() / ".sign_translator" / "models"
    model_path = ensure_hand_model(models_dir)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def parse_hands(results) -> dict[str, HandDetection]:
    """{"Left": HandDetection, "Right": HandDetection}, solo con lo detectado."""
    hands: dict[str, HandDetection] = {}
    if not results.hand_landmarks:
        return hands

    for i, hand_lms in enumerate(results.hand_landmarks):
        handedness = "Right"
        confidence = 0.0
        if results.handedness and i < len(results.handedness) and results.handedness[i]:
            cat = results.handedness[i][0]
            handedness = cat.category_name
            confidence = cat.score

        lm_2d = np.array([[lm.x, lm.y] for lm in hand_lms], dtype=np.float32)

        lm_3d = np.zeros((21, 3), dtype=np.float32)
        if results.hand_world_landmarks and i < len(results.hand_world_landmarks):
            world = results.hand_world_landmarks[i]
            lm_3d = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)

        if handedness not in hands:
            hands[handedness] = HandDetection(
                handedness=handedness, confidence=confidence,
                landmarks_2d=lm_2d, landmarks_3d=lm_3d,
            )
    return hands


def build_feature_vector(hands: dict[str, HandDetection]) -> np.ndarray:
    """126 = [mano izquierda normalizada (63)] + [mano derecha normalizada (63)].

    Mismo orden de slots que _update_keypoint_buffer en senas.py, para que el
    esquema de features sea el mismo si en el futuro se re-usa ese codigo.
    Una mano ausente queda en ceros (normalize_keypoints ya maneja ese caso
    sin dividir entre cero).
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for slot_idx, handedness in enumerate(("Left", "Right")):
        hand = hands.get(handedness)
        if hand is None:
            continue
        raw = hand_to_feature_vector(hand.landmarks_2d, hand.landmarks_3d)
        norm = normalize_keypoints(raw)
        offset = slot_idx * N_FEATURES_PER_HAND
        vec[offset:offset + N_FEATURES_PER_HAND] = norm
    return vec


def count_existing_samples(label: str) -> int:
    if not OUTPUT_CSV.exists():
        return 0
    count = 0
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("etiqueta") == label:
                count += 1
    return count


def csv_header_matches() -> bool:
    """False si OUTPUT_CSV ya existe con otras columnas (p. ej. uno grabado
    antes de agregar b0..b8). Agregar filas de 135 valores bajo un encabezado
    de 126 desalinearia todo el archivo, asi que main() se niega a seguir."""
    if not OUTPUT_CSV.exists():
        return True
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    return header is None or header == FIELDNAMES


def append_sample(vec: np.ndarray, body_vec: np.ndarray, label: str, recorder: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        row = {f"v{i}": float(vec[i]) for i in range(N_FEATURES)}
        row.update({f"b{i}": float(body_vec[i]) for i in range(N_BODY_FEATURES)})
        row["etiqueta"] = label
        row["quien_grabo"] = recorder
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recolector de senas estaticas (1-2 manos)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("=== Recolector de senas ESTATICAS ===")
    if not csv_header_matches():
        print(
            f"ERROR: {OUTPUT_CSV} tiene columnas de una version anterior (sin b0..b{N_BODY_FEATURES - 1},\n"
            f"la ubicacion respecto al cuerpo). Renombralo o muevelo a otra carpeta y vuelve a correr\n"
            f"este programa; no se agregan filas encima para no desalinear el archivo.",
            file=sys.stderr,
        )
        return 1
    recorder = input("Tu nombre (para rastrear quien grabo cada muestra): ").strip() or "anonimo"

    landmarker = init_hand_landmarker(max_num_hands=2)
    body_tracker = BodyTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {args.camera}", file=sys.stderr)
        return 1

    window = "Recolector estatico - LSM"
    cv2.namedWindow(window)
    timestamp_ms = 0

    try:
        while True:
            # Normalizacion de la etiqueta: strip() + upper(). Se compara
            # el comando de salida ANTES de decidir si esta vacia, y en
            # minusculas para aceptar "/salir", "/SALIR", etc.
            raw = input(f"\nPalabra/sena a grabar ('{EXIT_COMMAND}' para salir): ")
            candidate = raw.strip()
            if candidate.lower() == EXIT_COMMAND:
                break
            if not candidate:
                print("  (etiqueta vacia, se ignora)")
                continue
            label = candidate.upper()
            print(f"Grabando '{label}'.")

            existing = count_existing_samples(label)
            session_count = 0
            print("ESPACIO=capturar  n=cambiar de palabra  ESC=salir del todo")

            quit_all = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: fallo al leer de la camara", file=sys.stderr)
                    break
                frame = cv2.flip(frame, 1)  # mismo mirror que CameraThread en senas.py

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 1
                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = parse_hands(results)
                body = body_tracker.detect(mp_image)   # tiempo real, no timestamp_ms (ver BodyTracker)
                frame_h, frame_w = frame.shape[:2]
                body_vec = body_location_features(hands, body, frame_w, frame_h)
                body_ok = bool(body_vec[-1])

                display = frame.copy()
                if body is not None:
                    draw_body_skeleton(display, body, hands.values())
                for hand in hands.values():
                    draw_hand_landmarks(display, hand)

                total = existing + session_count
                status = (
                    f"Palabra: {label}  |  Muestras (equipo): {total}/"
                    f"{TARGET_SAMPLES[0]}-{TARGET_SAMPLES[1]}  |  manos: {len(hands)}"
                )
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102, 255, 102), 1, cv2.LINE_AA)
                cv2.putText(display, "ESPACIO=capturar  n=nueva palabra  ESC=salir", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                draw_body_status(display, body_ok)

                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    if not hands:
                        print("  (sin manos detectadas, no se guardo)")
                    else:
                        vec = build_feature_vector(hands)
                        append_sample(vec, body_vec, label, recorder)
                        session_count += 1
                        print(f"  capturada muestra #{existing + session_count} de '{label}'")
                        if not body_ok:
                            print("  aviso: no se veian hombros y boca, la ubicacion quedo en ceros")
                elif key == ord('n'):
                    break
                elif key == 27:  # ESC. 'q' ya no sale: es una etiqueta valida (letra Q).
                    quit_all = True
                    break

            if quit_all:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        body_tracker.close()

    print(f"\nListo. Datos guardados en {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
