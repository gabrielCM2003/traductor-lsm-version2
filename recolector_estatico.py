"""Recolector de datos para señas ESTATICAS (postura fija, 1 o 2 manos).

Reutiliza la deteccion de manos (MediaPipe HandLandmarker) y la normalizacion
de landmarks ya definidas en senas.py / sign_classifier.py, para que los
vectores generados aqui sean compatibles con ese mismo esquema de features.
No reimplementa esa logica: la importa directamente.

Uso:
    python recolector_estatico.py [--camera 0]

Controles:
    ESPACIO  -> captura y guarda una muestra de la palabra actual
    n        -> termina la palabra actual y pide una nueva
    q / ESC  -> sale del programa
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

OUTPUT_DIR = Path(__file__).resolve().parent / "datos_palabras"
OUTPUT_CSV = OUTPUT_DIR / "dataset_palabras.csv"
N_FEATURES_PER_HAND = 63
N_FEATURES = N_FEATURES_PER_HAND * 2
FIELDNAMES = [f"v{i}" for i in range(N_FEATURES)] + ["etiqueta", "quien_grabo"]
TARGET_SAMPLES = (150, 200)


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


def append_sample(vec: np.ndarray, label: str, recorder: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        row = {f"v{i}": float(vec[i]) for i in range(N_FEATURES)}
        row["etiqueta"] = label
        row["quien_grabo"] = recorder
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recolector de senas estaticas (1-2 manos)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("=== Recolector de senas ESTATICAS ===")
    recorder = input("Tu nombre (para rastrear quien grabo cada muestra): ").strip() or "anonimo"

    landmarker = init_hand_landmarker(max_num_hands=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {args.camera}", file=sys.stderr)
        return 1

    window = "Recolector estatico - LSM"
    cv2.namedWindow(window)
    timestamp_ms = 0

    try:
        while True:
            label = input("\nPalabra/sena a grabar ('q' para salir): ").strip()
            if label.lower() in ("q", "salir", "exit"):
                break
            if not label:
                continue

            existing = count_existing_samples(label)
            session_count = 0
            print(f"Grabando '{label}'. ESPACIO=capturar  n=cambiar de palabra  q/ESC=salir del todo")

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

                display = frame.copy()
                for hand in hands.values():
                    draw_hand_landmarks(display, hand)

                total = existing + session_count
                status = (
                    f"Palabra: {label}  |  Muestras (equipo): {total}/"
                    f"{TARGET_SAMPLES[0]}-{TARGET_SAMPLES[1]}  |  manos: {len(hands)}"
                )
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102, 255, 102), 1, cv2.LINE_AA)
                cv2.putText(display, "ESPACIO=capturar  n=nueva palabra  q=salir", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    if not hands:
                        print("  (sin manos detectadas, no se guardo)")
                    else:
                        vec = build_feature_vector(hands)
                        append_sample(vec, label, recorder)
                        session_count += 1
                        print(f"  capturada muestra #{existing + session_count} de '{label}'")
                elif key == ord('n'):
                    break
                elif key in (ord('q'), 27):
                    quit_all = True
                    break

            if quit_all:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print(f"\nListo. Datos guardados en {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
