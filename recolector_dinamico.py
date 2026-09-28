"""Recolector de datos para senas DINAMICAS (con movimiento).

Igual que recolector_estatico.py, reutiliza la deteccion de manos y la
normalizacion de senas.py / sign_classifier.py, pero en vez de un solo vector
guarda una secuencia completa (una muestra = lista de vectores de 126, uno
por frame) mientras se mantiene presionada una tecla.

OpenCV no expone un evento nativo de "tecla soltada" (waitKey solo informa
teclas presionadas, via el auto-repeat del sistema operativo). Por eso la
"suelta" se simula: si la tecla de grabar no vuelve a detectarse durante
RELEASE_TIMEOUT_S segundos, se asume que se solto y se guarda la secuencia.
Esto anade un pequeno colchon de frames al final de cada muestra (no afecta
el entrenamiento, solo agrega unos frames de la pose final).

Uso:
    python recolector_dinamico.py [--camera 0]

Controles:
    mantener 'g'  -> graba la secuencia mientras se sostiene
    n             -> termina la sena actual y pide una nueva
    q / ESC       -> sale del programa (no durante una grabacion activa)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from sign_classifier import normalize_keypoints, hand_to_feature_vector
from senas import ensure_hand_model, HandDetection, draw_hand_landmarks

OUTPUT_ROOT = Path(__file__).resolve().parent / "datos_dinamicas"
N_FEATURES_PER_HAND = 63
N_FEATURES = N_FEATURES_PER_HAND * 2
RECORD_KEY = ord('g')
RELEASE_TIMEOUT_S = 0.4
MIN_FRAMES = 3
RECOMMENDED_SAMPLES = (3, 5)


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

    Mismo orden de slots y misma normalizacion que recolector_estatico.py,
    para que ambos tipos de datos compartan exactamente el mismo esquema.
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


def next_sample_path(word_dir: Path) -> Path:
    word_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(word_dir.glob("muestra_*.json"))) + 1
    return word_dir / f"muestra_{n}.json"


def save_sequence(word_dir: Path, sequence: list[np.ndarray]) -> Path:
    path = next_sample_path(word_dir)
    payload = {
        "n_frames": len(sequence),
        "n_features": N_FEATURES,
        "frames": [vec.tolist() for vec in sequence],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Recolector de senas dinamicas (con movimiento)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("=== Recolector de senas DINAMICAS ===")
    input("Tu nombre (solo como referencia, no se guarda en el archivo): ")

    landmarker = init_hand_landmarker(max_num_hands=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {args.camera}", file=sys.stderr)
        return 1

    window = "Recolector dinamico - LSM"
    cv2.namedWindow(window)
    timestamp_ms = 0

    try:
        while True:
            word = input("\nSena a grabar ('q' para salir): ").strip()
            if word.lower() in ("q", "salir", "exit"):
                break
            if not word:
                continue

            word_dir = OUTPUT_ROOT / word
            existing_n = len(list(word_dir.glob("muestra_*.json"))) if word_dir.exists() else 0
            print(f"Grabando '{word}'. Manten 'g' presionada durante toda la sena.")
            print(f"Con {RECOMMENDED_SAMPLES[0]}-{RECOMMENDED_SAMPLES[1]} repeticiones basta, no grabes de mas.")

            recording = False
            sequence: list[np.ndarray] = []
            last_key_seen = 0.0
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

                now = time.monotonic()
                key = cv2.waitKey(1) & 0xFF

                if key == RECORD_KEY:
                    last_key_seen = now
                    if not recording:
                        recording = True
                        sequence = []
                        print("  grabando...")

                if recording:
                    sequence.append(build_feature_vector(hands))
                    if now - last_key_seen > RELEASE_TIMEOUT_S:
                        recording = False
                        if len(sequence) < MIN_FRAMES:
                            print(f"  secuencia muy corta ({len(sequence)} frames), descartada")
                        else:
                            path = save_sequence(word_dir, sequence)
                            existing_n += 1
                            print(f"  guardada {path.name} ({len(sequence)} frames) - muestras de '{word}': {existing_n}")
                            if existing_n >= RECOMMENDED_SAMPLES[1]:
                                print(f"  ya tienes {existing_n}, con eso basta para '{word}'")
                        sequence = []

                display = frame.copy()
                for hand in hands.values():
                    draw_hand_landmarks(display, hand)

                status = f"Sena: {word}  |  muestras: {existing_n}  |  {'GRABANDO' if recording else 'listo'}"
                color = (0, 0, 255) if recording else (102, 255, 102)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(display, "manten 'g'=grabar  n=nueva sena  q=salir", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(window, display)

                if key == ord('n') and not recording:
                    break
                if key in (ord('q'), 27) and not recording:
                    quit_all = True
                    break

            if quit_all:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print(f"\nListo. Datos guardados en {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
