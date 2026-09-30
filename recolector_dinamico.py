"""Recolector de datos para senas DINAMICAS (con movimiento).

Igual que recolector_estatico.py, reutiliza la deteccion de manos y la
normalizacion de senas.py / sign_classifier.py, pero en vez de un solo vector
guarda una secuencia completa (una muestra = lista de vectores de 126, uno
por frame) durante el tiempo que dura la seña.

Grabacion con INICIO/FIN explicitos (no "mantener presionada"): la primera
vez que se presiona 'g' arranca la grabacion, la segunda vez la termina. Se
elimino el esquema anterior de "mantener presionada + tiempo de espera para
detectar la suelta" (RELEASE_TIMEOUT_S ~0.4s): el auto-repeat del teclado de
Windows (delay inicial antes de repetir la tecla) podia dejar un hueco justo
al presionar, y ese hueco se interpretaba como "se solto", grabando una
muestra fantasma de ~12-14 frames antes de la real. Con inicio/fin explicitos
ese problema desaparece: la duracion de la grabacion la decide el usuario, no
un temporizador.

Al elegir la letra se pide una confirmacion grande antes de empezar a grabar
("Vas a grabar: <LETRA> - ¿correcto?"): ENTER confirma, cualquier otra cosa
vuelve a pedir la letra. Evita quedarse grabando en la letra equivocada sin
darse cuenta (ver confirm_letter()).

Al terminar una grabacion se pide confirmacion en la propia ventana (no hace
falta volver a la terminal): 's' = guardar, 'd' = descartar. Si la seña dura
menos de MIN_FRAMES_OK frames se avisa "demasiado corta" y se descarta por
defecto (se puede forzar el guardado con 'f').

Cada muestra guardada se escribe DOS veces, con el mismo indice N:
  - datos_dinamicas/<LETRA>/muestra_N.json   (formato que ya usa el proyecto,
    vectores normalizados de 126 valores, sin cambios)
  - Dataset_CICESE/propias_crudas/<LETRA>/muestra_N.npz (landmarks crudos:
    timestamp por frame, y por cada mano detectada su etiqueta Left/Right,
    score, los 21 landmarks de imagen (x,y,z) y los 21 world (x,y,z); los
    frames sin ninguna mano se guardan igual, vacios, no se omiten)

Uso:
    python recolector_dinamico.py [--camera 0]

Controles:
    'g'      -> primera vez: empieza a grabar. segunda vez: termina.
    's'      -> (al terminar de grabar) guardar la muestra
    'd'      -> (al terminar de grabar) descartar la muestra
    'f'      -> (solo si salio "demasiado corta") forzar el guardado
    n        -> termina la seña actual y pide una nueva
    ESC      -> sale del programa (no durante una grabacion activa)
    /salir   -> sale del programa (escrito en el prompt de texto)

Nota: el comando para salir del prompt de sena es "/salir", con diagonal, a
proposito. Las etiquetas aqui son justo las letras dinamicas (J, K, Ñ, Q, X,
Z), y "Q" en mayusculas o minusculas es una de ellas: un comando de un solo
caracter como "q" jamas puede usarse para salir sin bloquear esa letra. La
diagonal inicial nunca puede aparecer en una etiqueta normalizada.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import mediapipe as mp

from sign_classifier import normalize_keypoints, hand_to_feature_vector
from senas import ensure_hand_model, HandDetection, draw_hand_landmarks

OUTPUT_ROOT = Path(__file__).resolve().parent / "datos_dinamicas"
N_FEATURES_PER_HAND = 63
N_FEATURES = N_FEATURES_PER_HAND * 2
RECORD_KEY = ord('g')
RECOMMENDED_SAMPLES = (3, 5)

# Debajo de esto se avisa "demasiado corta" y se descarta por defecto al
# confirmar (el minimo observado en el dataset real es 27 frames).
MIN_FRAMES_OK = 25

# Teclas de la confirmacion al terminar de grabar (ver docstring del modulo).
CONFIRM_SAVE_KEY = ord('s')
CONFIRM_DISCARD_KEY = ord('d')
CONFIRM_FORCE_KEY = ord('f')

# Respaldo crudo (landmarks sin normalizar) para el extractor del dataset.
# Ruta absoluta fuera del proyecto, a proposito: es un dataset compartido con
# otro trabajo (Dataset_CICESE), no un artefacto de este repo.
RAW_DATASET_ROOT = Path(r"C:\Proyectos\Dataset_CICESE\propias_crudas")

# Comando para terminar el programa desde el prompt de texto. Con diagonal a
# proposito: nunca puede coincidir con una etiqueta real (J, K, Ñ, Q, X, Z u
# otra letra/palabra que se agregue despues), a diferencia de "q" a secas.
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


# =========================================================================== #
# Respaldo crudo (landmarks sin normalizar) para Dataset_CICESE/propias_crudas
# =========================================================================== #

@dataclass
class RawHandSample:
    """Una mano detectada en un frame, con los landmarks TAL CUAL los entrega
    MediaPipe (sin normalizar), a diferencia de HandDetection/build_feature_vector
    que solo guardan lo que hace falta para clasificar (x,y de imagen + z de
    world) y descartan la z de imagen. Este respaldo la conserva."""
    label: str          # "Left" o "Right"
    score: float
    image_xyz: np.ndarray   # (21, 3): x,y,z tal como los da MediaPipe en la imagen
    world_xyz: np.ndarray   # (21, 3): x,y,z en el sistema "world" (metros aprox.)


@dataclass
class RawFrameSample:
    """Un frame completo de la grabacion cruda. hands puede estar vacio
    (frame sin ninguna mano detectada) - se guarda igual, no se omite."""
    timestamp_ms: float
    hands: dict[str, RawHandSample] = field(default_factory=dict)


def capture_raw_hands(results) -> dict[str, RawHandSample]:
    """Como parse_hands, pero sin descartar informacion: conserva la z de los
    landmarks de imagen (parse_hands no la necesita para el vector de 126 y
    la tira). Se usa solo para el respaldo crudo, no para clasificar."""
    hands: dict[str, RawHandSample] = {}
    if not results.hand_landmarks:
        return hands

    for i, hand_lms in enumerate(results.hand_landmarks):
        handedness = "Right"
        score = 0.0
        if results.handedness and i < len(results.handedness) and results.handedness[i]:
            cat = results.handedness[i][0]
            handedness = cat.category_name
            score = cat.score

        if handedness in hands:
            continue

        image_xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

        world_xyz = np.zeros((21, 3), dtype=np.float32)
        if results.hand_world_landmarks and i < len(results.hand_world_landmarks):
            world = results.hand_world_landmarks[i]
            world_xyz = np.array([[lm.x, lm.y, lm.z] for lm in world], dtype=np.float32)

        hands[handedness] = RawHandSample(
            label=handedness, score=score, image_xyz=image_xyz, world_xyz=world_xyz,
        )
    return hands


def save_raw_npz(path: Path, raw_frames: list[RawFrameSample], fps_medido: float) -> None:
    """Empaqueta la grabacion cruda en arreglos de forma fija (T, 2, 21, 3):
    el slot 0 es la mano "Left" y el slot 1 la "Right" (misma convencion de
    slots que el resto del proyecto), en vez de una lista de largo variable
    por frame. Así un frame sin manos queda representado igual (todo en
    ceros/etiqueta vacia) en vez de tener que omitirse."""
    total_frames = len(raw_frames)
    timestamps_ms = np.zeros(total_frames, dtype=np.float64)
    hand_labels = np.full((total_frames, 2), "", dtype="<U5")
    hand_scores = np.zeros((total_frames, 2), dtype=np.float32)
    landmarks_image = np.zeros((total_frames, 2, 21, 3), dtype=np.float32)
    landmarks_world = np.zeros((total_frames, 2, 21, 3), dtype=np.float32)

    for t, frame in enumerate(raw_frames):
        timestamps_ms[t] = frame.timestamp_ms
        for slot_idx, lado in enumerate(("Left", "Right")):
            hand = frame.hands.get(lado)
            if hand is None:
                continue
            hand_labels[t, slot_idx] = hand.label
            hand_scores[t, slot_idx] = hand.score
            landmarks_image[t, slot_idx] = hand.image_xyz
            landmarks_world[t, slot_idx] = hand.world_xyz

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        timestamps_ms=timestamps_ms,
        hand_labels=hand_labels,
        hand_scores=hand_scores,
        landmarks_image=landmarks_image,
        landmarks_world=landmarks_world,
        fps_medido=np.float32(fps_medido),
    )


# =========================================================================== #
# Decision de guardar/descartar y escritura de una muestra (funciones puras,
# sin camara ni teclado, para poder probarlas con datos sinteticos)
# =========================================================================== #

def confirm_letter(word: str, read_line: Callable[[str], str] = input) -> bool:
    """Confirmacion grande antes de empezar a grabar `word`: evita que una
    letra mal escrita/mal seleccionada se quede grabando sin que nadie se
    de cuenta (ya paso: varias muestras destinadas a J/Q/Z terminaron
    grabadas en K por error).

    ENTER (linea vacia) confirma. Cualquier otra cosa escrita antes de
    ENTER se interpreta como "no, vuelve a pedir la letra" - no hace falta
    que sea una tecla en particular.

    `read_line` es inyectable (por defecto `input`) para poder probar esta
    funcion con entrada simulada, sin teclado real ni camara."""
    print("\n" + "=" * 60)
    print(f"  VAS A GRABAR: '{word}'")
    print("  ¿Correcto? ENTER = confirmar   |   cualquier tecla + ENTER = volver a escribir")
    print("=" * 60)
    respuesta = read_line("> ")
    return respuesta == ""


def should_discard_by_default(n_frames: int) -> bool:
    """True si, sin intervencion del usuario, esta grabacion se descartaria
    por ser demasiado corta (por debajo de MIN_FRAMES_OK). La confirmacion
    interactiva puede forzar el guardado de todas formas con 'f'."""
    return n_frames < MIN_FRAMES_OK


def compute_recording_stats(raw_frames: list[RawFrameSample], duration_s: float) -> dict:
    """Frames, duracion real y % de frames sin ninguna mano detectada."""
    n_frames = len(raw_frames)
    sin_mano = sum(1 for f in raw_frames if not f.hands)
    pct_sin_mano = (sin_mano / n_frames * 100.0) if n_frames else 0.0
    return {"n_frames": n_frames, "duracion_s": duration_s, "pct_sin_mano": pct_sin_mano}


def next_muestra_index(word_dir: Path) -> int:
    """Siguiente indice N libre para muestra_N.json, robusto a huecos (si se
    borro a mano una muestra intermedia, usar solo len(archivos)+1 podia
    volver a usar un numero ya ocupado; aqui se toma el maximo existente+1)."""
    word_dir.mkdir(parents=True, exist_ok=True)
    numeros = []
    for p in word_dir.glob("muestra_*.json"):
        try:
            numeros.append(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(numeros) + 1) if numeros else 1


def save_muestra(
    word_dir: Path,
    letra: str,
    feature_sequence: list[np.ndarray],
    raw_frames: list[RawFrameSample],
    fps_medido: float,
    raw_dataset_root: Path = RAW_DATASET_ROOT,
) -> tuple[Path, Path]:
    """Guarda la muestra en datos_dinamicas/<LETRA>/muestra_N.json (formato
    del proyecto, sin cambios) Y en Dataset_CICESE/propias_crudas/<LETRA>/
    muestra_N.npz (landmarks crudos), con el MISMO indice N en ambas, tomado
    una sola vez a partir de los .json existentes (para que ambos formatos
    queden sincronizados aunque uno de los dos directorios se limpie aparte)."""
    n = next_muestra_index(word_dir)

    json_path = word_dir / f"muestra_{n}.json"
    payload = {
        "n_frames": len(feature_sequence),
        "n_features": N_FEATURES,
        "frames": [vec.tolist() for vec in feature_sequence],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    npz_path = raw_dataset_root / letra / f"muestra_{n}.npz"
    save_raw_npz(npz_path, raw_frames, fps_medido)

    return json_path, npz_path


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
            # Normalizacion de la etiqueta: strip() + upper(). El comando de
            # salida se compara ANTES de decidir si esta vacia, y en
            # minusculas para aceptar "/salir", "/SALIR", etc.
            raw_input_text = input(f"\nSena a grabar ('{EXIT_COMMAND}' para salir): ")
            candidate = raw_input_text.strip()
            if candidate.lower() == EXIT_COMMAND:
                break
            if not candidate:
                print("  (etiqueta vacia, se ignora)")
                continue
            word = candidate.upper()
            if not confirm_letter(word):
                print("  (cancelado, vuelve a escribir la letra)")
                continue
            print(f"Grabando '{word}'.")

            word_dir = OUTPUT_ROOT / word
            if OUTPUT_ROOT.is_dir() and not word_dir.is_dir():
                carpetas = sorted(p.name for p in OUTPUT_ROOT.iterdir() if p.is_dir())
                print(f"  aviso: no existe todavia datos_dinamicas/{word}/ (carpetas actuales: {carpetas}); "
                      f"se creara nueva.")

            existing_n = len(list(word_dir.glob("muestra_*.json"))) if word_dir.exists() else 0
            print("Presiona 'g' para EMPEZAR a grabar y otra vez para TERMINAR.")
            print(f"Con {RECOMMENDED_SAMPLES[0]}-{RECOMMENDED_SAMPLES[1]} repeticiones basta, no grabes de mas.")

            recording = False
            sequence: list[np.ndarray] = []
            raw_frames: list[RawFrameSample] = []
            record_start = 0.0
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
                    if not recording:
                        # Primera pulsacion: EMPIEZA. Ya no se "mantiene
                        # presionada" ni se espera un timeout para la suelta.
                        recording = True
                        sequence = []
                        raw_frames = []
                        record_start = now
                        print("  grabando... (presiona 'g' de nuevo para terminar)")
                    else:
                        # Segunda pulsacion: TERMINA y pasa a confirmacion.
                        recording = False
                        duracion_s = now - record_start
                        stats = compute_recording_stats(raw_frames, duracion_s)
                        print(
                            f"  grabacion terminada: {stats['n_frames']} frames, "
                            f"{stats['duracion_s']:.2f}s, "
                            f"{stats['pct_sin_mano']:.0f}% de frames sin mano"
                        )

                        demasiado_corta = should_discard_by_default(stats["n_frames"])
                        if demasiado_corta:
                            print(f"  AVISO: demasiado corta (<{MIN_FRAMES_OK} frames), se descartara.")
                            print("  presiona 'f' para forzar el guardado, cualquier otra tecla para descartar.")
                        else:
                            print("  presiona 's' para guardar, 'd' para descartar.")

                        # Mini-bucle de confirmacion: sigue mostrando camara en
                        # vivo (sin acumular mas frames a la secuencia) hasta
                        # que se decida que hacer con la grabacion.
                        decidido = False
                        guardar = False
                        while not decidido:
                            ret2, frame2 = cap.read()
                            if not ret2:
                                guardar = False
                                break
                            frame2 = cv2.flip(frame2, 1)
                            aviso = (
                                f"Confirmar '{word}': {stats['n_frames']} frames, "
                                f"{stats['duracion_s']:.1f}s"
                            )
                            accion = (
                                "'f'=forzar guardar, otra tecla=descartar" if demasiado_corta
                                else "'s'=guardar  'd'=descartar"
                            )
                            cv2.putText(frame2, aviso, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.putText(frame2, aviso, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 165, 255), 1, cv2.LINE_AA)
                            cv2.putText(frame2, accion, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                        (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.imshow(window, frame2)

                            key2 = cv2.waitKey(1) & 0xFF
                            if demasiado_corta:
                                if key2 == CONFIRM_FORCE_KEY:
                                    guardar, decidido = True, True
                                elif key2 != 255:  # cualquier OTRA tecla real descarta
                                    guardar, decidido = False, True
                            else:
                                if key2 == CONFIRM_SAVE_KEY:
                                    guardar, decidido = True, True
                                elif key2 == CONFIRM_DISCARD_KEY:
                                    guardar, decidido = False, True
                                elif key2 == 27:  # ESC tambien descarta y sale del todo
                                    guardar, decidido = False, True
                                    quit_all = True

                        if guardar:
                            fps_medido = stats["n_frames"] / stats["duracion_s"] if stats["duracion_s"] > 0 else 0.0
                            json_path, npz_path = save_muestra(
                                word_dir, word, sequence, raw_frames, fps_medido,
                            )
                            existing_n += 1
                            print(f"  guardada {json_path.name} + {npz_path.name} "
                                  f"({stats['n_frames']} frames) - muestras de '{word}': {existing_n}")
                            if existing_n >= RECOMMENDED_SAMPLES[1]:
                                print(f"  ya tienes {existing_n}, con eso basta para '{word}'")
                        else:
                            print("  descartada.")

                        sequence = []
                        raw_frames = []
                        if quit_all:
                            break

                if recording:
                    sequence.append(build_feature_vector(hands))
                    raw_frames.append(RawFrameSample(
                        timestamp_ms=(now - record_start) * 1000.0,
                        hands=capture_raw_hands(results),
                    ))

                display = frame.copy()
                for hand in hands.values():
                    draw_hand_landmarks(display, hand)

                if recording:
                    elapsed = now - record_start
                    status = f"Sena: {word}  |  GRABANDO  |  frames: {len(sequence)}  |  {elapsed:.1f}s"
                    color = (0, 0, 255)
                else:
                    status = f"Sena: {word}  |  muestras: {existing_n}  |  listo"
                    color = (102, 255, 102)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(display, "'g'=empezar/terminar  n=nueva sena  ESC=salir", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(window, display)

                if key == ord('n') and not recording:
                    break
                # ESC. 'q' ya no sale: es una etiqueta valida (letra Q).
                if key == 27 and not recording:
                    quit_all = True
                    break

            if quit_all:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print(f"\nListo. Datos guardados en {OUTPUT_ROOT} y {RAW_DATASET_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
