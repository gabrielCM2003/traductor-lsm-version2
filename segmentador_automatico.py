"""Segmentador automatico de senas dinamicas: detecta solo, sin teclas, cuando
empieza y termina una sena, y la reconoce o la graba como muestra.

Corre la camara, MediaPipe (manos + cuerpo) y una maquina de estados simple
(AutoSegmenter, abajo). Dos modos de corte:

  --modo letras (por defecto): la sena dura mientras haya una mano en cuadro.
      Es el MISMO corte con el que se hicieron las plantillas del alfabeto
      dinamico (procesar_dataset_dinamico.py recorta cada video del CICESE del
      primer al ultimo frame con mano), para que lo que se captura en vivo se
      parezca a ellas. Mismos umbrales que antes.
  --modo palabras: la sena dura mientras alguna mano este por ENCIMA de la
      linea de reposo (MediaPipe Pose, ver rest_line_y en body_tracker.py) y
      termina al bajar las manos, sin tener que sacarlas de cuadro. Asi se
      pueden hacer palabras seguidas, y no entra a la sena el tramo de subir
      las manos desde el regazo. Si no se ven los hombros, vuelve al criterio
      de letras (hay mano o no).

Sin --grabar, cada sena se reconoce con DTWRecognizer (letras: plantillas de
datos_dinamicas/; palabras: de datos_palabras_dinamicas/). Con --grabar
ETIQUETA, cada sena valida se guarda como muestra, con el mismo formato que
recolector_dinamico.py (126 de manos en "frames", 9 de ubicacion en
"body_frames", crudos en .npz), en datos_dinamicas/<ETIQUETA> (letras) o en
datos_palabras_dinamicas/<ETIQUETA> (palabras). Las palabras van en otra
carpeta porque DTWRecognizer toma cada subcarpeta como una clase: mezcladas
con las letras, el modo dinamico de senas.py empezaria a confundirlas. Grabar
con el mismo corte automatico que se usa al reconocer hace que las muestras y
las senas en vivo se corten igual.

El DTW sigue comparando solo los 126 valores de las manos: las plantillas de
letras (CICESE) no tienen cuerpo, y para palabras el peso de la ubicacion se
podra ajustar cuando haya vocabulario grabado. Por eso se guarda desde ahora.

Los umbrales se miden en tiempo real (ms), no en conteo de frames: un umbral
en frames representa distinto tiempo real segun que tan rapido procese cada
dispositivo (ej. laptop de desarrollo vs. Raspberry Pi 5 de despliegue), lo
que haria que la segmentacion se sintiera mas agresiva o mas lenta con solo
cambiar de maquina sin haber tocado nada.

Uso:
    python segmentador_automatico.py                              # letras, reconoce
    python segmentador_automatico.py --modo palabras              # palabras, reconoce
    python segmentador_automatico.py --modo palabras --grabar HOLA
    python segmentador_automatico.py --grabar J                   # graba la letra J
    En la ventana: ESC sale; 'd' descarta la ultima muestra grabada (se mueve
    a datos_descartados/, no se borra).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import mediapipe as mp

# recolector_dinamico, dtw_recognizer, body_tracker y senas se importan dentro
# de main(), no aqui arriba: recolector_dinamico.py ya importa cosas de
# senas.py, y senas.py importa AutoSegmenter de este archivo para el modo
# dinamico, asi que un import a nivel de modulo aqui crearia un ciclo (senas
# -> segmentador automatico -> recolector_dinamico -> senas). AutoSegmenter en
# si no depende de ninguno, solo main() los necesita para correr standalone.

# Cuanto tiempo SEGUIDO sin manos (en ms de reloj real) se necesita para dar
# por terminada la sena. Punto de partida equivalente al umbral original de
# 8 frames a 30fps (8/30*1000 ≈ 267ms). Ver nota de calibracion al final.
NO_HAND_MS_TO_END = 270

# Senas mas cortas que esto (duracion real desde que aparece la mano hasta el
# ultimo frame en que se detecto) se descartan como ruido/falso positivo en
# vez de mandarlas a clasificar. Equivalente al umbral original de 5 frames a
# 30fps (5/30*1000 ≈ 167ms).
MIN_SEQUENCE_MS = 170

# Modo palabras: tiempo SEGUIDO con las manos en reposo (bajo la linea) o
# fuera de cuadro para terminar la sena. Mas que en letras, porque dentro de
# una palabra una mano puede bajar un instante sin que la sena haya terminado.
PALABRAS_REST_MS_TO_END = 400
# Una palabra dura mas que una letra: menos que esto se toma como ruido.
PALABRAS_MIN_SEQUENCE_MS = 300
# Tope por si las manos nunca bajan (igual idea que DYN_MAX_SEQUENCE_MS).
PALABRAS_MAX_SEQUENCE_MS = 8000

PROJECT_DIR = Path(__file__).resolve().parent
LETRAS_DIR = PROJECT_DIR / "datos_dinamicas"
PALABRAS_DIR = PROJECT_DIR / "datos_palabras_dinamicas"
# 'd' mueve aqui la ultima muestra grabada (no se borra: se puede recuperar).
DESCARTADAS_DIR = PROJECT_DIR / "datos_descartados"


class AutoSegmenter:
    """Maquina de estados pura (sin camara), para poder probarla sin hardware.

        esperando -> (hay actividad) -> grabando
        grabando  -> (sin actividad no_hand_ms_to_end ms seguidos, en tiempo
                     real, o se alcanzo max_duration_ms) -> esperando
                     + entrega el buffer (recortando la cola sin actividad que
                       disparo el fin), o lo descarta si duro menos de
                       min_sequence_ms.

    push(has_hand, vector, now) se llama una vez por frame:
      has_hand: si en este frame hay actividad. En senas.py y en el modo letras,
          que haya una mano; en el modo palabras, que alguna mano este por
          encima de la linea de reposo.
      vector: lo que se quiera acumular por frame (senas.py pasa el vector de
          126; main(), un FrameCapture con manos, cuerpo y datos crudos).
      now: timestamp en segundos (ej. time.perf_counter()) provisto por el
          llamador en vez de leido internamente, para que se pueda probar con
          timestamps sinteticos sin depender de un reloj real.
    Devuelve:
      None                          -> nada que reportar todavia
      ("inicio", [])                -> se acaba de detectar el arranque de una sena
      ("fin_valida", secuencia)     -> sena terminada, lista para clasificar
      ("fin_descartada", secuencia) -> sena terminada pero demasiado corta

    max_duration_ms (opcional, None por defecto = sin tope, igual que antes):
    corta la grabacion por la fuerza si se excede, aunque la mano siga
    presente. Lo usa senas.py en su integracion con la GUI (constante
    DYN_MAX_SEQUENCE_MS) y el modo palabras de main() (PALABRAS_MAX_SEQUENCE_MS),
    para no grabar indefinidamente si el usuario no baja la mano.
    """

    def __init__(
        self,
        no_hand_ms_to_end: float = NO_HAND_MS_TO_END,
        min_sequence_ms: float = MIN_SEQUENCE_MS,
        max_duration_ms: Optional[float] = None,
    ):
        self.no_hand_ms_to_end = no_hand_ms_to_end
        self.min_sequence_ms = min_sequence_ms
        self.max_duration_ms = max_duration_ms
        self.state = "esperando"
        self.buffer: list[Any] = []
        self.last_hand_idx = -1
        self.start_time = 0.0
        self.last_hand_time = 0.0

    def push(self, has_hand: bool, vector: Any, now: float) -> Optional[tuple[str, list[Any]]]:
        if self.state == "esperando":
            if not has_hand:
                return None
            self.state = "grabando"
            self.buffer = [vector]
            self.last_hand_idx = 0
            self.start_time = now
            self.last_hand_time = now
            return ("inicio", [])

        # state == "grabando"
        self.buffer.append(vector)
        if has_hand:
            self.last_hand_idx = len(self.buffer) - 1
            self.last_hand_time = now

        elapsed_since_hand_ms = (now - self.last_hand_time) * 1000.0
        duration_so_far_ms = (now - self.start_time) * 1000.0
        tope_alcanzado = (
            self.max_duration_ms is not None and duration_so_far_ms >= self.max_duration_ms
        )
        if elapsed_since_hand_ms < self.no_hand_ms_to_end and not tope_alcanzado:
            return None

        # Se acumulan frames vacios mientras se decide si la sena termino;
        # se recortan antes de clasificar para no diluir la secuencia con
        # ceros (igual que hace procesar_dataset_dinamico.py con los videos).
        trimmed = self.buffer[: self.last_hand_idx + 1]
        duration_ms = (self.last_hand_time - self.start_time) * 1000.0

        self.state = "esperando"
        self.buffer = []
        self.last_hand_idx = -1

        if duration_ms < self.min_sequence_ms:
            return ("fin_descartada", trimmed)
        return ("fin_valida", trimmed)


@dataclass
class FrameCapture:
    """Lo que main() acumula por frame: el vector de 126 que va al DTW, los 9
    de ubicacion respecto al cuerpo y lo crudo para el .npz de la muestra."""
    t: float                 # time.perf_counter() del frame
    hands_vec: np.ndarray    # 126
    body_vec: np.ndarray     # 9 (ver body_location_features)
    raw_hands: dict          # recolector_dinamico.capture_raw_hands
    body: Any                # BodyDetection o None


def load_recognizer(palabras: bool):
    """DTWRecognizer del modo, o None si no hay plantillas."""
    from dtw_recognizer import DTWRecognizer

    if not palabras:
        return DTWRecognizer.try_load()
    if not PALABRAS_DIR.is_dir():
        return None
    # Directo y sin auto_save_labels: try_load reescribiria labels_dinamicas.json
    # (la lista de LETRAS que usa senas.py) con las palabras.
    try:
        recognizer = DTWRecognizer(data_dir=PALABRAS_DIR, auto_save_labels=False)
    except Exception as e:
        print(f"No se pudo cargar el reconocedor de palabras: {e}", file=sys.stderr)
        return None
    return recognizer if recognizer.labels else None


def move_to_discarded(paths: list[Path], label: str) -> Path:
    """Mueve los archivos de una muestra a datos_descartados/<ETIQUETA>/, con la
    hora en el nombre para que no choquen (el indice N se reusa al grabar la
    siguiente). Devuelve la carpeta destino."""
    target_dir = DESCARTADAS_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in paths:
        if path.exists():
            shutil.move(str(path), str(target_dir / f"{path.stem}_{stamp}{path.suffix}"))
    return target_dir


def put_text(image: np.ndarray, text: str, y: int, color: tuple[int, int, int], scale: float = 0.6) -> None:
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> int:
    import recolector_dinamico as rd
    from body_tracker import (
        BodyTracker, body_location_features, draw_body_skeleton, draw_body_status,
        draw_rest_line, hands_in_signing_space,
    )
    from senas import HandDistanceWarning, draw_hand_landmarks, draw_warning

    parser = argparse.ArgumentParser(
        description="Segmentador automatico de senas dinamicas (sin teclas): reconoce o graba"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--k", type=int, default=3, help="cuantos candidatos mostrar por prediccion")
    parser.add_argument(
        "--modo", choices=("letras", "palabras"), default="letras",
        help="letras: la sena dura mientras haya mano (corte de las plantillas CICESE). "
             "palabras: dura mientras una mano este sobre la linea de reposo (pose)",
    )
    parser.add_argument(
        "--grabar", metavar="ETIQUETA",
        help="guardar cada sena detectada como muestra de ETIQUETA (ej. HOLA o J) en vez de reconocerla",
    )
    args = parser.parse_args()

    palabras = args.modo == "palabras"
    data_dir = PALABRAS_DIR if palabras else LETRAS_DIR
    label: Optional[str] = None
    if args.grabar is not None:
        label = args.grabar.strip().upper()
        if not label:
            print("ERROR: --grabar necesita una etiqueta, ej. --grabar HOLA", file=sys.stderr)
            return 1

    recognizer = None
    n_saved = 0
    if label is None:
        print("Cargando reconocedor DTW...")
        recognizer = load_recognizer(palabras)
        if recognizer is None:
            if palabras:
                print("ERROR: no hay plantillas de palabras en datos_palabras_dinamicas/. Grabalas "
                      "primero con: python segmentador_automatico.py --modo palabras --grabar ETIQUETA",
                      file=sys.stderr)
            else:
                print("ERROR: no hay plantillas en datos_dinamicas/. Corre recolector_dinamico.py "
                      "o procesar_dataset_dinamico.py primero.", file=sys.stderr)
            return 1
        print(f"Senas disponibles ({len(recognizer.labels)}): {recognizer.labels}")
    else:
        label_dir = data_dir / label
        n_saved = len(list(label_dir.glob("muestra_*.json"))) if label_dir.is_dir() else 0
        print(f"Grabando '{label}' en {data_dir.name}/{label}/ (ya hay {n_saved} muestras). "
              f"Cada sena detectada se guarda sola; 'd' descarta la ultima.")

    landmarker = rd.init_hand_landmarker(max_num_hands=2)
    body_tracker = BodyTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {args.camera}", file=sys.stderr)
        landmarker.close()
        body_tracker.close()
        return 1

    if palabras:
        segmenter = AutoSegmenter(
            no_hand_ms_to_end=PALABRAS_REST_MS_TO_END,
            min_sequence_ms=PALABRAS_MIN_SEQUENCE_MS,
            max_duration_ms=PALABRAS_MAX_SEQUENCE_MS,
        )
        print(f"\nLa sena empieza al subir una mano sobre la linea de reposo y termina con las "
              f"manos {PALABRAS_REST_MS_TO_END} ms abajo (o fuera de cuadro).")
    else:
        segmenter = AutoSegmenter()
        print(f"\nUmbral de fin de sena: {NO_HAND_MS_TO_END} ms sin manos seguidos (tiempo real).")
    print(f"Senas de menos de {segmenter.min_sequence_ms:.0f} ms de duracion se descartan como ruido.")
    print("ESC en la ventana para salir.\n")

    window = f"Segmentador ({args.modo}) - LSM"
    cv2.namedWindow(window)
    distance_warning = HandDistanceWarning()
    timestamp_ms = 0  # timestamp entero que exige la API de video de MediaPipe (solo manos)
    status = "Esperando: sube la mano" if palabras else "Esperando: muestra la mano"
    last_saved: Optional[list[Path]] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: fallo al leer de la camara", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)  # mismo mirror que el resto del proyecto
            frame_h, frame_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 1
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands = rd.parse_hands(results)
            body = body_tracker.detect(mp_image)   # tiempo real, no timestamp_ms (ver BodyTracker)
            now = time.perf_counter()

            active = bool(hands)
            if palabras:
                in_space = hands_in_signing_space(hands.values(), body, frame_w, frame_h)
                if in_space is not None:
                    active = in_space

            capture = FrameCapture(
                t=now,
                hands_vec=rd.build_feature_vector(hands),
                body_vec=body_location_features(hands, body, frame_w, frame_h),
                raw_hands=rd.capture_raw_hands(results),
                body=body,
            )
            event = segmenter.push(active, capture, now)

            if event is not None:
                kind, sequence = event
                if kind == "inicio":
                    print("[inicio] grabando...")
                elif kind == "fin_descartada":
                    status = "Descartada: muy corta"
                    print(f"[fin] descartada: sena muy corta "
                          f"(<{segmenter.min_sequence_ms:.0f}ms), probablemente ruido ({len(sequence)} frames)")
                elif label is not None:
                    if len(sequence) < rd.MIN_FRAMES_OK:
                        status = f"Muy corta ({len(sequence)} frames), no se guardo"
                        print(f"[fin] {status} (minimo {rd.MIN_FRAMES_OK} frames)")
                    else:
                        t0 = sequence[0].t
                        duration_s = sequence[-1].t - t0
                        raw_frames = [
                            rd.RawFrameSample(timestamp_ms=(c.t - t0) * 1000.0, hands=c.raw_hands, body=c.body)
                            for c in sequence
                        ]
                        json_path, npz_path = rd.save_muestra(
                            data_dir / label, label,
                            [c.hands_vec for c in sequence], raw_frames,
                            len(sequence) / duration_s if duration_s > 0 else 0.0,
                            body_sequence=[c.body_vec for c in sequence],
                        )
                        last_saved = [json_path, npz_path]
                        n_saved += 1
                        status = f"Guardada {json_path.name} ({len(sequence)} frames)"
                        print(f"[fin] {status} - muestras de '{label}': {n_saved}")
                else:
                    print(f"[fin] sena de {len(sequence)} frames, clasificando...")
                    try:
                        topk = recognizer.predict_topk([c.hands_vec for c in sequence], k=args.k)
                    except Exception as e:
                        status = "Error al clasificar"
                        print(f"  ERROR al clasificar: {e}")
                    else:
                        best_word, best_conf = topk[0]
                        status = f"{best_word} ({best_conf * 100:.0f}%)"
                        print(f"  -> {best_word} ({best_conf * 100:.1f}%)")
                        for rank, (word, conf) in enumerate(topk[1:], start=2):
                            print(f"     {rank}. {word} ({conf * 100:.1f}%)")

            display = frame.copy()
            if body is not None:
                draw_body_skeleton(display, body, hands.values())
            if palabras:
                draw_rest_line(display, body)
            for hand in hands.values():
                draw_hand_landmarks(display, hand)

            what = f"grabando '{label}': {n_saved} muestras" if label else "reconociendo"
            put_text(display, f"Modo {args.modo}  |  {what}", 30, (102, 255, 102))
            if segmenter.state == "grabando":
                put_text(display, "GRABANDO...", 60, (0, 0, 255))
            else:
                put_text(display, status, 60, (255, 255, 255))
            draw_body_status(display, bool(capture.body_vec[-1]))
            put_text(display, "ESC=salir" + ("  d=descartar la ultima" if label else ""), 115, (200, 200, 200), 0.5)
            warning = distance_warning.update(hands.values(), frame_w, frame_h, time.monotonic())
            if warning is not None:
                draw_warning(display, warning)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('d') and label is not None and last_saved is not None:
                target = move_to_discarded(last_saved, label)
                n_saved -= 1
                status = f"Descartada la ultima (en {target.parent.name}/{target.name}/)"
                print(f"  {status}")
                last_saved = None
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        body_tracker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
