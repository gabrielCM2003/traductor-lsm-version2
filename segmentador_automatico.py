"""Segmentador automatico de senas dinamicas: sin teclas, solo consola.

Corre la camara y MediaPipe en vivo y usa una maquina de estados simple para
detectar solo, sin que el usuario presione nada, cuando empieza y termina una
seña, para despues clasificarla con DTWRecognizer (dtw_recognizer.py).

Reutiliza directamente init_hand_landmarker/parse_hands/build_feature_vector
de recolector_dinamico.py (misma deteccion de manos y misma normalizacion de
126 features en todo el proyecto) y DTWRecognizer para la clasificacion.

Esto es solo un componente de prueba por consola, sin interfaz grafica y sin
tocar senas.py: sirve para validar si la segmentacion automatica se siente
natural (que no corte la sena a medias ni tarde de mas en reaccionar) antes
de integrarla a la app.

Maquina de estados:
    esperando -> (aparece una mano) -> grabando
    grabando  -> (manos ausentes NO_HAND_MS_TO_END ms seguidos, en tiempo real)
                 -> esperando
                 + clasifica el buffer acumulado (recortando la cola sin manos
                   que disparo el fin de sena), o lo descarta si la sena duro
                   menos de MIN_SEQUENCE_MS.

Los umbrales se miden en tiempo real (ms), no en conteo de frames: un umbral
en frames representa distinto tiempo real segun que tan rapido procese cada
dispositivo (ej. laptop de desarrollo vs. Raspberry Pi 5 de despliegue), lo
que haria que la segmentacion se sintiera mas agresiva o mas lenta con solo
cambiar de maquina sin haber tocado nada.

Uso:
    python segmentador_automatico.py [--camera 0] [--k 3]
    Ctrl+C para salir.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# recolector_dinamico y dtw_recognizer se importan dentro de main(), no aqui
# arriba: recolector_dinamico.py ya importa cosas de senas.py, y senas.py
# importa AutoSegmenter de este archivo para el modo dinamico, asi que un
# import a nivel de modulo aqui crearia un ciclo (senas -> segmentador
# automatico -> recolector_dinamico -> senas). AutoSegmenter en si no depende
# de ninguno de los dos, solo main() los necesita para correr standalone.

# Cuanto tiempo SEGUIDO sin manos (en ms de reloj real) se necesita para dar
# por terminada la sena. Punto de partida equivalente al umbral original de
# 8 frames a 30fps (8/30*1000 ≈ 267ms). Ver nota de calibracion al final.
NO_HAND_MS_TO_END = 270

# Senas mas cortas que esto (duracion real desde que aparece la mano hasta el
# ultimo frame en que se detecto) se descartan como ruido/falso positivo en
# vez de mandarlas a clasificar. Equivalente al umbral original de 5 frames a
# 30fps (5/30*1000 ≈ 167ms).
MIN_SEQUENCE_MS = 170


class AutoSegmenter:
    """Maquina de estados pura (sin camara), para poder probarla sin hardware.

    push(has_hand, vector, now) se llama una vez por frame, con `now` un
    timestamp en segundos (ej. time.perf_counter()) provisto por el llamador
    en vez de leido internamente, para que se pueda probar con timestamps
    sinteticos sin depender de un reloj real. Devuelve:
      None                          -> nada que reportar todavia
      ("inicio", [])                -> se acaba de detectar el arranque de una sena
      ("fin_valida", secuencia)     -> sena terminada, lista para clasificar
      ("fin_descartada", secuencia) -> sena terminada pero demasiado corta

    max_duration_ms (opcional, None por defecto = sin tope, igual que antes):
    corta la grabacion por la fuerza si se excede, aunque la mano siga
    presente. Lo usa senas.py en su integracion con la GUI (constante
    DYN_MAX_SEQUENCE_MS) para no grabar indefinidamente si el usuario no baja
    la mano; el modo consola de este archivo (main(), mas abajo) no lo pasa,
    asi que su comportamiento no cambia.
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
        self.buffer: list[np.ndarray] = []
        self.last_hand_idx = -1
        self.start_time = 0.0
        self.last_hand_time = 0.0

    def push(self, has_hand: bool, vector: np.ndarray, now: float) -> Optional[tuple[str, list[np.ndarray]]]:
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


def main() -> int:
    import recolector_dinamico as rd
    from dtw_recognizer import DTWRecognizer

    parser = argparse.ArgumentParser(
        description="Segmentador automatico de senas dinamicas (sin teclas, solo consola)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--k", type=int, default=3, help="cuantos candidatos mostrar por prediccion")
    args = parser.parse_args()

    print("Cargando reconocedor DTW...")
    recognizer = DTWRecognizer.try_load()
    if recognizer is None:
        print("ERROR: no hay plantillas en datos_dinamicas/. Corre recolector_dinamico.py "
              "o procesar_dataset_dinamico.py primero.", file=sys.stderr)
        return 1
    print(f"Senas disponibles ({len(recognizer.labels)}): {recognizer.labels}")

    landmarker = rd.init_hand_landmarker(max_num_hands=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {args.camera}", file=sys.stderr)
        landmarker.close()
        return 1

    segmenter = AutoSegmenter()
    timestamp_ms = 0  # timestamp entero que exige la API de video de MediaPipe

    print(f"\nUmbral de fin de sena: {NO_HAND_MS_TO_END} ms sin manos seguidos (tiempo real).")
    print(f"Senas de menos de {MIN_SEQUENCE_MS} ms de duracion se descartan como ruido.")
    print("Muestra las manos frente a la camara para empezar. Ctrl+C para salir.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: fallo al leer de la camara", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)  # mismo mirror que el resto del proyecto

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 1
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands = rd.parse_hands(results)
            vector = rd.build_feature_vector(hands)

            event = segmenter.push(bool(hands), vector, time.perf_counter())
            if event is None:
                continue
            kind, sequence = event

            if kind == "inicio":
                print("[inicio] mano detectada, grabando...")
            elif kind == "fin_descartada":
                print(f"[fin] descartada: sena muy corta "
                      f"(<{MIN_SEQUENCE_MS}ms), probablemente ruido ({len(sequence)} frames)")
            elif kind == "fin_valida":
                print(f"[fin] sena de {len(sequence)} frames, clasificando...")
                try:
                    topk = recognizer.predict_topk(sequence, k=args.k)
                except Exception as e:
                    print(f"  ERROR al clasificar: {e}")
                    continue
                best_word, best_conf = topk[0]
                print(f"  -> {best_word} ({best_conf * 100:.1f}%)")
                for rank, (word, conf) in enumerate(topk[1:], start=2):
                    print(f"     {rank}. {word} ({conf * 100:.1f}%)")
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        cap.release()
        landmarker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
