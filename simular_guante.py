"""ESP32 falsa para probar el traductor sin el guante: como el firmware, manda
sin handshake, a ~25 lecturas por segundo, muestras del dataset en el mismo
formato JSON ("h": "D" o "I") al puerto 4210 de este equipo. No abre el puerto
4210 (ese es solo del receptor de guante.py).

    python senas.py                                  # en otra terminal
    python simular_guante.py                         # repite A, B, C... en ciclo
    python simular_guante.py --senas L,A              # solo esas
    python simular_guante.py --mano I                # guante izquierdo ("h": "I")
    python simular_guante.py --mano DI               # frases: los dos guantes a la vez

Para letras de los dos guantes a la vez, un simulador por mano (en dos
terminales); --mano DI manda las frases del dataset de dos manos.

Cada seña se sostiene --segundos (4 por defecto) y luego hay 1.5 s de ruido
(mano en movimiento).
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import time
from pathlib import Path

import numpy as np

from guante import (GLOVE_BOTH, GLOVE_HAND, GLOVE_HANDS, GLOVE_PORT, HAND_NAMES, N_VALUES, SENSOR_NAMES,
                    VALUES_PER_SENSOR, default_dataset, load_dataset, parse_glove)

RATE_HZ = 25.0


def packet(vec: np.ndarray, n: int = 0, t_ms: int = 0, hand: str = GLOVE_HAND) -> bytes:
    v = np.asarray(vec, dtype=float).reshape(len(SENSOR_NAMES), VALUES_PER_SENSOR)
    d = {"n": n, "t": t_ms, "h": hand, "err": 0}
    d.update({name: [round(float(x), 2) for x in row] for name, row in zip(SENSOR_NAMES, v)})
    return json.dumps(d).encode()


def main() -> None:
    p = argparse.ArgumentParser(description="ESP32 falsa del guante")
    p.add_argument("--destino", default="127.0.0.1", help="IP del receptor (por defecto este equipo)")
    p.add_argument("--port", type=int, default=GLOVE_PORT)
    p.add_argument("--mano", type=parse_glove, default=GLOVE_HAND,
                   help="D (derecho), I (izquierdo) o DI (frases, los dos)")
    p.add_argument("--dataset", type=Path, help="por defecto el de esa mano en datos_guante/")
    p.add_argument("--senas", default="", help="señas a repetir, separadas por coma")
    p.add_argument("--segundos", type=float, default=4.0)
    args = p.parse_args()

    dataset = args.dataset or default_dataset(args.mano)
    if not dataset.exists():
        raise SystemExit(f"No existe {dataset}: graba el guante {HAND_NAMES[args.mano]} con "
                         f"python grabar_guante.py --mano {args.mano}")
    frames, labels, _ = load_dataset(dataset, 2 * N_VALUES if args.mano == GLOVE_BOTH else N_VALUES)
    wanted = [s.strip().upper() for s in args.senas.split(",") if s.strip()] or sorted(set(labels))
    samples = {s: [f for f, l in zip(frames, labels) if l == s] for s in wanted}
    samples = {s: v for s, v in samples.items() if v}
    if not samples:
        raise SystemExit("Ninguna de esas señas está en el dataset")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.destino, args.port)
    print(f"ESP32 falsa (guante {HAND_NAMES[args.mano]}, \"h\":\"{args.mano}\") -> {dest[0]}:{dest[1]}; señas: {', '.join(samples)}. Ctrl+C para salir.")
    t_start = time.time()
    n = 0

    def send(vec) -> None:
        nonlocal n
        # DI: 96 valores = derecho + izquierdo, un paquete de cada mano.
        hands = GLOVE_HANDS if args.mano == GLOVE_BOTH else (args.mano,)
        for k, hand in enumerate(hands):
            try:
                sock.sendto(packet(vec[k * N_VALUES:(k + 1) * N_VALUES], n,
                                   int((time.time() - t_start) * 1000), hand), dest)
            except OSError:
                pass        # como la ESP: si no hay a quien mandar, se sigue
        n += 1
        time.sleep(1 / RATE_HZ)

    try:
        while True:
            for sign, options in samples.items():
                sample = random.choice(options)
                print(f"-> {sign}")
                t0 = time.time()
                i = 0
                while time.time() - t0 < args.segundos:
                    send(sample[i % len(sample)])
                    i += 1
                # transicion: de la ultima lectura a ruido, como una mano moviendose
                t0 = time.time()
                while time.time() - t0 < 1.5:
                    send(sample[-1] + np.random.normal(0, 1, sample.shape[1]) * np.tile(
                        [0.3, 0.3, 0.3, 80, 80, 80, 25, 25], sample.shape[1] // VALUES_PER_SENSOR))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
