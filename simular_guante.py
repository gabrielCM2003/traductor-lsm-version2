"""ESP32 falsa para probar el traductor sin el guante: responde al "hola" en
UDP y manda, a ~21 lecturas por segundo, muestras del dataset en el mismo
formato JSON que el firmware.

    python simular_guante.py                         # repite A, B, C... en ciclo
    python simular_guante.py --senas L,A              # solo esas
    python senas.py --glove-ip 127.0.0.1             # en otra terminal

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

from guante import DEFAULT_DATASET, GLOVE_PORT, SENSOR_NAMES, VALUES_PER_SENSOR, load_dataset

RATE_HZ = 21.0


def packet(vec: np.ndarray) -> bytes:
    v = np.asarray(vec, dtype=float).reshape(len(SENSOR_NAMES), VALUES_PER_SENSOR)
    d = {name: [round(float(x), 2) for x in row] for name, row in zip(SENSOR_NAMES, v)}
    d["err"] = 0
    return json.dumps(d).encode()


def main() -> None:
    p = argparse.ArgumentParser(description="ESP32 falsa del guante")
    p.add_argument("--port", type=int, default=GLOVE_PORT)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--senas", default="", help="señas a repetir, separadas por coma")
    p.add_argument("--segundos", type=float, default=4.0)
    args = p.parse_args()

    frames, labels, _ = load_dataset(args.dataset)
    wanted = [s.strip().upper() for s in args.senas.split(",") if s.strip()] or sorted(set(labels))
    samples = {s: [f for f, l in zip(frames, labels) if l == s] for s in wanted}
    samples = {s: v for s, v in samples.items() if v}
    if not samples:
        raise SystemExit("Ninguna de esas señas está en el dataset")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    client = None
    print(f"ESP32 falsa en el puerto {args.port}; señas: {', '.join(samples)}. Ctrl+C para salir.")

    def send(vec) -> None:
        nonlocal client
        try:
            while True:
                _, addr = sock.recvfrom(64)
                if client != addr:
                    print(f"hola de {addr}")
                client = addr
        except BlockingIOError:
            pass
        if client:
            sock.sendto(packet(vec), client)
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
                        [0.3, 0.3, 0.3, 80, 80, 80, 25, 25], len(SENSOR_NAMES)))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
