"""Graba muestras del guante (ESP32) en datos_guante/dataset_guante.jsonl.

Cada muestra son 2 s de lecturas (cuenta atras 2, 1, ¡ya!). El traductor
(senas.py, boton 🧤 Guante) reconoce las señas contra este archivo.

    python grabar_guante.py
    python grabar_guante.py --ip 127.0.0.1          # contra simular_guante.py
    python grabar_guante.py --probar                # reconocer sin grabar
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

try:
    import readline  # noqa: F401  (flechas en input() sin meter "\x1b[D" en la etiqueta)
except ImportError:
    pass

import numpy as np

from guante import (DEFAULT_DATASET, GLOVE_IP, GLOVE_PORT, MIN_FRAMES, WINDOW_S,
                    GloveClassifier, GloveReceiver, clean_label, load_dataset)

COUNTDOWN = 2


def countdown(n: int = COUNTDOWN) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    print(" ¡YA!", flush=True)


def capture(rec: GloveReceiver) -> np.ndarray:
    t0 = time.time()
    time.sleep(WINDOW_S)
    return rec.frames_since(t0)


def count_samples(path: Path) -> Counter:
    return Counter(load_dataset(path)[1]) if path.exists() else Counter()


def show(counts: Counter) -> None:
    if not counts:
        print("  (todavía no hay muestras)")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}" + ("   <-- pocas, graba más" if n < 3 else ""))


def remove_last_line(path: Path) -> None:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines(keepends=True) if l.strip()]
    path.write_text("".join(lines[:-1]), encoding="utf-8")


def wait_for_glove(rec: GloveReceiver) -> bool:
    print("Esperando datos del guante...")
    t0 = time.time()
    while not rec.connected():
        if time.time() - t0 > 8:
            print("No llegan datos. Revisa: red GUANTE_LSM (sudo nmcli device wifi connect GUANTE_LSM "
                  "password lsm12345), ESP encendida y que no haya otro programa usando el guante.")
            return False
        time.sleep(0.2)
    print(f"Guante conectado ({rec.rate_hz():.0f} lecturas/s).\n")
    return True


def record(rec: GloveReceiver, path: Path) -> None:
    person = input("Nombre de quien graba (ej. cesar): ").strip().lower() or "anonimo"
    if not wait_for_glove(rec):
        return
    counts = count_samples(path)
    print("Muestras guardadas hasta ahora:")
    show(counts)
    last = None
    last_saved = None
    while True:
        txt = input(f"\nEtiqueta (Enter = repetir '{last}' | ver | borrar | salir): ").strip()
        if txt.lower() == "salir":
            break
        if txt.lower() == "ver":
            show(counts)
            continue
        if txt.lower() == "borrar":
            if last_saved is None:
                print("  No hay nada que borrar en esta sesión.")
            else:
                remove_last_line(path)
                counts[last_saved] -= 1
                print(f"  Borrada la última muestra de '{last_saved}'.")
                last_saved = None
            continue
        label = clean_label(txt) if txt else last
        if not label:
            print("  Escribe una etiqueta primero (ej. A, HOLA, POR FAVOR).")
            continue
        last = label
        print(f"Grabando '{label}'. Estáticas: ten la seña lista desde el 1. Con movimiento: empieza al ¡YA!")
        countdown()
        frames = capture(rec)
        if len(frames) < MIN_FRAMES:
            print(f"  Solo llegaron {len(frames)} lecturas (mínimo {MIN_FRAMES}). Descartada.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"persona": person, "etiqueta": label,
                                "frames": np.round(frames, 2).tolist()}) + "\n")
        counts[label] += 1
        last_saved = label
        print(f"  Guardada ({len(frames)} lecturas). Total de '{label}': {counts[label]}")


def test(rec: GloveReceiver, path: Path) -> None:
    clf = GloveClassifier.from_file(path)
    ok, total = clf.leave_one_out()
    print(f"{total} muestras de {len(clf.labels)} señas; dejando una fuera acierta {ok}/{total}. "
          f"Umbral de distancia: {clf.max_distance:.2f}")
    if not wait_for_glove(rec):
        return
    while input("\nEnter = detectar seña | salir: ").strip().lower() != "salir":
        countdown()
        res = clf.classify(capture(rec))
        print("  Resultado:" + ("" if res.accepted else f"  (no se escribiría: {res.reason})"))
        for i, (label, p) in enumerate(res.topk, 1):
            print(f"   {i}. {label:<12} {p * 100:5.1f}%")
        print(f"   distancia {res.distance:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Grabar o probar señas del guante")
    p.add_argument("--ip", default=GLOVE_IP, help=f"IP de la ESP32 (por defecto {GLOVE_IP})")
    p.add_argument("--port", type=int, default=GLOVE_PORT)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--probar", action="store_true", help="reconocer señas en vez de grabar")
    args = p.parse_args()

    rec = GloveReceiver(args.ip, args.port)
    rec.start()
    try:
        (test if args.probar else record)(rec, args.dataset)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rec.stop()
        print("\nSesión terminada.")


if __name__ == "__main__":
    main()
