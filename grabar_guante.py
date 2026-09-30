"""Graba muestras del guante (ESP32). Cada mano tiene su dataset, y las
frases de dos manos el suyo:

    derecho   ("h": "D")  datos_guante/dataset_guante.jsonl
    izquierdo ("h": "I")  datos_guante/dataset_guante_izquierdo.jsonl
    las dos   (D + I)     datos_guante/dataset_guante_ambas.jsonl

Cada muestra son 2 s de lecturas (cuenta atras 2, 1, ¡ya!); las frases, 3 s
(--segundos para cambiarlo; todas las de un dataset deben durar lo mismo).
El traductor (senas.py) reconoce las señas de cada guante contra su archivo.

    python grabar_guante.py                         # guante derecho
    python grabar_guante.py --mano I                # guante izquierdo
    python grabar_guante.py --mano DI               # frases con los dos guantes
    python simular_guante.py --mano I               # (otra terminal) ESP32 falsa
    python grabar_guante.py --mano I --probar       # reconocer sin grabar
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

from guante import (GLOVE_BOTH, GLOVE_HANDS, GLOVE_PORT, HAND_NAMES, MIN_FRAMES, PHRASE_WINDOW_S, WINDOW_S,
                    N_VALUES, GloveClassifier, GloveHand, GloveReceiver, clean_label, default_dataset, load_dataset,
                    parse_glove, sample_seconds)

COUNTDOWN = 2


def countdown(n: int = COUNTDOWN) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    print(" ¡YA!", flush=True)


def capture(rec: GloveHand, seconds: float = WINDOW_S) -> np.ndarray:
    t0 = time.time()
    time.sleep(seconds)
    return rec.frames_since(t0)


def width(rec) -> int:
    """Valores por lectura: 48 (un guante) o 96 (los dos)."""
    return 2 * N_VALUES if rec.hand == GLOVE_BOTH else N_VALUES


def count_samples(path: Path, n_values: int = N_VALUES) -> Counter:
    return Counter(load_dataset(path, n_values)[1]) if path.exists() else Counter()


def show(counts: Counter) -> None:
    if not counts:
        print("  (todavía no hay muestras)")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}" + ("   <-- pocas, graba más" if n < 3 else ""))


def remove_last_line(path: Path) -> None:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines(keepends=True) if l.strip()]
    path.write_text("".join(lines[:-1]), encoding="utf-8")


def wait_for_glove(rec: GloveHand) -> bool:
    name = HAND_NAMES[rec.hand]
    print(f"Esperando datos del guante {name}...")
    t0 = time.time()
    while not rec.connected():
        if time.time() - t0 > 8 and rec.hand == GLOVE_BOTH:
            live = rec.receiver.connected_hands()
            missing = [HAND_NAMES[h] for h in GLOVE_HANDS if h not in live]
            print(f"Para frases hacen falta los dos guantes; no llegan datos del {' ni del '.join(missing)}. "
                  "Revisa que cada ESP mande su \"h\" (\"D\" o \"I\") con err 0.")
            for problem in rec.receiver.problems():
                print("  - " + problem)
            return False
        if time.time() - t0 > 8:
            others = [h for h in rec.receiver.connected_hands() if h != rec.hand]
            if others:
                # Llega el otro guante: casi siempre es --mano equivocada.
                print(f"Llegan datos del guante {HAND_NAMES[others[0]]} (\"h\":\"{others[0]}\"), "
                      f"no del {name}. ¿Querías --mano {others[0]}? Si no, revisa que la ESP del "
                      f"guante {name} mande \"h\":\"{rec.hand}\".")
            else:
                print("No llegan datos. Revisa: hotspot GUANTE_LSM activo en esta Raspberry (10.42.0.1, "
                      f"nmcli con up Hotspot), ESP encendida y conectada, y que manda \"h\":\"{rec.hand}\" con err 0.")
            for problem in rec.receiver.problems():
                print("  - " + problem)
            return False
        time.sleep(0.2)
    print(f"Guante {name} conectado ({rec.rate_hz():.0f} lecturas/s).\n")
    return True


def record(rec: GloveHand, path: Path, seconds: float) -> None:
    person = input("Nombre de quien graba (ej. cesar): ").strip().lower() or "anonimo"
    if not wait_for_glove(rec):
        return
    counts = count_samples(path, width(rec))
    print("Muestras guardadas hasta ahora:")
    show(counts)
    if "NADA" not in counts:
        print("Consejo: graba también NADA (6 o más: manos quietas, moviéndose entre señas, deletreando "
              "con una mano). Nunca se escribe y evita que lo que no es seña salga como la seña más parecida."
              + (" Con frases de dos manos es casi obligatorio." if rec.hand == GLOVE_BOTH else ""))
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
            print("  Escribe una etiqueta primero (ej. A, HOLA, POR FAVOR, o NADA: manos en reposo o "
                  "entre señas, nunca se escribe y ayuda a no confundir).")
            continue
        last = label
        print(f"Grabando '{label}' ({seconds:g} s). Estáticas: ten la seña lista desde el 1. "
              "Con movimiento: empieza al ¡YA!")
        countdown()
        frames = capture(rec, seconds)
        if len(frames) < MIN_FRAMES:
            print(f"  Solo llegaron {len(frames)} lecturas (mínimo {MIN_FRAMES}). Descartada.")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"persona": person, "mano": rec.hand, "etiqueta": label, "segundos": seconds,
                                "frames": np.round(frames, 2).tolist()}) + "\n")
        counts[label] += 1
        last_saved = label
        print(f"  Guardada ({len(frames)} lecturas). Total de '{label}': {counts[label]}")


def test(rec: GloveHand, path: Path, seconds: float) -> None:
    clf = GloveClassifier.from_file(path, n_values=width(rec))
    ok, total = clf.leave_one_out()
    print(f"Guante {HAND_NAMES[rec.hand]} ({path.name}): {total} muestras de {len(clf.labels)} señas; dejando una fuera acierta {ok}/{total}. "
          f"Umbral de distancia: {clf.max_distance:.2f}")
    if not wait_for_glove(rec):
        return
    while input("\nEnter = detectar seña | salir: ").strip().lower() != "salir":
        countdown()
        res = clf.classify(capture(rec, clf.window_s))
        print("  Resultado:" + ("" if res.accepted else f"  (no se escribiría: {res.reason})"))
        for i, (label, p) in enumerate(res.topk, 1):
            print(f"   {i}. {label:<12} {p * 100:5.1f}%")
        print(f"   distancia {res.distance:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Grabar o probar señas del guante")
    p.add_argument("--port", type=int, default=GLOVE_PORT, help="puerto UDP donde se escucha al guante")
    p.add_argument("--mano", type=parse_glove, default="D",
                   help="guante a grabar: D (derecho, por defecto), I (izquierdo) o DI (frases con los dos)")
    p.add_argument("--segundos", type=float,
                   help=f"duración de cada muestra (por defecto {WINDOW_S:g} s; frases {PHRASE_WINDOW_S:g} s, "
                        "o lo que ya dure el dataset)")
    p.add_argument("--dataset", type=Path, help="archivo .jsonl (por defecto el de esa mano en datos_guante/)")
    p.add_argument("--probar", action="store_true", help="reconocer señas en vez de grabar")
    args = p.parse_args()
    path = args.dataset or default_dataset(args.mano)
    if args.segundos:
        seconds = args.segundos
    elif path.exists():
        seconds = sample_seconds(path)
    else:
        seconds = PHRASE_WINDOW_S if args.mano == GLOVE_BOTH else WINDOW_S
    if path.exists() and abs(seconds - sample_seconds(path)) > 0.01:
        print(f"Aviso: las muestras de {path.name} duran {sample_seconds(path):g} s y estas {seconds:g} s.")
    print(f"Guante {HAND_NAMES[args.mano]} (\"h\":\"{args.mano}\") -> {path}")
    if args.probar and not path.exists():
        raise SystemExit(f"Todavía no hay muestras del guante {HAND_NAMES[args.mano]} ({path}). "
                         f"Grábalas primero: python grabar_guante.py --mano {args.mano}")

    # Se escuchan las dos manos para avisar si llega la otra (--mano equivocada).
    rec = GloveReceiver(args.port, hands=GLOVE_HANDS)
    rec.start()     # falla si otro programa (senas.py) tiene el puerto
    try:
        (test if args.probar else record)(rec.hand(args.mano), path, seconds)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rec.stop()
        print("\nSesión terminada.")


if __name__ == "__main__":
    main()
