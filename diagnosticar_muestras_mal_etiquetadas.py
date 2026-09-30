"""Diagnóstico de muestras "propias" posiblemente mal etiquetadas en datos_dinamicas/K/.

Contexto: se sospecha que varias muestras grabadas con recolector_dinamico.py,
destinadas a J, K, Q y/o Z, quedaron guardadas por error en datos_dinamicas/K/.

SOLO LECTURA: este script no mueve, no borra ni modifica ningún archivo. No
toca senas.py, dtw_recognizer.py, ni ningún .json existente (ni siquiera
labels_dinamicas.json: se evita explícitamente auto_save_labels/__init__ de
DTWRecognizer para garantizar cero escrituras).

Qué hace:
1. Distingue dos patrones de archivo dentro de cada datos_dinamicas/<letra>/:
     - "dataset" : muestra_<sujeto>_<repeticion>.json  (CICESE, ej. muestra_10_3.json)
     - "propia"  : muestra_<N>.json                    (grabado a mano, ej. muestra_224.json)
   e imprime, para las 6 letras dinámicas, qué números "propios" tiene cada
   una — esto permite confirmar (no asumir) desde qué número exacto las
   muestras de K dejan de tener contraparte en J/Q/Z y se vuelven exclusivas
   de K (indicio de que ahí empezó el error de guardado).
2. Construye un set de plantillas SOLO con archivos "dataset" (de las 6
   letras), excluyendo TODAS las muestras "propias" de TODAS las letras -
   así ninguna muestra propia se compara contra sí misma ni contra otra
   propia, solo contra el dataset CICESE original.
3. Para cada muestra "propia" de datos_dinamicas/K/ (por defecto, desde
   muestra_222.json en adelante - ver --desde), corre
   DTWRecognizer.predict_topk() contra ese set de plantillas y reporta
   top-3 con distancia DTW y confianza (softmax).
4. Imprime una tabla resumen y un conteo agrupado por letra top-1.

Uso:
    python diagnosticar_muestras_mal_etiquetadas.py
    python diagnosticar_muestras_mal_etiquetadas.py --desde 216   # revisar más atrás
    python diagnosticar_muestras_mal_etiquetadas.py --carpeta Q   # revisar otra carpeta
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from dtw_recognizer import DTWRecognizer, N_FEATURES

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"
LETRAS_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]

# "propia": exactamente un número (muestra_224.json). NO confundir con las
# del dataset CICESE, que llevan sujeto_repeticion (muestra_10_3.json).
PROPIA_RE = re.compile(r"^muestra_(\d+)\.json$")
DATASET_RE = re.compile(r"^muestra_\d+_\d+\.json$")


def cargar_frames(path: Path) -> np.ndarray:
    """Lee un JSON de recolector_dinamico.py y devuelve su array (T, 126)."""
    content = json.loads(path.read_text(encoding="utf-8"))
    frames = content.get("frames", content.get("sequence", content))
    arr = np.array(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
        raise ValueError(f"{path.name}: shape inesperado {arr.shape}, se esperaba (T, {N_FEATURES})")
    return arr


def listar_propias_por_letra() -> dict[str, list[int]]:
    """Para cada letra dinámica, los números de sus muestras 'propias' (ordenados)."""
    resultado: dict[str, list[int]] = {}
    for letra in LETRAS_DINAMICAS:
        carpeta = DATA_DIR / letra
        if not carpeta.is_dir():
            resultado[letra] = []
            continue
        numeros = []
        for f in carpeta.glob("*.json"):
            m = PROPIA_RE.match(f.name)
            if m:
                numeros.append(int(m.group(1)))
        resultado[letra] = sorted(numeros)
    return resultado


def construir_templates_solo_dataset() -> dict[str, list[np.ndarray]]:
    """Carga SOLO archivos con patrón 'dataset' (sujeto_repeticion) de las 6 letras.

    Cualquier archivo 'propia' (de cualquier letra) queda excluido del set de
    plantillas, tal como pidió el usuario: ninguna muestra propia debe
    compararse contra sí misma ni contra otra muestra propia.
    """
    templates: dict[str, list[np.ndarray]] = {}
    for letra in LETRAS_DINAMICAS:
        carpeta = DATA_DIR / letra
        if not carpeta.is_dir():
            continue
        muestras = []
        for f in sorted(carpeta.glob("*.json")):
            if not DATASET_RE.match(f.name):
                continue  # excluye "propias" (y cualquier otro archivo raro)
            try:
                muestras.append(cargar_frames(f))
            except Exception as e:
                print(f"  [aviso] no se pudo leer {f}: {e}")
        if muestras:
            templates[letra] = muestras
    return templates


def recognizer_desde_templates(templates: dict[str, list[np.ndarray]]) -> DTWRecognizer:
    """Construye un DTWRecognizer sin pasar por __init__/load_templates(),
    para que use EXACTAMENTE el set de plantillas filtrado (solo dataset) y
    no escriba nada a disco (evita auto_save_labels)."""
    rec = DTWRecognizer.__new__(DTWRecognizer)
    rec.data_dir = DATA_DIR
    rec.labels_path = None
    rec.resample_len = None          # mismos defaults que _get_dtw_recognizer() en senas.py
    rec.hand_agnostic = False        # (DTWRecognizer.try_load() sin argumentos)
    rec._templates = templates
    rec._labels = sorted(templates.keys())
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", default="K", help="letra a diagnosticar (default: K)")
    parser.add_argument(
        "--desde", type=int, default=222,
        help="número de muestra 'propia' desde el cual reportar en detalle (default: 222)",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("PASO 1: números de muestras 'propias' por letra (patrón muestra_<N>.json)")
    print("=" * 90)
    propias_por_letra = listar_propias_por_letra()
    for letra, numeros in propias_por_letra.items():
        if not numeros:
            print(f"  {letra}: (ninguna)")
            continue
        # Compacta rangos consecutivos para que se lea fácil (ej. 216-221, 222-248).
        rangos = []
        ini = prev = numeros[0]
        for n in numeros[1:]:
            if n == prev + 1:
                prev = n
                continue
            rangos.append((ini, prev))
            ini = prev = n
        rangos.append((ini, prev))
        rango_str = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in rangos)
        print(f"  {letra}: {len(numeros)} muestra(s) -> {rango_str}")

    # Números que SOLO aparecen en la carpeta pedida (--carpeta, default K) y
    # en ninguna otra letra: son el indicio más fuerte de "quedaron aquí por
    # error", ya que si el mismo número existe también en J/Q/Z probablemente
    # sí se guardó correctamente en cada una por separado.
    objetivo = args.carpeta
    otros = set()
    for letra, numeros in propias_por_letra.items():
        if letra != objetivo:
            otros.update(numeros)
    exclusivos = sorted(n for n in propias_por_letra.get(objetivo, []) if n not in otros)
    compartidos = sorted(n for n in propias_por_letra.get(objetivo, []) if n in otros)
    print()
    print(f"  Números 'propios' de {objetivo} que TAMBIÉN existen en otra letra "
          f"(probablemente correctos, cada uno en su carpeta): {compartidos or '(ninguno)'}")
    print(f"  Números 'propios' de {objetivo} EXCLUSIVOS de {objetivo} "
          f"(sospechosos de mala carpeta): {exclusivos or '(ninguno)'}")
    if exclusivos and exclusivos[0] != args.desde:
        print(f"  [aviso] el primer número exclusivo es {exclusivos[0]}, no coincide "
              f"con --desde={args.desde}. Revisa si --desde debería ajustarse.")

    print()
    print("=" * 90)
    print(f"PASO 2: construyendo plantillas SOLO del dataset CICESE (excluye TODAS las propias)")
    print("=" * 90)
    templates = construir_templates_solo_dataset()
    for letra, muestras in templates.items():
        print(f"  {letra}: {len(muestras)} plantillas de dataset")
    if not templates:
        print("ERROR: no se cargó ninguna plantilla de dataset. Nada que comparar.")
        return 1

    recognizer = recognizer_desde_templates(templates)

    print()
    print("=" * 90)
    print(f"PASO 3: clasificando muestras 'propias' de {objetivo}/ desde muestra_{args.desde}.json")
    print("=" * 90)
    carpeta_objetivo = DATA_DIR / objetivo
    archivos = []
    for f in carpeta_objetivo.glob("*.json"):
        m = PROPIA_RE.match(f.name)
        if m and int(m.group(1)) >= args.desde:
            archivos.append((int(m.group(1)), f))
    archivos.sort()

    if not archivos:
        print(f"  No se encontraron muestras 'propias' en {objetivo}/ desde {args.desde}.")
        return 0

    filas = []  # (archivo, top1, conf1, dist1, top2, conf2, dist2, top3, conf3, dist3)
    for numero, f in archivos:
        try:
            frames = cargar_frames(f)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            continue

        top3_conf = recognizer.predict_topk(frames, k=3, return_distance=False)
        top3_dist = recognizer.predict_topk(frames, k=3, return_distance=True)
        # Mismo orden de candidatos en ambas listas (predict_topk ordena por
        # distancia ascendente en los dos casos), así que se pueden combinar por índice.
        dist_por_letra = dict(top3_dist)

        fila = [f.name]
        for letra, conf in top3_conf:
            fila.append(letra)
            fila.append(conf)
            fila.append(dist_por_letra.get(letra, float("nan")))
        while len(fila) < 10:
            fila.append(None)
        filas.append(fila)

        print(f"\n  {f.name} ({len(frames)} frames):")
        for letra, conf in top3_conf:
            print(f"      {letra:3s}  confianza={conf*100:5.1f}%  distancia_dtw={dist_por_letra.get(letra):.4f}")

    print()
    print("=" * 90)
    print("PASO 4: tabla resumen (archivo | top-1 | confianza | 2do lugar | confianza)")
    print("=" * 90)
    header = f"{'archivo':22s} | {'top-1':5s} | {'conf.':>7s} | {'2do lugar':9s} | {'conf.':>7s}"
    print(header)
    print("-" * len(header))
    for fila in filas:
        archivo, t1, c1, d1, t2, c2, d2, t3, c3, d3 = fila
        print(f"{archivo:22s} | {t1:5s} | {c1*100:6.1f}% | {t2:9s} | {c2*100:6.1f}%")

    print()
    print("=" * 90)
    print("PASO 5: conteo agrupado por letra más probable (top-1)")
    print("=" * 90)
    conteo: dict[str, list[str]] = {}
    for fila in filas:
        archivo, t1 = fila[0], fila[1]
        conteo.setdefault(t1, []).append(archivo)
    for letra in sorted(conteo, key=lambda l: -len(conteo[l])):
        archivos_letra = conteo[letra]
        print(f"  {letra}: {len(archivos_letra)} muestra(s) -> {', '.join(archivos_letra)}")

    print()
    print(f"Total de muestras propias analizadas: {len(filas)}")
    print("\nRecuerda: esto es solo diagnóstico (top-1 de un DTW contra el dataset "
          "CICESE). No se movió ni se borró ningún archivo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
