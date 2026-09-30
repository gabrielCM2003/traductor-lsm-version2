"""Evaluación LIMPIA (sin fuga de información) de la compensación angular de
la CONSULTA antes de clasificar con DTW.

Contexto: un experimento anterior (ver diagnostico_orientacion.py, PUNTO 4)
mostro que rotar las consultas "propias" hacia el angulo mediano del dataset
CICESE subia a K de 12.5% a 62.5%. Ese experimento tenia una fuga de
informacion: el angulo de referencia para cada letra se calculaba con
Mediana(dataset) - Mediana(PROPIAS) usando las mismas muestras propias que
luego se evaluaban como consultas (una fuga de estadistica agregada del
propio conjunto de prueba, no solo de la muestra individual).

Este script mide el mismo efecto SIN esa fuga:

  1. El angulo de referencia de cada letra se calcula UNICAMENTE con las
     plantillas del dataset CICESE (patron muestra_<sujeto>_<rep>.json).
     Las muestras propias (patron muestra_<N>.json, incluidas las ~150
     nuevas agregadas hoy para Ñ y X) NUNCA participan en ese calculo -
     solo se usan como consultas de prueba adicionales (ver PASO 4).
  2. LOSO limpio sobre el dataset: para cada sujeto de prueba, el angulo de
     referencia de cada letra se recalcula excluyendo a ESE sujeto (igual
     que la galeria DTW de evaluar_dtw.py). Nunca se usa el angulo propio
     del sujeto de prueba para construir su propia referencia.
  3. La rotacion aplicada a cada consulta es SIEMPRE especifica de esa
     consulta: delta = angulo_referencia(letra_real) - angulo_propio(consulta),
     donde angulo_propio(consulta) se calcula UNICAMENTE con los frames de
     esa misma consulta (normalizacion por muestra, no una fuga: es
     exactamente lo que haria un sistema en vivo con la secuencia que
     acaba de grabar, no usa nada de otras muestras ni del resto del
     conjunto de prueba).

CAVEAT IMPORTANTE (léase antes de decidir aplicar esto a producción): la
rotacion se hace hacia el angulo de referencia de la LETRA REAL de la
consulta (oraculo). En produccion, en el momento de rotar todavia no se
conoce que letra es la consulta (es justo lo que DTW va a decidir), asi que
este experimento mide un TECHO (best case: "si supieramos la letra correcta,
cuanto ayudaria corregir su angulo"), no un procedimiento directamente
desplegable. Si el techo no es prometedor, no vale la pena buscar una
version practica (ej. probar las 6 rotaciones candidatas y quedarse con la
de menor distancia DTW, que si seria desplegable pero mas cara en tiempo).
Si el techo SI es prometedor, ese seria el siguiente paso a evaluar aparte.

Uso:
    python evaluar_compensacion_angular.py
    python evaluar_compensacion_angular.py --skip-propias   # solo LOSO dataset (mas rapido)

NO modifica senas.py, dtw_recognizer.py ni datos_dinamicas/. Solo lee.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluar_compensacion_angular")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"
CLASES_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]

PAT_DATASET = re.compile(r"^muestra_([^_]+)_(\d+)\.json$")
PAT_PROPIA = re.compile(r"^muestra_(\d+)\.json$")


# =========================================================================== #
# DTW (Numba / NumPy) - identico en metodologia a evaluar_dtw.py
# =========================================================================== #

if HAVE_NUMBA:
    @njit(fastmath=True)
    def _dtw_core_numba(cost_matrix: np.ndarray) -> float:
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                prev_min = dp[i - 1, j - 1]
                if dp[i - 1, j] < prev_min:
                    prev_min = dp[i - 1, j]
                if dp[i, j - 1] < prev_min:
                    prev_min = dp[i, j - 1]
                dp[i, j] = cost_matrix[i - 1, j - 1] + prev_min
        total_dist = dp[n, m]
        i, j = n, m
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            diag = dp[i - 1, j - 1]
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1
            i -= 1
        while j > 0:
            path_len += 1
            j -= 1
        return total_dist / max(1, path_len)

    def dtw_distance(cost_mat: np.ndarray) -> float:
        return float(_dtw_core_numba(cost_mat))
else:
    def dtw_distance(cost_matrix: np.ndarray) -> float:
        n, m = cost_matrix.shape
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            c_row = cost_matrix[i - 1]
            d_prev = dp[i - 1]
            d_curr = dp[i]
            for j in range(1, m + 1):
                d_curr[j] = c_row[j - 1] + min(d_prev[j - 1], d_prev[j], d_curr[j - 1])
        total_dist = dp[n, m]
        i, j = n, m
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            diag, up, left = dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1
            i -= 1
        while j > 0:
            path_len += 1
            j -= 1
        return float(total_dist / max(1, path_len))


def dtw_min_distance(query: np.ndarray, templates: List[np.ndarray]) -> float:
    """Distancia DTW minima de `query` contra una lista de plantillas."""
    min_d = float("inf")
    for tmpl in templates:
        cost_mat = cdist(query, tmpl, metric="euclidean")
        d = dtw_distance(cost_mat)
        if d < min_d:
            min_d = d
    return min_d


# =========================================================================== #
# Angulo por muestra y rotacion 2D
# =========================================================================== #

def angulo_propio_muestra(frames: np.ndarray) -> Optional[float]:
    """Mediana del angulo (grados) muneca->base del dedo medio (landmark 9: x
    en indice 27, y en indice 28 DENTRO del bloque de 63) de ESTA muestra
    unicamente - nunca usa datos de otras muestras.

    Generalizacion sobre calcular_angulos_muestra() de diagnostico_orientacion.py:
    ese script solo miraba el primer bloque de mano (indices 0:63) y
    descartaba en silencio cualquier frame con ese bloque vacio. Aqui, si el
    primer bloque esta vacio en un frame pero el segundo (63:126) tiene
    datos, se usa el segundo. Solo se ignora un frame si AMBOS bloques estan
    vacios. Esto importa mas para las muestras propias nuevas (grabadas en
    condiciones reales, variadas) que para el dataset CICESE ya normalizado.
    """
    angulos = []
    for f in frames:
        bloque0_activo = not np.all(f[:63] == 0.0)
        bloque1_activo = not np.all(f[63:126] == 0.0)
        if bloque0_activo:
            offset = 0
        elif bloque1_activo:
            offset = 63
        else:
            continue
        x = f[offset + 27]
        y = f[offset + 28]
        angulos.append(float(np.degrees(np.arctan2(y, x))))
    if not angulos:
        return None
    return float(np.median(angulos))


def rotar_secuencia_2d(frames: np.ndarray, angulo_grados: float) -> np.ndarray:
    """Rotacion 2D en el plano (x, y) de cada landmark de cada mano, z intacto.
    Identica a diagnostico_orientacion.py (misma formula), reutilizada aqui
    para no divergir en metodologia."""
    theta = np.radians(angulo_grados)
    c, s = np.cos(theta), np.sin(theta)
    rotados = frames.copy()
    for t in range(len(rotados)):
        for slot in (0, 1):
            offset = slot * 63
            if np.all(rotados[t, offset:offset + 63] == 0.0):
                continue
            for k in range(21):
                idx_x = offset + k * 3
                idx_y = offset + k * 3 + 1
                x = rotados[t, idx_x]
                y = rotados[t, idx_y]
                rotados[t, idx_x] = c * x - s * y
                rotados[t, idx_y] = s * x + c * y
    return rotados


# =========================================================================== #
# Carga de datos
# =========================================================================== #

def cargar_dataset(data_dir: Path) -> List[dict]:
    """Carga UNICAMENTE plantillas CICESE (patron muestra_<sujeto>_<rep>.json)."""
    muestras = []
    for letra_dir in sorted(data_dir.iterdir()):
        if not letra_dir.is_dir():
            continue
        letra = letra_dir.name
        for f in sorted(letra_dir.glob("*.json")):
            m = PAT_DATASET.match(f.name)
            if not m:
                continue
            try:
                content = json.loads(f.read_text(encoding="utf-8"))
                frames = np.array(content["frames"], dtype=np.float64)
                if frames.ndim != 2 or frames.shape[1] != 126 or len(frames) == 0:
                    continue
                muestras.append({
                    "letra": letra,
                    "sujeto": m.group(1),
                    "rep": m.group(2),
                    "archivo": f.name,
                    "frames": frames,
                    "angulo_propio": angulo_propio_muestra(frames),
                })
            except Exception as e:
                log.warning("Error leyendo %s: %s", f.name, e)
    return muestras


def cargar_propias(data_dir: Path) -> List[dict]:
    """Carga UNICAMENTE muestras propias (patron muestra_<N>.json). Se usan
    SOLO como consultas de prueba (PASO 4) - nunca para el angulo de
    referencia."""
    muestras = []
    for letra_dir in sorted(data_dir.iterdir()):
        if not letra_dir.is_dir():
            continue
        letra = letra_dir.name
        for f in sorted(letra_dir.glob("*.json")):
            m = PAT_PROPIA.match(f.name)
            if not m:
                continue
            try:
                content = json.loads(f.read_text(encoding="utf-8"))
                frames = np.array(content["frames"], dtype=np.float64)
                if frames.ndim != 2 or frames.shape[1] != 126 or len(frames) == 0:
                    continue
                muestras.append({
                    "letra": letra,
                    "id": m.group(1),
                    "archivo": f.name,
                    "frames": frames,
                    "angulo_propio": angulo_propio_muestra(frames),
                })
            except Exception as e:
                log.warning("Error leyendo %s: %s", f.name, e)
    return muestras


# =========================================================================== #
# Angulo de referencia por letra (SOLO dataset CICESE)
# =========================================================================== #

def angulo_referencia_por_letra(
    dataset: List[dict], excluir_sujeto: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Mediana del angulo propio de las plantillas CICESE de cada letra,
    excluyendo (si se indica) las muestras de `excluir_sujeto`. NUNCA
    incluye muestras propias - esta funcion solo recibe `dataset`."""
    por_letra: Dict[str, List[float]] = defaultdict(list)
    for m in dataset:
        if excluir_sujeto is not None and m["sujeto"] == excluir_sujeto:
            continue
        if m["angulo_propio"] is None:
            continue
        por_letra[m["letra"]].append(m["angulo_propio"])
    return {l: (float(np.median(v)) if v else None) for l, v in por_letra.items()}


# =========================================================================== #
# Clasificacion DTW 1-NN (original vs rotada) contra una galeria dada
# =========================================================================== #

def clasificar_original_y_rotada(
    query: dict,
    gallery_by_letter: Dict[str, List[np.ndarray]],
    ref_angles: Dict[str, Optional[float]],
    clases: List[str],
    rot_time_acc: List[float],
) -> dict:
    """Clasifica `query` contra `gallery_by_letter` dos veces: (a) sin rotar,
    (b) rotada hacia ref_angles[letra_real] usando SOLO el angulo propio de
    la consulta (angulo_propio ya calculado al cargarla). Si no se pudo
    calcular angulo_propio o no hay angulo de referencia para la letra real,
    la version "rotada" cae de vuelta a la original (delta=0) y se marca
    rotacion_aplicada=False."""
    letra_real = query["letra"]
    q_frames = query["frames"]

    dists_orig: Dict[str, float] = {}
    for l in clases:
        dists_orig[l] = dtw_min_distance(q_frames, gallery_by_letter.get(l, []))
    ranked_orig = sorted(dists_orig.items(), key=lambda kv: kv[1])

    ref = ref_angles.get(letra_real)
    propio = query["angulo_propio"]
    rotacion_aplicada = ref is not None and propio is not None
    if rotacion_aplicada:
        delta = ref - propio
        t0 = time.perf_counter()
        q_rot = rotar_secuencia_2d(q_frames, delta)
        rot_time_acc.append(time.perf_counter() - t0)
    else:
        delta = 0.0
        q_rot = q_frames

    dists_rot: Dict[str, float] = {}
    for l in clases:
        dists_rot[l] = dtw_min_distance(q_rot, gallery_by_letter.get(l, []))
    ranked_rot = sorted(dists_rot.items(), key=lambda kv: kv[1])

    return {
        "archivo": query["archivo"],
        "letra_real": letra_real,
        "delta": delta,
        "rotacion_aplicada": rotacion_aplicada,
        "pred_orig": ranked_orig[0][0],
        "d1_orig": ranked_orig[0][1],
        "pred_rot": ranked_rot[0][0],
        "d1_rot": ranked_rot[0][1],
    }


# =========================================================================== #
# PASO 2/3: LOSO limpio sobre el dataset CICESE (original vs rotada)
# =========================================================================== #

def loso_limpio_dataset(dataset: List[dict], clases: List[str]) -> Tuple[List[dict], List[float], float]:
    sujetos = sorted(set(m["sujeto"] for m in dataset), key=lambda x: int(x) if x.isdigit() else x)
    log.info("LOSO limpio (dataset CICESE): %d muestras, %d sujetos", len(dataset), len(sujetos))

    resultados = []
    rot_times: List[float] = []
    t0 = time.perf_counter()

    for idx, sujeto_test in enumerate(sujetos, 1):
        gallery_by_letter: Dict[str, List[np.ndarray]] = defaultdict(list)
        for m in dataset:
            if m["sujeto"] != sujeto_test:
                gallery_by_letter[m["letra"]].append(m["frames"])

        # Angulo de referencia SIN el sujeto de prueba (sin fuga).
        ref_angles = angulo_referencia_por_letra(dataset, excluir_sujeto=sujeto_test)

        queries = [m for m in dataset if m["sujeto"] == sujeto_test]
        for q in queries:
            resultados.append(clasificar_original_y_rotada(q, gallery_by_letter, ref_angles, clases, rot_times))

        if idx % 5 == 0 or idx == len(sujetos):
            log.info("  LOSO dataset: %d/%d sujetos procesados", idx, len(sujetos))

    total_time = time.perf_counter() - t0
    return resultados, rot_times, total_time


# =========================================================================== #
# PASO 4: propias como consultas adicionales (galeria = dataset COMPLETO)
# =========================================================================== #

def evaluar_propias_vs_dataset_completo(
    propias: List[dict], dataset: List[dict], clases: List[str]
) -> Tuple[List[dict], List[float], float]:
    gallery_by_letter: Dict[str, List[np.ndarray]] = defaultdict(list)
    for m in dataset:
        gallery_by_letter[m["letra"]].append(m["frames"])

    # Angulo de referencia con el dataset COMPLETO (las propias nunca
    # participan en este calculo, sin importar si son antiguas o de hoy).
    ref_angles = angulo_referencia_por_letra(dataset, excluir_sujeto=None)

    log.info("Evaluando %d muestras propias contra el dataset completo (%d plantillas)...",
             len(propias), len(dataset))

    resultados = []
    rot_times: List[float] = []
    t0 = time.perf_counter()
    for i, q in enumerate(propias, 1):
        resultados.append(clasificar_original_y_rotada(q, gallery_by_letter, ref_angles, clases, rot_times))
        if i % 25 == 0 or i == len(propias):
            log.info("  propias: %d/%d procesadas", i, len(propias))
    total_time = time.perf_counter() - t0
    return resultados, rot_times, total_time


# =========================================================================== #
# Reportes
# =========================================================================== #

def matriz_confusion(resultados: List[dict], clases: List[str], campo_pred: str) -> Dict[str, Dict[str, int]]:
    mat = {tl: {pl: 0 for pl in clases} for tl in clases}
    for r in resultados:
        mat[r["letra_real"]][r[campo_pred]] += 1
    return mat


def imprimir_matriz(mat: Dict[str, Dict[str, int]], clases: List[str], titulo: str) -> None:
    print(f"\n{titulo}")
    header = "Real \\ Pred  " + "  ".join(f"{c:>6s}" for c in clases) + "   Total"
    print(header)
    print("-" * len(header))
    for tl in clases:
        fila = mat[tl]
        total = sum(fila.values())
        print(f"{tl:<12s}" + "  ".join(f"{fila[pl]:>6d}" for pl in clases) + f"   {total:>5d}")


def tabla_comparativa_por_letra(
    resultados: List[dict], clases: List[str], titulo: str
) -> Dict[str, dict]:
    print(f"\n{titulo}")
    header = (f"{'Letra':<6} | {'N':<4} | {'Top-1 Original':<16} | {'Top-1 Rotada':<16} | "
              f"{'Delta (pp)':<11} | {'Veredicto'}")
    print(header)
    print("-" * len(header))

    resumen: Dict[str, dict] = {}
    for l in clases:
        items = [r for r in resultados if r["letra_real"] == l]
        n = len(items)
        if n == 0:
            continue
        aciertos_orig = sum(1 for r in items if r["pred_orig"] == l)
        aciertos_rot = sum(1 for r in items if r["pred_rot"] == l)
        pct_orig = aciertos_orig / n * 100
        pct_rot = aciertos_rot / n * 100
        delta_pp = pct_rot - pct_orig

        if delta_pp >= 2.0:
            veredicto = "MEJORA"
        elif delta_pp <= -2.0:
            veredicto = "EMPEORA"
        else:
            veredicto = "neutral"

        resumen[l] = {
            "n": n, "pct_orig": pct_orig, "pct_rot": pct_rot,
            "delta_pp": delta_pp, "veredicto": veredicto,
        }
        print(f"{l:<6} | {n:<4} | {aciertos_orig:>3}/{n:<3} ({pct_orig:>5.1f}%) | "
              f"{aciertos_rot:>3}/{n:<3} ({pct_rot:>5.1f}%) | {delta_pp:>+9.1f}pp | {veredicto}")
    return resumen


# =========================================================================== #
# Programa principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-propias", action="store_true",
                         help="Omite el PASO 4 (propias vs dataset completo), solo corre el LOSO limpio del dataset.")
    args = parser.parse_args()

    print("=" * 100)
    print("EVALUACION LIMPIA DE COMPENSACION ANGULAR DE LA CONSULTA (sin fuga de informacion)")
    print("=" * 100)

    dataset = cargar_dataset(args.data_dir)
    propias = [] if args.skip_propias else cargar_propias(args.data_dir)

    print(f"\nDataset CICESE cargado: {len(dataset)} plantillas "
          f"({len(set(m['sujeto'] for m in dataset))} sujetos)")
    if not args.skip_propias:
        print(f"Muestras propias cargadas (consultas de prueba adicionales, PASO 4): {len(propias)}")
        por_letra_pr = defaultdict(int)
        for m in propias:
            por_letra_pr[m["letra"]] += 1
        print("  " + "  ".join(f"{l}={por_letra_pr.get(l, 0)}" for l in CLASES_DINAMICAS))

    sin_angulo_ds = sum(1 for m in dataset if m["angulo_propio"] is None)
    sin_angulo_pr = sum(1 for m in propias if m["angulo_propio"] is None)
    if sin_angulo_ds or sin_angulo_pr:
        print(f"\n[aviso] muestras sin angulo propio calculable (ambos bloques de mano vacios en "
              f"todos sus frames): dataset={sin_angulo_ds}, propias={sin_angulo_pr}. "
              f"Esas consultas NO se rotan (caen de vuelta a la version original).")

    # ----------------------------------------------------------------------- #
    # PASO 1: angulo mediano de referencia por letra (SOLO dataset completo,
    # informativo - el LOSO real recalcula esto excluyendo cada sujeto).
    # ----------------------------------------------------------------------- #
    ref_full = angulo_referencia_por_letra(dataset, excluir_sujeto=None)
    print("\n" + "-" * 100)
    print("PASO 1: ANGULO MEDIANO DE REFERENCIA POR LETRA (dataset CICESE completo, informativo)")
    print("-" * 100)
    for l in CLASES_DINAMICAS:
        v = ref_full.get(l)
        print(f"  {l}: {v:.1f}°" if v is not None else f"  {l}: (sin datos)")

    # ----------------------------------------------------------------------- #
    # PASO 2/3: LOSO limpio sobre el dataset
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("PASO 2/3: LOSO LIMPIO SOBRE EL DATASET CICESE (angulo de referencia excluye siempre al sujeto de prueba)")
    print("=" * 100)
    resultados_ds, rot_times_ds, tiempo_ds = loso_limpio_dataset(dataset, CLASES_DINAMICAS)

    n_ds = len(resultados_ds)
    aciertos_orig_ds = sum(1 for r in resultados_ds if r["pred_orig"] == r["letra_real"])
    aciertos_rot_ds = sum(1 for r in resultados_ds if r["pred_rot"] == r["letra_real"])
    print(f"\nTiempo total LOSO dataset (original + rotada, {n_ds} consultas x2): {tiempo_ds:.2f}s")
    print(f"Precision Top-1 GLOBAL sin rotar:  {aciertos_orig_ds}/{n_ds} ({aciertos_orig_ds/n_ds*100:.2f}%)")
    print(f"Precision Top-1 GLOBAL rotada:     {aciertos_rot_ds}/{n_ds} ({aciertos_rot_ds/n_ds*100:.2f}%)")

    resumen_ds = tabla_comparativa_por_letra(
        resultados_ds, CLASES_DINAMICAS,
        "TABLA A: LOSO dataset CICESE - Top-1 original vs rotada, por letra"
    )

    mat_orig_ds = matriz_confusion(resultados_ds, CLASES_DINAMICAS, "pred_orig")
    mat_rot_ds = matriz_confusion(resultados_ds, CLASES_DINAMICAS, "pred_rot")
    imprimir_matriz(mat_orig_ds, CLASES_DINAMICAS, "MATRIZ DE CONFUSION - LOSO dataset, SIN rotar")
    imprimir_matriz(mat_rot_ds, CLASES_DINAMICAS, "MATRIZ DE CONFUSION - LOSO dataset, ROTADA")

    # ----------------------------------------------------------------------- #
    # PASO 4: propias como consultas adicionales
    # ----------------------------------------------------------------------- #
    resumen_pr = {}
    rot_times_pr: List[float] = []
    if not args.skip_propias and propias:
        print("\n" + "=" * 100)
        print("PASO 4: MUESTRAS PROPIAS (incluye las nuevas de hoy) COMO CONSULTAS DE PRUEBA")
        print("        Galeria = dataset CICESE completo. Angulo de referencia = dataset completo")
        print("        (las propias NUNCA participan en el calculo del angulo de referencia).")
        print("=" * 100)
        resultados_pr, rot_times_pr, tiempo_pr = evaluar_propias_vs_dataset_completo(
            propias, dataset, CLASES_DINAMICAS
        )

        n_pr = len(resultados_pr)
        aciertos_orig_pr = sum(1 for r in resultados_pr if r["pred_orig"] == r["letra_real"])
        aciertos_rot_pr = sum(1 for r in resultados_pr if r["pred_rot"] == r["letra_real"])
        print(f"\nTiempo total propias (original + rotada, {n_pr} consultas x2): {tiempo_pr:.2f}s")
        print(f"Precision Top-1 GLOBAL sin rotar:  {aciertos_orig_pr}/{n_pr} ({aciertos_orig_pr/n_pr*100:.2f}%)")
        print(f"Precision Top-1 GLOBAL rotada:     {aciertos_rot_pr}/{n_pr} ({aciertos_rot_pr/n_pr*100:.2f}%)")

        resumen_pr = tabla_comparativa_por_letra(
            resultados_pr, CLASES_DINAMICAS,
            "TABLA B: propias vs dataset completo - Top-1 original vs rotada, por letra"
        )

        mat_orig_pr = matriz_confusion(resultados_pr, CLASES_DINAMICAS, "pred_orig")
        mat_rot_pr = matriz_confusion(resultados_pr, CLASES_DINAMICAS, "pred_rot")
        imprimir_matriz(mat_orig_pr, CLASES_DINAMICAS, "MATRIZ DE CONFUSION - propias, SIN rotar")
        imprimir_matriz(mat_rot_pr, CLASES_DINAMICAS, "MATRIZ DE CONFUSION - propias, ROTADA")

    # ----------------------------------------------------------------------- #
    # PASO 5: latencia de la rotacion
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("PASO 5: LATENCIA EXTRA DE ROTAR CADA CONSULTA (solo rotar_secuencia_2d, sin el DTW)")
    print("=" * 100)
    todos_rot_times = rot_times_ds + rot_times_pr
    if todos_rot_times:
        arr = np.array(todos_rot_times) * 1000.0  # a ms
        print(f"  Llamadas medidas: {len(arr)}")
        print(f"  Media: {arr.mean():.3f} ms | Mediana: {np.median(arr):.3f} ms | "
              f"p95: {np.percentile(arr, 95):.3f} ms | Max: {arr.max():.3f} ms")
        print(f"  Tiempo total acumulado en rotaciones: {arr.sum()/1000.0:.3f} s "
              f"(de un tiempo total de evaluacion de {tiempo_ds + (tiempo_pr if not args.skip_propias and propias else 0.0):.1f} s)")
    else:
        print("  No se registraron rotaciones (todas las consultas quedaron sin angulo de referencia o propio).")

    # ----------------------------------------------------------------------- #
    # PASO 6: conclusiones
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("PASO 6: CONCLUSION POR LETRA Y GLOBAL")
    print("=" * 100)
    print("\nRecordatorio del caveat: esta prueba rota cada consulta hacia el angulo de referencia")
    print("de su LETRA REAL (oraculo) - mide el TECHO del beneficio, no un procedimiento ya listo")
    print("para produccion (en vivo no se conoce la letra antes de clasificar).")

    print(f"\n{'Letra':<6} | {'LOSO dataset (oraculo)':<28} | {'Propias (oraculo)':<25}")
    print("-" * 65)
    for l in CLASES_DINAMICAS:
        v_ds = resumen_ds.get(l)
        v_pr = resumen_pr.get(l)
        s_ds = f"{v_ds['delta_pp']:+.1f}pp -> {v_ds['veredicto']}" if v_ds else "(sin datos)"
        s_pr = f"{v_pr['delta_pp']:+.1f}pp -> {v_pr['veredicto']}" if v_pr else "(sin datos)"
        print(f"{l:<6} | {s_ds:<28} | {s_pr:<25}")

    mejoras_ds = sum(1 for v in resumen_ds.values() if v["veredicto"] == "MEJORA")
    empeoras_ds = sum(1 for v in resumen_ds.values() if v["veredicto"] == "EMPEORA")
    print(f"\nLOSO dataset: {mejoras_ds} letra(s) mejoran, {empeoras_ds} letra(s) empeoran, "
          f"{len(resumen_ds) - mejoras_ds - empeoras_ds} neutral(es) (umbral +/-2pp).")
    if resumen_pr:
        mejoras_pr = sum(1 for v in resumen_pr.values() if v["veredicto"] == "MEJORA")
        empeoras_pr = sum(1 for v in resumen_pr.values() if v["veredicto"] == "EMPEORA")
        print(f"Propias:      {mejoras_pr} letra(s) mejoran, {empeoras_pr} letra(s) empeoran, "
              f"{len(resumen_pr) - mejoras_pr - empeoras_pr} neutral(es) (umbral +/-2pp).")

    print("\n" + "=" * 100)
    print("FIN. No se modifico senas.py, dtw_recognizer.py ni datos_dinamicas/.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
