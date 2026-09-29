"""Evaluación sistemática de features de curvatura por dedo (vector de 136 dimensiones)
frente al vector base de 126 dimensiones.

1. Reconstruye el dataset frontal completo (558 plantillas) y las 28 muestras de prueba
   con el vector de 136 dimensiones (126 normalizados + 10 curvaturas de dedos).
2. Ejecuta validación cruzada Leave-One-Subject-Out (LOSO) en los 20 sujetos del dataset:
   - Baseline 126 vs Ampliado 136 (radianes [0, pi]) vs Ampliado 136 (normalizado [0, 1]).
   - Matrices de confusión completas y comparación por letra (foco en Q y X).
3. Evalúa el impacto sobre las 28 muestras de prueba del usuario.
4. Mide la latencia de inferencia por consulta en DTW (126 vs 136).
5. Reporte con tablas y conclusiones sobre viabilidad de integración a sign_classifier.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from vector_con_curvatura import (
    frame_crudo_a_vector_136,
    calcular_curvatura_mano,
    N_FEATURES_126,
    N_FEATURES_136,
)

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluar_curvatura")

CLASES_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]


# =========================================================================== #
# Programación Dinámica DTW optimizada con Numba / NumPy (nogil para hilos)
# =========================================================================== #

if HAVE_NUMBA:
    @njit(fastmath=True, nogil=True)
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

        # Backtracking para normalizar por longitud de camino
        i, j = n, m
        path_len = 0
        while i > 0 and j > 0:
            path_len += 1
            diag = dp[i - 1, j - 1]
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1; j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1; i -= 1
        while j > 0:
            path_len += 1; j -= 1

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
                i -= 1; j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        while i > 0:
            path_len += 1; i -= 1
        while j > 0:
            path_len += 1; j -= 1
        return float(total_dist / max(1, path_len))


# =========================================================================== #
# Reconstrucción de secuencias desde .npz crudos
# =========================================================================== #

def cargar_muestra_npz(
    npz_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Carga y procesa un archivo .npz crudo recortando frames vacíos.

    Retorna:
        (seq_126, seq_136_rad, seq_136_norm) o None si no hay manos detectadas.
    """
    with np.load(npz_path) as data:
        hl = data["hand_labels"]
        lmi = data["landmarks_image"]
        lmw = data["landmarks_world"]

    total_frames = hl.shape[0]
    presentes = [t for t in range(total_frames) if np.any(hl[t] != "")]
    if not presentes:
        return None

    inicio, fin = presentes[0], presentes[-1]

    seq_126 = []
    seq_136_rad = []
    seq_136_norm = []

    for t in range(inicio, fin + 1):
        v136_r = frame_crudo_a_vector_136(hl[t], lmi[t], lmw[t], normalizar_curvatura=False)
        v136_n = frame_crudo_a_vector_136(hl[t], lmi[t], lmw[t], normalizar_curvatura=True)
        v126 = v136_r[:N_FEATURES_126].copy()

        seq_126.append(v126)
        seq_136_rad.append(v136_r)
        seq_136_norm.append(v136_n)

    return (
        np.array(seq_126, dtype=np.float64),
        np.array(seq_136_rad, dtype=np.float64),
        np.array(seq_136_norm, dtype=np.float64),
    )


def cargar_dataset_completo(
    dataset_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    """Carga el dataset frontal de CICESE y las muestras de prueba del usuario."""
    raw_root = dataset_dir / "landmarks_crudos"
    propias_root = dataset_dir / "propias_crudas"

    pat_ds = re.compile(r"^S(\d+)_frontal_(\d+)\.npz$")
    pat_pr = re.compile(r"^muestra_(\d+)\.npz$")

    ds_samples = []
    pr_samples = []

    # 1. Dataset frontal
    for c in CLASES_DINAMICAS:
        c_dir = raw_root / c
        if not c_dir.is_dir():
            continue
        for f in sorted(c_dir.glob("*_frontal_*.npz")):
            m = pat_ds.match(f.name)
            if not m:
                continue
            sub_id = m.group(1)
            rep_id = m.group(2)

            res = cargar_muestra_npz(f)
            if res is None:
                continue
            s126, s136_r, s136_n = res

            ds_samples.append({
                "letra": c,
                "sujeto": sub_id,
                "rep": rep_id,
                "archivo": f.name,
                "f126": s126,
                "f136_rad": s136_r,
                "f136_norm": s136_n,
            })

    # 2. Muestras de prueba propias
    for c in ["J", "K", "Q", "Z"]:
        c_dir = propias_root / c
        if not c_dir.is_dir():
            continue
        for f in sorted(c_dir.glob("*.npz")):
            m = pat_pr.match(f.name)
            if not m:
                continue
            res = cargar_muestra_npz(f)
            if res is None:
                continue
            s126, s136_r, s136_n = res

            pr_samples.append({
                "letra": c,
                "id": m.group(1),
                "archivo": f.name,
                "f126": s126,
                "f136_rad": s136_r,
                "f136_norm": s136_n,
            })

    return ds_samples, pr_samples


# =========================================================================== #
# Motor de Evaluación DTW
# =========================================================================== #

def resolver_clasificacion(
    q_seq: np.ndarray,
    gallery_by_class: Dict[str, List[np.ndarray]],
    clases: List[str] = CLASES_DINAMICAS,
) -> Tuple[str, str, float, float, float, Dict[str, float]]:
    """Calcula distancias DTW a todas las plantillas de la galería y resuelve la predicción."""
    dists = {}
    for c in clases:
        tmpls = gallery_by_class.get(c, [])
        if not tmpls:
            dists[c] = float("inf")
            continue
        min_d = float("inf")
        for t in tmpls:
            c_mat = cdist(q_seq, t, metric="euclidean")
            d = dtw_distance(c_mat)
            if d < min_d:
                min_d = d
        dists[c] = min_d

    ranking = sorted(dists.items(), key=lambda x: x[1])
    top1, d1 = ranking[0]
    top2, d2 = ranking[1]
    ratio = d2 / d1 if d1 > 0 else float("inf")
    return top1, top2, d1, d2, ratio, dists


def ejecutar_loso_evaluacion(
    ds_samples: List[dict],
    feat_key: str,
    n_workers: int = 16,
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """Ejecuta Leave-One-Subject-Out multihilo para una representación dada."""
    sujetos = sorted(list(set(s["sujeto"] for s in ds_samples)), key=lambda x: int(x) if x.isdigit() else x)
    por_sujeto = defaultdict(list)
    for s in ds_samples:
        por_sujeto[s["sujeto"]].append(s)

    resultados = []

    for test_sub in sujetos:
        queries = por_sujeto[test_sub]
        train_samples = [s for s in ds_samples if s["sujeto"] != test_sub]

        gal_by_c = defaultdict(list)
        for s in train_samples:
            gal_by_c[s["letra"]].append(s[feat_key])

        def _eval_q(q):
            top1, top2, d1, d2, ratio, dists = resolver_clasificacion(q[feat_key], gal_by_c, clases)
            return {
                "archivo": q["archivo"],
                "letra": q["letra"],
                "sujeto": test_sub,
                "pred": top1,
                "pred_top2": top2,
                "d1": d1,
                "d2": d2,
                "ratio": ratio,
                "dists": dists,
                "acierto": (top1 == q["letra"]),
            }

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            res_sub = list(ex.map(_eval_q, queries))
        resultados.extend(res_sub)

    return resultados


def evaluar_consultas_contra_galeria(
    consultas: List[dict],
    galeria: List[dict],
    feat_key: str,
    n_workers: int = 16,
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """Evalúa un conjunto de consultas contra la galería completa."""
    gal_by_c = defaultdict(list)
    for s in galeria:
        gal_by_c[s["letra"]].append(s[feat_key])

    def _eval_q(q):
        top1, top2, d1, d2, ratio, dists = resolver_clasificacion(q[feat_key], gal_by_c, clases)
        return {
            "archivo": q["archivo"],
            "letra": q["letra"],
            "pred": top1,
            "pred_top2": top2,
            "d1": d1,
            "d2": d2,
            "ratio": ratio,
            "dists": dists,
            "acierto": (top1 == q["letra"]),
        }

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        resultados = list(ex.map(_eval_q, consultas))
    return resultados


# =========================================================================== #
# Benchmark de Latencia
# =========================================================================== #

def medir_latencia_individual(
    consultas: List[dict],
    galeria: List[dict],
    feat_key: str,
    n_queries: int = 15,
) -> float:
    """Mide el tiempo de inferencia monohilo (en ms) por consulta contra toda la galería."""
    sample_q = consultas[:min(len(consultas), n_queries)]
    t0 = time.perf_counter()
    for q in sample_q:
        q_seq = q[feat_key]
        for tmpl in galeria:
            t_seq = tmpl[feat_key]
            c_mat = cdist(q_seq, t_seq, metric="euclidean")
            dtw_distance(c_mat)
    total_time = time.perf_counter() - t0
    return (total_time / len(sample_q)) * 1000.0


# =========================================================================== #
# Utilidades de reporte
# =========================================================================== #

def calcular_metricas(resultados: List[dict], clases: List[str] = CLASES_DINAMICAS):
    totales = defaultdict(int)
    aciertos = defaultdict(int)
    matriz = {tl: {pl: 0 for pl in clases} for tl in clases}

    for r in resultados:
        tl = r["letra"]
        pl = r["pred"]
        totales[tl] += 1
        matriz[tl][pl] += 1
        if tl == pl:
            aciertos[tl] += 1

    top1 = {}
    for c in clases:
        top1[c] = (aciertos[c] / totales[c]) * 100.0 if totales[c] > 0 else 0.0

    global_acc = (sum(aciertos.values()) / sum(totales.values())) * 100.0 if totales else 0.0
    return top1, global_acc, matriz, totales, aciertos


def imprimir_matriz(matriz: Dict[str, Dict[str, int]], letras_filas: List[str], titulo: str):
    print(f"\n{titulo}")
    header = f"{'Real \\ Pred':<12} | " + " | ".join(f"{c:>4}" for c in CLASES_DINAMICAS) + " | Total"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for tl in letras_filas:
        row = [f"{matriz[tl][c]:>4}" for c in CLASES_DINAMICAS]
        tot = sum(matriz[tl][c] for c in CLASES_DINAMICAS)
        print(f"{tl:<12} | " + " | ".join(row) + f" | {tot:>5}")
    print("-" * len(header))


# =========================================================================== #
# Programa Principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de features de curvatura en DTW")
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"C:\Proyectos\Dataset_CICESE"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    t_start = time.perf_counter()
    log.info("Cargando dataset frontal y muestras de prueba con vectores 126 y 136...")
    ds_samples, pr_samples = cargar_dataset_completo(args.dataset_dir)

    print("\n" + "=" * 95)
    print("EVALUACIÓN DE FEATURES DE CURVATURA POR DEDO (VECTOR DE 136 DIMENSIONES)")
    print("=" * 95)
    print(f"Muestras dataset frontal: {len(ds_samples)} (20 sujetos)")
    print(f"Muestras propias prueba:  {len(pr_samples)} (J: 3, K: 8, Q: 9, Z: 8)")

    # ----------------------------------------------------------------------- #
    # PUNTO 4: LOSO en Dataset
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 95)
    print("PUNTO 4: VALIDACIÓN LEAVE-ONE-SUBJECT-OUT (LOSO) - DATASET CICESE")
    print("=" * 95)

    log.info("Ejecutando LOSO en Baseline 126...")
    res_126 = ejecutar_loso_evaluacion(ds_samples, "f126", n_workers=args.workers)

    log.info("Ejecutando LOSO en Ampliado 136 (Radianes [0, pi])...")
    res_136_r = ejecutar_loso_evaluacion(ds_samples, "f136_rad", n_workers=args.workers)

    log.info("Ejecutando LOSO en Ampliado 136 (Normalizado [0, 1])...")
    res_136_n = ejecutar_loso_evaluacion(ds_samples, "f136_norm", n_workers=args.workers)

    top1_126, glob_126, cm_126, tot, ac_126 = calcular_metricas(res_126)
    top1_rad, glob_rad, cm_rad, _, ac_rad = calcular_metricas(res_136_r)
    top1_nrm, glob_nrm, cm_nrm, _, ac_nrm = calcular_metricas(res_136_n)

    print("\nTABLA 4.1: Comparativa de Rendimiento LOSO (Top-1 % por Letra):")
    print("-" * 95)
    header_loso = (
        f"{'Letra':<6} | {'N':<4} | {'126 Baseline':<16} | {'136 Radianes':<16} | "
        f"{'136 Normalizado':<18} | {'Delta (Rad - Base)':<18} | {'Delta (Norm - Base)'}"
    )
    print(header_loso)
    print("-" * len(header_loso))

    for c in CLASES_DINAMICAS:
        n = tot[c]
        p_base = top1_126[c]
        p_rad = top1_rad[c]
        p_nrm = top1_nrm[c]
        d_rad = p_rad - p_base
        d_nrm = p_nrm - p_base
        print(
            f"{c:<6} | {n:<4} | {p_base:>6.2f}% ({ac_126[c]:>2}/{n})  | "
            f"{p_rad:>6.2f}% ({ac_rad[c]:>2}/{n})  | "
            f"{p_nrm:>6.2f}% ({ac_nrm[c]:>2}/{n})    | "
            f"{d_rad:>+7.2f} pts         | {d_nrm:>+7.2f} pts"
        )
    print("-" * len(header_loso))
    d_glob_rad = glob_rad - glob_126
    d_glob_nrm = glob_nrm - glob_126
    print(
        f"{'GLOBAL':<6} | {len(ds_samples):<4} | {glob_126:>6.2f}% ({sum(ac_126.values())}/{len(ds_samples)}) | "
        f"{glob_rad:>6.2f}% ({sum(ac_rad.values())}/{len(ds_samples)}) | "
        f"{glob_nrm:>6.2f}% ({sum(ac_nrm.values())}/{len(ds_samples)})   | "
        f"{d_glob_rad:>+7.2f} pts         | {d_glob_nrm:>+7.2f} pts"
    )
    print("-" * len(header_loso))

    imprimir_matriz(cm_126, CLASES_DINAMICAS, "Matriz de Confusión LOSO - Baseline 126:")
    imprimir_matriz(cm_rad, CLASES_DINAMICAS, "Matriz de Confusión LOSO - Ampliado 136 (Radianes):")
    imprimir_matriz(cm_nrm, CLASES_DINAMICAS, "Matriz de Confusión LOSO - Ampliado 136 (Normalizado):")

    # ----------------------------------------------------------------------- #
    # PUNTO 3: Evaluación en muestras propias
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 95)
    print("PUNTO 3: EVALUACIÓN EN MUESTRAS DE PRUEBA PROPIAS (N=28)")
    print("=" * 95)

    pr_res_126 = evaluar_consultas_contra_galeria(pr_samples, ds_samples, "f126", n_workers=args.workers)
    pr_res_rad = evaluar_consultas_contra_galeria(pr_samples, ds_samples, "f136_rad", n_workers=args.workers)
    pr_res_nrm = evaluar_consultas_contra_galeria(pr_samples, ds_samples, "f136_norm", n_workers=args.workers)

    top_pr_126, glob_pr_126, cm_pr_126, tot_pr, ac_pr_126 = calcular_metricas(pr_res_126)
    top_pr_rad, glob_pr_rad, cm_pr_rad, _, ac_pr_rad = calcular_metricas(pr_res_rad)
    top_pr_nrm, glob_pr_nrm, cm_pr_nrm, _, ac_pr_nrm = calcular_metricas(pr_res_nrm)

    letras_pr = sorted(list(tot_pr.keys()))

    print("\nTABLA 3.1: Rendimiento en Muestras Propias (Top-1 % por Letra):")
    print("-" * 85)
    print(f"{'Letra':<6} | {'N':<4} | {'126 Baseline':<16} | {'136 Radianes':<16} | {'136 Normalizado'}")
    print("-" * 85)
    for c in letras_pr:
        n = tot_pr[c]
        print(
            f"{c:<6} | {n:<4} | {top_pr_126[c]:>6.1f}% ({ac_pr_126[c]}/{n})    | "
            f"{top_pr_rad[c]:>6.1f}% ({ac_pr_rad[c]}/{n})    | "
            f"{top_pr_nrm[c]:>6.1f}% ({ac_pr_nrm[c]}/{n})"
        )
    print("-" * 85)
    print(
        f"{'GLOBAL':<6} | {len(pr_samples):<4} | {glob_pr_126:>6.1f}% ({sum(ac_pr_126.values())}/{len(pr_samples)})    | "
        f"{glob_pr_rad:>6.1f}% ({sum(ac_pr_rad.values())}/{len(pr_samples)})    | "
        f"{glob_pr_nrm:>6.1f}% ({sum(ac_pr_nrm.values())}/{len(pr_samples)})"
    )
    print("-" * 85)

    imprimir_matriz(cm_pr_126, letras_pr, "Matriz de Confusión Propias - Baseline 126:")
    imprimir_matriz(cm_pr_rad, letras_pr, "Matriz de Confusión Propias - Ampliado 136 (Radianes):")

    # ----------------------------------------------------------------------- #
    # PUNTO 5: Medición de Latencia
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 95)
    print("PUNTO 5: MEDICIÓN DE LATENCIA DE INFERENCIA POR CONSULTA (DTW MONOHILO)")
    print("=" * 95)

    log.info("Midiendo latencias monohilo en consultas...")
    lat_126 = medir_latencia_individual(pr_samples, ds_samples, "f126")
    lat_136 = medir_latencia_individual(pr_samples, ds_samples, "f136_rad")

    print("\nTABLA 5.1: Latencia de Inferencia por Consulta (558 plantillas):")
    print("-" * 75)
    print(f"{'Representación':<20} | {'Features':<10} | {'Latencia Media (ms)':<20} | {'Incremento'}")
    print("-" * 75)
    print(f"{'Baseline':<20} | 126        | {lat_126:>15.1f} ms    | 1.00x (Base)")
    print(f"{'Con Curvatura':<20} | 136        | {lat_136:>15.1f} ms    | {lat_136/lat_126:>4.2f}x ({((lat_136/lat_126)-1)*100:+.1f}%)")
    print("-" * 75)

    t_total = time.perf_counter() - t_start
    log.info("Evaluación completada en %.1f segundos.", t_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
