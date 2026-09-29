"""Evaluación de muestras propias frente a vistas frontal y perfil del dataset CICESE,
y análisis de umbral por cociente d2/d1 para mitigación del atractor Z.

1. Carga y reconstrucción de landmarks de perfil (.npz) a vectores de 126 valores.
2. Comparación de 28 muestras propias limpias (J3, K8, Q9, Z8) contra:
   (a) Galería solo frontal (558 plantillas)
   (b) Galería solo perfil (550 plantillas)
   (c) Galería frontal + perfil (1108 plantillas)
3. Reporte de distancias d1, d2, cociente d2/d1, distancia a letra propia vs distancia a Z.
4. Análisis de umbral de cociente d2/d1 y evaluación de rechazo sobre muestras propias y LOSO del dataset.
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

from sign_classifier import normalize_keypoints, hand_to_feature_vector

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
log = logging.getLogger("evaluar_perfil")

N_FEATURES_PER_HAND = 63
N_FEATURES = 126
CLASES_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]


# =========================================================================== #
# Programación Dinámica DTW optimizada
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

        # Backtracking para normalizar por longitud de camino
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


# =========================================================================== #
# Reconstrucción de secuencias desde .npz de perfil
# =========================================================================== #

def reconstruir_desde_npz(npz_path: Path) -> Optional[np.ndarray]:
    """Reconstruye secuencia de 126 valores con el mismo procedimiento de verificar_landmarks_crudos.py:
    1. Carga hand_labels, landmarks_image y landmarks_world.
    2. Recorta frames sin mano al inicio y al final.
    3. Para cada frame y slot (0=Izquierda, 1=Derecha), aplica hand_to_feature_vector y normalize_keypoints.
    """
    with np.load(npz_path) as data:
        hand_labels = data["hand_labels"]          # (T, 2)
        landmarks_image = data["landmarks_image"]  # (T, 2, 21, 3)
        landmarks_world = data["landmarks_world"]  # (T, 2, 21, 3)

    total_frames = hand_labels.shape[0]
    presentes = [t for t in range(total_frames) if np.any(hand_labels[t] != "")]
    if not presentes:
        return None
    inicio, fin = presentes[0], presentes[-1]

    secuencia = []
    for t in range(inicio, fin + 1):
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        for slot in range(2):
            if hand_labels[t, slot] == "":
                continue
            lm2d = landmarks_image[t, slot, :, :2]
            lm3d = landmarks_world[t, slot]
            raw = hand_to_feature_vector(lm2d, lm3d)
            norm = normalize_keypoints(raw)
            offset = slot * N_FEATURES_PER_HAND
            vec[offset:offset + N_FEATURES_PER_HAND] = norm
        secuencia.append(vec)
    return np.array(secuencia, dtype=np.float64)


# =========================================================================== #
# Carga de datos
# =========================================================================== #

def cargar_datos_completos(
    repo_dir: Path, dataset_dir: Path
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Carga:
    1. Muestras del dataset frontal (.json en datos_dinamicas)
    2. Muestras propias limpias (.json en datos_dinamicas con patrón muestra_N.json)
    3. Muestras del dataset perfil (.npz en landmarks_crudos)
    """
    json_dir = repo_dir / "datos_dinamicas"
    raw_dir = dataset_dir / "landmarks_crudos"

    pat_dataset = re.compile(r"^muestra_(\d+)_(\d+)\.json$")
    pat_propia = re.compile(r"^muestra_(\d+)\.json$")

    frontal_dataset: List[dict] = []
    muestras_propias: List[dict] = []

    # Cargar frontal (dataset + propias)
    for l_dir in sorted(json_dir.iterdir()):
        if not l_dir.is_dir():
            continue
        letra = l_dir.name

        for f in sorted(l_dir.glob("*.json")):
            m_ds = pat_dataset.match(f.name)
            m_pr = pat_propia.match(f.name)
            if not (m_ds or m_pr):
                continue

            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                frames = np.array(data["frames"], dtype=np.float64)
                if frames.ndim != 2 or frames.shape[1] != 126 or len(frames) == 0:
                    continue

                if m_ds:
                    frontal_dataset.append({
                        "letra": letra,
                        "sujeto": m_ds.group(1),
                        "rep": m_ds.group(2),
                        "archivo": f.name,
                        "frames": frames,
                    })
                elif m_pr:
                    muestras_propias.append({
                        "letra": letra,
                        "id": m_pr.group(1),
                        "archivo": f.name,
                        "frames": frames,
                    })
            except Exception as e:
                log.warning("Error leyendo %s: %s", f.name, e)

    # Cargar perfil (.npz)
    perfil_dataset: List[dict] = []
    pat_perfil = re.compile(r"^S(\d+)_perfil_(\d+)\.npz$")

    for l_dir in sorted(raw_dir.iterdir()):
        if not l_dir.is_dir():
            continue
        letra = l_dir.name

        for f in sorted(l_dir.glob("*_perfil_*.npz")):
            m_pf = pat_perfil.match(f.name)
            if not m_pf:
                continue

            sujeto = m_pf.group(1)
            rep = m_pf.group(2)
            try:
                frames = reconstruir_desde_npz(f)
                if frames is not None and len(frames) > 0:
                    perfil_dataset.append({
                        "letra": letra,
                        "sujeto": sujeto,
                        "rep": rep,
                        "archivo": f.name,
                        "frames": frames,
                    })
            except Exception as e:
                log.warning("Error reconstruyendo %s: %s", f.name, e)

    return frontal_dataset, perfil_dataset, muestras_propias


# =========================================================================== #
# Clasificación contra galería
# =========================================================================== #

def evaluar_consultas_contra_galeria(
    consultas: List[dict],
    galeria: List[dict],
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """Evalúa cada consulta contra la galería dada por DTW nearest-neighbor.
    Para cada clase c, encuentra la mínima distancia DTW a plantillas de clase c.
    Devuelve lista con resultados detallados de cada consulta.
    """
    # Indexar plantillas por clase
    galeria_por_clase: Dict[str, List[np.ndarray]] = defaultdict(list)
    for item in galeria:
        galeria_por_clase[item["letra"]].append(item["frames"])

    resultados = []
    for q in consultas:
        q_frames = q["frames"]
        dists_por_clase: Dict[str, float] = {}

        for c in clases:
            tmpls = galeria_por_clase.get(c, [])
            if not tmpls:
                dists_por_clase[c] = float("inf")
                continue
            min_d = float("inf")
            for t_frames in tmpls:
                c_mat = cdist(q_frames, t_frames, metric="euclidean")
                d = dtw_distance(c_mat)
                if d < min_d:
                    min_d = d
            dists_por_clase[c] = min_d

        # Ordenar clases por distancia ascendente
        ranking = sorted(dists_por_clase.items(), key=lambda x: x[1])
        top1_letra, d1 = ranking[0]
        top2_letra, d2 = ranking[1]
        ratio = d2 / d1 if d1 > 0 else float("inf")

        d_propia = dists_por_clase.get(q["letra"], float("inf"))
        d_z = dists_por_clase.get("Z", float("inf"))

        top3_letras = [r[0] for r in ranking[:3]]

        resultados.append({
            "archivo": q.get("archivo", ""),
            "id": q.get("id", ""),
            "letra_real": q["letra"],
            "pred_top1": top1_letra,
            "pred_top2": top2_letra,
            "top3": top3_letras,
            "d1": d1,
            "d2": d2,
            "ratio_d2_d1": ratio,
            "d_propia": d_propia,
            "d_z": d_z,
            "delta_propia_menos_z": d_propia - d_z,
            "acierto": (top1_letra == q["letra"]),
            "dists_por_clase": dists_por_clase,
        })

    return resultados


def calcular_metricas_resumen(
    resultados: List[dict],
    clases: List[str] = CLASES_DINAMICAS,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """Calcula Top-1 por letra y matriz de confusión."""
    matriz = {tl: {pl: 0 for pl in clases} for tl in clases}
    totales_por_letra = defaultdict(int)
    aciertos_por_letra = defaultdict(int)

    for r in resultados:
        tl = r["letra_real"]
        pl = r["pred_top1"]
        matriz[tl][pl] += 1
        totales_por_letra[tl] += 1
        if tl == pl:
            aciertos_por_letra[tl] += 1

    top1_por_letra = {}
    for tl in sorted(totales_por_letra.keys()):
        top1_por_letra[tl] = (aciertos_por_letra[tl] / totales_por_letra[tl]) * 100.0

    return top1_por_letra, matriz


# =========================================================================== #
# LOSO sobre dataset frontal (para evaluar distribución de d2/d1 en Z)
# =========================================================================== #

def evaluar_loso_dataset_frontal(
    frontal_dataset: List[dict],
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """Calcula validación Leave-One-Subject-Out en el dataset frontal."""
    sujetos = sorted(list(set(s["sujeto"] for s in frontal_dataset)), key=lambda x: int(x) if x.isdigit() else x)
    por_sujeto = defaultdict(list)
    for s in frontal_dataset:
        por_sujeto[s["sujeto"]].append(s)

    resultados_loso = []
    for test_sub in sujetos:
        galeria_sub = [s for s in frontal_dataset if s["sujeto"] != test_sub]
        consultas_sub = por_sujeto[test_sub]
        res = evaluar_consultas_contra_galeria(consultas_sub, galeria_sub, clases)
        for r, orig in zip(res, consultas_sub):
            r["sujeto"] = test_sub
            r["rep"] = orig["rep"]
            resultados_loso.append(r)
    return resultados_loso


# =========================================================================== #
# Función principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de vistas frontal vs perfil y análisis de umbral d2/d1")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"C:\Proyectos\Dataset_CICESE"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    log.info("Cargando datos: dataset frontal, dataset perfil y muestras propias...")
    frontal_ds, perfil_ds, propias = cargar_datos_completos(args.repo_dir, args.dataset_dir)

    print("\n" + "=" * 80)
    print("CONTEO DE DATOS CARGADOS")
    print("=" * 80)
    print(f"Dataset Frontal: {len(frontal_ds)} plantillas")
    print(f"Dataset Perfil:  {len(perfil_ds)} plantillas (reconstruidas desde npz con token '_perfil_')")
    print(f"Muestras propias limpias: {len(propias)} muestras")

    conteo_propias = defaultdict(int)
    for p in propias:
        conteo_propias[p["letra"]] += 1
    print(f"  Detalle propias: {dict(sorted(conteo_propias.items()))}")

    # 1. Tres galerías
    galeria_a = frontal_ds
    galeria_b = perfil_ds
    galeria_c = frontal_ds + perfil_ds

    log.info("Evaluando contra Galería (a): Solo Frontal (%d plantillas)...", len(galeria_a))
    res_a = evaluar_consultas_contra_galeria(propias, galeria_a)

    log.info("Evaluando contra Galería (b): Solo Perfil (%d plantillas)...", len(galeria_b))
    res_b = evaluar_consultas_contra_galeria(propias, galeria_b)

    log.info("Evaluando contra Galería (c): Frontal + Perfil (%d plantillas)...", len(galeria_c))
    res_c = evaluar_consultas_contra_galeria(propias, galeria_c)

    # ======================================================================= #
    # TABLA COMPARATIVA DE GALERÍAS (Punto 2)
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("PUNTO 2: EVALUACIÓN DE MUESTRAS PROPIAS CONTRA LAS 3 GALERÍAS")
    print("=" * 80)

    top1_a, mat_a = calcular_metricas_resumen(res_a)
    top1_b, mat_b = calcular_metricas_resumen(res_b)
    top1_c, mat_c = calcular_metricas_resumen(res_c)

    total_a = sum(1 for r in res_a if r["acierto"])
    total_b = sum(1 for r in res_b if r["acierto"])
    total_c = sum(1 for r in res_c if r["acierto"])
    n_tot = len(propias)

    letras_eval = sorted(list(conteo_propias.keys()))

    print("\nTABLA 2.1: Top-1 (%) por Letra y Total en las 3 Galerías:")
    print("-" * 75)
    print(f"{'Letra':<8} | {'N':<4} | {'(a) Frontal':<15} | {'(b) Perfil':<15} | {'(c) Frontal+Perfil':<18}")
    print("-" * 75)
    for l in letras_eval:
        n_l = conteo_propias[l]
        acc_a = top1_a.get(l, 0.0)
        acc_b = top1_b.get(l, 0.0)
        acc_c = top1_c.get(l, 0.0)
        print(f"{l:<8} | {n_l:<4} | {acc_a:>6.1f}% ({int(round(acc_a*n_l/100))}/{n_l}) | "
              f"{acc_b:>6.1f}% ({int(round(acc_b*n_l/100))}/{n_l}) | "
              f"{acc_c:>6.1f}% ({int(round(acc_c*n_l/100))}/{n_l})")
    print("-" * 75)
    print(f"{'GLOBAL':<8} | {n_tot:<4} | {total_a/n_tot*100:>6.1f}% ({total_a}/{n_tot}) | "
          f"{total_b/n_tot*100:>6.1f}% ({total_b}/{n_tot}) | "
          f"{total_c/n_tot*100:>6.1f}% ({total_c}/{n_tot})")
    print("-" * 75)

    def print_matriz(mat: Dict[str, Dict[str, int]], titulo: str):
        print(f"\n{titulo}")
        header = f"{'Real \\ Pred':<12} | " + " | ".join(f"{c:>4}" for c in CLASES_DINAMICAS) + " | Total"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for tl in letras_eval:
            row = [f"{mat[tl][c]:>4}" for c in CLASES_DINAMICAS]
            tot_row = sum(mat[tl][c] for c in CLASES_DINAMICAS)
            print(f"{tl:<12} | " + " | ".join(row) + f" | {tot_row:>5}")
        print("-" * len(header))

    print_matriz(mat_a, "Matriz de Confusión - (a) Solo Frontal:")
    print_matriz(mat_b, "Matriz de Confusión - (b) Solo Perfil:")
    print_matriz(mat_c, "Matriz de Confusión - (c) Frontal + Perfil:")

    # ======================================================================= #
    # TABLA DETALLADA POR MUESTRA PROPIA (Punto 3)
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("PUNTO 3: DETALLE POR MUESTRA PROPIA (Galería a: Frontal)")
    print("=" * 80)
    print(f"{'Archivo':<18} | {'Real':<4} | {'Pred':<4} | {'d1':<7} | {'d2':<7} | {'d2/d1':<6} | {'d(propia)':<9} | {'d(Z)':<7} | {'d(prop)-d(Z)':<12}")
    print("-" * 92)
    for r in res_a:
        print(f"{r['archivo']:<18} | {r['letra_real']:<4} | {r['pred_top1']:<4} | "
              f"{r['d1']:>7.4f} | {r['d2']:>7.4f} | {r['ratio_d2_d1']:>6.3f} | "
              f"{r['d_propia']:>9.4f} | {r['d_z']:>7.4f} | {r['delta_propia_menos_z']:>+12.4f}")
    print("-" * 92)

    # ======================================================================= #
    # ANÁLISIS DE UMBRAL DE COCIENTE (Punto 4)
    # ======================================================================= #
    print("\n" + "=" * 80)
    print("PUNTO 4: ANÁLISIS DE UMBRAL POR COCIENTE d2/d1 (Galería Frontal)")
    print("=" * 80)

    z_correctas = [r for r in res_a if r["letra_real"] == "Z" and r["pred_top1"] == "Z"]
    kq_en_z = [r for r in res_a if r["letra_real"] in ("K", "Q") and r["pred_top1"] == "Z"]

    print(f"\n1. Muestras propias de Z clasificadas CORRECTAMENTE como Z ({len(z_correctas)}/{len([r for r in res_a if r['letra_real']=='Z'])}):")
    for r in z_correctas:
        print(f"   {r['archivo']} (Z -> Z): d1={r['d1']:.4f}, d2={r['d2']:.4f}, d2/d1={r['ratio_d2_d1']:.4f} (2da: {r['pred_top2']})")

    ratios_z = [r["ratio_d2_d1"] for r in z_correctas]
    print(f"   --> Ratios Z: min={min(ratios_z):.4f}, mediana={np.median(ratios_z):.4f}, max={max(ratios_z):.4f}, media={np.mean(ratios_z):.4f}")

    print(f"\n2. Muestras propias de K y Q clasificadas ERRÓNEAMENTE como Z ({len(kq_en_z)}/17):")
    for r in kq_en_z:
        print(f"   {r['archivo']} ({r['letra_real']} -> Z): d1={r['d1']:.4f}, d2={r['d2']:.4f}, d2/d1={r['ratio_d2_d1']:.4f} (2da: {r['pred_top2']})")

    ratios_kq_z = [r["ratio_d2_d1"] for r in kq_en_z]
    if ratios_kq_z:
        print(f"   --> Ratios K/Q->Z: min={min(ratios_kq_z):.4f}, mediana={np.median(ratios_kq_z):.4f}, max={max(ratios_kq_z):.4f}, media={np.mean(ratios_kq_z):.4f}")

    log.info("Calculando validación LOSO en el dataset frontal para obtener distribución de Z real...")
    loso_ds = evaluar_loso_dataset_frontal(frontal_ds)
    z_loso_correctas = [r for r in loso_ds if r["letra_real"] == "Z" and r["pred_top1"] == "Z"]
    ratios_z_loso = [r["ratio_d2_d1"] for r in z_loso_correctas]

    print(f"\n3. Validación LOSO del Dataset Frontal: Z correctas ({len(z_loso_correctas)}/{len([r for r in loso_ds if r['letra_real']=='Z'])}):")
    print(f"   --> Ratios Z LOSO: min={min(ratios_z_loso):.4f}, p5={np.percentile(ratios_z_loso, 5):.4f}, p10={np.percentile(ratios_z_loso, 10):.4f}, "
          f"mediana={np.median(ratios_z_loso):.4f}, max={max(ratios_z_loso):.4f}")

    # Barrido de posibles umbrales
    print("\nTABLA 4.1: Impacto de distintos umbrales theta (Aceptar si ratio d2/d1 >= theta):")
    print("-" * 90)
    print(f"{'Umbral theta':<12} | {'Z Propias Aceptadas':<20} | {'K/Q Erróneas Rechazadas':<24} | {'Z Dataset LOSO Rechazadas':<26}")
    print("-" * 90)

    umbrales_test = [1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.25]
    for th in umbrales_test:
        z_prop_ok = sum(1 for r in z_correctas if r["ratio_d2_d1"] >= th)
        kq_rechazadas = sum(1 for r in kq_en_z if r["ratio_d2_d1"] < th)
        z_loso_rechazadas = sum(1 for r in z_loso_correctas if r["ratio_d2_d1"] < th)

        pct_z_prop = (z_prop_ok / len(z_correctas)) * 100 if z_correctas else 0
        pct_kq_rech = (kq_rechazadas / len(kq_en_z)) * 100 if kq_en_z else 0
        pct_z_loso_rech = (z_loso_rechazadas / len(z_loso_correctas)) * 100 if z_loso_correctas else 0

        print(f"{th:<12.2f} | {z_prop_ok:>2}/{len(z_correctas)} ({pct_z_prop:>5.1f}%)         | "
              f"{kq_rechazadas:>2}/{len(kq_en_z)} ({pct_kq_rech:>5.1f}%)             | "
              f"{z_loso_rechazadas:>2}/{len(z_loso_correctas)} ({pct_z_loso_rech:>5.1f}%)")
    print("-" * 90)

    t_tot = time.perf_counter() - t0
    log.info("Evaluación completada con éxito en %.2f segundos.", t_tot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
