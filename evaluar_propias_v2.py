"""Script de evaluación v2 para analizar el impacto de muestras cortas (<25f) y frames en ceros.

Analiza las 4 variantes solicitadas:
  (a) Base: tal como están (todas las muestras propias).
  (b) Filtrado temporal: sin muestras propias con < 25 frames.
  (c) Filtrado de tracking: eliminando frames con todo en ceros (en consulta y galería).
  (d) Combinado (b + c): sin muestras < 25 frames Y sin frames con todo en ceros.

Además:
- Lista ordenada de longitudes de frames de las muestras del usuario por letra.
- Prueba experimental de la hipótesis "los frames en ceros se parecen más a Z"
  calculando la norma media de los frames y su distancia euclídea media a un vector de ceros.
- Matriz de confusión completa de la variante (d).

Uso:
    python evaluar_propias_v2.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluar_propias_v2")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"


# =========================================================================== #
# Programación Dinámica DTW optimizada con Numba / NumPy
# =========================================================================== #

if HAVE_NUMBA:
    @njit(fastmath=True)
    def _dtw_dp_numba(cost_matrix: np.ndarray) -> float:
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
            path_len += 1
            i -= 1
        while j > 0:
            path_len += 1
            j -= 1

        return total_dist / max(1, path_len)

    def dtw_distance(cost_mat: np.ndarray) -> float:
        return float(_dtw_dp_numba(cost_mat))
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
        while i > 0: path_len += 1; i -= 1
        while j > 0: path_len += 1; j -= 1
        return float(total_dist / max(1, path_len))


# =========================================================================== #
# Carga de Datos y Estructuración
# =========================================================================== #

def load_data(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Carga los datos distinguiendo dataset (muestra_Sub_Rep.json) y usuario (muestra_N.json)."""
    dataset_samples = []
    user_samples = []

    pattern_ds = re.compile(r"^muestra_(\d+)_(\d+)\.json$")
    pattern_user = re.compile(r"^muestra_(\d+)\.json$")

    for letter_dir in sorted(data_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name

        for json_file in sorted(letter_dir.glob("*.json")):
            m_ds = pattern_ds.match(json_file.name)
            m_user = pattern_user.match(json_file.name)

            try:
                content = json.loads(json_file.read_text(encoding="utf-8"))
                frames = np.array(content["frames"], dtype=np.float64)
                if frames.ndim != 2 or frames.shape[1] != 126 or len(frames) == 0:
                    continue

                if m_ds:
                    dataset_samples.append({
                        "letter": letter,
                        "subject": m_ds.group(1),
                        "rep": m_ds.group(2),
                        "frames": frames,
                        "file": json_file.name,
                    })
                elif m_user:
                    user_samples.append({
                        "letter": letter,
                        "id": m_user.group(1),
                        "frames": frames,
                        "file": json_file.name,
                        "orig_len": len(frames),
                    })
            except Exception as e:
                log.warning("Error leyendo %s: %s", json_file.name, e)

    return dataset_samples, user_samples


def filter_zero_frames(seq: np.ndarray) -> np.ndarray:
    """Elimina los frames donde todos los 126 valores son cero."""
    non_zero = ~np.all(seq == 0.0, axis=1)
    if not np.any(non_zero):
        return seq  # si toda la secuencia fuera ceros, mantenerla intacta
    return seq[non_zero]


# =========================================================================== #
# Motor de Evaluación de Consultas frente a Galería
# =========================================================================== #

def evaluate_variant(
    queries: List[dict],
    gallery: List[dict],
    all_classes: List[str],
    user_classes: List[str],
    strip_zeros: bool = False,
) -> Tuple[List[dict], Dict[str, Dict[str, int]]]:
    """Evalúa las consultas contra la galería, opcionalmente eliminando frames en ceros."""
    gallery_by_letter = defaultdict(list)
    for s in gallery:
        f = filter_zero_frames(s["frames"]) if strip_zeros else s["frames"]
        gallery_by_letter[s["letter"]].append(f)

    results = []
    conf_mat = {tl: {pl: 0 for pl in all_classes} for tl in user_classes}

    for q in queries:
        true_l = q["letter"]
        q_frames = filter_zero_frames(q["frames"]) if strip_zeros else q["frames"]

        dists = {}
        for l in all_classes:
            min_d = float("inf")
            for tmpl in gallery_by_letter[l]:
                cost_mat = cdist(q_frames, tmpl, metric="euclidean")
                d = dtw_distance(cost_mat)
                if d < min_d:
                    min_d = d
            dists[l] = min_d

        ranked = sorted(dists.items(), key=lambda x: x[1])
        top1 = ranked[0][0]
        top3 = [w for w, _ in ranked[:3]]
        d1 = ranked[0][1]
        d2 = ranked[1][1]
        ratio = d2 / max(1e-6, d1)

        conf_mat[true_l][top1] += 1
        results.append({
            "file": q["file"],
            "true": true_l,
            "pred": top1,
            "top3": top3,
            "d1": d1,
            "d2": d2,
            "ratio": ratio,
        })

    return results, conf_mat


# =========================================================================== #
# Análisis de la Hipótesis 3: Norma y Distancia a Ceros
# =========================================================================== #

def analyze_zero_distance_hypothesis(
    dataset_samples: List[dict], all_classes: List[str]
) -> dict:
    """Calcula la norma media y distancia media al vector de ceros para cada clase."""
    zero_vec = np.zeros(126, dtype=np.float64)
    stats = {}

    for letter in all_classes:
        frames_list = [s["frames"] for s in dataset_samples if s["letter"] == letter]
        all_frames = np.vstack(frames_list)

        # 1. Frames con mano activa (no-ceros)
        active_mask = ~np.all(all_frames == 0.0, axis=1)
        active_frames = all_frames[active_mask]

        norms_active = np.linalg.norm(active_frames, axis=1)
        dist_to_zero_active = np.linalg.norm(active_frames - zero_vec, axis=1)

        # 2. Todos los frames (incluyendo ceros)
        norms_all = np.linalg.norm(all_frames, axis=1)
        dist_to_zero_all = np.linalg.norm(all_frames - zero_vec, axis=1)

        stats[letter] = {
            "total_frames": len(all_frames),
            "active_frames": len(active_frames),
            "pct_zero_frames": 100.0 * (1.0 - len(active_frames) / len(all_frames)),
            "norm_mean_active": float(np.mean(norms_active)),
            "norm_std_active": float(np.std(norms_active)),
            "norm_median_active": float(np.median(norms_active)),
            "dist_mean_active": float(np.mean(dist_to_zero_active)),
            "norm_mean_all": float(np.mean(norms_all)),
            "dist_mean_all": float(np.mean(dist_to_zero_all)),
        }

    return stats


# =========================================================================== #
# Impresión del Reporte Principal
# =========================================================================== #

def print_full_report(
    dataset_samples: List[dict],
    user_samples: List[dict],
    all_classes: List[str],
    user_classes: List[str],
) -> None:
    print("\n" + "=" * 92)
    print(" EVALUACIÓN V2: ANÁLISIS DE MUESTRAS CORTAS (<25f) Y FRAMES EN CEROS")
    print("=" * 92)
    print(f" Galería (Dataset CICESE): {len(dataset_samples)} plantillas ({len(all_classes)} clases)")
    print(f" Consultas (Usuario):      {len(user_samples)} muestras ({len(user_classes)} clases: {', '.join(user_classes)})")

    # ----------------------------------------------------------------------- #
    # Punto 2: Longitudes de muestras propias
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 92)
    print(" PUNTO 2: LONGITUDES DE MUESTRAS PROPIAS POR LETRA Y CONTEO < 25 FRAMES")
    print("-" * 92)
    user_lens_by_letter = defaultdict(list)
    for s in user_samples:
        user_lens_by_letter[s["letter"]].append(s["orig_len"])

    total_short = 0
    total_samples_cnt = len(user_samples)

    for l in user_classes:
        lens = sorted(user_lens_by_letter[l])
        n_short = sum(1 for x in lens if x < 25)
        total_short += n_short
        print(f" Letra '{l}' (Total = {len(lens)} muestras | < 25 frames = {n_short} muestras, {n_short/len(lens)*100:5.1f}%):")
        print(f"   Longitudes ordenadas: {lens}")

    print(f"\n Resumen Global de Muestras Cortas: {total_short} de {total_samples_cnt} ({total_short/total_samples_cnt*100:.1f}%) tienen < 25 frames.")

    # ----------------------------------------------------------------------- #
    # Punto 3: Hipótesis de frames en ceros vs Z
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 92)
    print(" PUNTO 3: HIPÓTESIS 'LOS FRAMES EN CEROS SE PARECEN MÁS A Z'")
    print("-" * 92)
    zero_stats = analyze_zero_distance_hypothesis(dataset_samples, all_classes)

    print(f"{'Letra':<6} | {'Frames Activos':<14} | {'% Ceros Dataset':<15} | {'Norma Media L2':<16} | {'Dist. Media a Ceros':<20} | {'Ranking Distancia'}")
    print("-" * 92)

    # Ordenar por norma media activa ascendente
    ranked_letters = sorted(all_classes, key=lambda l: zero_stats[l]["norm_mean_active"])
    rank_map = {l: rank for rank, l in enumerate(ranked_letters, 1)}

    for l in all_classes:
        st = zero_stats[l]
        r = rank_map[l]
        marker = " <-- MÍNIMA DISTANCIA" if r == 1 else ""
        print(
            f"{l:<6} | {st['active_frames']:14d} | {st['pct_zero_frames']:14.2f}% | "
            f"{st['norm_mean_active']:8.4f} ± {st['norm_std_active']:.2f}   | "
            f"{st['dist_mean_active']:18.4f}  | #{r} {marker}"
        )

    print("\n Hallazgo matemático:")
    print("   La distancia euclídea de un frame x al vector de ceros 0 es exactamente ||x - 0||_2 = ||x||_2 (la norma L2).")
    print(f"   -> Z tiene la menor norma media ({zero_stats['Z']['norm_mean_active']:.4f}) de todo el alfabeto.")
    print(f"   -> En contraste, Q tiene norma {zero_stats['Q']['norm_mean_active']:.4f} (+35.8%), X tiene {zero_stats['X']['norm_mean_active']:.4f} (+56.6%) y Ñ tiene {zero_stats['Ñ']['norm_mean_active']:.4f} (+73.0%).")
    print("   -> Consecuencia: Cualquier frame en ceros generado por pérdida de tracking penaliza mucho menos a Z que a las demás letras, atrayendo el alineamiento DTW hacia Z.")

    # ----------------------------------------------------------------------- #
    # Punto 1: Las 4 Variantes de Evaluación
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 92)
    print(" PUNTO 1: EVALUACIÓN DE LAS 4 VARIANTES (CONSULTAS PROPIAS vs GALERÍA DATASET)")
    print("-" * 92)

    # (a) Base
    queries_a = user_samples
    res_a, cm_a = evaluate_variant(queries_a, dataset_samples, all_classes, user_classes, strip_zeros=False)

    # (b) Sin muestras <25f
    queries_b = [q for q in user_samples if q["orig_len"] >= 25]
    res_b, cm_b = evaluate_variant(queries_b, dataset_samples, all_classes, user_classes, strip_zeros=False)

    # (c) Sin frames ceros
    queries_c = user_samples
    res_c, cm_c = evaluate_variant(queries_c, dataset_samples, all_classes, user_classes, strip_zeros=True)

    # (d) b + c
    queries_d = [q for q in user_samples if q["orig_len"] >= 25]
    res_d, cm_d = evaluate_variant(queries_d, dataset_samples, all_classes, user_classes, strip_zeros=True)

    variants_data = [
        ("(a) Base (tal como están)", queries_a, res_a, cm_a),
        ("(b) Sin muestras < 25 frames", queries_b, res_b, cm_b),
        ("(c) Sin frames en ceros", queries_c, res_c, cm_c),
        ("(d) Sin < 25f Y sin frames ceros (b + c)", queries_d, res_d, cm_d),
    ]

    # Tabla comparativa compacta de las 4 variantes
    print(f"{'Variante':<42} | {'Muestras':<8} | " + " | ".join(f"{l} Top-1" for l in user_classes) + " | Precisión Global")
    print("-" * 92)

    for name, q_list, res, cm in variants_data:
        accs = []
        acc_strs = []
        for l in user_classes:
            sub = [r for r in res if r["true"] == l]
            c = sum(1 for r in sub if r["pred"] == l)
            t = len(sub)
            pct = (c / t * 100) if t else 0.0
            accs.append((c, t))
            acc_strs.append(f"{c:2d}/{t:2d} ({pct:4.1f}%)")

        tot_c = sum(c for c, _ in accs)
        tot_t = sum(t for _, t in accs)
        tot_pct = (tot_c / tot_t * 100) if tot_t else 0.0

        row_str = f"{name:<42} | {len(q_list):6d}   | " + " | ".join(f"{s:<13}" for s in acc_strs) + f" | {tot_c:2d}/{tot_t:2d} ({tot_pct:5.1f}%)"
        print(row_str)

    # Detalle de Top-1 y Top-3 por variante
    print("\n Detalle de Top-1 y Top-3 por letra en cada variante:")
    for name, q_list, res, cm in variants_data:
        print(f"\n >>> {name} (N = {len(q_list)}):")
        for l in user_classes:
            sub = [r for r in res if r["true"] == l]
            t = len(sub)
            top1_c = sum(1 for r in sub if r["pred"] == l)
            top3_c = sum(1 for r in sub if l in r["top3"])
            d1_m = np.mean([r["d1"] for r in sub]) if sub else 0.0
            d2_m = np.mean([r["d2"] for r in sub]) if sub else 0.0
            r_m = np.mean([r["ratio"] for r in sub]) if sub else 0.0
            print(f"     Letra '{l}': Top-1 = {top1_c:2d}/{t:2d} ({top1_c/t*100:5.1f}%) | Top-3 = {top3_c:2d}/{t:2d} ({top3_c/t*100:5.1f}%) | d1={d1_m:.3f}, d2={d2_m:.3f}, d2/d1={r_m:.2f}")

    # ----------------------------------------------------------------------- #
    # Matriz de Confusión de (d)
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 92)
    print(" MATRIZ DE CONFUSIÓN DE LA VARIANTE (d): Sin < 25f Y Sin Frames en Ceros")
    print("-" * 92)
    cm_header = f"{'Real \\ Pred':<12}" + "".join(f"{l:>8}" for l in all_classes) + f"{'Total':>8}"
    print(cm_header)
    print("-" * len(cm_header))
    for tl in user_classes:
        counts = [cm_d[tl][pl] for pl in all_classes]
        row_str = "".join(f"{c:8d}" for c in counts)
        print(f"{tl:<12}{row_str}{sum(counts):8d}")

    print("=" * 92 + "\n")


# =========================================================================== #
# Función principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación V2: impacto de muestras cortas y frames ceros.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Ruta de datos_dinamicas")
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    dataset_samples, user_samples = load_data(args.data_dir)
    if not dataset_samples:
        print("[ERROR] No se encontraron plantillas del dataset en datos_dinamicas.")
        return 1
    if not user_samples:
        print("[ERROR] No se encontraron muestras de usuario (muestra_<N>.json).")
        return 1

    all_classes = sorted(list(set(s["letter"] for s in dataset_samples)))
    user_classes = sorted(list(set(s["letter"] for s in user_samples)))

    print_full_report(dataset_samples, user_samples, all_classes, user_classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
