"""Script de evaluación de muestras propias grabadas con recolector_dinamico.py.

Evalúa las muestras propias (nombradas `muestra_<N>.json`) como consultas
frente a la galería de plantillas del dataset de CICESE (nombradas
`muestra_<SubjectId>_<Rep>.json`).

Compara:
1. Precisión Top-1 y Top-3, matriz de confusión y distancias (d1, d2, d2/d1).
2. Diagnóstico de calidad de captura: longitud de frames, porcentaje de ceros
   por bloque de mano, pérdidas totales de tracking, coordenadas disparadas
   por normalización (|x|, |y| > 4) y temblor frame a frame.
3. Sensibilidad a remuestreo (60 frames) y comparación agnóstica a la mano.

Uso:
    python evaluar_propias.py
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
log = logging.getLogger("evaluar_propias")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"


# =========================================================================== #
# Programación Dinámica DTW optimizada
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

        # Backtracking para normalizar exactamente por longitud de camino
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

    def dtw_distance(c_mat: np.ndarray) -> float:
        return float(_dtw_dp_numba(c_mat))
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
# Utilidades de transformación
# =========================================================================== #

def mirror_and_swap_hands(sequence: np.ndarray) -> np.ndarray:
    """Intercambia bloque izquierdo (63) y derecho (63) y refleja el eje X de cada mano."""
    out = np.zeros_like(sequence)
    out[:, :63] = sequence[:, 63:]
    out[:, 63:] = sequence[:, :63]
    for i in range(21):
        out[:, i * 3] = -out[:, i * 3]
        out[:, 63 + i * 3] = -out[:, 63 + i * 3]
    return out


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Interpola linealmente la secuencia a target_len frames."""
    n_frames, n_feats = sequence.shape
    if n_frames == target_len:
        return sequence
    x_old = np.linspace(0.0, 1.0, n_frames)
    x_new = np.linspace(0.0, 1.0, target_len)
    resampled = np.zeros((target_len, n_feats), dtype=sequence.dtype)
    for f in range(n_feats):
        resampled[:, f] = np.interp(x_new, x_old, sequence[:, f])
    return resampled


# =========================================================================== #
# Carga de datos separando Galería (Dataset) y Consultas (Propias)
# =========================================================================== #

def load_dataset_and_user_samples(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Carga los archivos diferenciando dataset (muestra_Sub_Rep.json) y usuario (muestra_N.json)."""
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
                    })
            except Exception as e:
                log.warning("Error leyendo %s: %s", json_file.name, e)

    return dataset_samples, user_samples


# =========================================================================== #
# Evaluación de consultas frente a galería
# =========================================================================== #

def evaluate_queries_against_gallery(
    queries: List[dict],
    gallery_by_letter: Dict[str, List[np.ndarray]],
    all_classes: List[str],
    hand_agnostic: bool = False,
    resample_len: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Dict[str, int]]]:
    """Evalúa cada consulta contra todas las plantillas de la galería."""
    # Preprocesar galería si se aplica remuestreo
    proc_gallery = defaultdict(list)
    for l, tmpls in gallery_by_letter.items():
        for t in tmpls:
            proc_gallery[l].append(resample_sequence(t, resample_len) if resample_len else t)

    results = []
    user_classes = sorted(list(set(q["letter"] for q in queries)))
    conf_mat = {tl: {pl: 0 for pl in all_classes} for tl in user_classes}

    for q in queries:
        true_l = q["letter"]
        q_frames = q["frames"]
        if resample_len:
            q_frames = resample_sequence(q_frames, resample_len)
        q_mirrored = mirror_and_swap_hands(q_frames) if hand_agnostic else None

        dists = {}
        for l in all_classes:
            min_d = float("inf")
            for tmpl in proc_gallery[l]:
                cost_mat = cdist(q_frames, tmpl, metric="euclidean")
                d = dtw_distance(cost_mat)
                if hand_agnostic and q_mirrored is not None:
                    cost_mat_m = cdist(q_mirrored, tmpl, metric="euclidean")
                    d_m = dtw_distance(cost_mat_m)
                    if d_m < d:
                        d = d_m
                if d < min_d:
                    min_d = d
            dists[l] = min_d

        ranked = sorted(dists.items(), key=lambda x: x[1])
        pred_top1 = ranked[0][0]
        top3 = [w for w, _ in ranked[:3]]
        d1 = ranked[0][1]
        d2 = ranked[1][1]
        ratio = d2 / max(1e-6, d1)
        margin = d2 - d1

        conf_mat[true_l][pred_top1] += 1
        results.append({
            "file": q["file"],
            "true": true_l,
            "pred": pred_top1,
            "top3": top3,
            "d1": d1,
            "d2": d2,
            "ratio": ratio,
            "margin": margin,
            "ranked": ranked,
        })

    return results, conf_mat


# =========================================================================== #
# Cálculo de Métricas de Calidad de Captura y Movimiento
# =========================================================================== #

def compute_quality_metrics(sample_list: List[dict]) -> dict:
    """Calcula estadísticas de frames, huecos de detección, escala y temblor."""
    lens = []
    pct_left_zeros = []
    pct_right_zeros = []
    pct_all_zeros = []
    pct_exploded_scale = []
    jitters = []

    # Coordenadas X e Y de ambas manos (descartando Z: indices % 3 != 2)
    xy_mask = np.ones(126, dtype=bool)
    xy_mask[2::3] = False

    for s in sample_list:
        frames = s["frames"]
        n = len(frames)
        lens.append(n)

        # Bloques en ceros
        left_z = np.all(frames[:, :63] == 0, axis=1)
        right_z = np.all(frames[:, 63:] == 0, axis=1)
        all_z = left_z & right_z

        pct_left_zeros.append(np.mean(left_z) * 100)
        pct_right_zeros.append(np.mean(right_z) * 100)
        pct_all_zeros.append(np.mean(all_z) * 100)

        # Escala disparada: coordenadas |x| > 4 o |y| > 4
        xy_vals = np.abs(frames[:, xy_mask])
        frame_exploded = np.any(xy_vals > 4.0, axis=1)
        pct_exploded_scale.append(np.mean(frame_exploded) * 100)

        # Temblor frame a frame: cambio absoluto medio
        if n > 1:
            diffs = np.abs(frames[1:] - frames[:-1])
            jitters.append(np.mean(diffs))
        else:
            jitters.append(0.0)

    return {
        "count": len(sample_list),
        "min_len": int(np.min(lens)) if lens else 0,
        "med_len": float(np.median(lens)) if lens else 0.0,
        "max_len": int(np.max(lens)) if lens else 0,
        "pct_left_z": float(np.mean(pct_left_zeros)) if pct_left_zeros else 0.0,
        "pct_right_z": float(np.mean(pct_right_zeros)) if pct_right_zeros else 0.0,
        "pct_all_z": float(np.mean(pct_all_zeros)) if pct_all_zeros else 0.0,
        "pct_exploded": float(np.mean(pct_exploded_scale)) if pct_exploded_scale else 0.0,
        "jitter": float(np.mean(jitters)) if jitters else 0.0,
    }


# =========================================================================== #
# Impresión estructurada de resultados
# =========================================================================== #

def print_evaluation_report(
    user_classes: List[str],
    all_classes: List[str],
    res_base: List[dict],
    cm_base: Dict[str, Dict[str, int]],
    res_resample: List[dict],
    res_agnostic: List[dict],
    dataset_samples: List[dict],
    user_samples: List[dict],
) -> None:
    print("\n" + "=" * 90)
    print(" EVALUACIÓN DE MUESTRAS PROPIAS FRENTE A GALERÍA DEL DATASET (CICESE)")
    print("=" * 90)
    print(f" Galería (Dataset): {len(dataset_samples)} plantillas ({len(all_classes)} clases: {', '.join(all_classes)})")
    print(f" Consultas (Usuario): {len(user_samples)} muestras ({len(user_classes)} clases: {', '.join(user_classes)})")

    # 1. Tabla de rendimiento comparable con LOSO
    print("\n" + "-" * 90)
    print(" 1. RENDIMIENTO TOP-1 Y TOP-3 POR LETRA (COMPARABLE CON LOSO)")
    print("-" * 90)
    header = f"{'Letra':<6} | {'Top-1 (Propias)':<16} | {'Top-3 (Propias)':<16} | {'d1 Medio':<10} | {'d2 Medio':<10} | {'Ratio d2/d1':<12} | {'Ref. LOSO Top-1'}"
    print(header)
    print("-" * len(header))

    loso_ref = {
        "J": "100.00%", "K": "84.21%", "Q": "58.24%", "X": "92.13%", "Z": "100.00%", "Ñ": "98.92%"
    }

    for l in user_classes:
        q_l = [r for r in res_base if r["true"] == l]
        top1_c = sum(1 for r in q_l if r["pred"] == l)
        top3_c = sum(1 for r in q_l if l in r["top3"])
        t = len(q_l)
        top1_pct = (top1_c / t) * 100 if t else 0.0
        top3_pct = (top3_c / t) * 100 if t else 0.0
        d1_m = float(np.mean([r["d1"] for r in q_l]))
        d2_m = float(np.mean([r["d2"] for r in q_l]))
        r_m = float(np.mean([r["ratio"] for r in q_l]))
        ref = loso_ref.get(l, "N/A")
        print(f"{l:<6} | {top1_c:2d}/{t:2d} ({top1_pct:5.1f}%)   | {top3_c:2d}/{t:2d} ({top3_pct:5.1f}%)   | {d1_m:8.3f}   | {d2_m:8.3f}   | {r_m:10.2f}   | {ref}")

    # 2. Matriz de Confusión
    print("\n" + "-" * 90)
    print(" 2. MATRIZ DE CONFUSIÓN (Filas: Letra Real Usuario, Columnas: Predicción Dataset)")
    print("-" * 90)
    cm_header = f"{'Real \\ Pred':<12}" + "".join(f"{l:>8}" for l in all_classes) + f"{'Total':>8}"
    print(cm_header)
    print("-" * len(cm_header))
    for tl in user_classes:
        counts = [cm_base[tl][pl] for pl in all_classes]
        row_str = "".join(f"{c:8d}" for c in counts)
        print(f"{tl:<12}{row_str}{sum(counts):8d}")

    # 3. Diagnóstico de Calidad y Movimiento
    print("\n" + "-" * 90)
    print(" 3. COMPARATIVA DE CALIDAD DE SEÑAL Y MOVIMIENTO (USUARIO vs DATASET)")
    print("-" * 90)
    diag_hdr = f"{'Letra / Grupo':<16} | {'Frames (Mín/Med/Máx)':<22} | {'% Izq Cero':<11} | {'% Der Cero':<11} | {'% Todo Cero':<12} | {'% Escala>4':<11} | {'Temblor'}"
    print(diag_hdr)
    print("-" * len(diag_hdr))

    for l in user_classes:
        u_m = compute_quality_metrics([s for s in user_samples if s["letter"] == l])
        d_m = compute_quality_metrics([s for s in dataset_samples if s["letter"] == l])

        u_len_str = f"{u_m['min_len']:2d} / {u_m['med_len']:4.1f} / {u_m['max_len']:3d}"
        d_len_str = f"{d_m['min_len']:2d} / {d_m['med_len']:4.1f} / {d_m['max_len']:3d}"

        print(f"{l + ' (Usuario)':<16} | {u_len_str:<22} | {u_m['pct_left_z']:9.2f}% | {u_m['pct_right_z']:9.2f}% | {u_m['pct_all_z']:10.2f}% | {u_m['pct_exploded']:9.2f}% | {u_m['jitter']:7.4f}")
        print(f"{l + ' (Dataset)':<16} | {d_len_str:<22} | {d_m['pct_left_z']:9.2f}% | {d_m['pct_right_z']:9.2f}% | {d_m['pct_all_z']:10.2f}% | {d_m['pct_exploded']:9.2f}% | {d_m['jitter']:7.4f}")
        print("-" * len(diag_hdr))

    # 4. Experimentos de Sensibilidad
    print("\n" + "-" * 90)
    print(" 4. SENSIBILIDAD: REMUESTREO (60 FRAMES) Y MODO AGNÓSTICO A LA MANO")
    print("-" * 90)
    sens_hdr = f"{'Letra':<6} | {'Base Top-1':<14} | {'Remuestreo 60f Top-1':<22} | {'Hand-Agnostic Top-1':<22}"
    print(sens_hdr)
    print("-" * len(sens_hdr))

    for l in user_classes:
        q_base = [r for r in res_base if r["true"] == l]
        q_res = [r for r in res_resample if r["true"] == l]
        q_agn = [r for r in res_agnostic if r["true"] == l]
        t = len(q_base)

        b_c = sum(1 for r in q_base if r["pred"] == l)
        r_c = sum(1 for r in q_res if r["pred"] == l)
        a_c = sum(1 for r in q_agn if r["pred"] == l)

        print(f"{l:<6} | {b_c:2d}/{t:2d} ({b_c/t*100:5.1f}%) | {r_c:2d}/{t:2d} ({r_c/t*100:5.1f}%) [Δ={r_c-b_c:+d}]   | {a_c:2d}/{t:2d} ({a_c/t*100:5.1f}%) [Δ={a_c-b_c:+d}]")

    print("=" * 90 + "\n")


# =========================================================================== #
# Función principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación de muestras dinámicas propias vs dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Ruta de datos_dinamicas")
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    dataset_samples, user_samples = load_dataset_and_user_samples(args.data_dir)
    if not dataset_samples:
        print("[ERROR] No se encontraron plantillas del dataset en datos_dinamicas.")
        return 1
    if not user_samples:
        print("[ERROR] No se encontraron muestras de usuario (muestra_<N>.json).")
        return 1

    all_classes = sorted(list(set(s["letter"] for s in dataset_samples)))
    user_classes = sorted(list(set(s["letter"] for s in user_samples)))

    # Agrupar galería
    gallery_by_letter = defaultdict(list)
    for s in dataset_samples:
        gallery_by_letter[s["letter"]].append(s["frames"])

    # 1. Base
    res_base, cm_base = evaluate_queries_against_gallery(
        user_samples, gallery_by_letter, all_classes, hand_agnostic=False, resample_len=None
    )

    # 2. Remuestreo a 60 frames
    res_resample, _ = evaluate_queries_against_gallery(
        user_samples, gallery_by_letter, all_classes, hand_agnostic=False, resample_len=60
    )

    # 3. Hand-agnostic
    res_agnostic, _ = evaluate_queries_against_gallery(
        user_samples, gallery_by_letter, all_classes, hand_agnostic=True, resample_len=None
    )

    print_evaluation_report(
        user_classes,
        all_classes,
        res_base,
        cm_base,
        res_resample,
        res_agnostic,
        dataset_samples,
        user_samples,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
