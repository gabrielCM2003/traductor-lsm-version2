"""Script de evaluación para el reconocedor dinámico DTW (DTWRecognizer).

Implementa validación cruzada Leave-One-Subject-Out (LOSO):
- Agrupa las muestras por SubjectId a partir del nombre del archivo:
    muestra_<SubjectId>_<Rep>.json
- En cada iteración, las muestras de un sujeto se evalúan como consultas contra
  las plantillas de los demás sujetos.
- Reporta la precisión global y por letra, la matriz de confusión completa (6x6),
  el análisis detallado de K, Q y Z, y la distribución de distancias Top-1 vs Top-2.

Uso:
    python evaluar_dtw.py [--hand-agnostic] [--resample 45]
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
log = logging.getLogger("evaluar_dtw")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"


# =========================================================================== #
# Programación Dinámica DTW (Numba / NumPy)
# =========================================================================== #

if HAVE_NUMBA:
    @njit(fastmath=True)
    def dtw_core_numba(cost_matrix: np.ndarray) -> float:
        """Cálculo DTW y longitud de path con Numba a nivel de C."""
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
        return float(dtw_core_numba(c_mat))
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
# Utilidades de secuencias
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
    """Interpola linealmente la secuencia a una longitud fija de frames."""
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
# Carga de Datos y Estructuración por Sujeto
# =========================================================================== #

def load_dynamic_dataset(data_dir: Path) -> List[dict]:
    """Carga todas las muestras y extrae el SubjectId del nombre del archivo."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")

    samples = []
    regex_pattern = re.compile(r"muestra_([^_]+)_(\d+)\.json")

    for letter_dir in sorted(data_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name

        for json_file in sorted(letter_dir.glob("*.json")):
            match = regex_pattern.match(json_file.name)
            if not match:
                log.warning("Archivo no coincide con formato esperado: %s", json_file.name)
                continue

            sub_id = match.group(1)
            rep_id = match.group(2)

            try:
                content = json.loads(json_file.read_text(encoding="utf-8"))
                frames = np.array(content["frames"], dtype=np.float64)
                if frames.ndim == 2 and frames.shape[1] == 126 and len(frames) > 0:
                    samples.append({
                        "letter": letter,
                        "subject": sub_id,
                        "rep": rep_id,
                        "frames": frames,
                        "mirrored": mirror_and_swap_hands(frames),
                        "file": json_file.name,
                    })
            except Exception as e:
                log.warning("Error leyendo %s: %s", json_file.name, e)

    return samples


# =========================================================================== #
# Evaluación Leave-One-Subject-Out (LOSO)
# =========================================================================== #

def evaluate_loso(
    samples: List[dict],
    hand_agnostic: bool = False,
    resample_len: Optional[int] = None,
    temperature: float = 1.0,
) -> dict:
    """Ejecuta la validación cruzada Leave-One-Subject-Out."""
    letters = sorted(list(set(s["letter"] for s in samples)))
    subjects = sorted(
        list(set(s["subject"] for s in samples)),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    log.info(
        "Iniciando LOSO: %d muestras, %d sujetos, %d letras (Agnóstico a mano: %s, Resample: %s)",
        len(samples),
        len(subjects),
        len(letters),
        hand_agnostic,
        resample_len,
    )

    # Pre-resamplear si está activo
    processed_samples = []
    for s in samples:
        f = s["frames"]
        if resample_len is not None:
            f = resample_sequence(f, resample_len)
        m = mirror_and_swap_hands(f) if hand_agnostic else None
        processed_samples.append({
            "letter": s["letter"],
            "subject": s["subject"],
            "frames": f,
            "mirrored": m,
            "file": s["file"],
        })

    # Mapeo sujeto -> muestras
    sub_to_samples = defaultdict(list)
    for s in processed_samples:
        sub_to_samples[s["subject"]].append(s)

    # Métricas a registrar
    y_true: List[str] = []
    y_pred: List[str] = []
    y_top3: List[List[str]] = []
    d1_list: List[float] = []
    d2_list: List[float] = []
    margins_list: List[float] = []
    ratios_list: List[float] = []
    conf_top1_list: List[float] = []

    conf_matrix = {tl: {pl: 0 for pl in letters} for tl in letters}
    detailed_per_letter = defaultdict(lambda: {"correct": 0, "total": 0, "margins": [], "ratios": []})

    t_start = time.perf_counter()

    for sub_idx, test_subject in enumerate(subjects, 1):
        # Galería de entrenamiento: todos los sujetos excepto el actual
        gallery_by_letter = defaultdict(list)
        for s in processed_samples:
            if s["subject"] != test_subject:
                gallery_by_letter[s["letter"]].append(s["frames"])

        test_queries = sub_to_samples[test_subject]

        for q in test_queries:
            true_lbl = q["letter"]
            q_frames = q["frames"]
            q_mirrored = q["mirrored"]

            letter_min_dists: Dict[str, float] = {}

            for l in letters:
                templates = gallery_by_letter[l]
                min_d = float("inf")
                for tmpl in templates:
                    # DTW normal
                    cost_mat = cdist(q_frames, tmpl, metric="euclidean")
                    d = dtw_distance(cost_mat)

                    # Si es agnóstico a la mano, evaluar también la versión en espejo
                    if hand_agnostic and q_mirrored is not None:
                        cost_mat_m = cdist(q_mirrored, tmpl, metric="euclidean")
                        d_m = dtw_distance(cost_mat_m)
                        if d_m < d:
                            d = d_m

                    if d < min_d:
                        min_d = d

                letter_min_dists[l] = min_d

            # Ordenar candidatos
            ranked = sorted(letter_min_dists.items(), key=lambda item: item[1])
            pred_lbl = ranked[0][0]
            top3 = [w for w, _ in ranked[:3]]
            d1 = ranked[0][1]
            d2 = ranked[1][1]
            margin = d2 - d1
            ratio = d2 / max(1e-6, d1)

            # Cálculo de confianza Softmax
            dists_arr = np.array([d for _, d in ranked], dtype=np.float64)
            logits = -dists_arr / max(1e-4, temperature)
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)
            top1_conf = float(probs[0])

            y_true.append(true_lbl)
            y_pred.append(pred_lbl)
            y_top3.append(top3)
            d1_list.append(d1)
            d2_list.append(d2)
            margins_list.append(margin)
            ratios_list.append(ratio)
            conf_top1_list.append(top1_conf)

            conf_matrix[true_lbl][pred_lbl] += 1
            detailed_per_letter[true_lbl]["total"] += 1
            if pred_lbl == true_lbl:
                detailed_per_letter[true_lbl]["correct"] += 1
            detailed_per_letter[true_lbl]["margins"].append(margin)
            detailed_per_letter[true_lbl]["ratios"].append(ratio)

    t_end = time.perf_counter()
    total_time = t_end - t_start
    avg_latency_ms = (total_time / len(samples)) * 1000

    # Resumen de métricas
    total_samples = len(y_true)
    total_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total_top3_correct = sum(1 for yt, t3 in zip(y_true, y_top3) if yt in t3)

    overall_acc = total_correct / total_samples
    top3_acc = total_top3_correct / total_samples

    return {
        "letters": letters,
        "subjects": subjects,
        "total_samples": total_samples,
        "total_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "overall_accuracy": overall_acc,
        "top3_accuracy": top3_acc,
        "conf_matrix": conf_matrix,
        "per_letter": detailed_per_letter,
        "d1_arr": np.array(d1_list),
        "d2_arr": np.array(d2_list),
        "margin_arr": np.array(margins_list),
        "ratio_arr": np.array(ratios_list),
        "conf_arr": np.array(conf_top1_list),
    }


# =========================================================================== #
# Impresión formateada de resultados
# =========================================================================== #

def print_evaluation_report(results: dict, title: str = "REPORTE DE EVALUACIÓN LOSO") -> None:
    letters = results["letters"]
    conf_mat = results["conf_matrix"]
    per_letter = results["per_letter"]

    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f" Total de muestras: {results['total_samples']} | Sujetos: {len(results['subjects'])}")
    print(f" Tiempo total: {results['total_time_s']:.2f} s | Latencia media: {results['avg_latency_ms']:.2f} ms/consulta")
    print(f" Precisión Top-1 Global: {results['overall_accuracy'] * 100:.2f}%")
    print(f" Precisión Top-3 Global: {results['top3_accuracy'] * 100:.2f}%")

    print("\n" + "-" * 80)
    print(" 1. PRECISIÓN TOP-1 POR LETRA")
    print("-" * 80)
    print(f"{'Letra':<6} | {'Aciertos':<10} | {'Total':<8} | {'Precisión':<12} | {'Margen Medio (d2-d1)':<22} | {'Ratio Medio (d2/d1)':<20}")
    print("-" * 80)
    for l in letters:
        c = per_letter[l]["correct"]
        t = per_letter[l]["total"]
        acc = (c / t) * 100 if t > 0 else 0.0
        m = np.mean(per_letter[l]["margins"])
        r = np.mean(per_letter[l]["ratios"])
        print(f"{l:<6} | {c:8d}   | {t:6d}   | {acc:10.2f}% | {m:20.4f}   | {r:18.2f}")

    print("\n" + "-" * 80)
    print(" 2. MATRIZ DE CONFUSIÓN COMPLETA (Filas: Real, Columnas: Predicho)")
    print("-" * 80)
    header = f"{'Real \\ Pred':<12}" + "".join(f"{l:>8}" for l in letters) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for tl in letters:
        row_counts = [conf_mat[tl][pl] for pl in letters]
        row_str = "".join(f"{cnt:8d}" for cnt in row_counts)
        print(f"{tl:<12}{row_str}{sum(row_counts):8d}")

    print("\n" + "-" * 80)
    print(" 3. FOCO EN SEÑAS PROBLEMÁTICAS: K, Q, Z")
    print("-" * 80)
    for focus_letter in ["K", "Q", "Z"]:
        t = per_letter[focus_letter]["total"]
        c = per_letter[focus_letter]["correct"]
        acc = (c / t) * 100 if t > 0 else 0.0
        print(f" Letra '{focus_letter}': Precisión Top-1 = {acc:.2f}% ({c}/{t})")
        errors = [(pl, conf_mat[focus_letter][pl]) for pl in letters if pl != focus_letter and conf_mat[focus_letter][pl] > 0]
        if errors:
            print("   Confundida con:")
            for pl, cnt in sorted(errors, key=lambda x: x[1], reverse=True):
                print(f"     - '{pl}': {cnt} veces ({cnt/t*100:.1f}%)")
        else:
            print("   Sin confusiones registradas en el dataset.")

    print("\n" + "-" * 80)
    print(" 4. DISTRIBUCIÓN DE DISTANCIAS TOP-1 VS TOP-2")
    print("-" * 80)
    d1 = results["d1_arr"]
    d2 = results["d2_arr"]
    m = results["margin_arr"]
    r = results["ratio_arr"]
    conf = results["conf_arr"]

    print(f" Distancia Top-1 (d1):   Media={d1.mean():.4f} ± {d1.std():.4f} | Mediana={np.median(d1):.4f} | Min={d1.min():.4f} | Max={d1.max():.4f}")
    print(f" Distancia Top-2 (d2):   Media={d2.mean():.4f} ± {d2.std():.4f} | Mediana={np.median(d2):.4f} | Min={d2.min():.4f} | Max={d2.max():.4f}")
    print(f" Margen (d2 - d1):       Media={m.mean():.4f} ± {m.std():.4f} | Mediana={np.median(m):.4f} | p10={np.percentile(m, 10):.4f} | p90={np.percentile(m, 90):.4f}")
    print(f" Ratio (d2 / d1):        Media={r.mean():.2f} ± {r.std():.2f} | Mediana={np.median(r):.2f}")
    print(f" Confianza Softmax Top1: Media={conf.mean()*100:.2f}% ± {conf.std()*100:.2f}% | Mediana={np.median(conf)*100:.2f}% | Min={conf.min()*100:.2f}%")
    print("=" * 80 + "\n")


# =========================================================================== #
# Función principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación Leave-One-Subject-Out de DTWRecognizer.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Ruta de datos_dinamicas")
    parser.add_argument("--hand-agnostic", action="store_true", help="Evaluar también con espejo e inversión de manos")
    parser.add_argument("--resample", type=int, default=None, help="Remuestrear secuencias a N frames fijos (ej. 45 o 60)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura del Softmax (default: 1.0)")
    args = parser.parse_args()

    # Si se corre en Windows, asegurar codificación UTF-8
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    samples = load_dynamic_dataset(args.data_dir)
    if not samples:
        print(f"[ERROR] No se encontraron muestras válidas en {args.data_dir}")
        return 1

    results = evaluate_loso(
        samples,
        hand_agnostic=args.hand_agnostic,
        resample_len=args.resample,
        temperature=args.temperature,
    )
    print_evaluation_report(results, title="VALIDACIÓN LEAVE-ONE-SUBJECT-OUT (LOSO) - DTW LSM")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
