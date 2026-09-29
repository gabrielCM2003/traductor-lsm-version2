"""Diagnóstico de orientación en el plano de la imagen (ángulo muñeca -> base del dedo medio).

1. Para cada muestra del dataset y propias: calcula el ángulo atan2(y, x) del vector muñeca->dedo medio
   (landmark 9: x en índice 27, y en índice 28 del bloque de 63).
2. Estadísticas de orientación por letra y origen (dataset vs propias): mediana, cuartiles (Q1, Q3),
   rango [min, max] y desv. estándar interna (cuánto gira la mano dentro de la seña).
3. Tabla comparativa de diferencias angulares (Q, K, Z, J).
4. Experimento de rotación 2D: rota las consultas propias por la diferencia mediana de ángulo respecto
   al dataset y evalúa si mejora el Top-1 de Q y K frente a la galería frontal.
5. Conclusiones y limitaciones estadísticas.
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
log = logging.getLogger("diagnostico_orientacion")

CLASES_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]


# =========================================================================== #
# Programación Dinámica DTW optimizada con Numba / NumPy
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
# Análisis de ángulos y orientación
# =========================================================================== #

def calcular_angulos_muestra(frames: np.ndarray) -> np.ndarray:
    """Calcula para cada frame con bloque izquierdo no nulo el ángulo en grados del
    vector muñeca -> base del dedo medio (landmark 9: x en 27, y en 28)."""
    angulos = []
    for f in frames:
        if np.all(f[:63] == 0.0):
            continue
        x = f[27]
        y = f[28]
        # atan2 devuelve radianes en [-pi, pi], convertimos a grados [-180, 180]
        ang = float(np.degrees(np.arctan2(y, x)))
        angulos.append(ang)
    return np.array(angulos, dtype=np.float64)


def rotar_secuencia_2d(frames: np.ndarray, angulo_grados: float) -> np.ndarray:
    """Aplica rotación 2D en los planos (x, y) de los landmarks de cada mano por angulo_grados.
    La componente z permanece inalterada."""
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

def cargar_muestras_con_angulos(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Carga todas las muestras de datos_dinamicas distinguiendo dataset y usuario."""
    pat_ds = re.compile(r"^muestra_(\d+)_(\d+)\.json$")
    pat_pr = re.compile(r"^muestra_(\d+)\.json$")

    muestras_ds = []
    muestras_pr = []

    for l_dir in sorted(data_dir.iterdir()):
        if not l_dir.is_dir():
            continue
        letra = l_dir.name

        for json_file in sorted(l_dir.glob("*.json")):
            m_ds = pat_ds.match(json_file.name)
            m_pr = pat_pr.match(json_file.name)
            if not (m_ds or m_pr):
                continue

            try:
                content = json.loads(json_file.read_text(encoding="utf-8"))
                frames = np.array(content["frames"], dtype=np.float64)
                if frames.ndim != 2 or frames.shape[1] != 126 or len(frames) == 0:
                    continue

                angulos = calcular_angulos_muestra(frames)
                if len(angulos) == 0:
                    continue

                info = {
                    "letra": letra,
                    "archivo": json_file.name,
                    "frames": frames,
                    "angulos": angulos,
                    "mediana_angulo": float(np.median(angulos)),
                    "media_angulo": float(np.mean(angulos)),
                    "std_angulo": float(np.std(angulos)),
                    "min_angulo": float(np.min(angulos)),
                    "max_angulo": float(np.max(angulos)),
                }

                if m_ds:
                    info["sujeto"] = m_ds.group(1)
                    info["rep"] = m_ds.group(2)
                    muestras_ds.append(info)
                elif m_pr:
                    info["id"] = m_pr.group(1)
                    muestras_pr.append(info)
            except Exception as e:
                log.warning("Error leyendo %s: %s", json_file.name, e)

    return muestras_ds, muestras_pr


# =========================================================================== #
# Evaluación DTW
# =========================================================================== #

def evaluar_clasificacion(
    consultas: List[dict],
    galeria_ds: List[dict],
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """Evalúa consultas contra galería usando DTW 1-NN por clase."""
    galeria_por_clase = defaultdict(list)
    for g in galeria_ds:
        galeria_por_clase[g["letra"]].append(g["frames"])

    resultados = []
    for q in consultas:
        q_frames = q["frames"]
        dists = {}
        for c in clases:
            tmpls = galeria_por_clase[c]
            min_d = float("inf")
            for t in tmpls:
                c_mat = cdist(q_frames, t, metric="euclidean")
                d = dtw_distance(c_mat)
                if d < min_d:
                    min_d = d
            dists[c] = min_d

        ranking = sorted(dists.items(), key=lambda x: x[1])
        top1, d1 = ranking[0]
        top2, d2 = ranking[1]
        ratio = d2 / d1 if d1 > 0 else float("inf")

        resultados.append({
            "archivo": q["archivo"],
            "letra_real": q["letra"],
            "pred_top1": top1,
            "pred_top2": top2,
            "d1": d1,
            "d2": d2,
            "ratio": ratio,
            "d_propia": dists.get(q["letra"], float("inf")),
            "d_z": dists.get("Z", float("inf")),
            "acierto": (top1 == q["letra"]),
        })
    return resultados


# =========================================================================== #
# Programa principal
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de orientación angular (dataset vs propias)")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    data_dir = args.repo_dir / "datos_dinamicas"
    log.info("Cargando muestras y calculando ángulos muñeca -> dedo medio...")
    muestras_ds, muestras_pr = cargar_muestras_con_angulos(data_dir)

    print("\n" + "=" * 95)
    print("DIAGNÓSTICO DE ORIENTACIÓN ANGULAR EN EL PLANO DE LA IMAGEN (atan2(y, x))")
    print("=" * 95)
    print(f"Dataset CICESE: {len(muestras_ds)} muestras cargadas")
    print(f"Muestras propias: {len(muestras_pr)} muestras cargadas (J: 3, K: 8, Q: 9, Z: 8)")

    # ----------------------------------------------------------------------- #
    # PUNTO 2: Distribución por letra y origen
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 95)
    print("PUNTO 2: ESTADÍSTICAS DE ORIENTACIÓN ANGULAR POR LETRA Y ORIGEN")
    print("         (Ángulo del vector muñeca -> dedo medio en grados [-180°, 180°])")
    print("         Nota: -90° = vertical hacia arriba; 0° = horizontal hacia la derecha")
    print("-" * 95)

    header_p2 = (
        f"{'Origen':<10} | {'Letra':<5} | {'N':<4} | {'Mediana (°)':<11} | "
        f"{'Q1 (°)':<8} | {'Q3 (°)':<8} | {'Rango [Min, Max] (°)':<22} | {'Std Interna Media (°)'}"
    )
    print(header_p2)
    print("-" * len(header_p2))

    estadisticas: Dict[Tuple[str, str], dict] = {}

    for origen_nombre, lista in [("Dataset", muestras_ds), ("Propias", muestras_pr)]:
        por_letra = defaultdict(list)
        for m in lista:
            por_letra[m["letra"]].append(m)

        for l in sorted(por_letra.keys()):
            items = por_letra[l]
            meds = np.array([it["mediana_angulo"] for it in items])
            stds = np.array([it["std_angulo"] for it in items])

            mediana_glob = float(np.median(meds))
            q1 = float(np.percentile(meds, 25))
            q3 = float(np.percentile(meds, 75))
            val_min = float(np.min(meds))
            val_max = float(np.max(meds))
            std_interna_m = float(np.mean(stds))

            estadisticas[(origen_nombre, l)] = {
                "mediana": mediana_glob,
                "q1": q1,
                "q3": q3,
                "min": val_min,
                "max": val_max,
                "std_interna": std_interna_m,
                "n": len(items),
            }

            rango_str = f"[{val_min:6.1f}°, {val_max:6.1f}°]"
            print(
                f"{origen_nombre:<10} | {l:<5} | {len(items):<4} | {mediana_glob:>10.1f}° | "
                f"{q1:>7.1f}° | {q3:>7.1f}° | {rango_str:<22} | {std_interna_m:>18.1f}°"
            )

    # ----------------------------------------------------------------------- #
    # PUNTO 3: Tabla comparativa de diferencias angulares
    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 95)
    print("PUNTO 3: TABLA COMPARATIVA DE DIFERENCIAS ANGULARES (PROPIAS vs DATASET)")
    print("-" * 95)

    header_p3 = (
        f"{'Letra':<6} | {'N (Prop)':<8} | {'Mediana Prop (°)':<16} | {'Mediana DS (°)':<15} | "
        f"{'Diferencia (DS - Prop)':<22} | {'Std Int. Prop (°)':<17} | {'Std Int. DS (°)'}"
    )
    print(header_p3)
    print("-" * len(header_p3))

    deltas_angulo = {}
    for l in ["J", "K", "Q", "Z"]:
        s_pr = estadisticas.get(("Propias", l))
        s_ds = estadisticas.get(("Dataset", l))
        if not (s_pr and s_ds):
            continue

        ang_pr = s_pr["mediana"]
        ang_ds = s_ds["mediana"]
        diff = ang_ds - ang_pr
        deltas_angulo[l] = diff

        diff_str = f"{diff:>+6.1f}° ({'horiz. en DS' if diff > 0 else 'vert. en DS'})"
        print(
            f"{l:<6} | {s_pr['n']:<8} | {ang_pr:>14.1f}°  | {ang_ds:>13.1f}° | "
            f"{diff_str:<22} | {s_pr['std_interna']:>15.1f}°  | {s_ds['std_interna']:>13.1f}°"
        )

    # ----------------------------------------------------------------------- #
    # PUNTO 4: Experimento de rotación 2D sobre consultas propias
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 95)
    print("PUNTO 4: EXPERIMENTO DE ROTACIÓN 2D POR DIFERENCIA MEDIANA DE ÁNGULO")
    print("         (Se rota la consulta propia en x,y por Delta = Mediana(DS) - Mediana(Prop); z intacto)")
    print("=" * 95)

    # Agrupar muestras propias por letra
    propias_por_letra = defaultdict(list)
    for m in muestras_pr:
        propias_por_letra[m["letra"]].append(m)

    # Evaluaciones con y sin rotación
    print("\nEvaluando clasificación DTW 1-NN antes y después de la rotación...")

    resumen_exp = []
    detalles_rotacion = []

    for l in ["K", "Q", "Z", "J"]:
        items = propias_por_letra[l]
        delta_rot = deltas_angulo[l]

        # 1. Sin rotar (original)
        res_orig = evaluar_clasificacion(items, muestras_ds)

        # 2. Rotadas
        items_rot = []
        for it in items:
            fr_rot = rotar_secuencia_2d(it["frames"], delta_rot)
            items_rot.append({
                "archivo": it["archivo"],
                "letra": it["letra"],
                "frames": fr_rot,
            })
        res_rot = evaluar_clasificacion(items_rot, muestras_ds)

        top1_orig = sum(1 for r in res_orig if r["acierto"])
        top1_rot = sum(1 for r in res_rot if r["acierto"])
        n_l = len(items)

        resumen_exp.append({
            "letra": l,
            "n": n_l,
            "delta_rot": delta_rot,
            "top1_orig": top1_orig,
            "top1_rot": top1_rot,
            "pct_orig": top1_orig / n_l * 100,
            "pct_rot": top1_rot / n_l * 100,
        })

        for ro, rr in zip(res_orig, res_rot):
            detalles_rotacion.append({
                "archivo": ro["archivo"],
                "letra": l,
                "pred_orig": ro["pred_top1"],
                "d1_orig": ro["d1"],
                "d_own_orig": ro["d_propia"],
                "d_z_orig": ro["d_z"],
                "pred_rot": rr["pred_top1"],
                "d1_rot": rr["d1"],
                "d_own_rot": rr["d_propia"],
                "d_z_rot": rr["d_z"],
            })

    print("\nTABLA 4.1: Resumen de Top-1 antes y después de rotar la consulta:")
    print("-" * 80)
    print(f"{'Letra':<6} | {'N':<4} | {'Rotación Aplicada':<18} | {'Top-1 Original':<18} | {'Top-1 Rotada (Experimento)'}")
    print("-" * 80)
    for r in resumen_exp:
        print(
            f"{r['letra']:<6} | {r['n']:<4} | {r['delta_rot']:>+6.1f}°            | "
            f"{r['top1_orig']:>2}/{r['n']} ({r['pct_orig']:>5.1f}%)        | "
            f"{r['top1_rot']:>2}/{r['n']} ({r['pct_rot']:>5.1f}%)"
        )
    print("-" * 80)

    print("\nTABLA 4.2: Detalle por muestra propia (Impacto en distancias hacia clase propia y hacia Z):")
    print("-" * 105)
    print(
        f"{'Archivo':<18} | {'Real':<4} | {'Pred Orig':<9} | {'d(prop) Orig':<12} | {'d(Z) Orig':<10} | "
        f"{'Pred Rot':<8} | {'d(prop) Rot':<11} | {'d(Z) Rot':<9} | {'Cambio d(prop)'}"
    )
    print("-" * 105)
    for d in detalles_rotacion:
        delta_d_own = d["d_own_rot"] - d["d_own_orig"]
        print(
            f"{d['archivo']:<18} | {d['letra']:<4} | {d['pred_orig']:<9} | "
            f"{d['d_own_orig']:>10.4f}   | {d['d_z_orig']:>8.4f}   | "
            f"{d['pred_rot']:<8} | {d['d_own_rot']:>9.4f}   | {d['d_z_rot']:>7.4f}   | "
            f"{delta_d_own:>+10.4f}"
        )
    print("-" * 105)

    # Experimento adicional: Barrido de ángulos para Q (0° a 90° en pasos de 10°)
    print("\nEXPERIMENTO ADICIONAL: Barrido angular sistemático en Q (-20° a +80° respecto a la propia):")
    print("-" * 80)
    q_items = propias_por_letra["Q"]
    for ang in range(-20, 91, 10):
        q_rot = [{"archivo": it["archivo"], "letra": "Q", "frames": rotar_secuencia_2d(it["frames"], ang)} for it in q_items]
        res_q_ang = evaluar_clasificacion(q_rot, muestras_ds)
        aciertos = sum(1 for r in res_q_ang if r["acierto"])
        preds = defaultdict(int)
        for r in res_q_ang:
            preds[r["pred_top1"]] += 1
        d_prop_m = np.mean([r["d_propia"] for r in res_q_ang])
        d_z_m = np.mean([r["d_z"] for r in res_q_ang])
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(preds.items()))
        print(f"  Rotación {ang:>+3d}°: Aciertos Q = {aciertos}/{len(q_items)} ({aciertos/len(q_items)*100:4.1f}%) | "
              f"d(Q) media={d_prop_m:.4f} | d(Z) media={d_z_m:.4f} | Predicciones: [{preds_str}]")

    print("\n" + "=" * 95)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
