"""LOSO (Leave-One-Subject-Out) comparando galeria SOLO FRONTAL vs FRONTAL+PERFIL,
ahora que datos_dinamicas/ ya tiene ambas vistas como plantillas de produccion
(ver convertir_perfil_a_plantillas.py).

No reinventa el motor de evaluacion: reutiliza evaluar_consultas_contra_galeria()
y calcular_metricas_resumen() de evaluar_perfil_y_umbral.py (mismo DTW, mismo
criterio top-1). Lo unico nuevo aqui es el LOSO por sujeto sobre DOS vistas a
la vez: al dejar fuera al sujeto S, se excluyen sus muestras de AMBAS vistas
de la galeria (nunca solo una), para no filtrar identidad de la misma persona
desde el otro angulo.

Dos comparaciones, mismo par de galerias (frontal-only vs frontal+perfil):
  1. Consultas FRONTALES (mismo criterio que el baseline ya documentado en
     ESTADO_PROYECTO_COMPLETO.md, seccion 5.1): mide si agregar perfil a la
     galeria ayuda a reconocer mejor una seña vista de frente.
  2. Consultas de PERFIL: mide si la galeria frontal-only (que nunca vio ese
     angulo) puede reconocer una seña de perfil, y si agregar perfil a la
     galeria lo resuelve - una capacidad nueva, no solo una mejora marginal.

Uso:
    python evaluar_loso_frontal_vs_combinado.py
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from evaluar_perfil_y_umbral import (
    CLASES_DINAMICAS,
    calcular_metricas_resumen,
    evaluar_consultas_contra_galeria,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("loso_combinado")

REPO_DIR = Path(__file__).resolve().parent
DATOS_DIR = REPO_DIR / "datos_dinamicas"

PAT_FRONTAL = re.compile(r"^muestra_(\d+)_(\d+)\.json$")
PAT_PERFIL = re.compile(r"^muestra_(\d+)_perfil_(\d+)\.json$")


def cargar_vista(patron: re.Pattern) -> List[dict]:
    """Carga todas las plantillas de datos_dinamicas/ que matchean `patron`
    (frontal o perfil), ya en formato {letra, sujeto, rep, archivo, frames}."""
    items: List[dict] = []
    for letra_dir in sorted(DATOS_DIR.iterdir()):
        if not letra_dir.is_dir() or letra_dir.name not in CLASES_DINAMICAS:
            continue
        letra = letra_dir.name
        for f in sorted(letra_dir.glob("*.json")):
            m = patron.match(f.name)
            if not m:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                frames = data.get("frames")
                if not frames:
                    continue
                import numpy as np
                arr = np.array(frames, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 126:
                    continue
            except Exception as e:
                log.warning("Error leyendo %s: %s", f.name, e)
                continue
            items.append({
                "letra": letra, "sujeto": m.group(1), "rep": m.group(2),
                "archivo": f.name, "frames": arr,
            })
    return items


def loso_multi_vista(
    consultas_base: List[dict],
    galeria_base: List[dict],
    clases: List[str] = CLASES_DINAMICAS,
) -> List[dict]:
    """LOSO: para cada sujeto presente en `consultas_base`, evalua sus
    muestras contra `galeria_base` SIN las muestras de ese mismo sujeto
    (en ninguna vista - `galeria_base` puede mezclar frontal+perfil)."""
    sujetos = sorted(set(c["sujeto"] for c in consultas_base), key=lambda x: int(x) if x.isdigit() else x)
    por_sujeto_consulta = defaultdict(list)
    for c in consultas_base:
        por_sujeto_consulta[c["sujeto"]].append(c)

    resultados: List[dict] = []
    for test_sub in sujetos:
        galeria_sin_sujeto = [g for g in galeria_base if g["sujeto"] != test_sub]
        consultas_sub = por_sujeto_consulta[test_sub]
        res = evaluar_consultas_contra_galeria(consultas_sub, galeria_sin_sujeto, clases)
        for r, orig in zip(res, consultas_sub):
            r["sujeto"] = test_sub
            r["rep"] = orig["rep"]
            resultados.append(r)
    return resultados


def imprimir_comparacion(titulo: str, top1_a: Dict[str, float], n_a: Dict[str, int],
                          top1_b: Dict[str, float], n_b: Dict[str, int]) -> None:
    print(f"\n{titulo}")
    print("-" * 70)
    print(f"{'Letra':<8} | {'(a) Solo frontal':<20} | {'(b) Frontal+Perfil':<20} | Delta")
    print("-" * 70)
    letras = sorted(set(list(top1_a.keys()) + list(top1_b.keys())))
    for l in letras:
        a, na = top1_a.get(l, 0.0), n_a.get(l, 0)
        b, nb = top1_b.get(l, 0.0), n_b.get(l, 0)
        delta = b - a
        print(f"{l:<8} | {a:>6.1f}% (n={na:<4}) | {b:>6.1f}% (n={nb:<4}) | {delta:+6.1f} pts")
    print("-" * 70)


def print_matriz(mat: Dict[str, Dict[str, int]], titulo: str, clases: List[str] = CLASES_DINAMICAS) -> None:
    print(f"\n{titulo}")
    header = f"{'Real \\ Pred':<12} | " + " | ".join(f"{c:>4}" for c in clases) + " | Total"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for tl in clases:
        row = [f"{mat[tl][c]:>4}" for c in clases]
        tot_row = sum(mat[tl][c] for c in clases)
        print(f"{tl:<12} | " + " | ".join(row) + f" | {tot_row:>5}")
    print("-" * len(header))


def guardar_cache(path: Path, resultados: List[dict]) -> None:
    """Guarda solo los campos serializables (sin dists_por_clase, que trae
    numpy floats) para poder reimprimir top-1/matriz sin correr el LOSO de
    nuevo (~35 min)."""
    ligero = [
        {k: v for k, v in r.items() if k != "dists_por_clase"}
        for r in resultados
    ]
    path.write_text(json.dumps(ligero), encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    log.info("Cargando plantillas frontales y de perfil desde datos_dinamicas/...")
    frontal = cargar_vista(PAT_FRONTAL)
    perfil = cargar_vista(PAT_PERFIL)
    log.info("Frontal: %d plantillas. Perfil: %d plantillas.", len(frontal), len(perfil))

    galeria_a = frontal                 # (a) baseline: solo frontal
    galeria_b = frontal + perfil        # (b) combinada: frontal + perfil

    # --------------------------------------------------------------- #
    # Comparacion 1: consultas FRONTALES (mismo criterio que el baseline
    # ya documentado, seccion 5.1 de ESTADO_PROYECTO_COMPLETO.md).
    # --------------------------------------------------------------- #
    log.info("LOSO (a) consultas frontales vs galeria SOLO frontal...")
    res_frontal_a = loso_multi_vista(frontal, galeria_a)
    guardar_cache(REPO_DIR / "_cache_loso_frontal_a.json", res_frontal_a)
    log.info("LOSO (b) consultas frontales vs galeria FRONTAL+PERFIL...")
    res_frontal_b = loso_multi_vista(frontal, galeria_b)
    guardar_cache(REPO_DIR / "_cache_loso_frontal_b.json", res_frontal_b)

    top1_fa, mat_fa = calcular_metricas_resumen(res_frontal_a)
    top1_fb, mat_fb = calcular_metricas_resumen(res_frontal_b)
    n_fa = {l: sum(1 for r in res_frontal_a if r["letra_real"] == l) for l in CLASES_DINAMICAS}
    n_fb = {l: sum(1 for r in res_frontal_b if r["letra_real"] == l) for l in CLASES_DINAMICAS}

    ok_fa = sum(1 for r in res_frontal_a if r["acierto"])
    ok_fb = sum(1 for r in res_frontal_b if r["acierto"])
    n_tot_f = len(res_frontal_a)

    print("\n" + "=" * 80)
    print("COMPARACION 1: consultas FRONTALES (LOSO, mismo criterio que el baseline documentado)")
    print("=" * 80)
    imprimir_comparacion("Top-1 (%) por letra:", top1_fa, n_fa, top1_fb, n_fb)
    print(f"\nGLOBAL: (a) solo frontal = {ok_fa}/{n_tot_f} ({ok_fa/n_tot_f*100:.2f}%)   "
          f"(b) frontal+perfil = {ok_fb}/{n_tot_f} ({ok_fb/n_tot_f*100:.2f}%)   "
          f"delta = {(ok_fb-ok_fa)/n_tot_f*100:+.2f} pts")
    print_matriz(mat_fa, "Matriz de confusion - Consultas FRONTALES, galeria (a) SOLO FRONTAL:")
    print_matriz(mat_fb, "Matriz de confusion - Consultas FRONTALES, galeria (b) FRONTAL+PERFIL:")

    # --------------------------------------------------------------- #
    # Comparacion 2: consultas de PERFIL (capacidad nueva: reconocer una
    # seña vista de costado, que la galeria solo-frontal nunca vio).
    # --------------------------------------------------------------- #
    log.info("LOSO (a) consultas de perfil vs galeria SOLO frontal...")
    res_perfil_a = loso_multi_vista(perfil, galeria_a)
    guardar_cache(REPO_DIR / "_cache_loso_perfil_a.json", res_perfil_a)
    log.info("LOSO (b) consultas de perfil vs galeria FRONTAL+PERFIL...")
    res_perfil_b = loso_multi_vista(perfil, galeria_b)
    guardar_cache(REPO_DIR / "_cache_loso_perfil_b.json", res_perfil_b)

    top1_pa, mat_pa = calcular_metricas_resumen(res_perfil_a)
    top1_pb, mat_pb = calcular_metricas_resumen(res_perfil_b)
    n_pa = {l: sum(1 for r in res_perfil_a if r["letra_real"] == l) for l in CLASES_DINAMICAS}
    n_pb = {l: sum(1 for r in res_perfil_b if r["letra_real"] == l) for l in CLASES_DINAMICAS}

    ok_pa = sum(1 for r in res_perfil_a if r["acierto"])
    ok_pb = sum(1 for r in res_perfil_b if r["acierto"])
    n_tot_p = len(res_perfil_a)

    print("\n" + "=" * 80)
    print("COMPARACION 2: consultas de PERFIL (¿la galeria puede reconocer ese angulo?)")
    print("=" * 80)
    imprimir_comparacion("Top-1 (%) por letra:", top1_pa, n_pa, top1_pb, n_pb)
    print(f"\nGLOBAL: (a) solo frontal = {ok_pa}/{n_tot_p} ({ok_pa/n_tot_p*100:.2f}%)   "
          f"(b) frontal+perfil = {ok_pb}/{n_tot_p} ({ok_pb/n_tot_p*100:.2f}%)   "
          f"delta = {(ok_pb-ok_pa)/n_tot_p*100:+.2f} pts")
    print_matriz(mat_pa, "Matriz de confusion - Consultas de PERFIL, galeria (a) SOLO FRONTAL:")
    print_matriz(mat_pb, "Matriz de confusion - Consultas de PERFIL, galeria (b) FRONTAL+PERFIL:")

    t_tot = time.perf_counter() - t0
    log.info("Evaluacion completada en %.1f segundos.", t_tot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
