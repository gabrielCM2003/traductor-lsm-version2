"""Convierte los landmarks crudos de la vista de PERFIL de CICESE
(C:\\Proyectos\\Dataset_CICESE\\landmarks_crudos\\<LETRA>\\S<sujeto>_perfil_<rep>.npz)
a plantillas de produccion en datos_dinamicas/<LETRA>/, mismo formato que ya
usa procesar_dataset_dinamico.py para la vista frontal.

No reinventa la reconstruccion: reutiliza reconstruir_desde_npz() de
evaluar_perfil_y_umbral.py (ya validada en ese experimento), que a su vez
usa hand_to_feature_vector/normalize_keypoints de sign_classifier.py sin
cambios. Mismo criterio de longitud minima que procesar_dataset_dinamico.py
(MIN_SEQUENCE_FRAMES = 3).

Nombre de archivo con "_perfil_" a proposito (muestra_<sujeto>_perfil_<rep>.json)
para no chocar nunca con:
  - las plantillas frontales del dataset: muestra_<sujeto>_<rep>.json
  - las muestras propias grabadas a mano: muestra_<N>.json (un solo numero)
dtw_recognizer.py no distingue por nombre de archivo (carga cualquier *.json
en la carpeta de la letra), asi que no hace falta tocar su logica de carga.

NO sobrescribe ningun archivo existente: si el nombre de destino ya existe,
se omite con aviso.

Uso:
    python convertir_perfil_a_plantillas.py [--dataset-dir RUTA] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from evaluar_perfil_y_umbral import reconstruir_desde_npz, CLASES_DINAMICAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("convertir_perfil")

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_DIR / "datos_dinamicas"
N_FEATURES = 126

# Mismo criterio que procesar_dataset_dinamico.py (vista frontal): descarta
# secuencias degeneradas de menos de 3 frames con mano.
MIN_SEQUENCE_FRAMES = 3

PAT_PERFIL = re.compile(r"^S(\d+)_perfil_(\d+)\.npz$")


def convertir(dataset_dir: Path, dry_run: bool = False) -> dict:
    raw_dir = dataset_dir / "landmarks_crudos"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No existe {raw_dir}")

    stats = {"convertidas": 0, "omitidas_ya_existe": 0, "omitidas_muy_corta": 0, "errores": 0}
    conteo_por_letra: dict[str, int] = {}

    for letra_dir in sorted(raw_dir.iterdir()):
        if not letra_dir.is_dir() or letra_dir.name not in CLASES_DINAMICAS:
            continue
        letra = letra_dir.name
        destino_dir = OUTPUT_ROOT / letra

        archivos = sorted(letra_dir.glob("*_perfil_*.npz"))
        for f in archivos:
            m = PAT_PERFIL.match(f.name)
            if not m:
                log.warning("  %s no sigue el patron S<sujeto>_perfil_<rep>.npz, se ignora", f.name)
                continue
            sujeto, rep = m.group(1), m.group(2)

            destino = destino_dir / f"muestra_{sujeto}_perfil_{rep}.json"
            if destino.exists():
                log.warning("  %s ya existe, se omite (no se sobrescribe)", destino.name)
                stats["omitidas_ya_existe"] += 1
                continue

            try:
                frames = reconstruir_desde_npz(f)
            except Exception as e:
                log.warning("  error reconstruyendo %s: %s", f.name, e)
                stats["errores"] += 1
                continue

            if frames is None or len(frames) < MIN_SEQUENCE_FRAMES:
                n = 0 if frames is None else len(frames)
                log.warning("  %s descartada: solo %d frames con mano (< %d)", f.name, n, MIN_SEQUENCE_FRAMES)
                stats["omitidas_muy_corta"] += 1
                continue

            if not dry_run:
                destino_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "n_frames": len(frames),
                    "n_features": N_FEATURES,
                    "frames": [vec.tolist() for vec in frames],
                }
                destino.write_text(json.dumps(payload), encoding="utf-8")

            stats["convertidas"] += 1
            conteo_por_letra[letra] = conteo_por_letra.get(letra, 0) + 1

    stats["por_letra"] = conteo_por_letra
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"C:\Proyectos\Dataset_CICESE"))
    parser.add_argument("--dry-run", action="store_true", help="no escribe nada, solo cuenta")
    args = parser.parse_args()

    stats = convertir(args.dataset_dir, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Conversion perfil -> datos_dinamicas/ completada")
    print("=" * 60)
    print(f"Convertidas:          {stats['convertidas']}")
    print(f"Omitidas (ya existia): {stats['omitidas_ya_existe']}")
    print(f"Omitidas (muy corta):  {stats['omitidas_muy_corta']}")
    print(f"Errores:               {stats['errores']}")
    print(f"Por letra: {dict(sorted(stats['por_letra'].items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
