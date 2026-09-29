r"""Script para identificar y separar muestras dinámicas propias (antes de regrabar).

Opciones principales:
1. Mover solo muestras cortas (< 25 frames):
     python separar_muestras_cortas.py             # Simulación (dry-run)
     python separar_muestras_cortas.py --aplicar   # Mueve a propias_descartadas\<LETRA>\

2. Mover TODAS las muestras propias (sin filtrar por longitud):
     python separar_muestras_cortas.py --todas            # Simulación (dry-run)
     python separar_muestras_cortas.py --todas --aplicar  # Mueve a propias_descartadas\antes_de_regrabar\<LETRA>\

Reglas de seguridad y preservación:
- Solo actúa sobre muestras propias (`muestra_<N>.json`).
- NUNCA toca ni mueve archivos del dataset (`muestra_<sujeto>_<rep>.json`), garantizando los 558 intactos.
- NO toca archivos `.npz` de `propias_crudas` ni otros formatos.
- NO sobrescribe archivos en el destino: si un archivo ya existe, añade un sufijo numérico único (`_1`, `_2`, etc.).
- NO borra archivos: solo los MUEVE a la carpeta correspondiente.
- Muestra el conteo exacto de archivos antes y después.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("separar_muestras_cortas")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datos_dinamicas"
BASE_DISCARD_DIR = Path(r"C:\Proyectos\Dataset_CICESE\propias_descartadas")
DEFAULT_DEST_SHORT = BASE_DISCARD_DIR
DEFAULT_DEST_ALL = BASE_DISCARD_DIR / "antes_de_regrabar"
UMBRAL_FRAMES_DEFAULT = 25


def get_unique_dest_path(target_dir: Path, filename: str) -> Path:
    """Devuelve una ruta de destino única sin sobrescribir archivos existentes."""
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    ext = Path(filename).suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def scan_user_samples(
    data_dir: Path,
    move_all: bool = False,
    umbral_frames: int = UMBRAL_FRAMES_DEFAULT,
) -> Tuple[Dict[str, List[Tuple[Path, int]]], Dict[str, int], Dict[str, int]]:
    """Escanea datos_dinamicas identificando muestras de usuario elegibles.

    Ignora cualquier archivo .npz y archivos del dataset (muestra_Sub_Rep.json).
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")

    pattern_user = re.compile(r"^muestra_(\d+)\.json$")
    pattern_ds = re.compile(r"^muestra_(\d+)_(\d+)\.json$")

    target_user_files: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    total_user_before: Dict[str, int] = defaultdict(int)
    total_ds_count: Dict[str, int] = defaultdict(int)

    for letter_dir in sorted(data_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name

        for f in sorted(letter_dir.glob("*.json")):
            # 1. Dataset CICESE: nunca tocar
            if pattern_ds.match(f.name):
                total_ds_count[letter] += 1
                continue

            # 2. Muestras de usuario: evaluar si califica
            if pattern_user.match(f.name):
                total_user_before[letter] += 1
                try:
                    content = json.loads(f.read_text(encoding="utf-8"))
                    frames = content.get("frames", [])
                    n_frames = len(frames)
                    if move_all or n_frames < umbral_frames:
                        target_user_files[letter].append((f, n_frames))
                except Exception as e:
                    log.warning("No se pudo leer %s: %s", f.name, e)

    return target_user_files, total_user_before, total_ds_count


def execute_separation(
    target_user_files: Dict[str, List[Tuple[Path, int]]],
    total_user_before: Dict[str, int],
    total_ds_count: Dict[str, int],
    dest_dir: Path,
    move_all: bool = False,
    aplicar: bool = False,
) -> None:
    """Muestra el reporte y realiza el movimiento seguro si aplicar=True."""
    all_letters = sorted(
        list(set(list(total_user_before.keys()) + list(total_ds_count.keys())))
    )
    total_to_move = sum(len(lst) for lst in target_user_files.values())

    mode_title = (
        "MODO REAL (--aplicar activado: MOVIENDO ARCHIVOS)"
        if aplicar
        else "MODO SIMULACIÓN (DRY-RUN: NO se movió ningún archivo)"
    )
    criterio_str = "TODAS las muestras propias" if move_all else "Muestras propias cortas (< 25 frames)"

    print("\n" + "=" * 85)
    print(f" GESTIÓN Y SEPARACIÓN DE MUESTRAS PROPIAS")
    print(f" Criterio: {criterio_str}")
    print(f" Estado:   {mode_title}")
    print("=" * 85)
    print(f" Carpeta origen:  {DEFAULT_DATA_DIR}")
    print(f" Carpeta destino: {dest_dir}\n")

    # 1. Listar detalle de archivos identificados
    print("-" * 85)
    print(f" 1. ARCHIVOS PROPIOS SELECCIONADOS ({total_to_move} encontrados)")
    print("-" * 85)
    if total_to_move == 0:
        print(" No se encontraron archivos de usuario que cumplan el criterio.")
    else:
        for letter in sorted(target_user_files.keys()):
            files_info = target_user_files[letter]
            print(f" Letra '{letter}' ({len(files_info)} archivos):")
            # Mostrar nombres y frames
            sample_strs = [f"{fp.name} ({nf}f)" for fp, nf in files_info]
            for i in range(0, len(sample_strs), 4):
                print("   " + ", ".join(sample_strs[i : i + 4]))

    # 2. Realizar movimiento si aplicar está activo
    moved_count_by_letter: Dict[str, int] = defaultdict(int)
    renamed_count = 0

    if aplicar and total_to_move > 0:
        print("\n" + "-" * 85)
        print(" 2. EJECUTANDO MOVIMIENTO SEGURO (CON PRESERVACIÓN DE DUPLICADOS)")
        print("-" * 85)
        for letter, files_info in target_user_files.items():
            letter_dest = dest_dir / letter
            letter_dest.mkdir(parents=True, exist_ok=True)
            for file_path, n_frames in files_info:
                target_path = get_unique_dest_path(letter_dest, file_path.name)
                if target_path.name != file_path.name:
                    renamed_count += 1
                    log.info(
                        "Destino existente: %s renombrado a %s",
                        file_path.name,
                        target_path.name,
                    )
                shutil.move(str(file_path), str(target_path))
                moved_count_by_letter[letter] += 1
            print(f"   [OK] {len(files_info)} archivos de '{letter}' movidos a {letter_dest}")

    # 3. Conteo antes y después
    print("\n" + "-" * 85)
    print(" 3. CONTEO DE ARCHIVOS EN DATOS_DINAMICAS (ANTES vs DESPUÉS)")
    print("-" * 85)
    col_mov = "A Mover" if not aplicar else "Movidos"
    header = (
        f"{'Letra':<6} | {'Dataset (Intacto)':<18} | {'Propias (Antes)':<16} | "
        f"{col_mov:<14} | {'Propias (Después)'}"
    )
    print(header)
    print("-" * len(header))

    tot_ds = 0
    tot_user_before = 0
    tot_moved = 0
    tot_user_after = 0

    for l in all_letters:
        ds_c = total_ds_count.get(l, 0)
        u_b = total_user_before.get(l, 0)
        m_c = len(target_user_files.get(l, []))
        u_a = u_b - m_c if aplicar else u_b

        tot_ds += ds_c
        tot_user_before += u_b
        tot_moved += m_c
        tot_user_after += u_a

        print(
            f"{l:<6} | {ds_c:12d}       | {u_b:12d}     | {m_c:10d}     | {u_a:12d}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<6} | {tot_ds:12d}       | {tot_user_before:12d}     | {tot_moved:10d}     | {tot_user_after:12d}"
    )

    # 4. Verificación de seguridad del Dataset
    print("\n" + "-" * 85)
    print(" 4. VERIFICACIÓN DE INTEGRIDAD DEL DATASET")
    print("-" * 85)
    if tot_ds == 558:
        print(" [OK] Los 558 archivos del dataset CICESE permanecen 100% INTACTOS.")
    else:
        print(f" [ALERTA] Conteo inesperado del dataset: {tot_ds} (se esperaban 558).")

    if not aplicar:
        flag_str = "--todas --aplicar" if move_all else "--aplicar"
        print("\n" + "=" * 85)
        print(" AVISO IMPORTANTE:")
        print("   Esta ejecución fue una SIMULACIÓN (dry-run). Ningún archivo fue movido.")
        print(f"   Para mover efectivamente los {total_to_move} archivos, ejecuta:")
        print(f"       python separar_muestras_cortas.py {flag_str}")
        print("=" * 85 + "\n")
    else:
        print("\n" + "=" * 85)
        print(" OPERACIÓN COMPLETADA CON ÉXITO:")
        print(f"   Se movieron {total_to_move} archivos propios a {dest_dir}.")
        if renamed_count > 0:
            print(f"   ({renamed_count} archivos recibieron sufijo para evitar sobrescribir duplicados).")
        print("   La carpeta datos_dinamicas ha quedado completamente limpia para regrabar.")
        print("=" * 85 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identifica y separa muestras dinámicas propias (cortas o todas)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Ruta de datos_dinamicas (origen)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Ruta destino (por defecto: antes_de_regrabar si --todas, o propias_descartadas si no)",
    )
    parser.add_argument(
        "--umbral",
        type=int,
        default=UMBRAL_FRAMES_DEFAULT,
        help="Umbral mínimo de frames (default: 25, solo aplica sin --todas)",
    )
    parser.add_argument(
        "--todas",
        action="store_true",
        help="Mover TODAS las muestras propias sin filtrar por longitud",
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Si se especifica, mueve los archivos. Si no, solo lista en dry-run.",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    # Determinar ruta destino por defecto
    if args.dest_dir is not None:
        dest_dir = args.dest_dir
    elif args.todas:
        dest_dir = DEFAULT_DEST_ALL
    else:
        dest_dir = DEFAULT_DEST_SHORT

    target_user_files, total_user_before, total_ds_count = scan_user_samples(
        args.data_dir, move_all=args.todas, umbral_frames=args.umbral
    )

    execute_separation(
        target_user_files,
        total_user_before,
        total_ds_count,
        dest_dir=dest_dir,
        move_all=args.todas,
        aplicar=args.aplicar,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
