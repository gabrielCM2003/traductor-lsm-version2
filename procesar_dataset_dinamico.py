"""Descarga y procesa el dataset abierto de letras dinamicas de la LSM (Zenodo)
para llenar datos_dinamicas/<LETRA>/ con el mismo formato que produce
recolector_dinamico.py, de modo que dtw_recognizer.py las reconozca sin cambios.

Dataset: "Mexican Sign Language Alphabet (dynamic signs only)"
DOI: 10.5281/zenodo.14689869 (CC BY 4.0) - Navarrete-Lopez & Lopez-Nava, CICESE.
Se descarga solo MSL-dynamic-signs-frontal-view.7z (~2.2 GB), se omite la vista
de perfil. Los videos siguen la convencion S<SubjectId>-<Letter>-<View>-<Rep>,
repartidos en subcarpetas train/ y test/ que aqui se recorren indistintamente
(no nos interesa esa particion, solo maximizar plantillas por letra).

La letra "Ñ" no se documenta en el nombre exacto que usa el dataset dentro del
archivo, asi que este script la detecta por eliminacion: cualquier token de
letra en los nombres de archivo que NO sea J/K/Q/X/Z se asume que es la
codificacion usada para "Ñ" (ver find_letter_tokens/build_letter_mapping).
El mapeo detectado se imprime siempre para poder verificarlo a simple vista.

No reescribe la normalizacion de keypoints ni la logica de deteccion de manos:
reutiliza directamente las funciones de recolector_dinamico.py (que a su vez
reutiliza sign_classifier.py y senas.py).

Uso:
    python procesar_dataset_dinamico.py [--skip-download] [--cleanup]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

import recolector_dinamico as rd

DATASET_URL = (
    "https://zenodo.org/api/records/14689869/files/"
    "MSL-dynamic-signs-frontal-view.7z/content"
)
ARCHIVE_NAME = "MSL-dynamic-signs-frontal-view.7z"

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "_dataset_dinamico_raw"
ARCHIVE_PATH = DOWNLOAD_DIR / ARCHIVE_NAME
EXTRACT_DIR = DOWNLOAD_DIR / "extracted"
OUTPUT_ROOT = BASE_DIR / "datos_dinamicas"

TARGET_LETTERS_ASCII = {"J", "K", "Q", "X", "Z"}
TARGET_LETTERS_ALL = TARGET_LETTERS_ASCII | {"Ñ"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

NAME_RE = re.compile(r"^S(\d+)-([^-]+)-([^-]+)-(\d+)$", re.IGNORECASE)
MIN_SEQUENCE_FRAMES = 3

# Zenodo bloquea con 403 los User-Agent genericos tipo "Mozilla/5.0" a secas
# (asi lo trata su WAF, no es un problema de permisos del dataset); hace
# falta un UA con pinta de navegador real.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


# =========================================================================== #
# 1. Descarga
# =========================================================================== #

def remote_content_length(url: str) -> Optional[int]:
    req = urllib.request.Request(url, headers=HTTP_HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.getheader("Content-Length")
            return int(cl) if cl else None
    except OSError:
        return None


def download_archive(url: str, dest: Path) -> None:
    expected_size = remote_content_length(url)
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"Ya existe {dest.name} ({expected_size / 1e6:.0f} MB), se omite la descarga.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Descargando {url}")
    if expected_size:
        print(f"  tamano esperado: {expected_size / 1e6:.0f} MB")

    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    downloaded = 0
    last_pct = -1
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if expected_size:
                pct = int(downloaded * 100 / expected_size)
                if pct != last_pct:
                    print(f"\r  {pct}% ({downloaded / 1e6:.0f}/{expected_size / 1e6:.0f} MB)",
                          end="", flush=True)
                    last_pct = pct
    print()
    tmp.rename(dest)
    print(f"Descarga completa: {dest}")


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Ya existe contenido extraido en {extract_dir}, se omite la extraccion.")
        return

    import py7zr  # import perezoso: solo hace falta si realmente hay que extraer

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extrayendo {archive_path.name} (puede tardar varios minutos)...")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)
    print("Extraccion completa.")


# =========================================================================== #
# 2. Descubrimiento de videos y mapeo de la letra "Ñ"
# =========================================================================== #

def find_videos(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]


def parse_video_name(path: Path) -> Optional[tuple[str, str, str, str]]:
    """Extrae (subject_id, letter_token, view_token, rep) de S<Id>-<Letra>-<Vista>-<Rep>."""
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    subject_id, letter_token, view_token, rep = m.groups()
    return subject_id, letter_token, view_token, rep


def build_letter_mapping(letter_tokens: set[str]) -> dict[str, str]:
    """Mapea cada token de letra encontrado en los nombres a su letra canonica.

    Los 5 tokens ASCII (J,K,Q,X,Z) son inequivocos. Cualquier otro token que
    aparezca se asume que es la codificacion usada por el dataset para "Ñ",
    ya que el dataset solo documenta estas 6 letras dinamicas.
    """
    mapping: dict[str, str] = {}
    leftover: list[str] = []
    for token in letter_tokens:
        upper = token.upper()
        if upper in TARGET_LETTERS_ASCII:
            mapping[token] = upper
        else:
            leftover.append(token)

    if leftover:
        print(f"Tokens sin match directo con J/K/Q/X/Z (se asumen como 'Ñ'): {leftover}")
        for token in leftover:
            mapping[token] = "Ñ"
    else:
        print("ADVERTENCIA: no aparecio ningun token adicional para mapear a 'Ñ'.")

    return mapping


# =========================================================================== #
# 3. Extraccion de landmarks por video (reutiliza recolector_dinamico.py)
# =========================================================================== #

def process_video(video_path: Path) -> Optional[list[np.ndarray]]:
    # Un HandLandmarker nuevo por video: en modo VIDEO, MediaPipe exige
    # timestamps estrictamente crecientes durante toda la vida del objeto y
    # ademas reutiliza el estado de tracking del frame anterior (ROI, etc.)
    # para el siguiente. Reusar un solo landmarker entre videos de sujetos
    # distintos rompia esa monotonicidad y ademas contaminaba el tracking
    # de un video con el estado del anterior.
    landmarker = rd.init_hand_landmarker(max_num_hands=2)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  no se pudo abrir {video_path.name}", file=sys.stderr)
        landmarker.close()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step_ms = max(1, round(1000.0 / fps)) if fps > 0 else 33

    raw: list[tuple[bool, np.ndarray]] = []
    timestamp_ms = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Mismo mirror que CameraThread/recolector_dinamico.py: el resto
            # del proyecto siempre extrae landmarks sobre el frame espejado,
            # asi que las plantillas deben construirse igual para que el
            # criterio Left/Right de las 126 features sea consistente.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += frame_step_ms
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands = rd.parse_hands(results)
            vec = rd.build_feature_vector(hands)
            raw.append((bool(hands), vec))
    finally:
        cap.release()
        landmarker.close()

    present = [i for i, (has_hand, _) in enumerate(raw) if has_hand]
    if not present:
        return None
    start, end = present[0], present[-1]
    trimmed = [vec for _, vec in raw[start:end + 1]]
    return trimmed if len(trimmed) >= MIN_SEQUENCE_FRAMES else None


def save_sample(letter: str, subject_id: str, rep: str, sequence: list[np.ndarray]) -> Path:
    word_dir = OUTPUT_ROOT / letter
    word_dir.mkdir(parents=True, exist_ok=True)
    path = word_dir / f"muestra_{subject_id}_{rep}.json"
    payload = {
        "n_frames": len(sequence),
        "n_features": rd.N_FEATURES,
        "frames": [vec.tolist() for vec in sequence],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# =========================================================================== #
# main
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga y procesa el dataset LSM de letras dinamicas (Zenodo)."
    )
    parser.add_argument("--skip-download", action="store_true",
                         help="Usa el .7z/extraido que ya exista en _dataset_dinamico_raw/")
    parser.add_argument("--cleanup", action="store_true",
                         help="Borra el .7z y los videos extraidos al terminar (libera disco)")
    args = parser.parse_args()

    if not args.skip_download:
        download_archive(DATASET_URL, ARCHIVE_PATH)
    elif not ARCHIVE_PATH.exists() and not EXTRACT_DIR.exists():
        print("ERROR: --skip-download pero no hay nada en _dataset_dinamico_raw/", file=sys.stderr)
        return 1

    if ARCHIVE_PATH.exists():
        extract_archive(ARCHIVE_PATH, EXTRACT_DIR)

    videos = find_videos(EXTRACT_DIR)
    print(f"\n{len(videos)} videos encontrados en {EXTRACT_DIR}")
    if not videos:
        print("Nada que procesar.", file=sys.stderr)
        return 1

    parsed: list[tuple[Path, str, str, str, str]] = []
    unmatched = 0
    for video in videos:
        info = parse_video_name(video)
        if info is None:
            unmatched += 1
            continue
        subject_id, letter_token, view_token, rep = info
        parsed.append((video, subject_id, letter_token, view_token, rep))
    if unmatched:
        print(f"  {unmatched} archivos no siguieron el patron S<Id>-<Letra>-<Vista>-<Rep>, se ignoraron.")

    letter_tokens = {letter_token for _, _, letter_token, _, _ in parsed}
    view_tokens = {view_token for _, _, _, view_token, _ in parsed}
    print(f"Tokens de letra encontrados: {sorted(letter_tokens)}")
    print(f"Tokens de vista encontrados: {sorted(view_tokens)}")

    letter_map = build_letter_mapping(letter_tokens)
    print("Mapeo final token -> letra:", letter_map)

    to_process = [
        (video, subject_id, letter_map[letter_token], rep)
        for video, subject_id, letter_token, _, rep in parsed
        if letter_map.get(letter_token) in TARGET_LETTERS_ALL
    ]
    print(f"\n{len(to_process)} videos corresponden a J/K/Ñ/Q/X/Z, procesando...")

    saved_count: dict[str, int] = defaultdict(int)
    skipped_existing = 0
    skipped_no_hand = 0

    for i, (video, subject_id, letter, rep) in enumerate(to_process, start=1):
        out_path = OUTPUT_ROOT / letter / f"muestra_{subject_id}_{rep}.json"
        if out_path.exists():
            skipped_existing += 1
            saved_count[letter] += 1
            continue

        print(f"  [{i}/{len(to_process)}] {video.name} -> {letter}", end="")
        sequence = process_video(video)
        if sequence is None:
            print("  (sin manos detectadas, se omite)")
            skipped_no_hand += 1
            continue

        save_sample(letter, subject_id, rep, sequence)
        saved_count[letter] += 1
        print(f"  ({len(sequence)} frames)")

    print("\n=== Resumen ===")
    for letter in sorted(TARGET_LETTERS_ALL):
        count = len(list((OUTPUT_ROOT / letter).glob("muestra_*.json"))) if (OUTPUT_ROOT / letter).exists() else 0
        print(f"  {letter}: {count} muestras")
    print(f"(saltadas por ya existir: {skipped_existing}, sin manos detectadas: {skipped_no_hand})")
    print(f"\nDatos listos en {OUTPUT_ROOT} - dtw_recognizer.py los detectara solo con instanciarse de nuevo.")

    if args.cleanup:
        import shutil
        print(f"\n--cleanup: borrando {DOWNLOAD_DIR} ...")
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
