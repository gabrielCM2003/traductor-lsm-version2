"""Extrae landmarks CRUDOS (sin normalizar) de todo el dataset dinamico de
CICESE (Zenodo, DOI 10.5281/zenodo.14689869), vista frontal y de perfil.

Por que un script aparte: procesar_dataset_dinamico.py ya proceso este mismo
dataset, pero guardo directamente vectores normalizados de 126 valores
(wrist-centered, escalados) para alimentar a DTWRecognizer. En ese proceso se
pierde la posicion real de la muneca en la imagen, la orientacion de la mano
y el tamano real (la escala se normaliza a proposito). Este script guarda los
landmarks TAL CUAL los entrega MediaPipe, una sola vez, para poder probar mas
adelante otros rasgos (trayectoria de la muneca, orientacion, velocidad) sin
tener que volver a descargar/decodificar los videos.

Reutiliza (no reimplementa):
  - de recolector_dinamico.py: init_hand_landmarker, capture_raw_hands,
    RawHandSample, RawFrameSample, save_raw_npz -> el MISMO esquema crudo por
    frame que ya se usa para las grabaciones propias (timestamp; por mano:
    etiqueta, score, 21 landmarks de imagen x,y,z y 21 world x,y,z; frames sin
    mano representados igual, no omitidos). No se usa parse_hands de ese
    modulo porque descarta la z de los landmarks de imagen (no la necesita
    para el vector de 126); capture_raw_hands es la version que SI la
    conserva, que es justo lo que hace falta aqui.
  - de procesar_dataset_dinamico.py: la descarga/extraccion del .7z (con el
    User-Agent que Zenodo exige), el parseo de nombres S<Id>-<Letra>-<Vista>-
    <Rep> y la deteccion de la letra "Ñ" por eliminacion.
  - el mismo cv2.flip(frame, 1) antes de MediaPipe que usa el resto del
    proyecto, para que el criterio Left/Right sea consistente en todos lados.

No modifica ninguno de esos archivos ni nada dentro de datos_dinamicas/: todo
lo que escribe este script vive fuera del repo, bajo --dataset-dir.

Orden de procesamiento: PRIMERO toda la vista frontal (con su propio reporte
parcial impreso al terminar) y SOLO DESPUES la de perfil, para que si hay que
interrumpir el script a medio proceso, la frontal ya haya quedado completa.
Reanudable: si el .npz de un video ya existe, se salta (sus estadisticas se
siguen contando para el reporte, leyendolas del archivo ya guardado).

Un HandLandmarker nuevo por video (igual que procesar_dataset_dinamico.py):
en modo VIDEO, MediaPipe exige timestamps crecientes durante toda la vida del
objeto y ademas arrastra el estado de tracking del frame anterior; reusar uno
solo entre videos de sujetos distintos rompia eso.

Los .7z NUNCA se borran (no hay opcion de limpieza en este script). Si un
video individual falla, se reporta y se sigue con el siguiente: no se
detiene todo el lote por un solo archivo problematico.

Uso:
    python extraer_landmarks_crudos.py
        [--dataset-dir "C:\\Proyectos\\Dataset_CICESE"]
        [--output-dir "C:\\Proyectos\\Dataset_CICESE\\landmarks_crudos"]
        [--skip-download]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from recolector_dinamico import (
    init_hand_landmarker,
    capture_raw_hands,
    RawFrameSample,
    save_raw_npz,
)
from procesar_dataset_dinamico import (
    remote_content_length,
    download_archive,
    extract_archive,
    find_videos,
    parse_video_name,
    build_letter_mapping,
    TARGET_LETTERS_ALL,
)

# Las dos vistas del mismo dataset (ver descripcion del registro de Zenodo,
# consultada via su API: title "Mexican Sign Language Alphabet (dynamic
# signs only)", licencia CC BY 4.0, Navarrete-Lopez & Lopez-Nava, CICESE).
ARCHIVOS = {
    "frontal": {
        "url": "https://zenodo.org/api/records/14689869/files/MSL-dynamic-signs-frontal-view.7z/content",
        "nombre": "MSL-dynamic-signs-frontal-view.7z",
    },
    "perfil": {
        "url": "https://zenodo.org/api/records/14689869/files/MSL-dynamic-signs-profile.7z/content",
        "nombre": "MSL-dynamic-signs-profile.7z",
    },
}

DEFAULT_DATASET_DIR = Path(r"C:\Proyectos\Dataset_CICESE")
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "landmarks_crudos"

# Reintentos para la descarga (download_archive, de procesar_dataset_dinamico.py,
# no se modifica: no tiene reintentos ni reanudacion por rangos, así que un
# corte de conexion a medio 2.2GB/1.4GB lo tira todo). Con archivos de este
# tamano un corte transitorio de la conexion es esperable; en vez de que el
# script se detenga y haya que relanzarlo a mano, reintenta unas cuantas
# veces con espera creciente antes de darse por vencido.
DESCARGA_MAX_INTENTOS = 5
DESCARGA_ESPERA_BASE_S = 10


def download_archive_con_reintentos(url: str, dest: Path, max_intentos: int = DESCARGA_MAX_INTENTOS) -> None:
    """Envuelve download_archive con reintentos y espera creciente.

    download_archive (de procesar_dataset_dinamico.py) no se toca: no hace
    descarga por rangos, asi que cada reintento vuelve a bajar el archivo
    completo desde cero. Con la velocidad observada (~2.2GB en menos de un
    minuto) esto sigue siendo mucho mas barato que dejar el proceso parado
    esperando que alguien lo relance a mano tras un corte de conexion."""
    ultimo_error: Optional[Exception] = None
    for intento in range(1, max_intentos + 1):
        try:
            download_archive(url, dest)
            return
        except Exception as e:
            ultimo_error = e
            espera = DESCARGA_ESPERA_BASE_S * intento
            print(f"\nERROR de descarga (intento {intento}/{max_intentos}): {e}")
            if intento < max_intentos:
                print(f"  reintentando en {espera}s...")
                time.sleep(espera)
    raise RuntimeError(f"la descarga de {url} fallo {max_intentos} veces seguidas") from ultimo_error


# =========================================================================== #
# Procesamiento de un video: TODOS los frames, sin recortar los vacios.
# =========================================================================== #

def process_video_raw(video_path: Path) -> tuple[list[RawFrameSample], float]:
    """Devuelve (frames_crudos, fps_del_video). A diferencia de
    procesar_dataset_dinamico.py, aqui NO se recorta el inicio/fin sin mano:
    se guarda la secuencia completa tal cual viene el video (item 1 del
    pedido), porque el objetivo es poder recalcular distintos rasgos despues
    sin haber tirado informacion de antemano."""
    landmarker = init_hand_landmarker(max_num_hands=2)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"no se pudo abrir el video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_period_ms = 1000.0 / fps if fps > 0 else 1000.0 / 30.0
    # Timestamp entero que exige la API de video de MediaPipe (monotono, no
    # es el dato que se guarda). El timestamp que SI se guarda en cada
    # RawFrameSample es el del propio frame dentro del video (frame_idx *
    # periodo), independiente de este contador interno.
    mp_timestamp_ms = 0

    raw_frames: list[RawFrameSample] = []
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Mismo mirror que el resto del proyecto (CameraThread, senas.py,
            # recolector_dinamico.py): asi el criterio Left/Right es el mismo
            # en todos los datos, propios o del dataset.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_timestamp_ms += max(1, round(frame_period_ms))
            results = landmarker.detect_for_video(mp_image, mp_timestamp_ms)

            hands = capture_raw_hands(results)
            raw_frames.append(RawFrameSample(
                timestamp_ms=frame_idx * frame_period_ms,
                hands=hands,
            ))
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    return raw_frames, fps


def read_npz_stats(path: Path) -> tuple[int, int]:
    """(n_frames, frames_sin_mano) leidos de un .npz ya guardado. Hace falta
    para que el reporte siga siendo correcto en una corrida reanudada: los
    videos que se saltan por ya estar procesados deben seguir contando."""
    with np.load(path) as data:
        labels = data["hand_labels"]
        n_frames = int(labels.shape[0])
        sin_mano = int(np.sum(np.all(labels == "", axis=1)))
    return n_frames, sin_mano


# =========================================================================== #
# Una vista completa (frontal o perfil)
# =========================================================================== #

def process_view(vista: str, extract_dir: Path, output_root: Path) -> dict:
    videos = find_videos(extract_dir)
    print(f"\n[{vista}] {len(videos)} videos encontrados en {extract_dir}")

    parsed = []
    unmatched = 0
    for video in videos:
        info = parse_video_name(video)
        if info is None:
            unmatched += 1
            continue
        subject_id, letter_token, view_token, rep = info
        parsed.append((video, subject_id, letter_token, rep))
    if unmatched:
        print(f"[{vista}]   {unmatched} archivos no siguieron el patron "
              f"S<Id>-<Letra>-<Vista>-<Rep>, se ignoraron.")

    letter_tokens = {letter_token for _, _, letter_token, _ in parsed}
    print(f"[{vista}] tokens de letra encontrados: {sorted(letter_tokens)}")
    letter_map = build_letter_mapping(letter_tokens)
    print(f"[{vista}] mapeo final token -> letra: {letter_map}")

    to_process = [
        (video, subject_id, letter_map[letter_token], rep)
        for video, subject_id, letter_token, rep in parsed
        if letter_map.get(letter_token) in TARGET_LETTERS_ALL
    ]
    print(f"[{vista}] {len(to_process)} videos corresponden a J/K/Ñ/Q/X/Z, procesando...")

    stats_por_letra: dict[str, dict] = defaultdict(lambda: {"videos": 0, "frames": 0, "sin_mano": 0})
    fallidos: list[str] = []
    saltados = 0

    for i, (video, subject_id, letra, rep) in enumerate(to_process, start=1):
        out_dir = output_root / letra
        out_path = out_dir / f"S{subject_id}_{vista}_{rep}.npz"

        if out_path.exists():
            saltados += 1
            try:
                n_frames, sin_mano = read_npz_stats(out_path)
                stats_por_letra[letra]["videos"] += 1
                stats_por_letra[letra]["frames"] += n_frames
                stats_por_letra[letra]["sin_mano"] += sin_mano
            except Exception as e:
                print(f"[{vista}]   aviso: no se pudieron leer stats de "
                      f"{out_path.name} (ya existente): {e}")
            continue

        print(f"[{vista}]   [{i}/{len(to_process)}] {video.name} -> "
              f"{letra}/{out_path.name}", end="", flush=True)
        try:
            raw_frames, fps = process_video_raw(video)
        except Exception as e:
            print(f"  ERROR: {e}")
            fallidos.append(f"{video.name}: {e}")
            continue

        n_frames = len(raw_frames)
        sin_mano = sum(1 for f in raw_frames if not f.hands)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_raw_npz(out_path, raw_frames, fps)
        except Exception as e:
            print(f"  ERROR al guardar: {e}")
            fallidos.append(f"{video.name}: error al guardar ({e})")
            continue

        stats_por_letra[letra]["videos"] += 1
        stats_por_letra[letra]["frames"] += n_frames
        stats_por_letra[letra]["sin_mano"] += sin_mano
        pct = (sin_mano / n_frames * 100.0) if n_frames else 0.0
        print(f"  ({n_frames} frames, {pct:.0f}% sin mano)")

    return {
        "vista": vista,
        "stats_por_letra": dict(stats_por_letra),
        "saltados": saltados,
        "fallidos": fallidos,
    }


def imprimir_reporte(resumen: dict) -> None:
    vista = resumen["vista"]
    print(f"\n=== Reporte parcial: vista {vista} ===")
    total_videos = 0
    total_frames = 0
    for letra in sorted(resumen["stats_por_letra"]):
        s = resumen["stats_por_letra"][letra]
        pct = (s["sin_mano"] / s["frames"] * 100.0) if s["frames"] else 0.0
        print(f"  {letra}: {s['videos']} videos, {s['frames']} frames, {pct:.1f}% sin mano")
        total_videos += s["videos"]
        total_frames += s["frames"]
    print(f"  TOTAL {vista}: {total_videos} videos, {total_frames} frames")
    if resumen["saltados"]:
        print(f"  ({resumen['saltados']} videos ya estaban procesados de antes, se saltaron)")
    if resumen["fallidos"]:
        print(f"  ADVERTENCIA: {len(resumen['fallidos'])} videos fallaron y se omitieron:")
        for f in resumen["fallidos"]:
            print(f"    - {f}")


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def imprimir_reporte_final(resumenes: list[dict], output_root: Path, archivos_7z: list[Path]) -> None:
    print("\n" + "=" * 70)
    print("REPORTE FINAL (todas las vistas)")
    print("=" * 70)

    letras = sorted(set().union(*(r["stats_por_letra"].keys() for r in resumenes)) | TARGET_LETTERS_ALL)
    total_videos = 0
    total_frames = 0
    print(f"\n{'letra':<6}" + "".join(f"{r['vista']:>22}" for r in resumenes) + f"{'combinado':>22}")
    for letra in letras:
        fila = [letra]
        videos_letra = 0
        frames_letra = 0
        sin_mano_letra = 0
        for r in resumenes:
            s = r["stats_por_letra"].get(letra, {"videos": 0, "frames": 0, "sin_mano": 0})
            pct = (s["sin_mano"] / s["frames"] * 100.0) if s["frames"] else 0.0
            fila.append(f"{s['videos']}v {s['frames']}f {pct:.0f}%sm")
            videos_letra += s["videos"]
            frames_letra += s["frames"]
            sin_mano_letra += s["sin_mano"]
        pct_comb = (sin_mano_letra / frames_letra * 100.0) if frames_letra else 0.0
        fila.append(f"{videos_letra}v {frames_letra}f {pct_comb:.0f}%sm")
        print(f"{fila[0]:<6}" + "".join(f"{c:>22}" for c in fila[1:]))
        total_videos += videos_letra
        total_frames += frames_letra

    print(f"\nTOTAL combinado: {total_videos} videos, {total_frames} frames")

    fallidos_totales = [f for r in resumenes for f in r["fallidos"]]
    if fallidos_totales:
        print(f"\n{len(fallidos_totales)} videos fallaron en total:")
        for f in fallidos_totales:
            print(f"  - {f}")

    print(f"\nTamano de {output_root}: {_dir_size_mb(output_root):.1f} MB")
    for archivo in archivos_7z:
        if archivo.exists():
            print(f"Tamano de {archivo.name}: {archivo.stat().st_size / 1e6:.0f} MB (conservado, no se borra)")


# =========================================================================== #
# main
# =========================================================================== #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrae landmarks crudos (sin normalizar) del dataset dinamico de CICESE (frontal + perfil)."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                         help="Donde estan/se descargan los .7z del dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Donde se escriben los .npz de landmarks crudos.")
    parser.add_argument("--skip-download", action="store_true",
                         help="No descargar: usar los .7z/extraidos que ya existan en --dataset-dir.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    resumenes: list[dict] = []
    archivos_7z: list[Path] = []

    # Frontal PRIMERO, siempre completa, con su reporte parcial, antes de
    # tocar el perfil (item de "Orden" del pedido: si hay que interrumpir,
    # la frontal debe quedar terminada).
    for vista in ("frontal", "perfil"):
        info = ARCHIVOS[vista]
        archivo_path = dataset_dir / info["nombre"]
        extract_dir = dataset_dir / f"_extraido_{vista}"
        archivos_7z.append(archivo_path)

        print(f"\n{'=' * 70}\nVISTA: {vista.upper()}\n{'=' * 70}")

        if not args.skip_download:
            try:
                download_archive_con_reintentos(info["url"], archivo_path)
            except Exception as e:
                print(f"ERROR al descargar {vista} (tras {DESCARGA_MAX_INTENTOS} intentos): {e}. "
                      f"Se guarda el progreso obtenido hasta ahora.", file=sys.stderr)
                if resumenes:
                    imprimir_reporte_final(resumenes, output_dir, archivos_7z)
                return 1
        elif not archivo_path.exists() and not extract_dir.exists():
            print(f"ERROR: --skip-download pero no hay nada de '{vista}' en {dataset_dir}", file=sys.stderr)
            return 1

        if archivo_path.exists():
            try:
                extract_archive(archivo_path, extract_dir)
            except Exception as e:
                print(f"ERROR al extraer {vista}: {e}. Se guarda el progreso obtenido hasta ahora.",
                      file=sys.stderr)
                if resumenes:
                    imprimir_reporte_final(resumenes, output_dir, archivos_7z)
                return 1

        resumen = process_view(vista, extract_dir, output_dir)
        resumenes.append(resumen)
        imprimir_reporte(resumen)

    imprimir_reporte_final(resumenes, output_dir, archivos_7z)
    print(f"\nListo. Landmarks crudos en {output_dir}. Los .7z se conservaron en {dataset_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
