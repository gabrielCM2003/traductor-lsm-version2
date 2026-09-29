"""Verifica que los landmarks CRUDOS guardados por extraer_landmarks_crudos.py
sean consistentes con los vectores normalizados que procesar_dataset_dinamico.py
ya guardo en datos_dinamicas/ para el MISMO video (misma letra, sujeto y
repeticion, vista frontal).

Metodo: toma un .npz crudo, recorta los frames sin mano al inicio/final
EXACTAMENTE como hace procesar_dataset_dinamico.py (mismo criterio: primer y
ultimo frame con alguna mano detectada), reconstruye el vector de 126 con
hand_to_feature_vector + normalize_keypoints -las MISMAS funciones de
sign_classifier.py que usa todo el proyecto, no una reimplementacion- y lo
compara contra el .json ya existente de ese video, dentro de una tolerancia
pequena (son dos corridas separadas de MediaPipe sobre el mismo video; el
modelo es determinista, pero se deja margen por seguridad numerica).

No modifica nada: solo lee datos_dinamicas/ y landmarks_crudos/.

Uso:
    python verificar_landmarks_crudos.py [--dataset-dir RUTA] [--n 3]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np

from sign_classifier import normalize_keypoints, hand_to_feature_vector

N_FEATURES_PER_HAND = 63
N_FEATURES = 126
DEFAULT_TOLERANCE = 1e-4

JSON_NAME_RE = re.compile(r"^muestra_(\d+)_(\d+)$")


def reconstruir_desde_npz(npz_path: Path) -> Optional[np.ndarray]:
    """Recorta los frames sin mano al inicio/final (igual que
    procesar_dataset_dinamico.py) y reconstruye la secuencia de vectores de
    126 a partir de los landmarks crudos. None si no queda ninguna mano."""
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
        for slot in range(2):  # 0=Left, 1=Right: misma convencion de slots que el resto del proyecto
            if hand_labels[t, slot] == "":
                continue
            lm2d = landmarks_image[t, slot, :, :2]  # x,y de imagen (se descarta la z, igual que parse_hands)
            lm3d = landmarks_world[t, slot]          # x,y,z world (hand_to_feature_vector solo usa la z)
            raw = hand_to_feature_vector(lm2d, lm3d)
            norm = normalize_keypoints(raw)
            offset = slot * N_FEATURES_PER_HAND
            vec[offset:offset + N_FEATURES_PER_HAND] = norm
        secuencia.append(vec)
    return np.array(secuencia, dtype=np.float32)


def cargar_json_existente(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return np.array(data["frames"], dtype=np.float32)


def encontrar_pares(raw_root: Path, json_root: Path, vista: str = "frontal") -> list[tuple]:
    """Empareja cada datos_dinamicas/<LETRA>/muestra_<Id>_<Rep>.json con su
    landmarks_crudos/<LETRA>/S<Id>_<vista>_<Rep>.npz correspondiente.

    Devuelve una lista de (letra, subject_id, rep, json_path, npz_path)."""
    pares = []
    if not json_root.exists():
        return pares
    for letra_dir in sorted(json_root.iterdir()):
        if not letra_dir.is_dir():
            continue
        letra = letra_dir.name
        npz_dir = raw_root / letra
        if not npz_dir.is_dir():
            continue
        for json_path in sorted(letra_dir.glob("muestra_*_*.json")):
            m = JSON_NAME_RE.match(json_path.stem)
            if not m:
                continue
            subject_id, rep = m.groups()
            npz_path = npz_dir / f"S{subject_id}_{vista}_{rep}.npz"
            if npz_path.exists():
                pares.append((letra, subject_id, rep, json_path, npz_path))
    return pares


def verificar_uno(
    letra: str, subject_id: str, rep: str, json_path: Path, npz_path: Path,
    tolerancia: float = DEFAULT_TOLERANCE,
) -> bool:
    print(f"\n--- letra {letra}, sujeto {subject_id}, repeticion {rep} ---")
    print(f"  json: {json_path}")
    print(f"  npz:  {npz_path}")

    esperado = cargar_json_existente(json_path)
    reconstruido = reconstruir_desde_npz(npz_path)

    if reconstruido is None:
        print("  FALLO: el npz no tiene ninguna mano detectada en ningun frame")
        return False

    if esperado.shape != reconstruido.shape:
        print(f"  FALLO: shapes distintas (json={esperado.shape}, reconstruido={reconstruido.shape})")
        return False

    diff = np.abs(esperado - reconstruido)
    max_diff = float(diff.max())
    ok = max_diff <= tolerancia
    print(f"  shape: {esperado.shape}, diferencia maxima entre json y reconstruido: {max_diff:.6f} "
          f"(tolerancia {tolerancia})")
    print(f"  -> {'OK' if ok else 'FALLO'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica los landmarks crudos contra el .json normalizado ya existente del mismo video."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"C:\Proyectos\Dataset_CICESE"))
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--vista", default="frontal")
    parser.add_argument("--n", type=int, default=3, help="cuantos videos verificar (de letras distintas si es posible)")
    parser.add_argument("--tolerancia", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--seed", type=int, default=None, help="fija la semilla para elegir siempre los mismos videos")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    raw_root = args.dataset_dir / "landmarks_crudos"
    json_root = args.repo_dir / "datos_dinamicas"

    pares = encontrar_pares(raw_root, json_root, vista=args.vista)
    if not pares:
        print("ERROR: no se encontraron pares json/npz para verificar. "
              "Corre extraer_landmarks_crudos.py primero (al menos la vista frontal).")
        return 1

    print(f"{len(pares)} pares json/npz disponibles para comparar.")

    # Hasta args.n videos, priorizando letras DISTINTAS entre si.
    por_letra: dict[str, list] = {}
    for p in pares:
        por_letra.setdefault(p[0], []).append(p)
    letras = list(por_letra.keys())
    random.shuffle(letras)

    elegidos = []
    for letra in letras:
        if len(elegidos) >= args.n:
            break
        elegidos.append(random.choice(por_letra[letra]))

    resultados = [
        verificar_uno(letra, subject_id, rep, json_path, npz_path, args.tolerancia)
        for letra, subject_id, rep, json_path, npz_path in elegidos
    ]

    print(f"\n=== RESUMEN: {sum(resultados)}/{len(resultados)} OK ===")
    return 0 if resultados and all(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
