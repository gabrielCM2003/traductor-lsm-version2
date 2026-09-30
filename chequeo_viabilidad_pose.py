"""Chequeo de viabilidad de MediaPipe Pose en videos de CICESE (frontal y perfil).

Evalúa si los videos recortados del dataset de CICESE (encuadre cerrado en mano y antebrazo)
permiten detectar de forma consistente hombro (11/12), codo (13/14) y muñeca (15/16).

Alcance: SOLO LECTURA de videos/modelos existentes. No toca nada de producción.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chequeo_pose")

DEFAULT_DATASET_DIR = Path(r"C:\Proyectos\Dataset_CICESE")
DEFAULT_MODEL_PATH = Path.home() / ".sign_translator" / "models" / "pose_landmarker_full.task"
ARTIFACT_DIR = Path(r"C:\Users\cesar\.gemini\antigravity\brain\c257d49a-5fac-4ec2-9b2a-7c847112dd02")

# Conexiones principales de brazos y hombros para dibujar
SKELETON_CONNECTIONS = [
    (11, 12),  # Hombro Izq - Hombro Der
    (11, 13),  # Hombro Izq - Codo Izq
    (13, 15),  # Codo Izq - Muñeca Izq
    (12, 14),  # Hombro Der - Codo Der
    (14, 16),  # Codo Der - Muñeca Der
]

KEY_LANDMARKS = [
    (11, "Hombro Izq", (0, 0, 255)),
    (12, "Hombro Der", (0, 100, 255)),
    (13, "Codo Izq", (0, 255, 0)),
    (14, "Codo Der", (100, 255, 0)),
    (15, "Muñeca Izq", (255, 0, 255)),
    (16, "Muñeca Der", (255, 100, 255)),
]


def init_pose_landmarker(model_path: Path) -> mp.tasks.vision.PoseLandmarker:
    """Inicializa PoseLandmarker en modo IMAGE para procesar frames independientes."""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return PoseLandmarker.create_from_options(options)


def dibujar_esqueleto(frame: np.ndarray, landmarks_proto) -> np.ndarray:
    """Dibuja hombros, codos y muñecas sobre el frame con texto explicativo."""
    h, w, _ = frame.shape
    out = frame.copy()

    if landmarks_proto is None:
        cv2.putText(
            out, "POSE NO DETECTADA (0 personas)", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
        )
        return out

    lm = landmarks_proto

    # Dibujar líneas de conexión
    for idx1, idx2 in SKELETON_CONNECTIONS:
        p1, p2 = lm[idx1], lm[idx2]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        # Solo dibujar si ambos puntos están dentro o cerca del encuadre
        if -0.2 <= p1.x <= 1.2 and -0.2 <= p1.y <= 1.2 and -0.2 <= p2.x <= 1.2 and -0.2 <= p2.y <= 1.2:
            cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Dibujar puntos y etiquetas
    for idx, name, color in KEY_LANDMARKS:
        p = lm[idx]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
        txt = f"{name} v={p.visibility:.2f} p={p.presence:.2f}"
        cv2.putText(out, txt, (x + 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (x + 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def analizar_frame(
    frame: np.ndarray,
    landmarker: mp.tasks.vision.PoseLandmarker,
    umbral_confianza: float = 0.5,
) -> dict:
    """Ejecuta PoseLandmarker sobre un frame y evalúa si hombro, codo y muñeca se detectan con confianza."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)

    has_pose = len(res.pose_landmarks) > 0
    if not has_pose:
        return {
            "pose_detectada": False,
            "landmarks_raw": None,
            "hombro_ok": False,
            "codo_ok": False,
            "muneca_ok": False,
            "detalles": {},
        }

    lm = res.pose_landmarks[0]

    # Evaluamos los dos brazos (11/13/15 Izq y 12/14/16 Der)
    # Consideramos un landmark detectado con confianza razonable si:
    # 1) Está dentro de los límites del encuadre [0, 1]
    # 2) presence >= umbral_confianza (indica que realmente está presente en la imagen)
    # 3) visibility >= umbral_confianza (no está ocluido)
    def _es_confiable(idx: int) -> bool:
        p = lm[idx]
        dentro_frame = (0.0 <= p.x <= 1.0) and (0.0 <= p.y <= 1.0)
        return dentro_frame and (p.presence >= umbral_confianza) and (p.visibility >= umbral_confianza)

    # El landmark se considera detectado si AL MENOS uno de los dos lados (izq o der) es confiable
    hombro_ok = _es_confiable(11) or _es_confiable(12)
    codo_ok = _es_confiable(13) or _es_confiable(14)
    muneca_ok = _es_confiable(15) or _es_confiable(16)

    detalles = {
        "sh_L": {"x": lm[11].x, "y": lm[11].y, "vis": lm[11].visibility, "pres": lm[11].presence, "ok": _es_confiable(11)},
        "sh_R": {"x": lm[12].x, "y": lm[12].y, "vis": lm[12].visibility, "pres": lm[12].presence, "ok": _es_confiable(12)},
        "el_L": {"x": lm[13].x, "y": lm[13].y, "vis": lm[13].visibility, "pres": lm[13].presence, "ok": _es_confiable(13)},
        "el_R": {"x": lm[14].x, "y": lm[14].y, "vis": lm[14].visibility, "pres": lm[14].presence, "ok": _es_confiable(14)},
        "wr_L": {"x": lm[15].x, "y": lm[15].y, "vis": lm[15].visibility, "pres": lm[15].presence, "ok": _es_confiable(15)},
        "wr_R": {"x": lm[16].x, "y": lm[16].y, "vis": lm[16].visibility, "pres": lm[16].presence, "ok": _es_confiable(16)},
    }

    return {
        "pose_detectada": True,
        "landmarks_raw": lm,
        "hombro_ok": hombro_ok,
        "codo_ok": codo_ok,
        "muneca_ok": muneca_ok,
        "detalles": detalles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chequeo de viabilidad de Pose en CICESE")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("inspeccion_pose"))
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de presencia y visibilidad")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_DIR.exists():
        (ARTIFACT_DIR / "inspeccion_pose").mkdir(parents=True, exist_ok=True)

    log.info("Inicializando MediaPipe PoseLandmarker...")
    landmarker = init_pose_landmarker(DEFAULT_MODEL_PATH)

    # Videos seleccionados: Sujeto S10, Letras J y Q, Frontal y Perfil
    videos_config = [
        {
            "letra": "J", "vista": "frontal",
            "path": args.dataset_dir / "_extraido_frontal" / "MSL-dynamic-signs" / "train" / "S10-J-frontal-1.mp4"
        },
        {
            "letra": "Q", "vista": "frontal",
            "path": args.dataset_dir / "_extraido_frontal" / "MSL-dynamic-signs" / "train" / "S10-Q-frontal-1.mp4"
        },
        {
            "letra": "J", "vista": "perfil",
            "path": args.dataset_dir / "_extraido_perfil" / "MSL dynamic-profile-signs" / "J" / "S10-J-perfil-1.mp4"
        },
        {
            "letra": "Q", "vista": "perfil",
            "path": args.dataset_dir / "_extraido_perfil" / "MSL dynamic-profile-signs" / "Q" / "S10-Q-perfil-1.mp4"
        },
    ]

    porcentajes_frames = [0.15, 0.32, 0.50, 0.68, 0.85]
    todos_los_resultados: List[dict] = []
    frames_guardados = []

    print("\n" + "=" * 105)
    print("CHEQUEO DE VIABILIDAD DE MEDIAPIPE POSE EN VIDEOS DE CICESE (SUJETO S10, LETRAS J Y Q)")
    print("=" * 105)

    for cfg in videos_config:
        letra = cfg["letra"]
        vista = cfg["vista"]
        vpath = cfg["path"]

        if not vpath.exists():
            print(f"ERROR: Video no encontrado: {vpath}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        indices_eval = [int(n_frames * p) for p in porcentajes_frames]

        for p_idx, f_idx in zip(porcentajes_frames, indices_eval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            info = analizar_frame(frame, landmarker, umbral_confianza=args.umbral)
            item = {
                "letra": letra,
                "vista": vista,
                "frame_idx": f_idx,
                "pct_video": p_idx * 100,
                "n_frames_totales": n_frames,
                "pose_detectada": info["pose_detectada"],
                "hombro_ok": info["hombro_ok"],
                "codo_ok": info["codo_ok"],
                "muneca_ok": info["muneca_ok"],
                "detalles": info["detalles"],
            }
            todos_los_resultados.append(item)

            # Guardar frames representativos (3 de frontal y 3 de perfil)
            es_representativo = (
                (vista == "frontal" and letra == "J" and f_idx in [indices_eval[1], indices_eval[2]]) or
                (vista == "frontal" and letra == "Q" and f_idx == indices_eval[2]) or
                (vista == "perfil" and letra == "J" and f_idx == indices_eval[2]) or
                (vista == "perfil" and letra == "Q" and f_idx in [indices_eval[1], indices_eval[2]])
            )

            if es_representativo:
                annotated = dibujar_esqueleto(frame, info["landmarks_raw"])
                nombre_img = f"{vista}_{letra}_f{f_idx}.jpg"
                out_path = args.output_dir / nombre_img
                cv2.imwrite(str(out_path), annotated)

                # Copiar al directorio de artefactos si existe
                if ARTIFACT_DIR.exists():
                    art_dest = ARTIFACT_DIR / "inspeccion_pose" / nombre_img
                    shutil.copy2(out_path, art_dest)

                frames_guardados.append((nombre_img, letra, vista, f_idx, info["pose_detectada"]))

        cap.release()

    # ======================================================================= #
    # TABLA DETALLADA DE LOS 20 FRAMES
    # ======================================================================= #
    print("\n" + "-" * 105)
    print("DETALLE FRAME A FRAME (20 FRAMES: 10 FRONTAL + 10 PERFIL)")
    print("-" * 105)
    header = (
        f"{'Letra':<5} | {'Vista':<8} | {'Frame':<7} | {'% Vid':<6} | {'¿Pose?':<8} | "
        f"{'Hombro (11/12)':<16} | {'Codo (13/14)':<16} | {'Muñeca (15/16)':<16}"
    )
    print(header)
    print("-" * len(header))

    for r in todos_los_resultados:
        pose_str = "SÍ" if r["pose_detectada"] else "NO (Fallo)"
        if not r["pose_detectada"]:
            sh_str = "NO DETECTADO"
            el_str = "NO DETECTADO"
            wr_str = "NO DETECTADO"
        else:
            d = r["detalles"]
            # Tomamos el valor de máxima presencia/visibilidad entre izq y der
            sh_p = max(d["sh_L"]["pres"], d["sh_R"]["pres"])
            sh_v = max(d["sh_L"]["vis"], d["sh_R"]["vis"])
            sh_status = "OK" if r["hombro_ok"] else "Dudoso"
            sh_str = f"{sh_status} (p={sh_p:.2f})"

            el_p = max(d["el_L"]["pres"], d["el_R"]["pres"])
            el_status = "OK" if r["codo_ok"] else "Dudoso"
            el_str = f"{el_status} (p={el_p:.2f})"

            wr_p = max(d["wr_L"]["pres"], d["wr_R"]["pres"])
            wr_status = "OK" if r["muneca_ok"] else "Dudoso"
            wr_str = f"{wr_status} (p={wr_p:.2f})"

        print(
            f"{r['letra']:<5} | {r['vista']:<8} | {r['frame_idx']:<7} | {r['pct_video']:>4.0f}%  | "
            f"{pose_str:<8} | {sh_str:<16} | {el_str:<16} | {wr_str:<16}"
        )
    print("-" * len(header))

    # ======================================================================= #
    # TABLA RESUMEN POR VISTA
    # ======================================================================= #
    print("\n" + "=" * 90)
    print("TABLA RESUMEN: % DE FRAMES CON LANDMARKS DETECTADOS POR VISTA (UMBRAL CONFIANZA >= 0.5)")
    print("=" * 90)
    print(f"{'Vista':<10} | {'Total Frames':<14} | {'Pose Detectada':<16} | {'Hombro Confiable':<18} | {'Codo Confiable':<16} | {'Muñeca Confiable'}")
    print("-" * 90)

    for v in ["frontal", "perfil"]:
        sub_r = [r for r in todos_los_resultados if r["vista"] == v]
        tot = len(sub_r)
        n_pose = sum(1 for r in sub_r if r["pose_detectada"])
        n_sh = sum(1 for r in sub_r if r["hombro_ok"])
        n_el = sum(1 for r in sub_r if r["codo_ok"])
        n_wr = sum(1 for r in sub_r if r["muneca_ok"])

        pct_pose = (n_pose / tot) * 100
        pct_sh = (n_sh / tot) * 100
        pct_el = (n_el / tot) * 100
        pct_wr = (n_wr / tot) * 100

        print(
            f"{v.capitalize():<10} | {tot:<14} | {pct_pose:>5.1f}% ({n_pose}/{tot})    | "
            f"{pct_sh:>5.1f}% ({n_sh}/{tot})      | {pct_el:>5.1f}% ({n_el}/{tot})   | {pct_wr:>5.1f}% ({n_wr}/{tot})"
        )
    print("-" * 90)

    print("\nFRAMES REPRESENTATIVOS GUARDADOS PARA INSPECCIÓN DIRECTA:")
    for img_name, l, v, f, ok in frames_guardados:
        estado = "Pose detectada" if ok else "Fallo total detector"
        print(f"  - inspeccion_pose/{img_name} ({v}, {l}, frame {f}) -> {estado}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
