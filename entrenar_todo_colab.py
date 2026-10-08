"""Entrena y compara, de principio a fin, los modelos de la app con los datasets
abiertos del CICESE (Zenodo, CC BY 4.0). Pensado para Google Colab con el
entorno TPU v5e-1 (que trae muchos nucleos de CPU), pero corre en cualquier
maquina: usa la GPU si hay (CUDA o la de una Mac) y si no, todos los nucleos.

Datasets (Navarrete-Lopez, Lopez-Nava y Morfin-Chavez, CICESE):
  - glosas:   121 palabras, 12 personas       (10.5281/zenodo.18330565)
  - estatico: 21 letras fijas, 20 personas    (10.5281/zenodo.10067509)
  - dinamico: J, K, N~, Q, X, Z, 20 personas  (10.5281/zenodo.14689869)
mas las 59 muestras de palabras del equipo (datos_palabras_dinamicas/).

Etapas. Cada una se salta si ya esta hecha: si Colab se desconecta, se vuelve
a correr el mismo comando y sigue donde se quedo (lo que importa se guarda en
--trabajo, que en Colab conviene que sea una carpeta de Google Drive). Lo unico
que se repite es la descarga, si se corta ANTES de terminar la extraccion (el
disco de la maquina de Colab se borra al desconectarse); los puntos ya
extraidos, cada pliegue de la comparacion y los modelos finales no se repiten.
  1. descarga    los .7z de Zenodo, verificando el md5
  2. descomprime
  3. extrae      los puntos de MediaPipe con el MISMO codigo de la app:
                 extraer_palabras_videos.extract_video para los videos (recorte
                 4:3, espejo, manos + ubicacion respecto al cuerpo, corte con el
                 segmentador automatico) y HandLandmarker para las fotos
  4. compara     por personas (validacion cruzada: cada persona se prueba con
                 un modelo que nunca la vio): transformer contra el DTW actual
                 en letras dinamicas y en palabras; red estatica nueva contra
                 la actual con las 2 personas de prueba del CICESE
  5. entrena     los modelos finales con todas las personas, en ONNX
  6. reporte     reporte.md y resultados.json en --trabajo

Uso:
    python entrenar_todo_colab.py --trabajo /content/drive/MyDrive/lsm_entrenamiento
    python entrenar_todo_colab.py --trabajo ... --sin-descarga --datos CARPETA
        (datos ya descomprimidos en CARPETA/glosas, CARPETA/estatico, CARPETA/dinamico)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mproc
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

ZENODO_RECORDS = {
    "glosas": "18330565",
    "estatico": "10067509",
    "dinamico": "14689869",
}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

# Secuencias (letras dinamicas y palabras): 126 de las manos + 9 de cuerpo,
# como las palabras en senas.py. El transformer las recibe remuestreadas a
# SEQ_LEN frames por interpolacion lineal (ver resample_linear).
N_HAND = 126
N_BODY = 9
N_SEQ = N_HAND + N_BODY
SEQ_LEN = 32
MIN_SEQ_FRAMES = 5
# El DTW se compara como lo usa senas.py: las letras con 1 de cada 2 frames
# (~15 fps) y solo las manos; las palabras con todos los frames y el cuerpo
# multiplicado por 4.
DTW_LETTER_STEP = 2
DTW_WORD_BODY_WEIGHT = 4.0
DYNAMIC_LETTERS = {"J", "K", "Q", "X", "Z", "Ñ"}
LIVE_ASPECT = 640 / 480

# Sesiones de los videos del equipo (Downloads/Entrenamiento, 2026-09-29):
# una persona por sesion, separadas por la hora de grabacion.
TEAM_SESSION_STARTS = ["19:20:00", "19:24:40"]


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def nfc_upper(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().upper().replace(" ", "_")


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# =========================================================================== #
# 1-2. Descarga y descompresion
# =========================================================================== #

def zenodo_files(record: str) -> list[dict]:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as resp:
        meta = json.load(resp)
    out = []
    for f in meta["files"]:
        algo, _, digest = f.get("checksum", "").partition(":")
        out.append({
            "name": f["key"],
            "size": int(f["size"]),
            "md5": digest if algo == "md5" else None,
            "url": f["links"]["self"],
        })
    return out


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, size: int, md5: Optional[str], tries: int = 6) -> None:
    """Descarga con reanudacion (Range): si se corta, sigue donde quedo."""
    ok_marker = target.with_name(target.name + ".ok")
    if ok_marker.exists() and target.exists():
        return
    for attempt in range(1, tries + 1):
        have = target.stat().st_size if target.exists() else 0
        if have > size:             # archivo de otra version o danado
            target.unlink()
            have = 0
        if have < size:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, target.open("ab" if have else "wb") as out:
                    if have and resp.status != 206:     # el servidor ignoro el Range
                        out.truncate(0)
                        have = 0
                    t0, last = time.time(), time.time()
                    done = have
                    for chunk in iter(lambda: resp.read(8 << 20), b""):
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 30:
                            rate = (done - have) / max(1e-6, time.time() - t0) / 1e6
                            log(f"  {target.name}: {done / 1e9:.2f}/{size / 1e9:.2f} GB ({rate:.0f} MB/s)")
                            last = time.time()
            except Exception as e:
                log(f"  {target.name}: se corto la descarga ({e}); reintento {attempt}/{tries}")
                time.sleep(min(60, 5 * attempt))
                continue
        if target.stat().st_size != size:
            log(f"  {target.name}: tamano {target.stat().st_size} != {size}; reintento")
            continue
        if md5 and file_md5(target) != md5:
            log(f"  {target.name}: md5 no coincide, se descarga de nuevo")
            target.unlink()
            continue
        ok_marker.touch()
        return
    raise RuntimeError(f"no se pudo descargar {target.name}")


def find_7z() -> list[str]:
    for exe in ("7z", "7zz", "7za"):
        if shutil.which(exe):
            return [exe]
    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        log("Instalando p7zip-full ...")
        subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=False,
                       stdout=subprocess.DEVNULL)
        if shutil.which("7z"):
            return ["7z"]
    raise RuntimeError("falta un descompresor de .7z (7z, 7zz o 7za); en Mac: brew install sevenzip")


def stage_download_and_unpack(datos: Path, keep_archives: bool) -> None:
    archives = datos / "_archivos"
    archives.mkdir(parents=True, exist_ok=True)
    for key, record in ZENODO_RECORDS.items():
        for f in zenodo_files(record):
            if not f["name"].lower().endswith(".7z"):
                continue
            stem = f["name"][:-3]
            dest = datos / key / stem
            if (dest / ".listo").exists():
                log(f"{f['name']}: ya descomprimido")
                continue
            archive = archives / f["name"]
            log(f"Descargando {f['name']} ({f['size'] / 1e9:.1f} GB) de Zenodo ...")
            t0 = time.time()
            download(f["url"], archive, f["size"], f["md5"])
            log(f"  listo en {(time.time() - t0) / 60:.1f} min; descomprimiendo ...")
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            subprocess.run(find_7z() + ["x", "-y", "-bso0", "-bsp0", f"-o{dest}", str(archive)], check=True)
            (dest / ".listo").touch()
            if not keep_archives:
                archive.unlink()
                archive.with_name(archive.name + ".ok").unlink(missing_ok=True)
            log(f"  {f['name']} descomprimido")


# =========================================================================== #
# 3. Inventario de muestras
# =========================================================================== #

# S<persona>-<letra>-<vista><sep><repeticion>; en el archivo real el separador
# de la repeticion a veces es "-" y a veces "." (S20-Z-Frontal.5, S21-J-frontal-2).
DYN_NAME_RE = re.compile(r"^S(\d+)[-_ ](.+?)[-_ ]([A-Za-z]+)[-._ ]?(\d+)$", re.IGNORECASE)
GLOSS_SUBJECT_RE = re.compile(r"(\d+)\s*$")
# Los videos del CICESE van a 60 fps; se procesan como la camara de la app (30).
MAX_VIDEO_FPS = 30.0
# Version de la extraccion: si cambia, los puntos viejos no se mezclan.
EXTRACTION_SETTINGS = {"version": 2, "max_fps": MAX_VIDEO_FPS}


def report_skipped(source: str, names: list[str]) -> None:
    if names:
        log(f"  {source}: {len(names)} archivos con nombre inesperado se omiten (p. ej. {names[:3]})")


def list_items(datos: Path, static_step: int, use_profile: bool) -> list[dict]:
    """Todas las muestras a extraer, con su etiqueta, persona y vista."""
    items: list[dict] = []

    # Glosas: <Palabra>/<Palabra>_<persona>.mp4
    skipped = []
    for path in sorted((datos / "glosas").rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        m = GLOSS_SUBJECT_RE.search(path.stem)
        if not m:
            skipped.append(path.name)
            continue
        subject = m.group(1)
        items.append({
            "id": "glosas/" + path.relative_to(datos / "glosas").as_posix(),
            "kind": "video", "path": str(path), "source": "glosas",
            "label": nfc_upper(path.parent.name), "family": "palabra",
            "person": f"glosas:{int(subject)}", "view": "frontal",
        })
    report_skipped("glosas", skipped)

    # Letras dinamicas: S<persona>-<letra>-<vista>-<repeticion>. La Ñ no viene
    # documentada en los nombres: el token que no sea J/K/Q/X/Z es la Ñ (como
    # en procesar_dataset_dinamico.py).
    dyn, skipped = [], []
    for path in sorted((datos / "dinamico").rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        m = DYN_NAME_RE.match(unicodedata.normalize("NFC", path.stem))
        if not m:
            skipped.append(path.name)
            continue
        dyn.append((path, m.groups()))
    report_skipped("dinamico", skipped)
    tokens = {g[1] for _, g in dyn}
    mapping = {t: (t.upper() if t.upper() in DYNAMIC_LETTERS - {"Ñ"} else "Ñ") for t in tokens}
    if tokens:
        log(f"  dinamico: tokens de letra -> {mapping}")
    for path, (subject, token, view_token, rep) in dyn:
        lower = path.as_posix().lower()
        view = "perfil" if ("profile" in lower or "perfil" in lower) else "frontal"
        if view == "perfil" and not use_profile:
            continue
        items.append({
            "id": "dinamico/" + path.relative_to(datos / "dinamico").as_posix(),
            "kind": "video", "path": str(path), "source": "dinamico",
            "label": mapping[token], "family": "letra",
            "person": f"dinamico:{int(subject)}", "view": view,
        })

    # Letras estaticas: <grupo A/B/C>/{train,test}/<LETRA>/*.jpg. Son cuadros
    # seguidos de video, casi repetidos: se toma 1 de cada static_step.
    by_dir = defaultdict(list)
    for path in (datos / "estatico").rglob("*"):
        if path.suffix.lower() in IMAGE_EXT:
            by_dir[path.parent].append(path)
    for folder in sorted(by_dir):
        parts = [p.lower() for p in folder.relative_to(datos / "estatico").parts]
        split = "test" if "test" in parts else "train"
        for path in sorted(by_dir[folder])[::max(1, static_step)]:
            items.append({
                "id": "estatico/" + path.relative_to(datos / "estatico").as_posix(),
                "kind": "image", "path": str(path), "source": "estatico",
                "label": nfc_upper(folder.name), "family": "estatica",
                "person": f"estatico:{split}", "view": "frontal",
            })
    return items


# =========================================================================== #
# 3. Extraccion de puntos (en paralelo, guardando por tandas)
# =========================================================================== #

_STATIC_DET = None


def _static_detector():
    global _STATIC_DET
    if _STATIC_DET is None:
        import mediapipe as mp
        import extraer_palabras_videos as ex
        vision = mp.tasks.vision
        _STATIC_DET = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ex.ensure_hand_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        ))
    return _STATIC_DET


def extract_one(item: dict) -> dict:
    """Corre en un proceso aparte. Devuelve los puntos de una muestra."""
    import cv2
    out = {"id": item["id"], "kind": item["kind"], "ok": False, "how": ""}
    try:
        if item["kind"] == "image":
            import mediapipe as mp
            bgr = cv2.imread(item["path"])
            if bgr is None:
                out["how"] = "ilegible"
                return out
            res = _static_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                     data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if not res.hand_landmarks:
                out["how"] = "sin_mano"
                return out
            best = max(range(len(res.hand_landmarks)),
                       key=lambda i: res.handedness[i][0].score if res.handedness and res.handedness[i] else 0.0)
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[:, :2] = [[p.x, p.y] for p in res.hand_landmarks[best]]
            if res.hand_world_landmarks:
                pts[:, 2] = [p.z for p in res.hand_world_landmarks[best]]
            out.update(ok=True, how="mano", points=pts, aspect=bgr.shape[1] / bgr.shape[0])
            return out

        import extraer_palabras_videos as ex
        from body_tracker import DEFAULT_POSE_MODEL
        # Detectores nuevos por video: en modo VIDEO arrastran el seguimiento
        # del frame anterior, y no debe pasar de un video (persona) a otro.
        hand_det, pose_det = ex.create_detectors(DEFAULT_POSE_MODEL)
        try:
            frames, signs, _, fps, _ = ex.extract_video(Path(item["path"]), hand_det, pose_det, 0,
                                                        max_fps=MAX_VIDEO_FPS)
        finally:
            hand_det.close()
            pose_det.close()
        index = {id(f): i for i, f in enumerate(frames)}
        if signs:
            # El video es una sola sena: de donde empieza el primer tramo que
            # corto el segmentador a donde termina el ultimo.
            start = min(index[id(s[0])] for s in signs)
            end = max(index[id(s[-1])] for s in signs)
            how = "segmentador"
        else:
            with_hands = [i for i, f in enumerate(frames) if f.hands]
            if not with_hands:
                out["how"] = "sin_manos"
                return out
            start, end, how = with_hands[0], with_hands[-1], "manos"
        seg = frames[start:end + 1]
        if len(seg) < MIN_SEQ_FRAMES:
            out["how"] = "muy_corta"
            return out
        seq = np.hstack([np.stack([f.hands_vec for f in seg]),
                         np.stack([f.body_vec for f in seg])]).astype(np.float32)
        out.update(ok=True, how=how, seq=seq, fps=float(fps), n_frames=len(frames))
    except Exception as e:      # un archivo danado no detiene todo
        out["how"] = f"error: {e}"[:200]
    return out


def save_shard(folder: Path, kind: str, results: list[dict]) -> None:
    n = len(list(folder.glob(f"{kind}_*.npz")))
    target = folder / f"{kind}_{n:05d}.npz"
    meta = [{k: v for k, v in r.items() if k not in ("seq", "points")} for r in results]
    arrays = {"meta": np.array(json.dumps(meta, ensure_ascii=False))}
    if kind == "video":
        seqs = [r["seq"] if r["ok"] else np.zeros((0, N_SEQ), np.float32) for r in results]
        arrays["lengths"] = np.array([len(s) for s in seqs], dtype=np.int32)
        arrays["data"] = np.concatenate(seqs) if seqs else np.zeros((0, N_SEQ), np.float32)
    else:
        arrays["points"] = np.stack([r["points"] if r["ok"] else np.zeros((21, 3), np.float32)
                                     for r in results])
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)


def load_shards(folder: Path) -> dict[str, dict]:
    """{id: resultado} de todo lo ya extraido."""
    done: dict[str, dict] = {}
    for path in sorted(folder.glob("*_*.npz")):
        if ".tmp" in path.name:
            continue
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if path.name.startswith("video_"):
                offsets = np.concatenate([[0], np.cumsum(z["lengths"])])
                data = z["data"]
                for i, m in enumerate(meta):
                    m["seq"] = data[offsets[i]:offsets[i + 1]]
                    done[m["id"]] = m
            else:
                points = z["points"]
                for i, m in enumerate(meta):
                    m["points"] = points[i]
                    done[m["id"]] = m
    return done


def _quiet_worker() -> None:
    """Los avisos de MediaPipe (W0000 ... Feedback manager) llenaban la salida
    de Colab; los errores de cada muestra se guardan en su resultado."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)


def stage_extract(items: list[dict], puntos: Path, n_proc: int) -> dict[str, dict]:
    settings_file = puntos / "ajustes.json"
    if puntos.exists() and any(puntos.glob("*.npz")) and (
            not settings_file.exists()
            or json.loads(settings_file.read_text(encoding="utf-8")) != EXTRACTION_SETTINGS):
        # Lo que se calculo con esos puntos (comparacion, modelos) tambien se aparta.
        old = puntos.with_name(f"anteriores_{time.strftime('%Y%m%d_%H%M%S')}")
        log(f"Los puntos ya guardados son de otra version de la extraccion: se apartan en {old.name}/")
        old.mkdir()
        for name in ("puntos", "comparacion", "modelos", "inventario.json"):
            if (puntos.parent / name).exists():
                (puntos.parent / name).rename(old / name)
    puntos.mkdir(parents=True, exist_ok=True)
    write_json(settings_file, EXTRACTION_SETTINGS)
    done = load_shards(puntos)
    pending = [it for it in items if it["id"] not in done]
    log(f"Extraccion: {len(items) - len(pending)} muestras ya hechas, faltan {len(pending)} "
        f"({sum(it['kind'] == 'video' for it in pending)} videos, "
        f"{sum(it['kind'] == 'image' for it in pending)} fotos) con {n_proc} procesos")
    if not pending:
        return done

    # Los modelos de MediaPipe se bajan una vez aqui, no en cada proceso a la vez.
    import extraer_palabras_videos as ex
    from body_tracker import DEFAULT_POSE_MODEL, ensure_pose_model
    ex.ensure_hand_model()
    ensure_pose_model(DEFAULT_POSE_MODEL)

    # Videos primero (los mas largos al inicio reparten mejor el trabajo).
    pending.sort(key=lambda it: (it["kind"] != "video", -Path(it["path"]).stat().st_size))
    buffers = {"video": [], "image": []}
    limits = {"video": 150, "image": 5000}
    t0, last_log, last_save, n_done = time.time(), time.time(), time.time(), 0
    total = Counter(it["kind"] for it in pending)
    count = Counter()
    first_image_t = None
    ctx = mproc.get_context("spawn")
    with ctx.Pool(n_proc, initializer=_quiet_worker, maxtasksperchild=400) as pool:
        for res in pool.imap_unordered(extract_one, pending, chunksize=1):
            buffers[res["kind"]].append(res)
            done[res["id"]] = res
            n_done += 1
            count[res["kind"]] += 1
            if res["kind"] == "image" and first_image_t is None:
                first_image_t = time.time()
            for kind, buf in buffers.items():
                if len(buf) >= limits[kind] or (buf and time.time() - last_save > 300):
                    save_shard(puntos, kind, buf)
                    buf.clear()
                    last_save = time.time()
            if time.time() - last_log > 60 or n_done == len(pending):
                # Videos y fotos tardan muy distinto: avance y tiempo por separado.
                if count["video"] < total["video"]:
                    rate = count["video"] / (time.time() - t0)
                    eta = (total["video"] - count["video"]) / max(rate, 1e-6) / 60
                    log(f"  videos {count['video']}/{total['video']} ({rate:.2f}/s, "
                        f"faltan ~{eta:.0f} min de videos; despues {total['image']} fotos, mas rapidas)")
                else:
                    rate = count["image"] / max(1e-6, time.time() - (first_image_t or t0))
                    eta = (total["image"] - count["image"]) / max(rate, 1e-6) / 60
                    log(f"  videos listos; fotos {count['image']}/{total['image']} ({rate:.0f}/s, "
                        f"faltan ~{eta:.0f} min)")
                last_log = time.time()
    for kind, buf in buffers.items():
        if buf:
            save_shard(puntos, kind, buf)
    return done


# =========================================================================== #
# Armado de los datos para entrenar
# =========================================================================== #

def fill_missing_body(seq: np.ndarray) -> np.ndarray:
    """Igual que senas.fill_missing_body: los frames sin cuerpo toman el
    bloque de cuerpo del frame con cuerpo mas cercano."""
    ok = seq[:, N_SEQ - 1] > 0
    if ok.all() or not ok.any():
        return seq
    idx = np.flatnonzero(ok)
    nearest = idx[np.abs(idx[None, :] - np.arange(len(seq))[:, None]).argmin(axis=1)]
    out = seq.copy()
    out[:, N_HAND:N_SEQ] = seq[nearest, N_HAND:N_SEQ]
    return out


def resample_linear(seq: np.ndarray, length: int, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    """Interpolacion lineal a `length` frames del tramo [start, end] (fracciones
    de la secuencia). Con 0 y 1 es lo que debe hacer la app antes del modelo:
    posiciones np.linspace(0, n-1, length)."""
    n = len(seq)
    pos = np.linspace(start * (n - 1), end * (n - 1), length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None].astype(np.float32)
    return seq[lo] * (1 - w) + seq[hi] * w


def team_samples() -> list[dict]:
    """Las 59 palabras del equipo (datos_palabras_dinamicas/), con su persona
    sacada de la hora del video."""
    out = []
    for path in sorted((APP_DIR / "datos_palabras_dinamicas").glob("*/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        hands = np.asarray(d["frames"], dtype=np.float32)
        body = np.asarray(d.get("body_frames", []), dtype=np.float32)
        if hands.ndim != 2 or body.shape != (len(hands), N_BODY) or len(hands) < MIN_SEQ_FRAMES:
            continue
        m = re.search(r"_(\d\d)_(\d\d)_(\d\d)_Pro", d.get("fuente", ""))
        hhmmss = ":".join(m.groups()) if m else "00:00:00"
        session = 1 + sum(hhmmss >= s for s in TEAM_SESSION_STARTS)
        out.append({
            "id": f"equipo/{path.parent.name}/{path.name}", "source": "equipo",
            "label": nfc_upper(path.parent.name), "family": "palabra",
            "person": f"equipo:{session}", "view": "frontal",
            "seq": np.hstack([hands, body]),
        })
    return out


def build_sequences(items: list[dict], done: dict[str, dict]) -> list[dict]:
    seqs = []
    for it in items:
        if it["kind"] != "video":
            continue
        r = done.get(it["id"])
        if r is None or not r["ok"]:
            continue
        seqs.append({**it, "seq": r["seq"]})
    seqs += team_samples()
    for s in seqs:
        s["seq"] = fill_missing_body(np.asarray(s["seq"], dtype=np.float32))
    return seqs


def static_vectors(points: np.ndarray, image_aspect: np.ndarray, mirror: bool, emulate_aspect: bool) -> np.ndarray:
    """(N, 21, 3) puntos de las fotos -> (N, 63) normalizados como en la app.
    En la app las coordenadas estan divididas entre el ancho y el alto de una
    imagen 4:3 en espejo; las fotos del CICESE son cuadradas y pueden venir sin
    espejo, asi que se prueba cual convencion reconoce mejor el modelo actual."""
    from sign_classifier import normalize_keypoints
    pts = points.copy()
    if mirror:
        pts[:, :, 0] = 1.0 - pts[:, :, 0]
    if emulate_aspect:
        pts[:, :, 0] *= (image_aspect / LIVE_ASPECT)[:, None]
    return np.stack([normalize_keypoints(p.reshape(63)) for p in pts])


# =========================================================================== #
# Modelos (PyTorch)
# =========================================================================== #

def usable_cpus() -> int:
    """Nucleos que de verdad se pueden usar. En Colab, nproc dice 24 pero la
    maquina esta limitada a 4 (cgroup cpu.max = 400000 100000)."""
    n = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, max(1, math.ceil(quota / period)))
        except (OSError, ValueError):
            pass
    return n


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(usable_cpus())
    return torch.device("cpu")


def make_seq_model(n_classes: int, d: int = 128, layers: int = 3, heads: int = 4, drop: float = 0.2):
    import torch
    import torch.nn as nn

    class SeqTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(N_SEQ, d), nn.LayerNorm(d))
            self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_classes)

        def forward(self, x):
            h = self.enc(self.inp(x) + self.pos)
            return self.head(self.norm(h.mean(dim=1)))

    return SeqTransformer()


def make_static_model(n_classes: int):
    import torch.nn as nn
    # Misma forma que lsm_alphabet.onnx (Gemm-Relu-Gemm-Relu-Gemm): se puede
    # cambiar por el actual sin tocar sign_classifier.py.
    return nn.Sequential(nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, n_classes))


def _rotate_hands(x: np.ndarray, rng: np.random.Generator, max_deg: float, n_hands: int) -> np.ndarray:
    """Gira (en el plano de la imagen) y escala cada mano alrededor de su
    muneca, y les suma ruido. x: (B, T, F) o (B, F); las manos ausentes (todo
    en cero) se quedan en cero."""
    x = x.copy()
    lead = x.shape[:-1]
    for h in range(n_hands):
        block = x[..., h * 63:(h + 1) * 63].reshape(*lead, 21, 3)
        present = np.abs(block).sum(axis=(-1, -2)) > 0
        theta = np.deg2rad(rng.uniform(-max_deg, max_deg, size=lead[:1]))
        scale = rng.uniform(0.9, 1.1, size=lead[:1])
        c, s = np.cos(theta) * scale, np.sin(theta) * scale
        shape = (-1,) + (1,) * (block.ndim - 2)
        bx, by = block[..., 0].copy(), block[..., 1].copy()
        block[..., 0] = c.reshape(shape) * bx - s.reshape(shape) * by
        block[..., 1] = s.reshape(shape) * bx + c.reshape(shape) * by
        noise = rng.normal(0, 0.01, size=block.shape).astype(np.float32)
        noise[..., 2] *= 0.3        # la z (metros, mundo) es mucho mas chica que x, y
        block += noise
        block *= present[..., None, None]
        x[..., h * 63:(h + 1) * 63] = block.reshape(*lead, 63)
    return x


def seq_batch(seqs: list[np.ndarray], rng: Optional[np.random.Generator]) -> np.ndarray:
    """Remuestrea a SEQ_LEN; al entrenar, con variaciones: tramo recortado
    (velocidad e inicio/fin distintos), giro, escala y ruido de las manos y
    ruido en la ubicacion respecto al cuerpo."""
    out = np.empty((len(seqs), SEQ_LEN, N_SEQ), dtype=np.float32)
    for i, s in enumerate(seqs):
        if rng is None:
            out[i] = resample_linear(s, SEQ_LEN)
        else:
            keep = rng.uniform(0.8, 1.0)
            start = rng.uniform(0, 1 - keep)
            out[i] = resample_linear(s, SEQ_LEN, start, start + keep)
    if rng is not None:
        out = _rotate_hands(out, rng, 15.0, 2)
        body = out[:, :, N_HAND:N_SEQ - 1]
        out[:, :, N_HAND:N_SEQ - 1] = body + rng.normal(0, 0.03, size=body.shape) * (body != 0)
    return out


def class_weights(y: np.ndarray, n_classes: int):
    import torch
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[counts > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


def warmup_fraction(steps: int) -> float:
    """10% de calentamiento, pero al menos 2 pasos (con menos, OneCycleLR
    divide entre cero en entrenamientos muy cortos)."""
    return min(0.3, max(0.1, 2.0 / max(1, steps)))


def train_seq_model(train: list[dict], classes: list[str], epochs: int, seed: int, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cls_idx = {c: i for i, c in enumerate(classes)}
    seqs = [s["seq"] for s in train]
    y = np.array([cls_idx[s["label"]] for s in train])
    model = make_seq_model(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, len(classes)).to(device), label_smoothing=0.1)
    batch = 64
    steps = epochs * math.ceil(len(seqs) / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=warmup_fraction(steps))
    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(len(seqs))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb = torch.from_numpy(seq_batch([seqs[j] for j in b], rng)).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            log(f"    epoca {epoch + 1}/{epochs}: perdida {total / len(seqs):.3f}")
    model.eval()
    return model


def predict_seq(model, seqs: list[np.ndarray], device) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            xb = torch.from_numpy(seq_batch(seqs[i:i + 256], None)).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def train_static_model(x: np.ndarray, y: np.ndarray, n_classes: int, epochs: int, seed: int, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_static_model(n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, n_classes).to(device), label_smoothing=0.05)
    batch = 256
    steps = epochs * math.ceil(len(x) / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=warmup_fraction(steps))
    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(len(x))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb = torch.from_numpy(_rotate_hands(x[b], rng, 10.0, 1)).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def predict_static(model, x: np.ndarray, device) -> np.ndarray:
    import torch
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + 4096]).to(device)).cpu().numpy()
                               for i in range(0, len(x), 4096)]) if len(x) else np.zeros((0,))


def export_onnx(model, example, path: Path, input_name: str, check_input: np.ndarray) -> None:
    """Exporta a un solo archivo .onnx y comprueba con onnxruntime que da lo
    mismo que PyTorch."""
    import onnx
    import torch
    model = model.cpu().eval()
    tmp = Path(tempfile.mkdtemp()) / "modelo.onnx"
    kwargs = dict(input_names=[input_name], output_names=["logits"],
                  dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (example,), str(tmp), dynamo=False, **kwargs)
    except Exception as e:     # PyTorch nuevo sin el exportador clasico
        log(f"  exportador clasico no disponible ({type(e).__name__}); se usa torch.export")
        kwargs.pop("dynamic_axes")
        batch = torch.export.Dim("batch")
        torch.onnx.export(model, (example,), str(tmp), dynamo=True,
                          dynamic_shapes=({0: batch},), **kwargs)
    onnx.save_model(onnx.load(str(tmp)), str(path), save_as_external_data=False)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {input_name: check_input})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(check_input)).numpy()
        diff = float(np.abs(got - want).max())
        log(f"  {path.name}: ONNX verificado (diferencia maxima {diff:.2e})")
    except ImportError:
        log(f"  {path.name}: exportado (sin onnxruntime para verificarlo)")


# =========================================================================== #
# DTW (el reconocedor actual de la app) para comparar
# =========================================================================== #

def make_dtw(templates: list[dict], with_body: bool):
    """DTW como en la app: con cuerpo = palabras (todos los frames), sin
    cuerpo = letras dinamicas (1 de cada DTW_LETTER_STEP)."""
    from dtw_recognizer import DTWRecognizer
    rec = DTWRecognizer(data_dir=Path(tempfile.mkdtemp()) / "vacio", auto_save_labels=False,
                        body_weight=DTW_WORD_BODY_WEIGHT if with_body else None)
    rec._templates = {}
    for t in templates:
        seq = t["seq"] if with_body else t["seq"][:, :N_HAND]
        step = 1 if with_body else DTW_LETTER_STEP
        rec._templates.setdefault(t["label"], []).append(rec._prepare(seq[::step]))
    rec._labels = sorted(rec._templates)
    return rec


def dtw_predict(rec, queries: list[dict], with_body: bool) -> list[str]:
    preds = []
    for q in queries:
        seq = q["seq"] if with_body else q["seq"][:, :N_HAND]
        d = rec.compute_distances(seq[::1 if with_body else DTW_LETTER_STEP])
        preds.append(min(d, key=d.get))
    return preds


# =========================================================================== #
# 4. Comparacion por personas
# =========================================================================== #

def assign_folds(samples: list[dict], n_folds: int) -> dict[str, int]:
    """Cada persona va a un pliegue, repartidas por fuente para que todos los
    pliegues tengan gente de cada dataset."""
    fold = {}
    by_source = defaultdict(set)
    for s in samples:
        by_source[s["source"]].add(s["person"])
    for source, persons in sorted(by_source.items()):
        for i, p in enumerate(sorted(persons, key=lambda p: (len(p), p))):
            fold[p] = i % n_folds
    return fold


def accuracy(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    ok = sum(a == b for a, b in pairs)
    return {"aciertos": ok, "total": n, "exactitud": round(ok / n, 4) if n else None}


def restricted_argmax(logits: np.ndarray, classes: list[str], allowed: set[str]) -> list[str]:
    mask = np.array([c in allowed for c in classes])
    sub = np.where(mask[None, :], logits, -np.inf)
    return [classes[i] for i in sub.argmax(axis=1)]


def stage_compare(seqs: list[dict], static: dict, n_folds: int, epochs: int, device, results: dict,
                  cache_dir: Path) -> None:
    classes = sorted({s["label"] for s in seqs})
    letters = {c for c in classes if c in DYNAMIC_LETTERS}
    words = set(classes) - letters
    fold_of = assign_folds(seqs, n_folds)
    log(f"Comparacion por personas: {len(seqs)} secuencias, {len(classes)} clases "
        f"({len(letters)} letras, {len(words)} palabras), {n_folds} pliegues")

    pairs = defaultdict(list)       # (metodo, familia, fuente) -> [(real, predicha)]
    top5_hits = defaultdict(int)
    per_label = defaultdict(lambda: defaultdict(list))
    fold_rows = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_folds):
        train = [s for s in seqs if fold_of[s["person"]] != k]
        test = [s for s in seqs if fold_of[s["person"]] == k and s["view"] == "frontal"]
        if not test:
            continue
        # Cada pliegue terminado se guarda: si Colab se desconecta, no se repite.
        cache = cache_dir / f"pliegue_{k + 1}.json"
        signature = {"pliegues": n_folds, "epocas": epochs, "muestras": len(seqs), "clases": len(classes)}
        saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        if saved is None or saved.get("firma") != signature:
            saved = {"firma": signature, **_run_fold(k, n_folds, train, test, classes, letters, words,
                                                     epochs, device)}
            write_json(cache, saved)
        else:
            log(f"  Pliegue {k + 1}/{n_folds}: ya estaba hecho")
        for method, fam, source, label, pred in saved["registros"]:
            pairs[(method, fam, source)].append((label, pred))
            per_label[method][label].append(pred == label)
        for fam, source, hit in saved["top5"]:
            top5_hits[(fam, source)] += hit
        fold_rows.append(saved["fila"])

    summary = {}
    for (method, fam, source), pr in sorted(pairs.items()):
        summary.setdefault(f"{fam}/{source}", {})[method] = accuracy(pr)
    for (fam, source), hits in top5_hits.items():
        key = f"{fam}/{source}"
        total = summary[key]["transformer"]["total"]
        summary[key]["transformer_top5"] = {"aciertos": hits, "total": total,
                                            "exactitud": round(hits / total, 4) if total else None}
    worst = {}
    for method, labels in per_label.items():
        acc = {lab: sum(v) / len(v) for lab, v in labels.items()}
        worst[method] = sorted(acc.items(), key=lambda kv: kv[1])[:10]
    results["secuencias"] = {"clases": classes, "pliegues": fold_rows, "resumen": summary,
                             "peores_clases": worst}

    # Letras estaticas: la division del CICESE (18 personas para entrenar,
    # 2 nunca vistas para probar).
    if static:
        xtr, ytr, xte, yte, labels = static["xtr"], static["ytr"], static["xte"], static["yte"], static["labels"]
        res = {"orientacion": static["orientation"], "modelo_actual": static.get("current_acc")}
        if len(xtr) and len(xte):
            model = train_static_model(xtr, ytr, len(labels), epochs=30, seed=0, device=device)
            pred = predict_static(model, xte, device).argmax(axis=1)
            res["modelo_nuevo"] = accuracy([(labels[a], labels[b]) for a, b in zip(yte, pred)])
            per = defaultdict(list)
            for a, b in zip(yte, pred):
                per[labels[a]].append(a == b)
            res["peores_letras_nuevo"] = sorted(((k2, round(sum(v) / len(v), 3)) for k2, v in per.items()),
                                                key=lambda kv: kv[1])[:6]
            log(f"  Estaticas: actual {res['modelo_actual']}, nuevo {res['modelo_nuevo']}")
        results["estaticas"] = res


def _run_fold(k: int, n_folds: int, train: list[dict], test: list[dict], classes: list[str],
              letters: set[str], words: set[str], epochs: int, device) -> dict:
    """Entrena el transformer sin las personas del pliegue k y las reconoce
    con el y con el DTW. Devuelve registros (metodo, familia, fuente, real,
    predicha), aciertos en el top-5 y la fila del reporte."""
    records, top5_list = [], []
    t0 = time.time()
    log(f"  Pliegue {k + 1}/{n_folds}: entrenando con {len(train)}, probando con {len(test)} "
        f"(personas: {', '.join(sorted({s['person'] for s in test}))})")
    model = train_seq_model(train, classes, epochs, seed=k, device=device)
    logits = predict_seq(model, [s["seq"] for s in test], device)
    t_train = time.time() - t0

    t0 = time.time()
    frontal_train = [s for s in train if s["view"] == "frontal"]
    dtw_letters = make_dtw([s for s in frontal_train if s["label"] in letters], with_body=False)
    dtw_words = make_dtw([s for s in frontal_train if s["label"] in words], with_body=True)
    row = {"pliegue": k + 1, "segundos_transformer": round(t_train, 1)}
    for fam, allowed, rec, body in (("letra", letters, dtw_letters, False),
                                    ("palabra", words, dtw_words, True)):
        idx = [i for i, s in enumerate(test) if s["label"] in allowed]
        if not idx or not rec._labels:
            continue
        sub = [test[i] for i in idx]
        tr_pred = restricted_argmax(logits[idx], classes, allowed)
        dtw_pred = dtw_predict(rec, sub, body)
        order = np.argsort(-logits[idx], axis=1)
        for j, s in enumerate(sub):
            records.append(["transformer", fam, s["source"], s["label"], tr_pred[j]])
            records.append(["dtw", fam, s["source"], s["label"], dtw_pred[j]])
            top5 = [classes[c] for c in order[j] if classes[c] in allowed][:5]
            top5_list.append([fam, s["source"], int(s["label"] in top5)])
        row[f"{fam}_transformer"] = accuracy(list(zip([s["label"] for s in sub], tr_pred)))
        row[f"{fam}_dtw"] = accuracy(list(zip([s["label"] for s in sub], dtw_pred)))
    row["segundos_dtw"] = round(time.time() - t0, 1)
    log(f"    {json.dumps({k2: v for k2, v in row.items() if 'segundos' not in k2}, ensure_ascii=False)}")
    return {"registros": records, "top5": top5_list, "fila": row}


def prepare_static(items: list[dict], done: dict[str, dict]) -> dict:
    """Vectores de las fotos con la convencion (espejo, proporcion) con que el
    modelo actual reconoce mejor las fotos de entrenamiento."""
    rows = [(it, done.get(it["id"])) for it in items if it["kind"] == "image"]
    rows = [(it, r) for it, r in rows if r is not None and r["ok"]]
    if not rows:
        return {}
    points = np.stack([r["points"] for _, r in rows]).astype(np.float32)
    aspect = np.array([r.get("aspect", 1.0) for _, r in rows], dtype=np.float32)
    labels_all = [it["label"] for it, _ in rows]
    is_test = np.array([it["person"].endswith(":test") for it, _ in rows])

    current, current_labels = None, []
    try:
        import onnxruntime as ort
        cur_path = APP_DIR / "lsm_alphabet.onnx"
        current = ort.InferenceSession(str(cur_path), providers=["CPUExecutionProvider"])
        current_labels = json.loads((APP_DIR / "lsm_labels.json").read_text(encoding="utf-8"))["labels"]
    except Exception as e:
        log(f"  No se pudo cargar el modelo estatico actual ({e}); se usa espejo + 4:3 sin verificar")

    def score(x, lab):
        logits = current.run(None, {current.get_inputs()[0].name: x.astype(np.float32)})[0]
        pred = [current_labels[i] for i in logits.argmax(axis=1)]
        return accuracy(list(zip(lab, pred)))

    best = (True, True)
    conventions = {}
    if current is not None:
        known = np.array([lab in current_labels for lab in labels_all])
        tr = np.flatnonzero(~is_test & known)
        sample = tr[np.random.default_rng(0).permutation(len(tr))[:20000]]
        for mirror in (True, False):
            for emulate in (True, False):
                x = static_vectors(points[sample], aspect[sample], mirror, emulate)
                conventions[f"espejo={mirror},4:3={emulate}"] = score(x, [labels_all[i] for i in sample])
        best_key = max(conventions, key=lambda k: conventions[k]["exactitud"] or 0)
        best = tuple(part.split("=")[1] == "True" for part in best_key.split(","))
        log(f"  Convenciones de las fotos (modelo actual): {conventions} -> {best_key}")

    x = static_vectors(points, aspect, *best).astype(np.float32)
    labels = sorted(set(labels_all))
    y = np.array([labels.index(lab) for lab in labels_all])
    out = {"labels": labels, "orientation": {"espejo": best[0], "proporcion_4_3": best[1],
                                             "pruebas": conventions},
           "xtr": x[~is_test], "ytr": y[~is_test], "xte": x[is_test], "yte": y[is_test],
           "x_all": x, "y_all": y}
    if current is not None and is_test.any():
        te = np.flatnonzero(is_test & np.array([lab in current_labels for lab in labels_all]))
        out["current_acc"] = score(x[te], [labels_all[i] for i in te])
    missing = sorted(set(current_labels) - set(labels))
    extra = sorted(set(labels) - set(current_labels))
    out["label_diff"] = {"faltan_en_cicese": missing, "sobran_en_cicese": extra}
    return out


# =========================================================================== #
# 5. Modelos finales
# =========================================================================== #

def stage_final(seqs: list[dict], static: dict, epochs: int, device, modelos: Path, results: dict) -> None:
    import torch
    modelos.mkdir(parents=True, exist_ok=True)
    classes = sorted({s["label"] for s in seqs})
    results["modelos"] = {}
    if (modelos / "lsm_secuencias.json").exists():      # se escribe al final, tras el .onnx
        log("Modelo final de secuencias: ya estaba hecho")
    else:
        log(f"Modelo final de secuencias: {len(seqs)} muestras, {len(classes)} clases")
        model = train_seq_model(seqs, classes, epochs, seed=123, device=device)
        check = seq_batch([s["seq"] for s in seqs[:4]], None)
        export_onnx(model, torch.from_numpy(check), modelos / "lsm_secuencias.onnx", "secuencia", check)
        write_json(modelos / "lsm_secuencias.json", {
        "labels": classes,
            "letras_dinamicas": sorted(c for c in classes if c in DYNAMIC_LETTERS),
            "seq_len": SEQ_LEN,
            "n_features": N_SEQ,
            "entrada": ("(1, seq_len, 135): por frame los 126 de build_dynamic_feature_vector + los 9 de "
                        "body_location_features; antes, fill_missing_body y remuestreo lineal de TODA la "
                        "sena a seq_len frames (posiciones np.linspace(0, n-1, seq_len))"),
            "salida": "logits (softmax para probabilidades)",
        })
    results["modelos"]["secuencias"] = "modelos/lsm_secuencias.onnx"

    if static:
        labels = static["labels"]
        if (modelos / "lsm_labels.json").exists():
            log("Modelo final estatico: ya estaba hecho")
        else:
            log(f"Modelo final estatico: {len(static['x_all'])} fotos de 20 personas, {len(labels)} letras")
            smodel = train_static_model(static["x_all"], static["y_all"], len(labels), epochs=30, seed=123,
                                        device=device)
            check = static["x_all"][:4].astype(np.float32)
            export_onnx(smodel, torch.from_numpy(check), modelos / "lsm_alphabet.onnx", "keypoints", check)
            write_json(modelos / "lsm_labels.json", {"labels": labels, "n_features": 63,
                                                     "normalization": "wrist_centered_middle_scaled"})
        results["modelos"]["estatico"] = "modelos/lsm_alphabet.onnx"


# =========================================================================== #
# 6. Reporte
# =========================================================================== #

def fmt(acc: Optional[dict]) -> str:
    if not acc or acc.get("total") in (None, 0):
        return "—"
    return f"{100 * acc['exactitud']:.1f}% ({acc['aciertos']}/{acc['total']})"


def write_report(trabajo: Path, results: dict) -> None:
    lines = ["# Entrenamiento con los datasets del CICESE", ""]
    hw = results.get("hardware", {})
    lines += [f"- Hardware: {hw.get('nucleos')} nucleos, dispositivo de entrenamiento `{hw.get('dispositivo')}`",
              "- Tiempos por etapa: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in results.get("tiempos", {}).items()),
              ""]
    d = results.get("datos", {})
    if d:
        lines += ["## Datos", "", "| Fuente | Muestras | Personas | Clases |", "|---|---|---|---|"]
        for src, row in d.get("por_fuente", {}).items():
            lines.append(f"| {src} | {row['muestras']} | {row['personas']} | {row['clases']} |")
        lines += ["", f"Extraccion: {json.dumps(d.get('resultado_extraccion', {}), ensure_ascii=False)}", ""]
    seq = results.get("secuencias")
    if seq:
        lines += ["## Letras dinamicas y palabras (personas nunca vistas)", "",
                  "| Grupo | DTW actual | Transformer | Transformer top-5 |", "|---|---|---|---|"]
        for key, row in seq["resumen"].items():
            lines.append(f"| {key} | {fmt(row.get('dtw'))} | {fmt(row.get('transformer'))} | "
                         f"{fmt(row.get('transformer_top5'))} |")
        lines += ["", "Clases con menos aciertos:"]
        for method, rows in seq["peores_clases"].items():
            lines.append(f"- {method}: " + ", ".join(f"{lab} {100 * a:.0f}%" for lab, a in rows))
        lines.append("")
    st = results.get("estaticas")
    if st:
        lines += ["## Letras estaticas (las 2 personas de prueba del CICESE)", "",
                  f"- Modelo actual (`lsm_alphabet.onnx`): {fmt(st.get('modelo_actual'))}",
                  f"- Modelo nuevo (mismo tamano, entrenado con las otras 18 personas): {fmt(st.get('modelo_nuevo'))}",
                  f"- Convencion con que se leyeron las fotos (la que mejor reconoce el modelo actual): "
                  f"espejo {'si' if st.get('orientacion', {}).get('espejo') else 'no'}, "
                  f"proporcion 4:3 {'si' if st.get('orientacion', {}).get('proporcion_4_3') else 'no'}",
                  f"- Letras: {json.dumps(results.get('datos', {}).get('letras_estaticas', {}), ensure_ascii=False)}",
                  ""]
    if results.get("modelos"):
        lines += ["## Archivos", ""] + [f"- `{v}`" for v in results["modelos"].values()] + [
            "", "Los modelos todavia no estan conectados a la app: falta integrarlos en senas.py "
            "(y calibrar con que confianza escribir) si la comparacion sale a su favor.", ""]
    lines += ["Datos: Navarrete-Lopez, Lopez-Nava y Morfin-Chavez, CICESE (Zenodo, CC BY 4.0).", ""]
    (trabajo / "reporte.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(trabajo / "resultados.json", results)


# =========================================================================== #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trabajo", type=Path, required=True,
                    help="carpeta que se conserva (en Colab, una de Google Drive): puntos, modelos, reporte")
    ap.add_argument("--datos", type=Path, default=Path("/content/datos_lsm"),
                    help="donde descargar y descomprimir (disco de la maquina, no Drive)")
    ap.add_argument("--sin-descarga", action="store_true", help="los datos ya estan en --datos")
    ap.add_argument("--conservar-7z", action="store_true", help="no borrar los .7z al descomprimir")
    ap.add_argument("--procesos", type=int, default=0, help="procesos para extraer (por defecto, nucleos - 1)")
    ap.add_argument("--paso-estatico", type=int, default=3, help="usar 1 de cada N fotos del alfabeto estatico")
    ap.add_argument("--sin-perfil", action="store_true", help="no usar la vista de perfil de las letras dinamicas")
    ap.add_argument("--pliegues", type=int, default=4)
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--hasta", choices=["descarga", "extrae", "compara", "entrena"], default="entrena")
    args = ap.parse_args()

    trabajo = args.trabajo
    trabajo.mkdir(parents=True, exist_ok=True)
    n_cpu = usable_cpus()
    n_proc = args.procesos or n_cpu
    if n_cpu < (os.cpu_count() or 1):
        log(f"La maquina reporta {os.cpu_count()} nucleos pero solo deja usar {n_cpu}")
    results: dict = {"hardware": {"nucleos": n_cpu, "procesos_extraccion": n_proc}, "tiempos": {}}
    log(f"Nucleos: {n_cpu}; carpeta de trabajo: {trabajo}")

    # Con la extraccion completa (inventario.json), los videos ya no hacen
    # falta: al reanudar no se vuelven a descargar los 11 GB.
    inventario = trabajo / "inventario.json"
    settings_file = trabajo / "puntos" / "ajustes.json"
    same_version = settings_file.exists() and json.loads(
        settings_file.read_text(encoding="utf-8")) == EXTRACTION_SETTINGS
    if inventario.exists() and same_version and args.hasta != "descarga":
        log("La extraccion ya estaba completa: se salta la descarga y la extraccion "
            f"(borra {inventario.name} para rehacerlas)")
        items = json.loads(inventario.read_text(encoding="utf-8"))
        done = load_shards(trabajo / "puntos")
    else:
        t0 = time.time()
        if not args.sin_descarga:
            stage_download_and_unpack(args.datos, args.conservar_7z)
        results["tiempos"]["descarga"] = time.time() - t0
        if args.hasta == "descarga":
            return 0

        t0 = time.time()
        items = list_items(args.datos, args.paso_estatico, not args.sin_perfil)
        log(f"Inventario: {Counter(it['source'] for it in items)}")
        done = stage_extract(items, trabajo / "puntos", n_proc)
        results["tiempos"]["extraccion"] = time.time() - t0
        if all(it["id"] in done for it in items):
            write_json(inventario, [{k: v for k, v in it.items() if k != "path"} for it in items])
    how = Counter(f"{it['source']}:{done[it['id']]['how'].split(':')[0]}" for it in items if it["id"] in done)
    log(f"Resultado de la extraccion: {dict(how)}")
    if args.hasta == "extrae":
        write_report(trabajo, results)
        return 0

    seqs = build_sequences(items, done)
    static = prepare_static(items, done)
    by_src = defaultdict(lambda: {"muestras": 0, "personas": set(), "clases": set()})
    for s in seqs:
        row = by_src[s["source"]]
        row["muestras"] += 1
        row["personas"].add(s["person"])
        row["clases"].add(s["label"])
    results["datos"] = {
        "por_fuente": {k: {"muestras": v["muestras"], "personas": len(v["personas"]), "clases": len(v["clases"])}
                       for k, v in by_src.items()},
        "resultado_extraccion": dict(how),
    }
    if static:
        results["datos"]["por_fuente"]["estatico"] = {"muestras": len(static["x_all"]), "personas": 20,
                                                      "clases": len(static["labels"])}
        results["datos"]["letras_estaticas"] = static["label_diff"]

    device = pick_device()
    results["hardware"]["dispositivo"] = str(device)
    log(f"Entrenamiento en: {device}")
    t0 = time.time()
    stage_compare(seqs, static, args.pliegues, args.epocas, device, results, trabajo / "comparacion")
    results["tiempos"]["comparacion"] = time.time() - t0
    write_report(trabajo, results)
    if args.hasta == "compara":
        return 0

    t0 = time.time()
    stage_final(seqs, static, args.epocas, device, trabajo / "modelos", results)
    results["tiempos"]["modelos_finales"] = time.time() - t0
    write_report(trabajo, results)
    log(f"Listo. Reporte: {trabajo / 'reporte.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
