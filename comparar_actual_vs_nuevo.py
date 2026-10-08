"""Compara lo que reconoce la app HOY contra lo que reconoceria con los datos
nuevos del CICESE, siempre con personas que el modelo nunca vio. Usa lo que
dejo entrenar_todo_colab.py en su carpeta de trabajo (puntos/, inventario.json,
resultados.json).

  - Palabras de la app (HOLA, GRACIAS, POR FAVOR, AYUDA, MAMA): cada persona del
    equipo contra las otras dos. Actual = DTW solo con las muestras del equipo
    (como senas.py). Nuevo = DTW con el equipo + las glosas del CICESE de esas
    palabras, y el transformer entrenado con todo el CICESE + el equipo.
  - Letras dinamicas: cada persona del CICESE (vista frontal, recortada como en
    vivo) contra las plantillas SIN esa persona. Actual = las plantillas de
    datos_dinamicas/ que usa la app hoy. Nuevo = plantillas sacadas con el
    recorte 4:3 de la app, y el transformer (de resultados.json).
  - Letras estaticas y palabras del CICESE: de resultados.json.

Uso:
    python comparar_actual_vs_nuevo.py --trabajo CARPETA_lsm_entrenamiento [--epocas 120]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

import entrenar_todo_colab as E


def acc(pairs: list[tuple[str, str]]) -> dict:
    return E.accuracy(pairs)


def current_letter_templates() -> list[dict]:
    """Plantillas de datos_dinamicas/ (las que usa la app). Las del CICESE se
    llaman muestra_<persona>_<rep>.json; las demas las grabo el equipo."""
    out = []
    for path in sorted((E.APP_DIR / "datos_dinamicas").glob("*/*.json")):
        frames = np.asarray(json.loads(path.read_text(encoding="utf-8"))["frames"], dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != E.N_HAND or len(frames) == 0:
            continue
        m = re.fullmatch(r"muestra_(\d+)_(\d+)", path.stem)
        out.append({"label": unicodedata.normalize("NFC", path.parent.name),
                    "person": f"dinamico:{int(m.group(1))}" if m else "equipo",
                    "seq": np.hstack([frames, np.zeros((len(frames), E.N_BODY), np.float32)])})
    return out


def as_template(s: dict, raw: dict[str, np.ndarray]) -> dict:
    """La app rellena el cuerpo faltante (fill_missing_body) solo en lo que
    ve la camara; las plantillas guardadas se comparan tal cual."""
    return {**s, "seq": raw[s["id"]]}


def team_words(seqs: list[dict], raw: dict[str, np.ndarray], epochs: int, device) -> dict:
    team = [s for s in seqs if s["source"] == "equipo"]
    words = sorted({s["label"] for s in team})
    cicese_same = [s for s in seqs if s["source"] == "glosas" and s["label"] in words]
    all_words = sorted({s["label"] for s in seqs if s["label"] not in E.DYNAMIC_LETTERS})
    classes = sorted({s["label"] for s in seqs})
    pairs = defaultdict(list)
    per_person = []
    for person in sorted({s["person"] for s in team}):
        test = [s for s in team if s["person"] == person]
        others = [s for s in team if s["person"] != person]
        row = {"persona": person, "muestras": len(test)}
        dtw_now = E.make_dtw([as_template(s, raw) for s in others], with_body=True)
        dtw_new = E.make_dtw([as_template(s, raw) for s in others + cicese_same], with_body=True)
        E.log(f"  {person}: entrenando el transformer sin esta persona ...")
        model = E.train_seq_model([s for s in seqs if s["person"] != person], classes, epochs, seed=7, device=device)
        logits = E.predict_seq(model, [s["seq"] for s in test], device)
        preds = {
            "actual_dtw": E.dtw_predict(dtw_now, test, True),
            "dtw_con_cicese": E.dtw_predict(dtw_new, test, True),
            "transformer_5_palabras": E.restricted_argmax(logits, classes, set(words)),
            "transformer_todas_las_palabras": E.restricted_argmax(logits, classes, set(all_words)),
        }
        for method, pred in preds.items():
            pp = [(s["label"], p) for s, p in zip(test, pred)]
            pairs[method] += pp
            row[method] = acc(pp)
        per_person.append(row)
        E.log(f"    {json.dumps(row, ensure_ascii=False)}")
    return {"palabras": words, "glosas_cicese_de_esas_palabras": len(cicese_same),
            "vocabulario_nuevo": len(all_words),
            "total": {m: acc(p) for m, p in pairs.items()}, "por_persona": per_person}


def dynamic_letters(seqs: list[dict]) -> dict:
    queries = [s for s in seqs if s["source"] == "dinamico" and s["view"] == "frontal"]
    new_templates = queries
    cur_templates = current_letter_templates()
    pairs = defaultdict(list)
    for person in sorted({q["person"] for q in queries}, key=lambda p: int(p.split(":")[1])):
        test = [q for q in queries if q["person"] == person]
        for method, templates in (("actual_dtw", cur_templates), ("dtw_plantillas_nuevas", new_templates)):
            rec = E.make_dtw([t for t in templates if t["person"] != person], with_body=False)
            pairs[method] += list(zip([q["label"] for q in test], E.dtw_predict(rec, test, False)))
    return {"personas": len({q["person"] for q in queries}), "muestras": len(queries),
            "plantillas_actuales": len(cur_templates),
            "total": {m: acc(p) for m, p in pairs.items()}}


def pct(a: dict | None) -> str:
    return "—" if not a or not a.get("total") else f"{100 * a['exactitud']:.1f}% ({a['aciertos']}/{a['total']})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajo", type=Path, required=True)
    ap.add_argument("--epocas", type=int, default=120)
    args = ap.parse_args()
    trabajo = args.trabajo
    items = json.loads((trabajo / "inventario.json").read_text(encoding="utf-8"))
    done = E.load_shards(trabajo / "puntos")
    seqs = E.build_sequences(items, done)
    raw = {**{k: v["seq"] for k, v in done.items() if v.get("ok") and "seq" in v},
           **{s["id"]: s["seq"] for s in E.team_samples()}}
    prev = json.loads((trabajo / "resultados.json").read_text(encoding="utf-8"))
    device = E.pick_device()
    E.log(f"{len(seqs)} secuencias; entrenamiento en {device}")

    E.log("Palabras de la app (cada persona del equipo contra las otras dos)")
    words = team_words(seqs, raw, args.epocas, device)
    E.log("Letras dinamicas (cada persona del CICESE contra las plantillas sin ella)")
    letters = dynamic_letters(seqs)
    summary = prev.get("secuencias", {}).get("resumen", {})
    static = prev.get("estaticas", {})
    out = {"palabras_app": words, "letras_dinamicas": letters,
           "letras_dinamicas_transformer": summary.get("letra/dinamico", {}).get("transformer"),
           "estaticas": static, "glosas_cicese": summary.get("palabra/glosas", {})}
    E.write_json(trabajo / "actual_vs_nuevo.json", out)

    w, l = words["total"], letters["total"]
    lines = [
        "# Actual contra nuevo (personas que el modelo nunca vio)", "",
        "| Qué | Con quién se prueba | Actual (app de hoy) | Nuevo |", "|---|---|---|---|",
        f"| Las 5 palabras de la app | Cada persona del equipo ({w['actual_dtw']['total']} muestras) | "
        f"DTW, muestras del equipo: {pct(w['actual_dtw'])} | DTW + CICESE: {pct(w['dtw_con_cicese'])}<br>"
        f"Transformer, 5 palabras: {pct(w['transformer_5_palabras'])}<br>"
        f"Transformer, {words['vocabulario_nuevo']} palabras: {pct(w['transformer_todas_las_palabras'])} |",
        f"| Letras dinámicas (6) | {letters['personas']} personas del CICESE, de frente ({letters['muestras']} muestras) | "
        f"DTW, plantillas actuales: {pct(l['actual_dtw'])} | DTW, plantillas nuevas: {pct(l['dtw_plantillas_nuevas'])}<br>"
        f"Transformer: {pct(out['letras_dinamicas_transformer'])} |",
        f"| Letras estáticas | Las 2 personas de prueba del CICESE | Modelo actual: {pct(static.get('modelo_actual'))} | "
        f"Modelo nuevo: {pct(static.get('modelo_nuevo'))} |",
        f"| Palabras del CICESE | 12 personas del CICESE | — (la app no las conoce) | "
        f"DTW: {pct(out['glosas_cicese'].get('dtw'))}<br>Transformer: {pct(out['glosas_cicese'].get('transformer'))} |",
        "",
        f"- Glosas del CICESE que son palabras de la app: {words['glosas_cicese_de_esas_palabras']} muestras.",
        "- Las letras estáticas se prueban con personas del mismo laboratorio con que se entrena el modelo "
        "nuevo, así que esa comparación lo favorece.",
        "- El transformer de letras dinámicas viene de la validación por pliegues (entrenado sin 5 personas a la vez); "
        "los DTW, dejando fuera una persona a la vez.",
        "- La columna Actual compara como senas.py: la seña probada con el cuerpo rellenado (fill_missing_body) y "
        "las plantillas tal cual. Sin rellenar, las palabras del equipo dan una muestra más (56/59, la cifra medida antes).",
        "- El DTW de las palabras del CICESE viene del reporte de Colab, que usó 1 de cada 2 cuadros.",
        "",
    ]
    (trabajo / "actual_vs_nuevo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
