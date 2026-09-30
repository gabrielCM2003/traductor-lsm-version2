"""Pruebas del modo automatico (letras estaticas + dinamicas + palabras) y del
soporte de cuerpo en DTWRecognizer. No usan camara: alimentan
HandTrackingThread._process_auto_frame con detecciones sinteticas y un reloj
falso.

    python -m unittest tests.test_modo_automatico
"""
import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtw_recognizer import N_BODY_FEATURES, N_FEATURES, DTWRecognizer, distances_to_topk, mirror_and_swap_hands  # noqa: E402


def write_sample(path: Path, n: int, seed: int, body: bool = True) -> None:
    rng = np.random.default_rng(seed)
    payload = {"frames": rng.normal(size=(n, N_FEATURES)).tolist()}
    if body:
        payload["body_frames"] = rng.normal(size=(n, N_BODY_FEATURES)).tolist()
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestDTWConCuerpo(unittest.TestCase):
    def test_espejo_con_cuerpo(self):
        seq = np.arange(3 * (N_FEATURES + N_BODY_FEATURES), dtype=np.float64).reshape(3, -1)
        out = mirror_and_swap_hands(seq)
        b = N_FEATURES
        # slots de mano del bloque de cuerpo intercambiados, dx negados, dy y bandera iguales
        np.testing.assert_array_equal(out[:, b], -seq[:, b + 4])
        np.testing.assert_array_equal(out[:, b + 1], seq[:, b + 5])
        np.testing.assert_array_equal(out[:, b + 6], -seq[:, b + 2])
        np.testing.assert_array_equal(out[:, b + 8], seq[:, b + 8])
        np.testing.assert_array_equal(mirror_and_swap_hands(out), seq)

    def test_espejo_sin_cuerpo_igual_que_antes(self):
        seq = np.random.default_rng(0).normal(size=(4, N_FEATURES))
        np.testing.assert_array_equal(mirror_and_swap_hands(mirror_and_swap_hands(seq)), seq)

    def test_carga_con_cuerpo_y_salta_sin_cuerpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, seed in (("HOLA", 1), ("GRACIAS", 2)):
                d = Path(tmp) / label
                d.mkdir()
                write_sample(d / "muestra_1.json", 20, seed)
                write_sample(d / "muestra_2.json", 22, seed + 10)
            write_sample(Path(tmp) / "HOLA" / "muestra_3.json", 20, 99, body=False)
            with self.assertLogs("dtw_recognizer", level="WARNING"):
                rec = DTWRecognizer(data_dir=tmp, auto_save_labels=False, body_weight=4.0)
            self.assertEqual(rec.templates_count, {"GRACIAS": 2, "HOLA": 2})
            self.assertEqual(rec.n_features, N_FEATURES + N_BODY_FEATURES)
            with self.assertRaises(ValueError):
                rec.predict_topk(np.zeros((10, N_FEATURES)))   # falta el bloque de cuerpo
            self.assertEqual(len(rec.leave_one_out()), 4)
            # Sin body_weight se leen las mismas muestras (y la de sin cuerpo), solo manos
            self.assertEqual(DTWRecognizer(data_dir=tmp, auto_save_labels=False).templates_count,
                             {"GRACIAS": 2, "HOLA": 3})

    def test_peso_del_cuerpo_cambia_la_distancia(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "HOLA"
            d.mkdir()
            write_sample(d / "muestra_1.json", 15, 3)
            data = json.loads((d / "muestra_1.json").read_text())
            query = np.hstack([np.array(data["frames"]), np.array(data["body_frames"]) + 1.0])
            d1 = DTWRecognizer(data_dir=tmp, auto_save_labels=False, body_weight=1.0).compute_distances(query)
            d4 = DTWRecognizer(data_dir=tmp, auto_save_labels=False, body_weight=4.0).compute_distances(query)
            self.assertAlmostEqual(d4["HOLA"], 4 * d1["HOLA"], places=6)

    def test_carpetas_nfc_y_nfd_se_suman(self):
        with tempfile.TemporaryDirectory() as tmp:
            nfc, nfd = Path(tmp) / "Ñ", Path(tmp) / "Ñ"
            nfc.mkdir()
            try:
                nfd.mkdir()
            except FileExistsError:
                self.skipTest("este sistema de archivos trata NFC y NFD como la misma carpeta (macOS)")
            write_sample(nfc / "muestra_1.json", 10, 1)
            write_sample(nfd / "muestra_1.json", 10, 2)
            rec = DTWRecognizer(data_dir=tmp, auto_save_labels=False)
            self.assertEqual(rec.templates_count, {"Ñ": 2})

    def test_distances_to_topk(self):
        topk = distances_to_topk({"A": 3.0, "B": 1.0, "C": 2.0}, k=2)
        self.assertEqual([w for w, _ in topk], ["B", "C"])
        full = distances_to_topk({"A": 3.0, "B": 1.0, "C": 2.0}, k=3)
        self.assertAlmostEqual(sum(c for _, c in full), 1.0)


def setUpModule():
    global senas, app
    from PyQt6.QtWidgets import QApplication
    import senas
    app = QApplication.instance() or QApplication([])


def hand(x: float, y: float, handedness: str = "Left"):
    """Mano sintetica con la muneca en (x, y) (normalizados) y tamano fijo."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[:, 0], pts[:, 1] = x, y
    pts[9] = (x, y - 0.1)       # base del dedo medio: tamano de mano = 0.1 del alto
    pts[8] = (x + 0.02, y - 0.2)
    return senas.HandDetection(handedness=handedness, confidence=0.9, landmarks_2d=pts,
                               landmarks_3d=np.zeros((21, 3), dtype=np.float32))


def result(kind, topk, best=2.0, letters=(), d_word=1.0, d_letter=5.0, duration=1.5):
    """Resultado de _classify_auto_sequence armado a mano."""
    return {"kind": kind, "topk": list(topk), "best_dist": best, "d_word": d_word, "d_letter": d_letter,
            "letters": list(letters), "stats": senas.SignStats(duration_s=duration), "duration_s": duration}


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestModoAutomatico(unittest.TestCase):
    W, H = 640, 480

    def setUp(self):
        self.thread = senas.HandTrackingThread(queue.Queue(), senas.AppConfig())
        self.thread.set_auto_mode(True)
        self.assertTrue(self.thread.auto_mode)
        self.committed: list[str] = []
        original = self.thread._commit_letter

        def record(letter):
            self.committed.append(letter)
            original(letter)
        self.thread._commit_letter = record
        # Clasificador estatico falso: siempre "A" con confianza alta.
        self.thread._classifier = object()
        self.thread._classify_static = lambda h: [("A", 0.95), ("B", 0.03), ("C", 0.02)]
        self.clock = FakeClock()
        self.patch = mock.patch.object(senas.time, "perf_counter", self.clock)
        self.patch.start()
        self.dispatched: list[tuple] = []
        self.thread._classify_auto_sequence = lambda seq, letters, duration=0.0: self.dispatched.append((seq, letters))
        self.results: list[dict] = []
        self.thread.auto_result_signal.connect(lambda info: self.results.append(info))
        self.retracted: list[list] = []
        self.thread.letters_retracted_signal.connect(lambda letters: self.retracted.append(letters))

    def tearDown(self):
        self.patch.stop()

    def frame(self, hands, in_space):
        self.clock.t += 1 / 30
        return self.thread._process_auto_frame(senas.FrameDetections(hands=hands), self.W, self.H, in_space)

    def rest(self, n=20):
        for _ in range(n):
            self.frame([], False)

    def test_mano_quieta_fija_letra_y_la_pasa_con_la_actividad(self):
        for _ in range(40):
            self.frame([hand(0.5, 0.4)], True)
        self.rest()
        self.assertEqual(self.committed, ["A"])
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0][1], ["A"])

    def test_dos_manos_arriba_no_fijan_letra(self):
        for _ in range(40):
            self.frame([hand(0.4, 0.4, "Left"), hand(0.6, 0.4, "Right")], True)
        self.assertEqual(self.committed, [])

    def test_deletreo_no_agrega_letra_dinamica(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(result("letra", [("J", 0.9), ("X", 0.05), ("Z", 0.05)], letters=["A"]))
        self.frame([], False)
        self.assertEqual(self.committed, [])

    def test_palabra_reemplaza_la_letra_de_su_pausa(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("palabra", [("HOLA", 0.9), ("MAMÁ", 0.06), ("AYUDA", 0.04)], letters=["R"]))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [["R"]])
        self.assertEqual(self.committed, ["HOLA"])

    def test_letra_dinamica_clara_reemplaza_su_letra_de_partida(self):
        """Misma prioridad: la J reemplaza la I que se fijo al empezarla, si su
        DTW es claro (distancia <= DYN_REPLACE_MAX_DISTANCE)."""
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("letra", [("J", 0.9), ("X", 0.05), ("Z", 0.05)], best=0.7, letters=["I"]))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [["I"]])
        self.assertEqual(self.committed, ["J"])

    def test_letra_fija_sostenida_no_se_vuelve_dinamica(self):
        """Una I sostenida se parece a la J (distancia > 1.0): se queda la I."""
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("letra", [("J", 0.9), ("X", 0.05), ("Z", 0.05)], best=1.6, letters=["I"]))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [])
        self.assertEqual(self.committed, [])
        self.assertEqual(self.results[-1]["code"], "deletreo")

    def test_palabra_sin_cuerpo_no_reemplaza_letra_sostenida(self):
        self.thread._auto_classifying = True
        r = result("palabra", [("MAMÁ", 0.9), ("HOLA", 0.06), ("AYUDA", 0.04)], best=1.4, letters=["W"])
        r["body_frac"] = 0.0
        self.thread._auto_result_queue.put(r)
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [])
        self.assertEqual(self.committed, [])

    def test_palabra_sin_cuerpo_muy_cercana_si_reemplaza(self):
        self.thread._auto_classifying = True
        r = result("palabra", [("POR_FAVOR", 0.9), ("HOLA", 0.06), ("AYUDA", 0.04)], best=0.4, letters=["B"])
        r["body_frac"] = 0.0
        self.thread._auto_result_queue.put(r)
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [["B"]])
        self.assertEqual(self.committed, ["POR FAVOR"])

    def test_guante_desempata_la_letra_dinamica(self):
        self.thread.set_glove(object(), {"J", "Z"})
        self.thread._auto_classifying = True
        self.thread._pending_activity_glove = [("Z", 0.9), ("J", 0.1)]
        self.thread._auto_result_queue.put(result("letra", [("J", 0.45), ("Z", 0.42), ("X", 0.13)]))
        self.frame([], False)
        self.assertEqual(self.committed, ["Z"])
        self.assertEqual(self.results[-1]["glove_status"], "coinciden")

    def test_camara_y_guante_no_coinciden_no_escribe(self):
        self.thread.set_glove(object(), {"J", "Z"})
        self.thread._auto_classifying = True
        self.thread._pending_activity_glove = [("Z", 0.95)]
        self.thread._auto_result_queue.put(result("letra", [("J", 0.9), ("X", 0.05), ("K", 0.05)]))
        self.frame([], False)
        self.assertEqual(self.committed, [])
        self.assertEqual(self.results[-1]["glove_status"], "no_coinciden")

    def test_camara_no_reconoce_y_guante_seguro(self):
        self.thread.set_glove(object(), {"HOLA"})
        self.thread._auto_classifying = True
        self.thread._pending_activity_glove = [("HOLA", 0.95)]
        self.thread._auto_result_queue.put(
            result("palabra", [("AYUDA", 0.5), ("GRACIAS", 0.3), ("MAMÁ", 0.2)], best=40.0))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.committed, ["HOLA"])
        self.assertEqual(self.results[-1]["glove_status"], "guante_solo")

    def test_letra_fija_con_guante(self):
        """Cada frame: si el guante siente otra letra que la camara, no se fija."""
        self.thread.set_glove(object(), {"A", "B"})
        for _ in range(40):
            self.thread.set_glove_opinion([("B", 0.95), ("A", 0.05)])
            self.frame([hand(0.5, 0.4)], True)
        self.assertEqual(self.committed, [])
        self.rest()
        for _ in range(40):
            self.thread.set_glove_opinion([("A", 0.95), ("B", 0.05)])
            self.frame([hand(0.5, 0.4)], True)
        self.assertEqual(self.committed, ["A"])

    def test_rellena_cuerpo_faltante(self):
        seq = np.zeros((5, 135))
        seq[1, 126:135] = 1.0
        seq[3, 126:135] = 2.0
        out = senas.fill_missing_body(seq)
        np.testing.assert_array_equal(out[:, 126], [1, 1, 1, 2, 2])
        self.assertTrue((out[:, 134] > 0).all())
        np.testing.assert_array_equal(senas.fill_missing_body(np.zeros((3, 135))), np.zeros((3, 135)))

    def test_deletreo_largo_no_se_reemplaza(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("palabra", [("HOLA", 0.9), ("MAMÁ", 0.06), ("AYUDA", 0.04)], letters=["A", "N", "A"]))
        text, _ = self.frame([], False)
        app.processEvents()
        self.assertEqual(self.retracted, [])
        self.assertEqual(self.committed, [])
        self.assertIn("no agregada", text)

    def test_mano_en_movimiento_no_fija_letra_y_va_al_dtw(self):
        for i in range(40):
            self.frame([hand(0.2 + 0.015 * i, 0.4)], True)   # ~4.5 tamanos de mano por segundo
        self.rest()
        self.assertEqual(self.committed, [])
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(np.asarray(self.dispatched[0][0]).shape[1], N_FEATURES + N_BODY_FEATURES)

    def test_mano_quieta_en_reposo_no_fija_letra(self):
        for _ in range(40):
            self.frame([hand(0.5, 0.9)], False)
        self.assertEqual(self.committed, [])

    def test_palabra_se_escribe_y_se_cierra(self):
        spaces = []
        self.thread.space_committed_signal.connect(lambda: spaces.append(1))
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("palabra", [("POR_FAVOR", 0.9), ("HOLA", 0.06), ("MAMÁ", 0.04)]))
        text, _ = self.frame([], False)
        app.processEvents()
        self.assertEqual(self.committed, ["POR FAVOR"])
        self.assertEqual(spaces, [1])
        self.assertIn("POR FAVOR", text)

    def test_palabra_dudosa_no_se_escribe(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("palabra", [("HOLA", 0.5), ("MAMÁ", 0.45), ("AYUDA", 0.05)]))
        text, _ = self.frame([], False)
        app.processEvents()
        self.assertEqual(self.committed, [])
        self.assertIn("no agregada", text)
        self.assertEqual(self.results[-1]["code"], "ambigua")

    def test_palabra_lejana_es_desconocida(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(
            result("palabra", [("HOLA", 0.9), ("MAMÁ", 0.05), ("AYUDA", 0.05)], best=senas.WORD_MAX_DISTANCE + 5))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(self.results[-1]["code"], "desconocida")

    def test_fases(self):
        phases = []
        self.thread.phase_signal.connect(phases.append)
        for i in range(20):
            self.frame([hand(0.2 + 0.015 * i, 0.4)], True)
        self.rest()
        app.processEvents()
        self.assertEqual(phases, ["seña", "clasificando"])
        # al llegar el resultado vuelve a reposo
        self.thread._auto_result_queue.put(result("letra", [("J", 0.8), ("X", 0.1), ("Z", 0.1)], best=1.0))
        self.frame([], False)
        app.processEvents()
        self.assertEqual(phases[-1], "reposo")

    def test_letra_dinamica_usa_las_reglas_del_modo_dinamico(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(result("letra", [("J", 0.8), ("X", 0.1), ("Z", 0.1)], best=1.0))
        self.frame([], False)
        self.assertEqual(self.committed, ["J"])

    def test_error_de_clasificacion_no_bloquea(self):
        self.thread._auto_classifying = True
        self.thread._auto_result_queue.put(None)
        text, _ = self.frame([], False)
        self.assertFalse(self.thread._auto_classifying)
        self.assertIn("Error", text)


class TestDecisionPalabra(unittest.TestCase):
    def test_reglas(self):
        ok, margin, _ = senas.word_commit_decision([("HOLA", 0.8), ("MAMÁ", 0.1)], 3.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(margin, 0.7)
        self.assertFalse(senas.word_commit_decision([("HOLA", 0.5), ("MAMÁ", 0.4)], 3.0)[0])
        self.assertFalse(senas.word_commit_decision([("HOLA", 0.9), ("MAMÁ", 0.1)],
                                                    senas.WORD_MAX_DISTANCE + 1)[0])
        self.assertEqual(senas.word_display("POR_FAVOR"), "POR FAVOR")


if __name__ == "__main__":
    unittest.main()
