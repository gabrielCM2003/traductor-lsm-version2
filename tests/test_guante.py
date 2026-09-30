"""Pruebas del guante (guante.py): paquetes de la ESP32, clasificador contra
el dataset, modo automatico, captura con cuenta atras y la conexion UDP
(contra una ESP32 falsa en localhost). No necesitan el guante.

    python -m unittest tests.test_guante
"""
import json
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guante  # noqa: E402
from guante import (  # noqa: E402
    DEFAULT_DATASET, MIN_FRAMES, N_VALUES, SENSOR_NAMES, GloveClassifier, GloveReceiver, GloveResult,
    GloveSession, GloveSpotter, clean_label, load_dataset, parse_packet,
)


def esp_packet(vec, err=0) -> bytes:
    v = np.asarray(vec, dtype=float).reshape(6, 8)
    d = {name: row.tolist() for name, row in zip(SENSOR_NAMES, v)}
    d["err"] = err
    return json.dumps(d).encode()


class TestPaquetes(unittest.TestCase):
    def test_etiqueta_con_flecha(self):
        self.assertEqual(clean_label("\x1b[DL"), "L")
        self.assertEqual(clean_label(" por favor "), "POR_FAVOR")
        self.assertEqual(clean_label("a\x1b[C"), "A")

    def test_paquete_valido(self):
        vec = np.arange(N_VALUES, dtype=float)
        self.assertEqual(parse_packet(esp_packet(vec)), vec.tolist())

    def test_paquetes_invalidos(self):
        vec = np.zeros(N_VALUES)
        self.assertIsNone(parse_packet(esp_packet(vec, err=1)))
        self.assertIsNone(parse_packet(b"{no es json"))
        self.assertIsNone(parse_packet(b'{"pulgar": [1, 2]}'))
        self.assertIsNone(parse_packet(b"[1, 2, 3]"))
        d = json.loads(esp_packet(vec))
        d["mano"] = [1] * 7
        self.assertIsNone(parse_packet(json.dumps(d).encode()))


@unittest.skipUnless(DEFAULT_DATASET.exists(), "falta datos_guante/dataset_guante.jsonl")
class TestClasificador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.labels, _ = load_dataset(DEFAULT_DATASET)
        cls.clf = GloveClassifier(cls.frames, cls.labels)

    def test_dataset_limpio(self):
        self.assertTrue(all(label == clean_label(label) for label in self.labels))

    def test_dejando_una_fuera(self):
        ok, total = self.clf.leave_one_out()
        self.assertGreaterEqual(ok / total, 0.9)

    def test_reconoce_cada_muestra(self):
        for frames, label in zip(self.frames, self.labels):
            res = self.clf.classify(frames)
            self.assertTrue(res.accepted, res.reason)
            self.assertEqual(res.label, label)
            self.assertAlmostEqual(sum(p for _, p in res.topk), 1.0, delta=0.05)

    def test_rechaza_lo_que_no_se_parece(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, (42, N_VALUES)) * np.tile([1, 1, 1, 200, 200, 200, 90, 90], 6)
        res = self.clf.classify(noise)
        self.assertFalse(res.accepted)

    def test_pocas_lecturas(self):
        res = self.clf.classify(self.frames[0][: MIN_FRAMES - 1])
        self.assertFalse(res.accepted)
        self.assertEqual(res.topk, [])


def result(label, accepted=True):
    return GloveResult([(label, 0.9)] if label else [], 0.1, 1.0, accepted)


class TestModoAutomatico(unittest.TestCase):
    def test_escribe_al_sostener(self):
        s = GloveSpotter(stable_ticks=3, release_ticks=2)
        self.assertEqual([s.update(result("A")) for _ in range(3)], [None, None, "A"])

    def test_no_repite_sin_soltar(self):
        s = GloveSpotter(stable_ticks=2, release_ticks=2)
        out = [s.update(result("A")) for _ in range(8)]
        self.assertEqual(out.count("A"), 1)
        # soltar (2 evaluaciones sin A) y volver a A: se escribe otra vez (LL)
        out = [s.update(None), s.update(None)] + [s.update(result("A")) for _ in range(2)]
        self.assertEqual(out, [None, None, None, "A"])

    def test_cambio_de_sena(self):
        s = GloveSpotter(stable_ticks=2, release_ticks=2)
        out = [s.update(result(x)) for x in "AABB"]
        self.assertEqual(out, [None, "A", None, "B"])

    def test_rechazadas_no_cuentan(self):
        s = GloveSpotter(stable_ticks=2)
        self.assertEqual([s.update(result("A", accepted=False)) for _ in range(5)], [None] * 5)

    def test_un_parpadeo_no_escribe(self):
        s = GloveSpotter(stable_ticks=3)
        out = [s.update(result(x)) for x in "AABAAB"]
        self.assertEqual(out, [None] * 6)


@unittest.skipUnless(DEFAULT_DATASET.exists(), "falta datos_guante/dataset_guante.jsonl")
class TestSesion(unittest.TestCase):
    def setUp(self):
        frames, labels, _ = load_dataset(DEFAULT_DATASET)
        self.clf = GloveClassifier(frames, labels)
        self.sample = {l: f for f, l in zip(frames, labels)}
        self.rec = GloveReceiver()      # sin start(): se alimenta con push()

    def feed(self, frames, t0, hz=21.0):
        for i, v in enumerate(frames):
            self.rec.push(list(v), now=t0 + i / hz)
        return t0 + len(frames) / hz

    def test_automatico_escribe_una_vez(self):
        session = GloveSession(self.rec, self.clf, auto=True)
        t = 1000.0
        commits = []
        for _ in range(4):
            t = self.feed(self.sample["B"], t)
            for tick in np.arange(t - 2.0, t, 0.25):
                for ev in session.tick(tick):
                    if ev.kind == "resultado":
                        commits.append(ev.data["commit"])
        self.assertEqual(commits, ["B"])

    def test_captura_con_cuenta_atras(self):
        session = GloveSession(self.rec, self.clf, auto=False)
        t0 = 2000.0
        self.feed(self.sample["C"][:5], t0)
        session.request_capture(now=t0)
        kinds = [ev.kind for ev in session.tick(t0 + 0.1)]
        self.assertIn("cuenta", kinds)
        start = t0 + GloveSession.COUNTDOWN_S
        self.feed(self.sample["C"], start + 0.01)
        kinds = [ev.kind for ev in session.tick(start + 0.5)]
        self.assertIn("capturando", kinds)
        events = session.tick(start + guante.WINDOW_S + 0.05)
        res = [ev for ev in events if ev.kind == "resultado"]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].data["commit"], "C")
        self.assertTrue(res[0].data["manual"])
        self.assertFalse(session.busy)

    def test_sin_datos_no_escribe(self):
        session = GloveSession(self.rec, self.clf, auto=True)
        events = session.tick(5000.0)
        self.assertEqual([ev.kind for ev in events], ["estado"])
        self.assertFalse(events[0].data["connected"])


class TestPalabrasGuante(unittest.TestCase):
    def test_palabra_necesita_menos_evaluaciones(self):
        s = GloveSpotter(stable_ticks=4, stable_ticks_word=2)
        self.assertEqual([s.update(result("HOLA")) for _ in range(2)], [None, "HOLA"])

    def test_etiqueta_sin_acentos(self):
        self.assertEqual(guante.plain_label("MAMÁ"), "MAMA")
        self.assertEqual(clean_label("mamá"), "MAMÁ")


class TestHolaMama(unittest.TestCase):
    """Desempate HOLA / MAMA por la punta del indice respecto a la boca."""

    @classmethod
    def setUpClass(cls):
        import senas
        cls.senas = senas

    def stats(self, near):
        return self.senas.SignStats(has_body=True, near_mouth=near)

    def test_cerca_de_la_boca_es_mama(self):
        topk, note = self.senas.hola_mama_rule([("HOLA", 0.52), ("MAMÁ", 0.46), ("AYUDA", 0.02)], self.stats(0.9))
        self.assertEqual(topk[0][0], "MAMÁ")
        self.assertGreaterEqual(topk[0][1] - topk[1][1], self.senas.WORD_MIN_MARGIN)
        self.assertEqual(topk[2], ("AYUDA", 0.02))
        self.assertTrue(note)

    def test_lejos_de_la_boca_es_hola(self):
        topk, _ = self.senas.hola_mama_rule([("MAMÁ", 0.55), ("HOLA", 0.43)], self.stats(0.05))
        self.assertEqual(topk[0][0], "HOLA")

    def test_en_medio_no_decide(self):
        orig = [("MAMÁ", 0.55), ("HOLA", 0.43)]
        self.assertEqual(self.senas.hola_mama_rule(orig, self.stats(0.45)), (orig, ""))

    def test_sin_cuerpo_u_otras_palabras_no_toca(self):
        orig = [("HOLA", 0.5), ("MAMÁ", 0.45)]
        self.assertEqual(self.senas.hola_mama_rule(orig, self.senas.SignStats(near_mouth=0.9))[0], orig)
        other = [("AYUDA", 0.5), ("GRACIAS", 0.45)]
        self.assertEqual(self.senas.hola_mama_rule(other, self.stats(0.9))[0], other)

    def test_plantillas(self):
        """La regla elige bien en las 24 plantillas de HOLA y MAMÁ, sin importar
        cual de las dos quedo primera en el DTW."""
        root = Path(__file__).resolve().parent.parent / "datos_palabras_dinamicas"
        n = 0
        for word in ("HOLA", "MAMÁ"):
            for f in sorted((root / word).glob("*.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                seq = np.hstack([np.array(d["frames"]), np.array(d["body_frames"])])
                st = self.senas.sign_stats(seq, 2.0)
                for first, second in (("HOLA", "MAMÁ"), ("MAMÁ", "HOLA")):
                    topk, _ = self.senas.hola_mama_rule([(first, 0.52), (second, 0.46)], st)
                    self.assertEqual(topk[0][0], word, f.name)
                    n += 1
        self.assertGreaterEqual(n, 40)


class TestManoConGuante(unittest.TestCase):
    """Aclarado de la mano con guante (senas.brighten_regions) y dibujo claro."""

    @classmethod
    def setUpClass(cls):
        import senas
        cls.senas = senas

    def test_aclara_lo_oscuro_y_no_el_fondo(self):
        frame = np.full((200, 200, 3), 200, dtype=np.uint8)     # fondo claro
        frame[80:120, 80:120] = 30                               # guante oscuro
        out = self.senas.brighten_regions(frame, [(40, 40, 160, 160)])
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(out[100, 100].mean(), 90)             # el guante se aclara
        self.assertLess(abs(int(out[45, 100].mean()) - 200), 10)  # el fondo casi igual
        np.testing.assert_array_equal(out[0:30, 0:30], frame[0:30, 0:30])   # fuera de la zona, igual
        self.assertEqual(frame[100, 100, 0], 30)                 # no toca el original

    def test_sin_zona_no_cambia_la_pantalla(self):
        """Sin mano no se aclara nada (antes la pantalla se ponia blanca)."""
        frame = np.random.default_rng(0).integers(0, 80, (60, 80, 3), dtype=np.uint8)
        np.testing.assert_array_equal(self.senas.brighten_regions(frame, []), frame)

    def test_flechas_de_los_sensores(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        reading = np.zeros(N_VALUES)
        self.senas.draw_glove_vectors(img, reading)                 # recuadro (sin mano)
        self.assertGreater(img[120:, :240].sum(), 0)
        img2 = np.zeros((240, 320, 3), dtype=np.uint8)
        pts = np.column_stack([np.linspace(100, 200, 21), np.linspace(80, 180, 21)])
        self.senas.draw_glove_vectors(img2, reading, pts, 50.0)     # sobre la mano
        self.assertGreater(img2[40:200, 80:220].sum(), 0)


class TestCamaraMasGuante(unittest.TestCase):
    """fuse_topk: la camara y el guante tienen que coincidir."""
    V = {"A", "B", "C", "L", "Y"}

    @classmethod
    def setUpClass(cls):
        import senas
        cls.senas = senas

    def fuse(self, cam, glove):
        return self.senas.fuse_topk(cam, glove, self.V)

    def test_coinciden_sube_la_confianza(self):
        topk, st = self.fuse([("B", 0.55), ("P", 0.35), ("D", 0.1)], [("B", 0.95)])
        self.assertEqual((topk[0][0], st), ("B", "coinciden"))
        self.assertGreater(topk[0][1], 0.55)

    def test_el_guante_desempata(self):
        topk, st = self.fuse([("A", 0.5), ("B", 0.4), ("D", 0.1)], [("B", 0.9), ("A", 0.08)])
        self.assertEqual((topk[0][0], st), ("B", "coinciden"))

    def test_no_coinciden_no_se_escribe(self):
        topk, st = self.fuse([("B", 0.55), ("P", 0.35), ("D", 0.1)], [("C", 0.95)])
        self.assertEqual(st, "no_coinciden")
        self.assertAlmostEqual(topk[0][1], topk[1][1])       # empate: ninguna regla lo acepta
        self.assertNotEqual(topk[0][0], "P")                  # nunca gana una tercera

    def test_guante_no_conoce_la_sena(self):
        cam = [("D", 0.6), ("R", 0.3), ("U", 0.1)]
        self.assertEqual(self.fuse(cam, [("L", 0.9)]), (cam, "solo_camara"))
        words = [("HOLA", 0.5), ("MAMÁ", 0.45)]
        self.assertEqual(self.fuse(words, [("B", 0.9)]), (words, "solo_camara"))

    def test_sin_guante(self):
        cam = [("A", 0.9)]
        self.assertEqual(self.fuse(cam, None), (cam, "solo_camara"))

    def test_zona_de_la_mano(self):
        pts = np.array([[0.4, 0.4], [0.6, 0.6]] + [[0.5, 0.5]] * 19)
        hand = self.senas.HandDetection("Left", 0.9, pts, None)
        x0, y0, x1, y1 = self.senas.hand_box(hand, 100, 100)
        self.assertTrue(x0 < 40 and y0 < 40 and x1 > 60 and y1 > 60)
        self.assertTrue(0 <= x0 and 0 <= y0 and x1 <= 100 and y1 <= 100)

    def test_dibujo_claro(self):
        pts = np.column_stack([np.linspace(0.2, 0.8, 21), np.linspace(0.2, 0.8, 21)])
        hand = self.senas.HandDetection("Left", 0.9, pts, None)
        normal = np.zeros((100, 100, 3), dtype=np.uint8)
        light = normal.copy()
        self.senas.draw_hand_landmarks(normal, hand)
        self.senas.draw_hand_landmarks(light, hand, light=True)
        self.assertGreater(light.max(axis=2).mean(), 0)
        c = self.senas.GLOVE_FINGER_COLORS["thumb"]
        self.assertTrue(min(c) > min(self.senas.FINGER_COLORS["thumb"]))


class TestUDP(unittest.TestCase):
    """GloveReceiver contra una ESP32 falsa en localhost: manda "hola" y
    recibe los paquetes que la ESP le regresa."""

    def test_hola_y_datos(self):
        esp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        esp.bind(("127.0.0.1", 0))
        esp.settimeout(2.0)
        port = esp.getsockname()[1]
        vec = np.linspace(-1, 1, N_VALUES)
        stop = threading.Event()

        def serve():
            try:
                msg, addr = esp.recvfrom(64)
            except socket.timeout:
                return
            self.assertEqual(msg, b"hola")
            esp.sendto(b"basura", addr)
            while not stop.is_set():
                esp.sendto(esp_packet(vec), addr)
                time.sleep(0.02)

        th = threading.Thread(target=serve, daemon=True)
        th.start()
        rec = GloveReceiver("127.0.0.1", port)
        rec.start()
        try:
            deadline = time.time() + 3.0
            while not rec.connected() and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
            self.assertTrue(rec.connected())
            frames = rec.window(1.0)
            self.assertGreater(len(frames), 5)
            np.testing.assert_allclose(frames[0], vec)
            self.assertGreaterEqual(rec.bad_packets, 1)
        finally:
            stop.set()
            rec.stop()
            th.join(1.0)
            esp.close()


if __name__ == "__main__":
    unittest.main()
