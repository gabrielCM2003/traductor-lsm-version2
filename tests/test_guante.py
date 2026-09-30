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
    DEFAULT_DATASET, GLOVE_BOTH, MIN_FRAMES, N_VALUES, SENSOR_NAMES, GloveBoth, GloveClassifier, GloveReceiver,
    GloveResult, GloveSession, GloveSpotter, clean_label, decode_packet, default_dataset, load_dataset,
    pair_frames, parse_glove, parse_hand, parse_packet, sample_seconds,
)


def esp_packet(vec, err=0, hand="D") -> bytes:
    v = np.asarray(vec, dtype=float).reshape(6, 8)
    d = {"n": 1, "t": 40, "h": hand, "err": err}
    d.update({name: row.tolist() for name, row in zip(SENSOR_NAMES, v)})
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
        # parse_packet es de una sola mano (la derecha por defecto); sin "h"
        # tampoco se acepta.
        self.assertIsNone(parse_packet(esp_packet(vec, hand="I")))
        d = json.loads(esp_packet(vec))
        del d["h"]
        self.assertIsNone(parse_packet(json.dumps(d).encode()))
        self.assertIsNone(parse_packet(b"{no es json"))
        self.assertIsNone(parse_packet(b'{"pulgar": [1, 2]}'))
        self.assertIsNone(parse_packet(b"[1, 2, 3]"))
        d = json.loads(esp_packet(vec))
        d["mano"] = [1] * 7
        self.assertIsNone(parse_packet(json.dumps(d).encode()))

    def test_paquete_izquierdo(self):
        vec = np.arange(N_VALUES, dtype=float)
        self.assertEqual(parse_packet(esp_packet(vec, hand="I"), hand="I"), vec.tolist())
        self.assertEqual(decode_packet(esp_packet(vec, hand="I")), (vec.tolist(), "I", ""))
        self.assertEqual(decode_packet(esp_packet(vec, hand="I", err=3))[1:], ("I", "err"))
        self.assertEqual(decode_packet(esp_packet(vec, hand="X"))[1:], (None, "mano"))

    def test_nombre_de_mano(self):
        for txt in ("D", "d", "der", "derecha", "right"):
            self.assertEqual(parse_hand(txt), "D")
        for txt in ("I", "izq", "Izquierda", "left", "L"):
            self.assertEqual(parse_hand(txt), "I")
        with self.assertRaises(ValueError):
            parse_hand("ambas")
        self.assertEqual(default_dataset("D"), DEFAULT_DATASET)
        self.assertEqual(default_dataset("I").name, "dataset_guante_izquierdo.jsonl")


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

    def test_guante_izquierdo_con_su_dataset(self):
        """Dos sesiones sobre el mismo receptor: la izquierda (hand("I"), con
        su propio dataset) solo ve las lecturas de "h": "I"."""
        frames, labels, _ = load_dataset(DEFAULT_DATASET)
        # Guante izquierdo de prueba: las mismas señas con las lecturas en espejo.
        left_clf = GloveClassifier([-f for f in frames], labels)
        right = GloveSession(self.rec, self.clf, auto=True)
        left = GloveSession(self.rec.hand("I"), left_clf, auto=True)
        t = 3000.0
        commits = {"D": [], "I": []}
        for _ in range(3):
            for i, v in enumerate(self.sample["L"]):
                self.rec.push(list(-v), now=t + i / 21.0, hand="I")
            t += len(self.sample["L"]) / 21.0
            for tick in np.arange(t - 2.0, t, 0.25):
                for hand, session in (("D", right), ("I", left)):
                    commits[hand] += [ev.data["commit"] for ev in session.tick(tick) if ev.kind == "resultado"]
        self.assertEqual(commits, {"D": [], "I": ["L"]})

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


class TestDiagnostico(unittest.TestCase):
    """Un guante que llega mal no debe parecer apagado: se acepta su "h" con
    otra forma, se muestran sus lecturas con err y se avisa si dos ESP32
    mandan la misma mano."""

    def setUp(self):
        self.rec = GloveReceiver()      # sin start(): se alimenta con handle_packet()
        self.vec = np.linspace(-1, 1, N_VALUES)

    def test_otras_formas_de_h(self):
        for h, hand in (("L", "I"), ("i", "I"), ("izq", "I"), ("R", "D"), ("d", "D")):
            self.assertEqual(decode_packet(esp_packet(self.vec, hand=h))[1:], (hand, ""), h)
        self.assertEqual(decode_packet(esp_packet(self.vec, hand="X"))[1:], (None, "mano"))

    def test_h_desconocida_se_avisa(self):
        self.rec.handle_packet(esp_packet(self.vec, hand="X"), ("10.42.0.30", 5000), now=100.0)
        self.assertFalse(self.rec.connected(100.1, "I"))
        self.assertEqual(self.rec.problems(100.1), ['10.42.0.30 manda "h":"X" (usa "D" o "I")'])
        self.assertEqual(self.rec.problems(110.0), [])

    def test_solo_err_se_muestra(self):
        for k in range(5):
            self.rec.handle_packet(esp_packet(self.vec, hand="I", err=4), ("10.42.0.31", 5000), now=100.0 + k * 0.04)
        self.assertFalse(self.rec.connected(100.3, "I"))
        err, values = self.rec.err_reading(100.3, "I")
        self.assertEqual(err, 4)
        np.testing.assert_allclose(values, self.vec)
        self.assertIn("I: solo llegan paquetes con err=4", self.rec.problems(100.3))
        # Con paquetes buenos otra vez, se reconoce y deja de avisar.
        self.rec.handle_packet(esp_packet(self.vec, hand="I"), ("10.42.0.31", 5000), now=100.4)
        self.assertTrue(self.rec.connected(100.5, "I"))
        self.assertIsNone(self.rec.err_reading(100.5, "I"))

    def test_dos_esp_con_la_misma_mano(self):
        """El firmware del izquierdo quedo con "h":"D": no se ve el izquierdo."""
        for k in range(10):
            self.rec.handle_packet(esp_packet(self.vec), ("10.42.0.30", 5000), now=100.0 + k * 0.04)
            self.rec.handle_packet(esp_packet(-self.vec), ("10.42.0.31", 5001), now=100.02 + k * 0.04)
        self.assertEqual(self.rec.connected_hands(100.5), ["D"])
        self.assertEqual(self.rec.problems(100.5), ['10.42.0.30 y 10.42.0.31 mandan los dos "h":"D"'])


class TestDosManos(unittest.TestCase):
    """Frases con los dos guantes: GloveBoth junta las lecturas de las dos
    manos (96 valores) y se reconocen contra un dataset de dos manos."""

    @classmethod
    def setUpClass(cls):
        frames, labels, _ = load_dataset(DEFAULT_DATASET)
        by_label: dict[str, list] = {}
        for f, l in zip(frames, labels):
            by_label.setdefault(l, []).append(f)
        # Frases de prueba: una postura en cada mano (el izquierdo, en espejo).
        cls.phrases = {"BUENOS_DIAS": ("A", "B"), "TE_QUIERO": ("L", "Y"), "MUCHAS_GRACIAS": ("C", "C")}
        cls.samples, cls.labels = [], []
        for phrase, (d, i) in cls.phrases.items():
            for fd, fi in zip(by_label[d], by_label[i]):
                n = min(len(fd), len(fi))
                cls.samples.append(np.hstack([fd[:n], -fi[:n]]))
                cls.labels.append(phrase)
        cls.clf = GloveClassifier(cls.samples, cls.labels)

    def test_nombres(self):
        for txt in ("DI", "ambas", "Ambos", "2"):
            self.assertEqual(parse_glove(txt), GLOVE_BOTH)
        self.assertEqual(parse_glove("izq"), "I")
        self.assertEqual(default_dataset(GLOVE_BOTH).name, "dataset_guante_ambas.jsonl")

    def test_junta_por_tiempo(self):
        t_d = np.arange(0, 1, 1 / 21)
        t_i = np.arange(0.01, 1, 1 / 25)
        f_d = np.ones((len(t_d), N_VALUES))
        f_i = 2 * np.ones((len(t_i), N_VALUES))
        both = pair_frames(t_d, f_d, t_i, f_i)
        self.assertEqual(both.shape, (len(t_d), 2 * N_VALUES))
        self.assertTrue(np.all(both[:, :N_VALUES] == 1) and np.all(both[:, N_VALUES:] == 2))
        # El izquierdo se corta a la mitad: las lecturas del derecho sin pareja se sueltan.
        both = pair_frames(t_d, f_d, t_i[t_i < 0.5], f_i[t_i < 0.5])
        self.assertLess(len(both), len(t_d))
        self.assertEqual(pair_frames(t_d, f_d, t_i[:0], f_i[:0]).shape, (0, 2 * N_VALUES))

    def test_reconoce_frases(self):
        ok, total = self.clf.leave_one_out()
        self.assertGreaterEqual(ok / total, 0.9)

    def feed(self, rec, sample, t0):
        """Cada mano a su ritmo y desfasada, como dos ESP32 sin sincronizar."""
        right, left = sample[:, :N_VALUES], sample[:, N_VALUES:]
        for k, v in enumerate(right):
            rec.push(list(v), now=t0 + k / 21.0, hand="D")
        for k, v in enumerate(left):
            rec.push(list(v), now=t0 + 0.017 + k / 21.0, hand="I")
        return t0 + len(sample) / 21.0

    def test_sesion_escribe_la_frase(self):
        rec = GloveReceiver()
        both = rec.hand(GLOVE_BOTH)
        self.assertIsInstance(both, GloveBoth)
        session = GloveSession(both, self.clf, auto=True, live_window_s=2.0)
        sample = self.samples[self.labels.index("TE_QUIERO")]
        t = 1000.0
        commits = []
        for _ in range(3):
            t = self.feed(rec, sample, t)
            for tick in np.arange(t - 2.0, t, 0.25):
                commits += [ev.data["commit"] for ev in session.tick(tick) if ev.kind == "resultado"]
        self.assertEqual(commits, ["TE_QUIERO"])

    def test_no_dos_frases_seguidas(self):
        """Tras escribir una frase, otra en menos de lo que dura una frase no
        se escribe (la ventana pasa por la frase parecida al sostenerla)."""
        rec = GloveReceiver()
        session = GloveSession(rec.hand(GLOVE_BOTH), self.clf, auto=True, live_window_s=2.0, min_gap_s=2.0)
        t = 1000.0
        commits = []
        for name in ("TE_QUIERO", "BUENOS_DIAS", "BUENOS_DIAS"):
            sample = self.samples[self.labels.index(name)]
            t = self.feed(rec, sample, t)
            for tick in (t - 0.25, t):
                commits += [(ev.data["commit"], tick) for ev in session.tick(tick) if ev.kind == "resultado"]
        self.assertEqual([c for c, _ in commits], ["TE_QUIERO", "BUENOS_DIAS"])
        self.assertGreaterEqual(commits[1][1] - commits[0][1], 2.0)

    def test_una_sola_mano_no_basta(self):
        rec = GloveReceiver()
        session = GloveSession(rec.hand(GLOVE_BOTH), self.clf, auto=True, live_window_s=2.0)
        sample = self.samples[0]
        for k, v in enumerate(sample[:, :N_VALUES]):
            rec.push(list(v), now=2000.0 + k / 21.0, hand="D")
        events = session.tick(2000.0 + len(sample) / 21.0)
        self.assertEqual([ev.kind for ev in events], ["estado"])
        self.assertFalse(events[0].data["connected"])

    def test_mano_distinta_pesa(self):
        """Con los dos guantes, cada mano tiene que parecerse: la distancia
        es la de la mano que peor coincide, no el promedio."""
        sample = self.samples[self.labels.index("TE_QUIERO")]
        wrong = sample.copy()
        wrong[:, N_VALUES:] = self.samples[self.labels.index("BUENOS_DIAS")][: len(sample), N_VALUES:]
        feat = guante.frames_to_features(wrong)
        d = self.clf._distances(feat)[self.clf.y == "TE_QUIERO"].min()
        d_mean = np.sqrt(((self.clf.X - feat) ** 2).mean(axis=1))[self.clf.y == "TE_QUIERO"].min()
        self.assertGreater(d, d_mean)

    def test_nada_no_se_escribe(self):
        """NADA (manos sin seña) se reconoce, pero nunca se acepta ni se
        escribe, y no cuenta como seña del guante."""
        rng = np.random.default_rng(1)
        nada = [rng.normal(0, 1, (42, 2 * N_VALUES)) * 0.05 + 3 for _ in range(6)]
        clf = GloveClassifier(self.samples + nada, self.labels + ["NADA"] * 6)
        self.assertNotIn("NADA", clf.sign_labels)
        res = clf.classify(nada[0])
        self.assertEqual(res.label, "NADA")
        self.assertFalse(res.accepted)
        self.assertEqual(GloveSpotter(stable_ticks_word=1).update(res), None)

    def test_dataset_de_frases(self):
        """El dataset de dos manos se lee con 96 valores y dura lo grabado."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_guante_ambas.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for sample, label in zip(self.samples, self.labels):
                    f.write(json.dumps({"persona": "prueba", "mano": "DI", "etiqueta": label, "segundos": 3.0,
                                        "frames": sample.tolist()}) + "\n")
            self.assertEqual(sample_seconds(path), 3.0)
            clf = GloveClassifier.from_file(path)
            self.assertEqual(clf.X.shape[1], 8 * N_VALUES)
            self.assertEqual(clf.window_s, 3.0)
            self.assertEqual(sorted(clf.labels), sorted(self.phrases))


class TestUDP(unittest.TestCase):
    """GloveReceiver escuchando en localhost: una ESP32 falsa le manda los
    paquetes sin handshake (como el firmware), mezclados con basura, la mano
    izquierda y paquetes con err."""

    def free_port(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_separa_las_dos_manos(self):
        """Los dos guantes mandan al mismo puerto: cada mano a su buffer; se
        descarta basura, una mano desconocida y los paquetes con err."""
        port = self.free_port()
        rec = GloveReceiver(port, bind_ip="127.0.0.1")
        rec.start()
        esp_d = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        esp_i = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        vec_d = np.linspace(-1, 1, N_VALUES)
        vec_i = np.linspace(1, -1, N_VALUES)
        try:
            for extra in (b"basura", esp_packet(vec_d, hand="X"), esp_packet(vec_d, err=2)):
                esp_d.sendto(extra, ("127.0.0.1", port))
            for _ in range(20):
                esp_d.sendto(esp_packet(vec_d), ("127.0.0.1", port))
                esp_i.sendto(esp_packet(vec_i, hand="I"), ("127.0.0.1", port))
                time.sleep(0.01)
            deadline = time.time() + 2.0
            while (len(rec.window(2.0)) < 20 or len(rec.window(2.0, hand="I")) < 20) and time.time() < deadline:
                time.sleep(0.05)
            left = rec.hand("I")
            self.assertTrue(rec.connected())
            self.assertTrue(left.connected())
            self.assertEqual(rec.connected_hands(), ["D", "I"])
            self.assertEqual(len(rec.window(2.0)), 20)
            self.assertEqual(len(left.window(2.0)), 20)
            np.testing.assert_allclose(rec.window(2.0)[0], vec_d)
            np.testing.assert_allclose(left.latest(), vec_i)
            self.assertEqual(dict(rec.dropped), {"json": 1, "mano": 1, "err": 1})
            self.assertTrue(rec.status_line().startswith("D "))
            self.assertIn("err 1", rec.status_line())
            self.assertTrue(left.status_line().startswith("I "))
            self.assertIn("err 0", left.status_line())
        finally:
            esp_d.close()
            esp_i.close()
            rec.stop()

    def test_solo_una_mano(self):
        """Un receptor de una sola mano descarta la otra."""
        port = self.free_port()
        rec = GloveReceiver(port, bind_ip="127.0.0.1", hands="D")
        rec.start()
        esp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            esp.sendto(esp_packet(np.zeros(N_VALUES), hand="I"), ("127.0.0.1", port))
            deadline = time.time() + 2.0
            while not rec.dropped and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(dict(rec.dropped), {"mano": 1})
            self.assertFalse(rec.connected())
            with self.assertRaises(ValueError):
                rec.hand("I")
        finally:
            esp.close()
            rec.stop()

    def test_puerto_ocupado_falla(self):
        """Sin SO_REUSEADDR: un segundo receptor no se abre en silencio."""
        port = self.free_port()
        rec = GloveReceiver(port, bind_ip="127.0.0.1")
        rec.start()
        try:
            with self.assertRaises(OSError) as ctx:
                GloveReceiver(port, bind_ip="127.0.0.1").start()
            self.assertIn("Address already in use", str(ctx.exception))
        finally:
            rec.stop()

    def test_guante_se_desconecta(self):
        """Sin paquetes, connected() pasa a False y el hilo sigue vivo."""
        rec = GloveReceiver(self.free_port(), bind_ip="127.0.0.1")
        rec.start()
        try:
            rec.push(np.zeros(N_VALUES).tolist(), now=time.time() - 5)
            self.assertFalse(rec.connected())
            self.assertTrue(rec._thread.is_alive())
        finally:
            rec.stop()


if __name__ == "__main__":
    unittest.main()
