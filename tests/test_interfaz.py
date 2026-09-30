"""Pruebas de la interfaz: textos de retroalimentacion (funciones puras de
interfaz_lsm.py) y el flujo de la ventana al recibir resultados del hilo.

    python -m unittest tests.test_interfaz
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

import interfaz_lsm as ui  # noqa: E402
import senas  # noqa: E402

# Perfiles como los que salen de las plantillas (ver senas.word_profiles).
PROFILES = {
    "HOLA": senas.SignStats(wrist_dx=0.58, wrist_dy=-0.39, tip_mouth=0.66, two_hands=0.0, duration_s=1.8, has_body=True),
    "MAMÁ": senas.SignStats(wrist_dx=0.39, wrist_dy=0.05, tip_mouth=0.22, two_hands=0.0, duration_s=1.9, has_body=True),
    "AYUDA": senas.SignStats(wrist_dx=0.15, wrist_dy=0.65, tip_mouth=0.96, two_hands=0.6, duration_s=2.1, has_body=True),
}
EN_EL_PECHO = senas.SignStats(wrist_dy=0.6, tip_mouth=0.9, two_hands=0.0, duration_s=1.8, has_body=True)


class TestTextos(unittest.TestCase):
    def test_zonas(self):
        self.assertEqual(ui.zone_of(PROFILES["HOLA"]), "a la altura de la cabeza")
        self.assertEqual(ui.zone_of(PROFILES["MAMÁ"]), "junto a la boca")
        self.assertEqual(ui.zone_of(PROFILES["AYUDA"]), "frente al pecho")
        self.assertEqual(ui.zone_of(senas.SignStats()), "")          # sin cuerpo: no se inventa zona

    def test_comparacion(self):
        tips = ui.compare_to("HOLA", PROFILES["HOLA"], EN_EL_PECHO)
        self.assertIn("Tu mano quedó frente al pecho; HOLA se hace a la altura de la cabeza.", tips)
        self.assertIn("AYUDA se hace con las dos manos.", ui.compare_to("AYUDA", PROFILES["AYUDA"], EN_EL_PECHO))
        rapido = senas.SignStats(wrist_dy=-0.4, tip_mouth=0.7, duration_s=0.4, has_body=True)
        self.assertTrue(any("rápido" in t for t in ui.compare_to("HOLA", PROFILES["HOLA"], rapido)))

    def test_resultados(self):
        ok = ui.feedback_for_result({"kind": "palabra", "code": "ok", "label": "HOLA",
                                     "topk": [("HOLA", 0.95), ("MAMÁ", 0.05)], "stats": PROFILES["HOLA"]}, PROFILES)
        self.assertEqual((ok.level, ok.title, ok.body), ("ok", "HOLA", ""))
        justa = ui.feedback_for_result({"kind": "palabra", "code": "ok", "label": "HOLA",
                                        "topk": [("HOLA", 0.6), ("MAMÁ", 0.4)], "stats": EN_EL_PECHO}, PROFILES)
        self.assertIn("MAMÁ", justa.body)
        amb = ui.feedback_for_result({"kind": "palabra", "code": "ambigua", "label": "HOLA",
                                      "topk": [("HOLA", 0.52), ("MAMÁ", 0.44)], "stats": EN_EL_PECHO}, PROFILES)
        self.assertEqual((amb.level, amb.title), ("warn", "¿HOLA o MAMÁ?"))
        self.assertIn("a la altura de la cabeza", amb.body)
        desc = ui.feedback_for_result({"kind": "palabra", "code": "desconocida", "label": "HOLA",
                                       "topk": [("HOLA", 0.9)], "stats": EN_EL_PECHO, "too_long": True}, PROFILES)
        self.assertEqual(desc.title, "No reconocí la seña")
        self.assertIn("Baja las manos", desc.body)
        self.assertEqual(ui.feedback_for_result({"code": "corta"}, PROFILES).title, "Seña muy corta")
        self.assertIsNone(ui.feedback_for_result({"kind": "letra", "code": "deletreo"}, PROFILES))
        letra = ui.feedback_for_result({"kind": "letra", "code": "letra_dudosa", "topk": [("J", 0.4)]}, PROFILES)
        self.assertEqual(letra.title, "¿J?")

    def test_consejos_en_vivo(self):
        base = {"hands": 1, "raised": 1, "body_tracking": True, "body_visible": True}
        self.assertIsNone(ui.guidance_feedback(base, {}))
        self.assertEqual(ui.guidance_feedback(dict(base, too_close=True), {}).title, "Mano muy cerca de la cámara")
        sin_cuerpo = dict(base, body_visible=False)
        self.assertIsNone(ui.guidance_feedback(sin_cuerpo, {"sin_cuerpo": 0.5}))     # todavia no: 2 s
        self.assertEqual(ui.guidance_feedback(sin_cuerpo, {"sin_cuerpo": 2.5}).title, "No veo tus hombros")
        duda = dict(base, static_unsure=[("B", 0.4), ("P", 0.35)])
        self.assertEqual(ui.guidance_feedback(duda, {"duda": 2.0}).title, "¿B o P?")

    def test_texto_con_pendientes(self):
        html = ui.sentence_html(["HOLA"], "AN", pending=1)
        self.assertIn("HOLA", html)
        self.assertIn(">A<", html)
        self.assertIn("underline'>N<", html)
        self.assertIn("aparecerá aquí", ui.sentence_html([], "", 0))

    def test_rgba(self):
        self.assertEqual(ui.rgba("#22c55e", 0.1), "rgba(34, 197, 94, 0.10)")


class TestVentana(unittest.TestCase):
    def setUp(self):
        self.win = senas.SignLanguageApp(senas.AppConfig(), Path(tempfile.mkdtemp()) / "config.json")

    def tearDown(self):
        self.win.close()

    def test_letra_retenida_no_se_ve_si_la_reemplaza_una_palabra(self):
        w = self.win
        w._on_phase("seña")
        w.on_letter_committed("R")
        self.assertEqual((w.current_word, w._held), ("", ["R"]))   # solo en la tarjeta
        self.assertEqual(w.sign_label.text(), "R")
        w.on_letters_retracted(["R"])
        w.on_letter_committed("HOLA")
        w.on_space_committed()
        w._on_phase("reposo")
        self.assertEqual(w.history, ["HOLA"])
        self.assertEqual((w.current_word, w._pending), ("", 0))

    def test_deletreo_queda_fijo_al_bajar_las_manos(self):
        w = self.win
        w._on_phase("seña")
        w.on_letter_committed("A")
        self.assertEqual(w.current_word, "")                # la primera se retiene
        w.on_letter_committed("N")                          # la segunda confirma el deletreo
        w.on_letter_committed("A")
        self.assertEqual((w.current_word, w._pending), ("ANA", 3))
        w._on_phase("reposo")
        self.assertEqual((w.current_word, w._pending), ("ANA", 0))

    def test_letra_sola_se_escribe_al_bajar_la_mano(self):
        w = self.win
        w._on_phase("seña")
        w.on_letter_committed("B")
        self.assertEqual(w.current_word, "")
        w._on_phase("reposo")
        self.assertEqual(w.current_word, "B")

    def test_retroceso_borra_la_letra_retenida(self):
        w = self.win
        w._on_phase("seña")
        w.on_letter_committed("B")
        w.delete_last_letter()
        w._on_phase("reposo")
        self.assertEqual(w.current_word, "")

    def test_resultado_muestra_retroalimentacion(self):
        w = self.win
        w._on_auto_result({"kind": "palabra", "code": "ambigua", "label": "HOLA",
                           "topk": [("HOLA", 0.52), ("MAMÁ", 0.44)], "stats": EN_EL_PECHO})
        self.assertEqual(w.sign_label.text(), "¿HOLA?")
        self.assertEqual(len(w.feedback._items), 1)
        w._on_auto_result({"kind": "palabra", "code": "ok", "label": "GRACIAS",
                           "topk": [("GRACIAS", 0.9), ("AYUDA", 0.1)], "stats": EN_EL_PECHO})
        self.assertEqual(w.sign_label.text(), "GRACIAS")
        self.assertEqual(len(w.feedback._items), 2)

    def test_limpiar_y_sin_camara(self):
        w = self.win
        w.history = ["HOLA"]
        w.current_word = "A"
        w.clear_all()
        self.assertEqual((w.history, w.current_word), ([], ""))
        self.assertEqual(w.start_button.text().strip(), "▶  Iniciar".strip())


class TestMenuYManual(unittest.TestCase):
    def setUp(self):
        self.win = senas.SignLanguageApp(senas.AppConfig(), Path(tempfile.mkdtemp()) / "config.json")
        self.started = []
        self.win.start_system = lambda: self.started.append(1)     # sin camara en las pruebas

    def tearDown(self):
        self.win.close()

    def test_menu_manual_traductor(self):
        w = self.win
        self.assertIs(w.stack.currentWidget(), w.start_page)
        w.start_page.start_requested.emit()          # "Iniciar programa" muestra primero el manual
        self.assertIs(w.stack.currentWidget(), w._manual_page)
        self.assertEqual(self.started, [])
        w._manual_page.continue_requested.emit()     # "Continuar al traductor" enciende la camara
        self.assertIs(w.stack.currentWidget(), w.app_page)
        self.assertEqual(self.started, [1])

    def test_manual_con_el_traductor_corriendo(self):
        w = self.win
        w.enter_translator()
        w.open_manual()
        self.assertTrue(w._manual_window.isVisible())
        self.assertFalse(w._manual_window.isModal())
        self.assertIs(w.stack.currentWidget(), w.app_page)   # el traductor sigue en pantalla

    def test_manual_trae_todo_el_abecedario_y_las_palabras(self):
        manual = ui.ManualWidget(senas.MANUAL_DIR, {"HOLA": "a la altura de la cabeza"})
        letras = manual.tabs.widget(0).widget()
        textos = [lbl.text() for lbl in letras.findChildren(ui.QLabel)]
        for letra in ui.ALPHABET:
            self.assertIn(letra, textos)
        self.assertEqual(textos.count("Ilustración pendiente"),
                         sum(1 for l in ui.ALPHABET if not (senas.MANUAL_DIR / "letras" / f"{l}.png").exists()))
        self.assertEqual(len(manual._sprites.get(0, [])), 6)       # J K Ñ Q X Z animadas
        self.assertEqual(len(manual._sprites.get(1, [])), 5)       # 5 palabras animadas

    def test_animacion_solo_con_el_manual_visible(self):
        manual = ui.ManualWidget(senas.MANUAL_DIR, {})
        self.assertFalse(manual._timer.isActive())
        manual.show()
        self.assertTrue(manual._timer.isActive())
        manual.hide()
        self.assertFalse(manual._timer.isActive())


if __name__ == "__main__":
    unittest.main()
