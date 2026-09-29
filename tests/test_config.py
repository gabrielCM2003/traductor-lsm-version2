"""Tests de validación de AppConfig (carga de config.json y CLI).

Importan señas.py, así que necesitan PyQt6, OpenCV y MediaPipe instalados;
si faltan, se omiten.
"""
import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))


def _load_app_module():
    spec = importlib.util.spec_from_file_location("senas", APP_DIR / "señas.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["senas"] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        del sys.modules["senas"]
        return None
    return module


senas = _load_app_module()


@unittest.skipIf(senas is None, "faltan dependencias de la app (PyQt6/cv2/mediapipe)")
class AppConfigTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.WARNING)
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "config.json"

    def tearDown(self):
        logging.disable(logging.NOTSET)
        self._tmp.cleanup()

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return senas.AppConfig.load(self.path)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(senas.AppConfig.load(self.path), senas.AppConfig())

    def test_valid_values_are_loaded(self):
        cfg = self.load({"stable_frames_to_commit": 20, "min_detection_confidence": 0.48,
                         "draw_landmarks": False, "dominant_hand": "Left"})
        self.assertEqual(cfg.stable_frames_to_commit, 20)
        self.assertAlmostEqual(cfg.min_detection_confidence, 0.48)
        self.assertFalse(cfg.draw_landmarks)
        self.assertEqual(cfg.dominant_hand, "Left")

    def test_wrong_types_keep_defaults(self):
        cfg = self.load({"stable_frames_to_commit": "20", "draw_landmarks": "false",
                         "min_letter_margin": None, "camera_index": True})
        default = senas.AppConfig()
        self.assertEqual(cfg.stable_frames_to_commit, default.stable_frames_to_commit)
        self.assertEqual(cfg.draw_landmarks, default.draw_landmarks)
        self.assertEqual(cfg.min_letter_margin, default.min_letter_margin)
        self.assertEqual(cfg.camera_index, default.camera_index)

    def test_out_of_range_is_clamped(self):
        cfg = self.load({"queue_maxsize": 0, "stable_frames_to_commit": 99,
                         "min_detection_confidence": 1.5})
        self.assertEqual(cfg.queue_maxsize, 1)
        self.assertEqual(cfg.stable_frames_to_commit, 25)
        self.assertAlmostEqual(cfg.min_detection_confidence, 0.95)

    def test_integral_float_accepted_for_int(self):
        self.assertEqual(self.load({"camera_index": 2.0}).camera_index, 2)
        self.assertEqual(self.load({"camera_index": 2.5}).camera_index, 0)

    def test_int_accepted_for_float(self):
        cfg = self.load({"watchdog_timeout_s": 10})
        self.assertIsInstance(cfg.watchdog_timeout_s, float)
        self.assertEqual(cfg.watchdog_timeout_s, 10.0)

    def test_choices(self):
        self.assertEqual(self.load({"dominant_hand": "left"}).dominant_hand, "Left")
        self.assertEqual(self.load({"dominant_hand": "both"}).dominant_hand, "Right")

    def test_unknown_and_removed_keys_are_ignored(self):
        cfg = self.load({"model_complexity": 1, "keypoint_buffer_size": 30, "foo": 1})
        self.assertEqual(cfg, senas.AppConfig())

    def test_non_object_json_gives_defaults(self):
        self.assertEqual(self.load([1, 2, 3]), senas.AppConfig())

    def test_invalid_json_gives_defaults(self):
        self.path.write_text("{no es json", encoding="utf-8")
        self.assertEqual(senas.AppConfig.load(self.path), senas.AppConfig())

    def test_save_roundtrip(self):
        cfg = senas.AppConfig(min_letter_confidence=0.7, dominant_hand="Left", speak_words=True)
        cfg.save(self.path)
        self.assertEqual(senas.AppConfig.load(self.path), cfg)


if __name__ == "__main__":
    unittest.main()
