"""Tests de la lógica pura de sign_classifier (no necesitan cámara ni ONNX).

Ejecutar desde la carpeta del proyecto:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sign_classifier import (  # noqa: E402
    LetterCommitter,
    PredictionSmoother,
    SignClassifier,
    hand_to_feature_vector,
    normalize_keypoints,
)


def topk(letter="A", conf=0.9, second="B", conf2=0.05):
    return [(letter, conf), (second, conf2), ("C", 0.01)]


class PredictionSmootherTests(unittest.TestCase):
    def test_requires_two_entries(self):
        with self.assertRaises(ValueError):
            PredictionSmoother().push([("A", 1.0)])

    def test_needs_window_majority_before_emitting(self):
        s = PredictionSmoother(window_size=7)
        # Mayoría de 7 = 4 votos: los 3 primeros frames no bastan.
        for _ in range(3):
            self.assertIsNone(s.push(topk()).letter)
        self.assertEqual(s.push(topk()).letter, "A")

    def test_majority_vote_with_noise(self):
        s = PredictionSmoother(window_size=7)
        for letter in "AABAACA":
            result = s.push(topk(letter))
        self.assertEqual(result.letter, "A")

    def test_low_confidence_blocks(self):
        s = PredictionSmoother(window_size=5, min_confidence=0.55)
        for _ in range(5):
            result = s.push(topk(conf=0.5, conf2=0.1))
        self.assertIsNone(result.letter)
        self.assertAlmostEqual(result.confidence, 0.5)

    def test_low_margin_blocks(self):
        s = PredictionSmoother(window_size=5, min_margin=0.15)
        for _ in range(5):
            result = s.push(topk(conf=0.6, conf2=0.5))
        self.assertIsNone(result.letter)

    def test_per_letter_confidence_overrides_global(self):
        s = PredictionSmoother(window_size=5, min_confidence=0.55,
                               per_letter_confidence={"M": 0.8})
        for _ in range(5):
            result = s.push(topk("M", conf=0.7, conf2=0.1))
        self.assertIsNone(result.letter)
        for _ in range(5):
            result = s.push(topk("A", conf=0.7, conf2=0.1))
        self.assertEqual(result.letter, "A")

    def test_reset_clears_history(self):
        s = PredictionSmoother(window_size=5)
        for _ in range(5):
            s.push(topk())
        s.reset()
        self.assertIsNone(s.push(topk()).letter)
        self.assertEqual(s.last_topk, topk())

    def test_raw_values_are_per_frame(self):
        result = PredictionSmoother().push(topk(conf=0.7, conf2=0.2))
        self.assertEqual(result.raw_top1, ("A", 0.7))
        self.assertEqual(result.raw_top2, ("B", 0.2))
        self.assertAlmostEqual(result.margin, 0.5)


class LetterCommitterTests(unittest.TestCase):
    def make(self, **kw):
        params = dict(stable_frames=3, release_frames=2, space_frames=6,
                      hand_lost_reset_frames=2)
        params.update(kw)
        return LetterCommitter(**params)

    @staticmethod
    def feed(c, letters, hand=True):
        """Pasa una secuencia ('.' = sin letra) y devuelve lo confirmado."""
        out = []
        for ch in letters:
            r = c.update(None if ch == "." else ch, hand_present=hand)
            if r is not None:
                out.append(r)
        return "".join(out)

    def test_commits_after_stable_frames(self):
        c = self.make()
        self.assertEqual(self.feed(c, "AA"), "")
        self.assertEqual(self.feed(c, "A"), "A")

    def test_holding_does_not_repeat(self):
        self.assertEqual(self.feed(self.make(), "A" * 30), "A")

    def test_double_letter_after_short_pause(self):
        # AAA -> pausa de 2 frames (release_frames) -> AAA = "AA" (LL, RR, EE...)
        self.assertEqual(self.feed(self.make(), "AAA..AAA"), "AA")

    def test_pause_shorter_than_release_does_not_repeat(self):
        c = self.make(release_frames=3)
        self.assertEqual(self.feed(c, "AAA..AAA"), "A")

    def test_brief_other_letter_does_not_allow_repeat(self):
        c = self.make(release_frames=3)
        self.assertEqual(self.feed(c, "AAABBAAA"), "A")

    def test_different_letter_commits_without_pause(self):
        self.assertEqual(self.feed(self.make(), "AAABBB"), "AB")

    def test_unstable_prediction_resets_count(self):
        self.assertEqual(self.feed(self.make(), "AA.AA.AA"), "")

    def test_space_after_hand_absent(self):
        c = self.make()
        self.feed(c, "AAA")
        self.assertEqual(self.feed(c, "....." , hand=False), "")
        self.assertEqual(self.feed(c, ".", hand=False), LetterCommitter.SPACE)
        # Solo un espacio por palabra.
        self.assertEqual(self.feed(c, "." * 20, hand=False), "")

    def test_no_space_without_letters(self):
        self.assertEqual(self.feed(self.make(), "." * 20, hand=False), "")

    def test_same_letter_allowed_after_space(self):
        c = self.make()
        self.feed(c, "AAA")
        self.feed(c, "." * 6, hand=False)
        self.assertEqual(self.feed(c, "AAA"), "A")

    def test_brief_dropout_keeps_progress(self):
        c = self.make(stable_frames=4, hand_lost_reset_frames=3)
        self.feed(c, "AA")
        self.feed(c, "..", hand=False)       # < 3 frames sin mano
        self.assertEqual(self.feed(c, "AA"), "A")

    def test_long_dropout_discards_progress(self):
        c = self.make(stable_frames=4, hand_lost_reset_frames=2)
        self.feed(c, "AAA")
        self.feed(c, "..", hand=False)       # llega al umbral: se descarta
        self.assertEqual(self.feed(c, "A"), "")
        self.assertEqual(self.feed(c, "AAA"), "A")

    def test_hand_just_lost_fires_once(self):
        c = self.make(hand_lost_reset_frames=2)
        flags = []
        for _ in range(5):
            c.update(None, hand_present=False)
            flags.append(c.hand_just_lost)
        self.assertEqual(flags, [False, True, False, False, False])

    def test_reset_keeps_space_if_word_has_letters(self):
        c = self.make()
        self.feed(c, "AAA")
        c.reset(has_letters=True)            # p. ej. tras borrar una letra
        self.assertEqual(self.feed(c, "AAA"), "A")   # se puede volver a signar
        self.assertEqual(self.feed(c, "." * 6, hand=False), LetterCommitter.SPACE)

    def test_reset_without_letters_disables_space(self):
        c = self.make()
        self.feed(c, "AAA")
        c.reset()
        self.assertEqual(self.feed(c, "." * 10, hand=False), "")


class KeypointTests(unittest.TestCase):
    def test_feature_vector_layout(self):
        l2d = np.arange(42, dtype=np.float32).reshape(21, 2)
        l3d = np.zeros((21, 3), dtype=np.float32)
        l3d[:, 2] = np.arange(21) * -1.0
        vec = hand_to_feature_vector(l2d, l3d)
        self.assertEqual(vec.shape, (63,))
        np.testing.assert_array_equal(vec.reshape(21, 3)[:, :2], l2d)
        np.testing.assert_array_equal(vec.reshape(21, 3)[:, 2], l3d[:, 2])

    def test_normalize_centers_on_wrist_and_scales_by_middle_mcp(self):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.5, 0.5, 0.1]          # muñeca
        pts[9] = [0.5, 0.3, 0.2]          # base del dedo medio: dist 0.2
        pts[8] = [0.6, 0.2, 0.3]
        out = normalize_keypoints(pts.reshape(63)).reshape(21, 3)
        np.testing.assert_allclose(out[0, :2], [0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(out[9, :2])), 1.0, places=5)
        np.testing.assert_allclose(out[8, :2], [0.5, -1.5], rtol=1e-5)
        np.testing.assert_allclose(out[:, 2], pts[:, 2])   # z sin tocar

    def test_normalize_degenerate_hand_does_not_divide_by_zero(self):
        out = normalize_keypoints(np.zeros(63, dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(out)))


class _FakeIO:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Imita onnxruntime.InferenceSession devolviendo logits fijos."""
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.last_input = None

    def get_inputs(self):
        return [_FakeIO("keypoints")]

    def get_outputs(self):
        return [_FakeIO("logits")]

    def run(self, outputs, feeds):
        self.last_input = feeds["keypoints"]
        return [self.logits]


class SignClassifierTests(unittest.TestCase):
    def test_topk_is_sorted_softmax(self):
        session = _FakeSession([1.0, 3.0, 2.0])
        clf = SignClassifier(session, ["A", "B", "C"])
        result = clf.predict_topk(np.ones(63, dtype=np.float32), k=3)
        self.assertEqual([r[0] for r in result], ["B", "C", "A"])
        self.assertAlmostEqual(sum(r[1] for r in result), 1.0, places=5)
        self.assertEqual(session.last_input.shape, (1, 63))

    def test_k_larger_than_classes(self):
        clf = SignClassifier(_FakeSession([0.0, 1.0]), ["A", "B"])
        self.assertEqual(len(clf.predict_topk(np.ones(63, dtype=np.float32), k=5)), 2)

    def test_rejects_wrong_shape(self):
        clf = SignClassifier(_FakeSession([0.0, 1.0]), ["A", "B"])
        with self.assertRaises(ValueError):
            clf.predict_topk(np.ones(62, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
