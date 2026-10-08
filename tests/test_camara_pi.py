"""Pruebas de la camara CSI de la Raspberry Pi (Picamera2), en especial la
Raspberry Pi AI Camera (IMX500), y de sus failsafes: corren en cualquier
computadora con una Picamera2 falsa que simula lo que hace la real cuando
falla (sin camaras, ocupada, sin firmware, sensor que deja de mandar imagen,
Picamera2 viejo...).

Importan senas.py, asi que necesitan PyQt6, OpenCV y MediaPipe; si faltan,
se omiten.
"""
import logging
import queue
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import senas
except ImportError:
    senas = None


class FakeJob:
    def __init__(self, cam):
        self.cam = cam


class FakePicamera2:
    """Lo minimo de Picamera2 que usa PiCameraCapture. Los atributos de clase
    se cambian en cada prueba para simular cada falla."""
    cameras = [{"Model": "imx500", "Num": 0}]
    init_error = None           # excepcion al crear Picamera2(0)
    frame = None                # lo que entrega cada captura
    deliver = True              # False: el sensor nunca entrega (timeout)
    old_api = False             # Picamera2 viejo: wait() sin timeout
    reject_sensor = False       # create_preview_configuration(sensor=...) falla
    close_hangs = False
    instances = []

    def __init__(self, num=0):
        if FakePicamera2.init_error is not None:
            raise FakePicamera2.init_error
        self.configs = []
        self.captures = 0
        self.started = False
        self.closed = False
        FakePicamera2.instances.append(self)

    @classmethod
    def global_camera_info(cls):
        return cls.cameras

    def create_preview_configuration(self, **kwargs):
        if "sensor" in kwargs and FakePicamera2.reject_sensor:
            raise TypeError("unexpected keyword argument 'sensor'")
        return kwargs

    def configure(self, config):
        self.configs.append(config)

    def set_controls(self, controls):
        pass

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        if FakePicamera2.close_hangs:
            time.sleep(30)
        self.closed = True

    def capture_array(self, name="main", wait=None):
        self.captures += 1
        if wait is False:
            return FakeJob(self)
        return self._frame_or_hang()

    def _frame_or_hang(self, timeout=None):
        if not FakePicamera2.deliver:
            if timeout is None:
                raise AssertionError("captura bloqueante sin imagen: se colgaria")
            time.sleep(timeout)
            raise TimeoutError()
        return FakePicamera2.frame.copy()

    def wait(self, job, timeout=None, **kwargs):
        if FakePicamera2.old_api and timeout is not None:
            raise TypeError("wait() got an unexpected keyword argument 'timeout'")
        return self._frame_or_hang(timeout)


def bgr_frame(width=640, height=480):
    """Cuadro azul puro en BGR ("RGB888" de Picamera2 es [B, G, R])."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    return frame


@unittest.skipIf(senas is None, "faltan dependencias de la app (PyQt6/cv2/mediapipe)")
class PiCameraCaptureTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        FakePicamera2.cameras = [{"Model": "imx500", "Num": 0}]
        FakePicamera2.init_error = None
        FakePicamera2.frame = bgr_frame()
        FakePicamera2.deliver = True
        FakePicamera2.old_api = False
        FakePicamera2.reject_sensor = False
        FakePicamera2.close_hangs = False
        FakePicamera2.instances = []
        module = types.ModuleType("picamera2")
        module.Picamera2 = FakePicamera2
        patches = [
            mock.patch.dict(sys.modules, {"picamera2": module}),
            mock.patch.object(senas, "IMX500_FIRST_FRAME_TIMEOUT_S", 0.2),
            mock.patch.object(senas, "PICAM_FIRST_FRAME_TIMEOUT_S", 0.2),
            mock.patch.object(senas, "PICAM_READ_TIMEOUT_S", 0.05),
            mock.patch.object(senas, "PICAM_CLOSE_TIMEOUT_S", 0.2),
            mock.patch.object(senas, "_active_picam", None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_ai_camera_opens_in_30fps_mode_and_keeps_colors(self):
        cap = senas.PiCameraCapture()
        self.assertTrue(cap.isOpened(), cap.error)
        self.assertTrue(cap.is_imx500)
        self.assertEqual(cap.short_name, "AI Camera")
        cam = FakePicamera2.instances[-1]
        self.assertEqual(cam.configs[0]["sensor"], {"output_size": senas.IMX500_SENSOR_SIZE})
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (480, 640, 3))
        # "RGB888" ya es BGR: el azul tiene que seguir en el canal 0.
        self.assertEqual(tuple(frame[0, 0]), (255, 0, 0))
        cap.release()
        self.assertTrue(cam.closed)
        self.assertFalse(cap.isOpened())

    def test_other_csi_camera_is_not_treated_as_ai_camera(self):
        FakePicamera2.cameras = [{"Model": "imx708", "Num": 0}]
        cap = senas.PiCameraCapture()
        self.assertTrue(cap.isOpened())
        self.assertFalse(cap.is_imx500)
        self.assertNotIn("sensor", FakePicamera2.instances[-1].configs[0])
        cap.release()

    def test_no_camera_detected(self):
        FakePicamera2.cameras = []
        cap = senas.PiCameraCapture()
        self.assertFalse(cap.isOpened())
        self.assertIn("rpicam-hello --list-cameras", cap.error)
        self.assertEqual(cap.read(), (False, None))

    def test_camera_busy(self):
        FakePicamera2.init_error = RuntimeError("Failed to acquire camera: Device or resource busy")
        cap = senas.PiCameraCapture()
        self.assertFalse(cap.isOpened())
        self.assertIn("ocupada", cap.error)

    def test_ai_camera_without_firmware_never_delivers(self):
        FakePicamera2.deliver = False
        t0 = time.time()
        cap = senas.PiCameraCapture()
        self.assertLess(time.time() - t0, 2.0)
        self.assertFalse(cap.isOpened())
        self.assertIn("imx500-all", cap.error)
        self.assertTrue(FakePicamera2.instances[-1].closed)

    def test_sensor_that_stops_does_not_freeze_read(self):
        cap = senas.PiCameraCapture()
        self.assertTrue(cap.isOpened())
        cam = FakePicamera2.instances[-1]
        FakePicamera2.deliver = False
        before = cam.captures
        t0 = time.time()
        for _ in range(3):
            self.assertEqual(cap.read(), (False, None))
        self.assertLess(time.time() - t0, 1.0)
        # Se vuelve a esperar la misma captura pendiente, sin apilar pedidos.
        self.assertEqual(cam.captures, before + 1)
        FakePicamera2.deliver = True
        self.assertTrue(cap.read()[0])
        cap.release()

    def test_old_picamera2_without_timeout_still_works(self):
        FakePicamera2.old_api = True
        cap = senas.PiCameraCapture()
        self.assertTrue(cap.isOpened(), cap.error)
        self.assertTrue(cap.read()[0])
        cap.release()

    def test_sensor_mode_rejected_falls_back_to_automatic(self):
        FakePicamera2.reject_sensor = True
        cap = senas.PiCameraCapture()
        self.assertTrue(cap.isOpened(), cap.error)
        self.assertNotIn("sensor", FakePicamera2.instances[-1].configs[0])
        cap.release()

    def test_odd_frames_are_normalized_or_rejected(self):
        cap = senas.PiCameraCapture()
        FakePicamera2.frame = np.full((240, 320, 4), 200, dtype=np.uint8)     # XRGB8888, otro tamaño
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertTrue(frame.flags["C_CONTIGUOUS"])
        FakePicamera2.frame = np.zeros((480, 640, 3), dtype=np.float32)        # no es imagen
        self.assertEqual(cap.read(), (False, None))
        cap.release()

    def test_stale_instance_is_closed_before_opening_again(self):
        first = senas.PiCameraCapture()
        old = FakePicamera2.instances[-1]
        second = senas.PiCameraCapture()      # el hilo viejo nunca llamo release()
        self.assertTrue(old.closed)
        self.assertTrue(second.isOpened())
        second.release()
        first.release()

    def test_hung_close_does_not_block(self):
        cap = senas.PiCameraCapture()
        FakePicamera2.close_hangs = True
        t0 = time.time()
        cap.release()
        self.assertLess(time.time() - t0, 2.0)


class FakeCap:
    """Captura de OpenCV falsa: entrega `frames` (None = lectura fallida,
    Exception = el driver lanza) y despues detiene el hilo."""

    def __init__(self, frames, thread=None):
        self.frames = list(frames)
        self.thread = thread
        self.released = False

    def read(self):
        if not self.frames:
            self.thread._run_flag = False
            return False, None
        item = self.frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return (item is not None), item

    def release(self):
        self.released = True


@unittest.skipIf(senas is None, "faltan dependencias de la app (PyQt6/cv2/mediapipe)")
class CameraThreadFailsafeTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.q = queue.Queue(maxsize=1)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def thread(self, backend="auto"):
        return senas.CameraThread(self.q, 0, backend=backend)

    def test_ai_camera_failure_falls_back_to_usb_webcam(self):
        t = self.thread()
        webcam = FakeCap([])
        with mock.patch.object(senas, "_is_raspberry_pi", return_value=True), \
                mock.patch.object(senas, "_picamera2_available", return_value=True), \
                mock.patch.object(t._opener, "_open_picamera", return_value=None), \
                mock.patch.object(t._opener, "_open_opencv", return_value=webcam):
            self.assertIs(t._open_camera(), webcam)

    def test_forced_picamera2_does_not_fall_back(self):
        t = self.thread("picamera2")
        with mock.patch.object(t._opener, "_open_picamera", return_value=None), \
                mock.patch.object(t._opener, "_open_opencv") as opencv:
            self.assertIsNone(t._open_camera())
        opencv.assert_not_called()

    def test_forced_opencv_never_touches_picamera2(self):
        t = self.thread("opencv")
        webcam = FakeCap([])
        with mock.patch.object(senas, "_is_raspberry_pi", return_value=True), \
                mock.patch.object(senas, "_picamera2_available", return_value=True), \
                mock.patch.object(t._opener, "_open_picamera") as picam, \
                mock.patch.object(t._opener, "_open_opencv", return_value=webcam):
            self.assertIs(t._open_camera(), webcam)
        picam.assert_not_called()

    def test_unexpected_error_while_opening_is_contained(self):
        t = self.thread("picamera2")
        with mock.patch.object(t._opener, "_open_picamera", side_effect=OSError("libcamera explotó")):
            self.assertIsNone(t._open_camera())
        self.assertIn("libcamera explotó", t.last_error)

    def test_initial_failure_reports_reason(self):
        t = self.thread("picamera2")
        errors = []
        t.error_signal.connect(errors.append)
        with mock.patch.object(t._opener, "_open_picamera",
                               side_effect=lambda: setattr(t._opener, "last_error", "sudo apt install imx500-all")):
            t.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("imx500-all", errors[0])

    def test_read_errors_and_failed_reconnect_do_not_kill_thread(self):
        t = self.thread()
        t._reconnect_delay_s = 0.01
        frame = bgr_frame()
        broken = FakeCap([RuntimeError("driver")] * 5, t)
        good = FakeCap([frame], t)
        # 1.a apertura: camara que lanza; reconexion 1: falla (None); 2: bien.
        opens = iter([broken, None, good])
        with mock.patch.object(t, "_open_camera", side_effect=lambda: next(opens)):
            runner = threading.Thread(target=t.run)
            runner.start()
            runner.join(5)
        self.assertFalse(runner.is_alive())
        self.assertTrue(broken.released)
        self.assertEqual(self.q.get_nowait().shape, (480, 640, 3))


@unittest.skipIf(senas is None, "faltan dependencias de la app (PyQt6/cv2/mediapipe)")
class LiveCameraTests(unittest.TestCase):
    """La camara de los scripts con bucle propio (recolectores, segmentador)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def live(self, opens):
        """LiveCamera cuyas aperturas devuelven, en orden, lo de `opens`."""
        opens = iter(opens)
        with mock.patch.object(senas.CameraOpener, "open", lambda self: next(opens)):
            cam = senas.LiveCamera(0, "auto")
        cam._opener.open = lambda: next(opens)
        return cam

    def test_initial_failure_is_reported_without_raising(self):
        cam = self.live([None, None])         # al abrir y al reintentar en read()
        self.assertFalse(cam.isOpened())
        self.assertEqual(cam.read(), (False, None))
        cam.release()

    def test_reconnects_after_failures_and_exceptions(self):
        frame = bgr_frame()
        broken = FakeCap([RuntimeError("driver"), None, None, None, None])
        good = FakeCap([frame])
        cam = self.live([broken, None, good])
        for _ in range(5):                      # 5 fallos (uno lanza) -> se cierra
            self.assertEqual(cam.read(), (False, None))
        self.assertTrue(broken.released)
        self.assertFalse(cam.gave_up)
        cam._retry_at = 0
        self.assertEqual(cam.read(), (False, None))   # la reapertura falla
        cam._retry_at = 0
        ok, got = cam.read()                    # la siguiente da imagen
        self.assertTrue(ok)
        self.assertIs(got, frame)
        self.assertEqual(cam.lost_for_s, 0.0)

    def test_gives_up_after_timeout(self):
        cam = self.live([FakeCap([None] * 5), None])
        for _ in range(5):
            cam.read()
        cam._last_frame_at -= senas.CAMERA_GIVE_UP_S + 1
        self.assertTrue(cam.gave_up)
        self.assertEqual(cam.placeholder().shape, (480, 640, 3))


@unittest.skipIf(senas is None, "faltan dependencias de la app (PyQt6/cv2/mediapipe)")
class CameraBackendConfigTests(unittest.TestCase):
    def test_backend_is_validated(self):
        logging.disable(logging.WARNING)
        try:
            cfg = senas.AppConfig()
            self.assertEqual(cfg.camera_backend, "auto")
            self.assertTrue(cfg.set_validated("camera_backend", " PiCamera2 "))
            self.assertEqual(cfg.camera_backend, "picamera2")
            self.assertFalse(cfg.set_validated("camera_backend", "v4l2"))
            self.assertEqual(cfg.camera_backend, "picamera2")
        finally:
            logging.disable(logging.NOTSET)


if __name__ == "__main__":
    unittest.main()
