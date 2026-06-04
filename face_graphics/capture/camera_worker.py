from __future__ import annotations

import sys
import threading
import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ..config import (
    AppConfig,
    EYE_CASCADE_PATH,
    FACE_CASCADE_PATH,
    FACE_LANDMARKER_PATH,
    HAND_LANDMARKER_PATH,
)
from ..overlay.renderer import OverlayRenderer, RenderSettings
from ..vision.face_tracker import FaceTracker
from ..vision.hand_tracker import HandTracker
from ..utils.smoothing import SmoothAngle, SmoothValue, angle_delta


class CameraWorker(QThread):
    frameReady = Signal(QImage)
    metricsReady = Signal(float, bool)
    statusReady = Signal(str)
    error = Signal(str)

    def __init__(self, config: AppConfig, settings: RenderSettings) -> None:
        super().__init__()
        self._config = config
        self._renderer = OverlayRenderer(config.smoothing_alpha)
        self._tracker = FaceTracker(FACE_CASCADE_PATH, EYE_CASCADE_PATH, FACE_LANDMARKER_PATH)
        self._hand_tracker = HandTracker(HAND_LANDMARKER_PATH)
        self._settings = settings.copy()
        self._settings_lock = threading.Lock()
        self._running = True
        self._last_state = None
        self._last_state_time = 0.0
        self._last_hand = None
        self._last_hand_time = 0.0
        self._hand_angle = SmoothAngle(config.smoothing_alpha)
        self._hand_scale = SmoothValue(config.smoothing_alpha)
        self._pinch_active = False
        self._pinch_start_hand_angle = 0.0
        self._pinch_start_rotation = 0.0

    def update_settings(self, settings: RenderSettings) -> None:
        with self._settings_lock:
            self._settings = settings.copy()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        capture = cv2.VideoCapture(self._config.camera_index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            self.error.emit("Unable to open camera.")
            return

        frame_count = 0
        start_time = time.perf_counter()
        last_metrics = start_time

        try:
            while self._running and not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera frame not available.")
                    break

                frame = cv2.flip(frame, 1)

                with self._settings_lock:
                    settings = self._settings.copy()

                if frame_count % self._config.detection_interval == 0:
                    state = self._tracker.detect(frame)
                    hand_state = None
                    if state is not None:
                        self._last_state = state
                        self._last_state_time = time.perf_counter()
                    else:
                        self._last_state = None

                    if settings.gesture_rotation:
                        hand_state = self._hand_tracker.detect(frame)
                        if hand_state is not None:
                            self._last_hand = hand_state
                            self._last_hand_time = time.perf_counter()
                        else:
                            self._last_hand = None
                            self._hand_angle.reset()
                            self._hand_scale.reset()
                            self._pinch_active = False
                    else:
                        self._last_hand = None
                        self._hand_angle.reset()
                        self._hand_scale.reset()
                        self._pinch_active = False

                    status = f"Face: {state.source}" if state is not None else "No face"
                    if settings.gesture_rotation:
                        if hand_state is None:
                            status = f"{status} | Hand: none"
                        elif hand_state.pinch_strength > 0.2:
                            status = f"{status} | Hand: pinch rotate"
                        else:
                            status = f"{status} | Hand: open scale"
                    self.statusReady.emit(status)

                now = time.perf_counter()
                if self._last_state is not None and (now - self._last_state_time) > self._config.max_face_staleness_s:
                    self._last_state = None

                if self._last_hand is not None and (now - self._last_hand_time) > self._config.max_hand_staleness_s:
                    self._last_hand = None
                    self._hand_angle.reset()
                    self._hand_scale.reset()
                    self._pinch_active = False

                if settings.gesture_rotation and self._last_hand is not None:
                    if self._last_hand.pinch_strength > 0.2:
                        if not self._pinch_active:
                            self._pinch_active = True
                            self._pinch_start_hand_angle = self._last_hand.angle_deg
                            self._pinch_start_rotation = settings.rotation_deg
                        twist = angle_delta(self._pinch_start_hand_angle, self._last_hand.angle_deg)
                        target_rotation = self._pinch_start_rotation + twist * settings.gesture_sensitivity
                        settings.rotation_deg = self._hand_angle.update(target_rotation)
                    else:
                        self._hand_angle.reset()
                        self._pinch_active = False
                    open_scale = 0.9 + self._last_hand.openness * 0.35
                    settings.gesture_scale = self._hand_scale.update(open_scale)
                else:
                    settings.gesture_scale = 1.0
                    self._pinch_active = False

                rendered = self._renderer.render(frame, self._last_state, settings)
                qimage = self._to_qimage(rendered)
                self.frameReady.emit(qimage)

                frame_count += 1
                if now - last_metrics >= 0.5:
                    fps = frame_count / max(0.001, now - start_time)
                    self.metricsReady.emit(fps, self._last_state is not None)
                    last_metrics = now

                self.msleep(1)
        finally:
            capture.release()
            self._tracker.close()
            self._hand_tracker.close()

    @staticmethod
    def _to_qimage(frame_bgr) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
