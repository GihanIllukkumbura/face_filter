from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

    _HAS_TASKS = True
except Exception:
    vision = None
    BaseOptions = None
    Image = None
    ImageFormat = None
    _HAS_TASKS = False


_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

Point = Tuple[float, float]


@dataclass
class HandState:
    angle_deg: float
    pinch_strength: float
    openness: float
    center: Point
    timestamp: float
    source: str


class HandTracker:
    def __init__(self, model_path: str) -> None:
        self._landmarker = None
        if _HAS_TASKS:
            self._landmarker = self._create_landmarker(model_path)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> HandState | None:
        if self._landmarker is None or Image is None or ImageFormat is None:
            return None

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None

        best_hand = None
        best_area = 0.0
        for hand_landmarks in result.hand_landmarks:
            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = (max_x - min_x) * (max_y - min_y)
            if area > best_area:
                best_area = area
                best_hand = hand_landmarks

        if best_hand is None:
            return None

        def point(index: int) -> Point:
            lm = best_hand[index]
            return (lm.x * width, lm.y * height)

        thumb_tip = point(4)
        index_tip = point(8)
        middle_tip = point(12)
        ring_tip = point(16)
        pinky_tip = point(20)
        wrist = point(0)
        middle_mcp = point(9)

        angle = math.degrees(math.atan2(index_tip[1] - thumb_tip[1], index_tip[0] - thumb_tip[0]))
        palm_size = max(1.0, math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]))
        pinch_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        pinch_ratio = pinch_distance / palm_size
        pinch_strength = max(0.0, min(1.0, 1.0 - pinch_ratio / 0.45))
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        spread = sum(math.hypot(pt[0] - wrist[0], pt[1] - wrist[1]) for pt in fingertips) / (len(fingertips) * palm_size)
        openness = max(0.0, min(1.0, (spread - 1.15) / 0.9))
        center = (
            sum(pt[0] for pt in fingertips + [wrist]) / 6.0,
            sum(pt[1] for pt in fingertips + [wrist]) / 6.0,
        )

        return HandState(
            angle_deg=angle,
            pinch_strength=pinch_strength,
            openness=openness,
            center=center,
            timestamp=time.perf_counter(),
            source="tasks",
        )

    def _create_landmarker(self, model_path: str):
        if not _HAS_TASKS or vision is None or BaseOptions is None:
            return None

        path = Path(model_path)
        if not path.exists():
            if not self._download_model(path):
                return None

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _download_model(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(_HAND_LANDMARKER_URL, path)
            return True
        except Exception:
            return False
