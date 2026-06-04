from __future__ import annotations

from dataclasses import dataclass
import os
import cv2

FACE_CASCADE_PATH = os.getenv(
    "FACE_CASCADE_PATH",
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
)
EYE_CASCADE_PATH = os.getenv(
    "EYE_CASCADE_PATH",
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml",
)
FACE_LANDMARKER_PATH = os.getenv(
    "FACE_LANDMARKER_PATH",
    os.path.join("models", "face_landmarker.task"),
)
HAND_LANDMARKER_PATH = os.getenv(
    "HAND_LANDMARKER_PATH",
    os.path.join("models", "hand_landmarker.task"),
)


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    detection_interval: int = 2
    smoothing_alpha: float = 0.7
    overlay_opacity: float = 0.85
    overlay_scale: float = 1.0
    overlay_rotation_deg: float = 0.0
    overlay_tint_strength: float = 0.0
    overlay_shadow_strength: float = 0.35
    gesture_rotation: bool = False
    max_face_staleness_s: float = 0.6
    max_hand_staleness_s: float = 0.4
