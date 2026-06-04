from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    import mediapipe as mp

    _HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    mp = None
    _HAS_SOLUTIONS = False

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


_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

Point = Tuple[float, float]


@dataclass
class FaceLandmarks:
    left_eye: Point
    right_eye: Point
    nose: Point
    forehead: Point
    chin: Point
    left_temple: Point
    right_temple: Point


@dataclass
class FaceState:
    bbox: Tuple[int, int, int, int]
    landmarks: FaceLandmarks
    confidence: float
    timestamp: float
    source: str


class FaceTracker:
    def __init__(self, face_cascade_path: str, eye_cascade_path: str, model_path: str) -> None:
        self._face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        self._mesh = None
        self._landmarker = None
        if _HAS_SOLUTIONS and mp is not None:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        elif _HAS_TASKS:
            self._landmarker = self._create_landmarker(model_path)

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> FaceState | None:
        mesh_state = self._detect_mesh(frame_bgr)
        if mesh_state is not None:
            return mesh_state

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self._detect_haar(gray)

    def _detect_mesh(self, frame_bgr: np.ndarray) -> FaceState | None:
        if self._mesh is None and self._landmarker is not None:
            return self._detect_landmarker(frame_bgr)
        if self._mesh is None:
            return None

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        best_landmarks = None
        best_area = 0.0
        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = (max_x - min_x) * (max_y - min_y)
            if area > best_area:
                best_area = area
                best_landmarks = face_landmarks

        if best_landmarks is None:
            return None

        points = self._mesh_points(best_landmarks.landmark, width, height)
        bbox = self._bbox_from_points(points)
        return FaceState(
            bbox=bbox,
            landmarks=points,
            confidence=1.0,
            timestamp=time.perf_counter(),
            source="mesh",
        )

    def _detect_landmarker(self, frame_bgr: np.ndarray) -> FaceState | None:
        if self._landmarker is None or Image is None or ImageFormat is None:
            return None

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None

        best_landmarks = None
        best_area = 0.0
        for face_landmarks in result.face_landmarks:
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = (max_x - min_x) * (max_y - min_y)
            if area > best_area:
                best_area = area
                best_landmarks = face_landmarks

        if best_landmarks is None:
            return None

        points = self._mesh_points(best_landmarks, width, height)
        bbox = self._bbox_from_points(points)
        return FaceState(
            bbox=bbox,
            landmarks=points,
            confidence=1.0,
            timestamp=time.perf_counter(),
            source="tasks",
        )

    def _mesh_points(self, landmarks, width: int, height: int) -> FaceLandmarks:
        idx = {
            "left_outer": 33,
            "left_inner": 133,
            "right_inner": 362,
            "right_outer": 263,
            "nose": 168,
            "forehead": 10,
            "chin": 152,
            "left_temple": 127,
            "right_temple": 356,
        }

        def point(i: int) -> Point:
            lm = landmarks[i]
            return (lm.x * width, lm.y * height)

        left_eye = self._midpoint(point(idx["left_outer"]), point(idx["left_inner"]))
        right_eye = self._midpoint(point(idx["right_outer"]), point(idx["right_inner"]))

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=point(idx["nose"]),
            forehead=point(idx["forehead"]),
            chin=point(idx["chin"]),
            left_temple=point(idx["left_temple"]),
            right_temple=point(idx["right_temple"]),
        )

    def _bbox_from_points(self, landmarks: FaceLandmarks) -> Tuple[int, int, int, int]:
        xs = [
            landmarks.left_eye[0],
            landmarks.right_eye[0],
            landmarks.nose[0],
            landmarks.forehead[0],
            landmarks.chin[0],
            landmarks.left_temple[0],
            landmarks.right_temple[0],
        ]
        ys = [
            landmarks.left_eye[1],
            landmarks.right_eye[1],
            landmarks.nose[1],
            landmarks.forehead[1],
            landmarks.chin[1],
            landmarks.left_temple[1],
            landmarks.right_temple[1],
        ]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _detect_haar(self, gray: np.ndarray) -> FaceState | None:
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi_gray = gray[y : y + h, x : x + w]
        eyes = self._eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        left_eye, right_eye = self._estimate_eyes(eyes, x, y, w, h)
        landmarks = FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=(x + w * 0.5, y + h * 0.55),
            forehead=(x + w * 0.5, y + h * 0.1),
            chin=(x + w * 0.5, y + h * 0.95),
            left_temple=(x + w * 0.1, y + h * 0.35),
            right_temple=(x + w * 0.9, y + h * 0.35),
        )
        return FaceState(
            bbox=(x, y, w, h),
            landmarks=landmarks,
            confidence=0.6,
            timestamp=time.perf_counter(),
            source="haar",
        )

    def _estimate_eyes(
        self, eyes: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Tuple[Point, Point]:
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            sorted_eyes = sorted(sorted_eyes, key=lambda e: e[0])
            left = sorted_eyes[0]
            right = sorted_eyes[1]
            left_eye = (x + left[0] + left[2] * 0.5, y + left[1] + left[3] * 0.5)
            right_eye = (x + right[0] + right[2] * 0.5, y + right[1] + right[3] * 0.5)
            return left_eye, right_eye

        if len(eyes) == 1:
            eye = eyes[0]
            center = (x + eye[0] + eye[2] * 0.5, y + eye[1] + eye[3] * 0.5)
            offset = w * 0.2
            return (center[0] - offset, center[1]), (center[0] + offset, center[1])

        return (x + w * 0.32, y + h * 0.4), (x + w * 0.68, y + h * 0.4)

    @staticmethod
    def _midpoint(a: Point, b: Point) -> Point:
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    def _create_landmarker(self, model_path: str):
        if not _HAS_TASKS or vision is None or BaseOptions is None:
            return None

        path = Path(model_path)
        if not path.exists():
            if not self._download_model(path):
                return None

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _download_model(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(_LANDMARKER_URL, path)
            return True
        except Exception:
            return False
