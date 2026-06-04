from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.geometry import angle_deg, distance, midpoint
from ..utils.image import alpha_blend_bgr, apply_tint, create_shadow_rgba, rotate_rgba
from ..utils.smoothing import Transform, TransformSmoother
from ..vision.face_tracker import FaceState
from .overlays import OverlayAssets, build_assets


@dataclass
class RenderSettings:
    show_sunglasses: bool = True
    show_hat: bool = True
    show_mustache: bool = False
    show_halo: bool = False
    show_sparkles: bool = False
    show_bow_tie: bool = False
    show_face_frame: bool = False
    opacity: float = 0.85
    scale: float = 1.0
    rotation_deg: float = 0.0
    rotation_target: str = "all"
    offset_x: float = 0.0
    offset_y: float = 0.0
    tint_color: Optional[Tuple[int, int, int]] = None
    tint_strength: float = 0.0
    accent_color: Optional[Tuple[int, int, int]] = None
    accent_strength: float = 0.0
    shadow_strength: float = 0.35
    style: str = "classic"
    frame_effect: str = "none"
    auto_motion: bool = False
    rotation_motion: str = "sway"
    rotation_speed: float = 1.0
    rotation_amplitude: float = 22.0
    gesture_rotation: bool = False
    gesture_sensitivity: float = 1.0
    gesture_scale: float = 1.0

    def copy(self) -> "RenderSettings":
        return RenderSettings(
            show_sunglasses=self.show_sunglasses,
            show_hat=self.show_hat,
            show_mustache=self.show_mustache,
            show_halo=self.show_halo,
            show_sparkles=self.show_sparkles,
            show_bow_tie=self.show_bow_tie,
            show_face_frame=self.show_face_frame,
            opacity=self.opacity,
            scale=self.scale,
            rotation_deg=self.rotation_deg,
            rotation_target=self.rotation_target,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            tint_color=self.tint_color,
            tint_strength=self.tint_strength,
            accent_color=self.accent_color,
            accent_strength=self.accent_strength,
            shadow_strength=self.shadow_strength,
            style=self.style,
            frame_effect=self.frame_effect,
            auto_motion=self.auto_motion,
            rotation_motion=self.rotation_motion,
            rotation_speed=self.rotation_speed,
            rotation_amplitude=self.rotation_amplitude,
            gesture_rotation=self.gesture_rotation,
            gesture_sensitivity=self.gesture_sensitivity,
            gesture_scale=self.gesture_scale,
        )


class OverlayRenderer:
    def __init__(self, smoothing_alpha: float) -> None:
        self._assets: OverlayAssets = build_assets()
        self._smoothers = {
            "sunglasses": TransformSmoother(smoothing_alpha),
            "hat": TransformSmoother(smoothing_alpha),
            "mustache": TransformSmoother(smoothing_alpha),
            "halo": TransformSmoother(smoothing_alpha),
            "sparkles": TransformSmoother(smoothing_alpha),
            "bow_tie": TransformSmoother(smoothing_alpha),
            "face_frame": TransformSmoother(smoothing_alpha),
        }

    def render(self, frame_bgr: np.ndarray, face_state: FaceState | None, settings: RenderSettings) -> np.ndarray:
        output = self._apply_frame_effect(frame_bgr.copy(), settings.frame_effect)
        if face_state is None:
            self._reset_smoothers()
            return output

        face_mean = self._mean_face_color(output, face_state.bbox)
        overlays = [
            (
                "face_frame",
                settings.show_face_frame,
                self._assets.face_frame,
                self._compute_face_frame_transform,
            ),
            (
                "sunglasses",
                settings.show_sunglasses,
                self._assets.sunglasses,
                self._compute_glasses_transform,
            ),
            ("hat", settings.show_hat, self._assets.hat, self._compute_hat_transform),
            (
                "mustache",
                settings.show_mustache,
                self._assets.mustache,
                self._compute_mustache_transform,
            ),
            ("halo", settings.show_halo, self._assets.halo, self._compute_halo_transform),
            (
                "sparkles",
                settings.show_sparkles,
                self._assets.sparkles,
                self._compute_sparkles_transform,
            ),
            (
                "bow_tie",
                settings.show_bow_tie,
                self._assets.bow_tie,
                self._compute_bow_tie_transform,
            ),
        ]

        for name, enabled, asset, transform_fn in overlays:
            if not enabled:
                self._smoothers[name].reset()
                continue

            transform = transform_fn(
                face_state,
                self._scale_for(name, settings),
                self._rotation_for(name, settings),
            )
            transform = self._offset_transform(name, transform, settings)
            smoothed = self._smoothers[name].update(transform)
            if smoothed is not None:
                output = self._apply_overlay(output, asset, smoothed, settings, face_mean, name)

        return output

    def _compute_glasses_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        eye_dist = distance(lm.left_eye, lm.right_eye)
        if eye_dist <= 1:
            return None

        base_h, base_w = self._assets.sunglasses.shape[:2]
        target_w = eye_dist * 2.2 * scale
        target_h = target_w * (base_h / base_w)
        center = midpoint(lm.left_eye, lm.right_eye)
        center = (center[0], center[1] + target_h * 0.05)
        angle = angle_deg(lm.left_eye, lm.right_eye) + rotation_deg
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _compute_hat_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        temple_dist = distance(lm.left_temple, lm.right_temple)
        if temple_dist <= 1:
            return None

        base_h, base_w = self._assets.hat.shape[:2]
        target_w = temple_dist * 1.6 * scale
        target_h = target_w * (base_h / base_w)
        center = midpoint(lm.left_temple, lm.right_temple)
        center = (center[0], lm.forehead[1] - target_h * 0.15)
        angle = angle_deg(lm.left_temple, lm.right_temple) + rotation_deg
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _compute_mustache_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        eye_dist = distance(lm.left_eye, lm.right_eye)
        if eye_dist <= 1:
            return None

        base_h, base_w = self._assets.mustache.shape[:2]
        target_w = eye_dist * 1.35 * scale
        target_h = target_w * (base_h / base_w)
        center = (lm.nose[0], lm.nose[1] + target_h * 0.55)
        angle = angle_deg(lm.left_eye, lm.right_eye) + rotation_deg
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _compute_halo_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        temple_dist = distance(lm.left_temple, lm.right_temple)
        if temple_dist <= 1:
            return None

        base_h, base_w = self._assets.halo.shape[:2]
        target_w = temple_dist * 1.25 * scale
        target_h = target_w * (base_h / base_w)
        center = (lm.forehead[0], lm.forehead[1] - target_h * 0.7)
        angle = angle_deg(lm.left_temple, lm.right_temple) + rotation_deg * 0.55
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _compute_sparkles_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        face_h = distance(lm.forehead, lm.chin)
        temple_dist = distance(lm.left_temple, lm.right_temple)
        size = max(face_h, temple_dist) * 1.45 * scale
        if size <= 1:
            return None

        center = (lm.nose[0], (lm.forehead[1] + lm.chin[1]) * 0.5)
        angle = angle_deg(lm.left_temple, lm.right_temple) + rotation_deg
        return Transform(center=center, width=size, height=size, angle=angle)

    def _compute_bow_tie_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        temple_dist = distance(lm.left_temple, lm.right_temple)
        if temple_dist <= 1:
            return None

        base_h, base_w = self._assets.bow_tie.shape[:2]
        target_w = temple_dist * 1.05 * scale
        target_h = target_w * (base_h / base_w)
        center = (lm.chin[0], lm.chin[1] + target_h * 0.52)
        angle = angle_deg(lm.left_temple, lm.right_temple) + rotation_deg
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _compute_face_frame_transform(
        self, face_state: FaceState, scale: float, rotation_deg: float
    ) -> Transform | None:
        lm = face_state.landmarks
        face_h = distance(lm.forehead, lm.chin)
        temple_dist = distance(lm.left_temple, lm.right_temple)
        if face_h <= 1 or temple_dist <= 1:
            return None

        base_h, base_w = self._assets.face_frame.shape[:2]
        target_h = face_h * 1.45 * scale
        target_w = target_h * (base_w / base_h)
        center = (lm.nose[0], (lm.forehead[1] + lm.chin[1]) * 0.5)
        angle = angle_deg(lm.left_temple, lm.right_temple) + rotation_deg * 0.35
        return Transform(center=center, width=target_w, height=target_h, angle=angle)

    def _apply_overlay(
        self,
        frame_bgr: np.ndarray,
        base_rgba: np.ndarray,
        transform: Transform,
        settings: RenderSettings,
        face_mean: Optional[np.ndarray],
        name: str,
    ) -> np.ndarray:
        if transform.width <= 2 or transform.height <= 2:
            return frame_bgr

        resized = cv2.resize(base_rgba, (int(transform.width), int(transform.height)), interpolation=cv2.INTER_AREA)
        auto_tint = 0.12 if name in {"sunglasses", "hat", "mustache"} else 0.03
        tinted = apply_tint(
            resized,
            face_mean,
            strength=auto_tint,
            tint_bgr=settings.tint_color,
            tint_strength=settings.tint_strength,
        )
        tinted = self._stylize_overlay(tinted, settings, name)
        rotated = rotate_rgba(tinted, transform.angle)

        shadow = create_shadow_rgba(rotated, offset=(4, 4), blur=9, opacity=settings.shadow_strength)
        top_left = (
            int(transform.center[0] - rotated.shape[1] / 2),
            int(transform.center[1] - rotated.shape[0] / 2),
        )

        frame_bgr = alpha_blend_bgr(frame_bgr, shadow, top_left, opacity=settings.opacity)
        frame_bgr = alpha_blend_bgr(frame_bgr, rotated, top_left, opacity=settings.opacity)
        return frame_bgr

    def _scale_for(self, name: str, settings: RenderSettings) -> float:
        scale = settings.scale
        if self._targets_name(settings.rotation_target, name):
            scale *= settings.gesture_scale
        return scale

    def _offset_transform(
        self, name: str, transform: Transform | None, settings: RenderSettings
    ) -> Transform | None:
        if transform is None or not self._targets_name(settings.rotation_target, name):
            return transform

        return Transform(
            center=(
                transform.center[0] + settings.offset_x * transform.width,
                transform.center[1] + settings.offset_y * transform.height,
            ),
            width=transform.width,
            height=transform.height,
            angle=transform.angle,
        )

    def _rotation_for(self, name: str, settings: RenderSettings) -> float:
        if not self._targets_name(settings.rotation_target, name):
            return 0.0

        angle = settings.rotation_deg
        if settings.auto_motion:
            elapsed = time.perf_counter()
            speed = max(0.05, settings.rotation_speed)
            amplitude = max(0.0, settings.rotation_amplitude)
            if settings.rotation_motion == "spin":
                angle += (elapsed * speed * 90.0) % 360.0
            elif settings.rotation_motion == "pulse":
                phase = elapsed * speed * 2.0
                angle += math.sin(phase) * amplitude + math.sin(phase * 2.0) * amplitude * 0.25
            else:
                phase = elapsed * speed * 1.8
                angle += math.sin(phase) * amplitude
        return angle

    @staticmethod
    def _targets_name(target: str, name: str) -> bool:
        return target == "all" or target == name

    def _stylize_overlay(self, overlay_rgba: np.ndarray, settings: RenderSettings, name: str) -> np.ndarray:
        styled = overlay_rgba.copy()
        accent_strength = settings.accent_strength
        if name in {"halo", "sparkles"}:
            accent_strength = max(accent_strength, 0.35 if settings.accent_color is not None else 0.0)

        if settings.accent_color is not None and accent_strength > 0:
            alpha = styled[:, :, 3:4].astype(np.float32) / 255.0
            color = np.array(settings.accent_color, dtype=np.float32).reshape(1, 1, 3)
            rgb = styled[:, :, :3].astype(np.float32)
            mix = min(1.0, accent_strength)
            styled[:, :, :3] = (rgb * (1.0 - mix * alpha) + color * (mix * alpha)).clip(0, 255).astype(np.uint8)

        if settings.style == "mono":
            gray = cv2.cvtColor(styled[:, :, :3], cv2.COLOR_BGR2GRAY)
            styled[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif settings.style == "neon":
            styled[:, :, :3] = cv2.convertScaleAbs(styled[:, :, :3], alpha=1.25, beta=18)
            alpha = styled[:, :, 3].astype(np.float32)
            styled[:, :, 3] = np.minimum(255, alpha * 1.08).astype(np.uint8)
        elif settings.style == "pop":
            hsv = cv2.cvtColor(styled[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.minimum(255, hsv[:, :, 1] * 1.25)
            hsv[:, :, 2] = np.minimum(255, hsv[:, :, 2] * 1.08)
            styled[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return styled

    def _apply_frame_effect(self, frame_bgr: np.ndarray, effect: str) -> np.ndarray:
        if effect == "grayscale":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if effect == "cinematic":
            tinted = frame_bgr.astype(np.float32)
            tinted[:, :, 0] *= 1.08
            tinted[:, :, 1] *= 1.0
            tinted[:, :, 2] *= 0.92
            height = frame_bgr.shape[0]
            gradient = np.linspace(0.84, 1.08, height, dtype=np.float32).reshape(height, 1, 1)
            return (tinted * gradient).clip(0, 255).astype(np.uint8)
        if effect == "cartoon":
            smooth = cv2.bilateralFilter(frame_bgr, 9, 80, 80)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
            return cv2.bitwise_and(smooth, smooth, mask=edges)
        if effect == "edges":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 70, 150)
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(frame_bgr, 0.55, edge_bgr, 0.65, 0)
        if effect == "soft":
            blur = cv2.GaussianBlur(frame_bgr, (0, 0), 7)
            return cv2.addWeighted(frame_bgr, 0.72, blur, 0.28, 0)
        return frame_bgr

    def _reset_smoothers(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()

    @staticmethod
    def _mean_face_color(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_bgr.shape[1], x + w)
        y2 = min(frame_bgr.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi.mean(axis=(0, 1))
