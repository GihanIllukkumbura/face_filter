from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def rotate_rgba(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def alpha_blend_bgr(
    base_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
    top_left: Tuple[int, int],
    opacity: float = 1.0,
) -> np.ndarray:
    x, y = top_left
    if opacity <= 0:
        return base_bgr

    base_h, base_w = base_bgr.shape[:2]
    ov_h, ov_w = overlay_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(base_w, x + ov_w)
    y2 = min(base_h, y + ov_h)

    if x1 >= x2 or y1 >= y2:
        return base_bgr

    overlay_crop = overlay_rgba[y1 - y : y2 - y, x1 - x : x2 - x]
    alpha = (overlay_crop[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    if alpha.size == 0:
        return base_bgr

    base_crop = base_bgr[y1:y2, x1:x2].astype(np.float32)
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)

    blended = base_crop * (1.0 - alpha) + overlay_rgb * alpha
    base_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    return base_bgr


def create_shadow_rgba(
    overlay_rgba: np.ndarray,
    offset: Tuple[int, int] = (4, 4),
    blur: int = 9,
    opacity: float = 0.35,
) -> np.ndarray:
    if blur % 2 == 0:
        blur += 1

    alpha = overlay_rgba[:, :, 3].astype(np.float32)
    shadow = np.zeros_like(overlay_rgba)
    shadow[:, :, 3] = (alpha * opacity).astype(np.uint8)
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)

    matrix = np.array([[1, 0, offset[0]], [0, 1, offset[1]]], dtype=np.float32)
    return cv2.warpAffine(shadow, matrix, (shadow.shape[1], shadow.shape[0]), borderValue=(0, 0, 0, 0))


def apply_tint(
    overlay_rgba: np.ndarray,
    mean_bgr: Optional[np.ndarray],
    strength: float = 0.15,
    tint_bgr: Optional[Tuple[int, int, int]] = None,
    tint_strength: float = 0.0,
) -> np.ndarray:
    tinted = overlay_rgba.copy()
    if mean_bgr is not None and strength > 0:
        mean = mean_bgr.astype(np.float32)
        rgb = tinted[:, :, :3].astype(np.float32)
        tinted[:, :, :3] = (rgb * (1.0 - strength) + mean * strength).clip(0, 255).astype(np.uint8)

    if tint_bgr is not None and tint_strength > 0:
        color = np.array(tint_bgr, dtype=np.float32).reshape(1, 1, 3)
        rgb = tinted[:, :, :3].astype(np.float32)
        tinted[:, :, :3] = (rgb * (1.0 - tint_strength) + color * tint_strength).clip(0, 255).astype(np.uint8)
    return tinted
