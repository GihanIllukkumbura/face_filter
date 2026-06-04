from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class OverlayAssets:
    sunglasses: np.ndarray
    hat: np.ndarray
    mustache: np.ndarray
    halo: np.ndarray
    sparkles: np.ndarray
    bow_tie: np.ndarray
    face_frame: np.ndarray


def build_assets() -> OverlayAssets:
    return OverlayAssets(
        sunglasses=_create_sunglasses(),
        hat=_create_hat(),
        mustache=_create_mustache(),
        halo=_create_halo(),
        sparkles=_create_sparkles(),
        bow_tie=_create_bow_tie(),
        face_frame=_create_face_frame(),
    )


def _create_sunglasses(width: int = 512, height: int = 256) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    lens_color = (30, 30, 30, 210)
    frame_color = (5, 5, 5, 255)

    left_center = (int(width * 0.28), int(height * 0.52))
    right_center = (int(width * 0.72), int(height * 0.52))
    axes = (int(width * 0.18), int(height * 0.18))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, left_center, axes, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, right_center, axes, 0, 0, 360, 255, -1)

    cv2.ellipse(img, left_center, axes, 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, right_center, axes, 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, left_center, axes, 0, 0, 360, frame_color, 12)
    cv2.ellipse(img, right_center, axes, 0, 0, 360, frame_color, 12)

    bridge_tl = (int(width * 0.46), int(height * 0.46))
    bridge_br = (int(width * 0.54), int(height * 0.58))
    cv2.rectangle(img, bridge_tl, bridge_br, frame_color, -1)

    _apply_highlight(img, mask)
    _feather_alpha(img, 7)
    return img


def _create_hat(width: int = 512, height: int = 320) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    crown_color = (70, 50, 30, 240)
    brim_color = (45, 35, 25, 255)
    band_color = (20, 20, 20, 200)

    crown = np.array(
        [
            [int(width * 0.2), int(height * 0.4)],
            [int(width * 0.8), int(height * 0.4)],
            [int(width * 0.7), int(height * 0.08)],
            [int(width * 0.3), int(height * 0.08)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [crown], crown_color)

    cv2.ellipse(
        img,
        (width // 2, int(height * 0.45)),
        (int(width * 0.38), int(height * 0.12)),
        0,
        0,
        360,
        brim_color,
        -1,
    )

    cv2.rectangle(
        img,
        (int(width * 0.25), int(height * 0.3)),
        (int(width * 0.75), int(height * 0.35)),
        band_color,
        -1,
    )

    gradient = np.linspace(1.05, 0.85, height).astype(np.float32)
    img[:, :, :3] = (img[:, :, :3].astype(np.float32) * gradient[:, None, None]).clip(0, 255).astype(
        np.uint8
    )

    _feather_alpha(img, 9)
    return img


def _create_mustache(width: int = 512, height: int = 220) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    hair = (18, 16, 14, 245)
    sheen = (80, 70, 58, 120)
    center = (width // 2, int(height * 0.42))

    left = np.array(
        [
            center,
            (int(width * 0.37), int(height * 0.18)),
            (int(width * 0.15), int(height * 0.18)),
            (int(width * 0.05), int(height * 0.48)),
            (int(width * 0.2), int(height * 0.68)),
            (int(width * 0.42), int(height * 0.6)),
        ],
        dtype=np.int32,
    )
    right = left.copy()
    right[:, 0] = width - right[:, 0]

    cv2.fillPoly(img, [left], hair)
    cv2.fillPoly(img, [right], hair)
    cv2.circle(img, center, int(height * 0.12), hair, -1)
    cv2.ellipse(
        img,
        (int(width * 0.28), int(height * 0.36)),
        (int(width * 0.16), int(height * 0.06)),
        -12,
        190,
        345,
        sheen,
        4,
    )
    cv2.ellipse(
        img,
        (int(width * 0.72), int(height * 0.36)),
        (int(width * 0.16), int(height * 0.06)),
        12,
        195,
        350,
        sheen,
        4,
    )

    _feather_alpha(img, 7)
    return img


def _create_halo(width: int = 512, height: int = 220) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    glow = (80, 210, 255, 70)
    ring = (90, 235, 255, 235)
    core = (230, 255, 255, 210)

    center = (width // 2, height // 2)
    axes = (int(width * 0.38), int(height * 0.22))
    for thickness, color in ((28, glow), (16, ring), (5, core)):
        cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness)

    cv2.line(
        img,
        (int(width * 0.18), int(height * 0.4)),
        (int(width * 0.38), int(height * 0.34)),
        (255, 255, 255, 120),
        4,
    )
    _feather_alpha(img, 11)
    return img


def _create_sparkles(width: int = 512, height: int = 512) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    colors = [
        (255, 220, 80, 230),
        (80, 230, 255, 210),
        (255, 120, 210, 210),
        (255, 255, 255, 190),
    ]
    points = [
        (0.2, 0.22, 0.09),
        (0.78, 0.25, 0.07),
        (0.14, 0.62, 0.06),
        (0.86, 0.66, 0.08),
        (0.5, 0.12, 0.045),
        (0.54, 0.83, 0.055),
    ]

    for i, (x, y, size) in enumerate(points):
        cx = int(width * x)
        cy = int(height * y)
        radius = int(width * size)
        color = colors[i % len(colors)]
        _draw_sparkle(img, (cx, cy), radius, color)

    cv2.ellipse(
        img,
        (width // 2, height // 2),
        (int(width * 0.42), int(height * 0.38)),
        0,
        195,
        345,
        (120, 235, 255, 90),
        5,
    )
    _feather_alpha(img, 5)
    return img


def _create_bow_tie(width: int = 512, height: int = 260) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    left_color = (38, 42, 210, 245)
    right_color = (35, 135, 245, 245)
    knot_color = (18, 18, 30, 255)
    edge_color = (12, 12, 24, 220)

    center = (width // 2, height // 2)
    left = np.array(
        [
            (int(width * 0.05), int(height * 0.16)),
            (int(width * 0.47), int(height * 0.36)),
            (int(width * 0.47), int(height * 0.64)),
            (int(width * 0.05), int(height * 0.84)),
            (int(width * 0.16), int(height * 0.5)),
        ],
        dtype=np.int32,
    )
    right = left.copy()
    right[:, 0] = width - right[:, 0]

    cv2.fillPoly(img, [left], left_color)
    cv2.fillPoly(img, [right], right_color)
    cv2.polylines(img, [left, right], True, edge_color, 8)
    cv2.ellipse(img, center, (int(width * 0.08), int(height * 0.2)), 0, 0, 360, knot_color, -1)
    cv2.line(
        img,
        (int(width * 0.2), int(height * 0.28)),
        (int(width * 0.38), int(height * 0.42)),
        (255, 255, 255, 95),
        5,
    )
    cv2.line(
        img,
        (int(width * 0.8), int(height * 0.28)),
        (int(width * 0.62), int(height * 0.42)),
        (255, 255, 255, 95),
        5,
    )
    _feather_alpha(img, 7)
    return img


def _create_face_frame(width: int = 512, height: int = 620) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    ring = (235, 235, 245, 220)
    glow = (70, 220, 255, 75)
    accent = (255, 160, 70, 210)
    center = (width // 2, int(height * 0.48))
    axes = (int(width * 0.39), int(height * 0.41))

    cv2.ellipse(img, center, axes, 0, 0, 360, glow, 28)
    cv2.ellipse(img, center, axes, 0, 0, 360, ring, 9)
    cv2.ellipse(img, center, (int(width * 0.43), int(height * 0.45)), 0, 210, 330, accent, 8)
    cv2.ellipse(img, center, (int(width * 0.43), int(height * 0.45)), 0, 30, 150, accent, 8)

    for x, y in ((0.14, 0.24), (0.86, 0.24), (0.16, 0.74), (0.84, 0.74)):
        cv2.circle(img, (int(width * x), int(height * y)), int(width * 0.035), accent, -1)

    _feather_alpha(img, 9)
    return img


def _draw_sparkle(img: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int, int]) -> None:
    x, y = center
    cv2.line(img, (x - radius, y), (x + radius, y), color, 3)
    cv2.line(img, (x, y - radius), (x, y + radius), color, 3)
    cv2.line(img, (x - radius // 2, y - radius // 2), (x + radius // 2, y + radius // 2), color, 2)
    cv2.line(img, (x - radius // 2, y + radius // 2), (x + radius // 2, y - radius // 2), color, 2)
    cv2.circle(img, center, max(2, radius // 6), color, -1)


def _apply_highlight(img: np.ndarray, mask: np.ndarray) -> None:
    height, width = mask.shape
    gradient = np.linspace(120, 0, height).astype(np.uint8)
    highlight = np.zeros((height, width, 4), dtype=np.uint8)
    highlight[:, :, :3] = 255
    highlight[:, :, 3] = (gradient[:, None] * (mask / 255.0)).astype(np.uint8)

    alpha = highlight[:, :, 3:4].astype(np.float32) / 255.0
    img[:, :, :3] = (img[:, :, :3].astype(np.float32) * (1.0 - alpha) + highlight[:, :, :3] * alpha).astype(
        np.uint8
    )
    img[:, :, 3] = np.maximum(img[:, :, 3], highlight[:, :, 3])


def _feather_alpha(img: np.ndarray, ksize: int) -> None:
    if ksize % 2 == 0:
        ksize += 1
    alpha = cv2.GaussianBlur(img[:, :, 3], (ksize, ksize), 0)
    img[:, :, 3] = alpha
