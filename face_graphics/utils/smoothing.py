from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SmoothValue:
    alpha: float
    value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.value * self.alpha + new_value * (1.0 - self.alpha)
        return self.value

    def reset(self) -> None:
        self.value = None


def normalize_angle(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def angle_delta(start: float, end: float) -> float:
    return normalize_angle(end - start)


@dataclass
class SmoothAngle:
    alpha: float
    value: Optional[float] = None

    def update(self, new_value: float) -> float:
        new_value = normalize_angle(new_value)
        if self.value is None:
            self.value = new_value
        else:
            self.value = normalize_angle(self.value + angle_delta(self.value, new_value) * (1.0 - self.alpha))
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class Transform:
    center: Tuple[float, float]
    width: float
    height: float
    angle: float


class TransformSmoother:
    def __init__(self, alpha: float) -> None:
        self._x = SmoothValue(alpha)
        self._y = SmoothValue(alpha)
        self._w = SmoothValue(alpha)
        self._h = SmoothValue(alpha)
        self._a = SmoothAngle(alpha)

    def update(self, transform: Transform | None) -> Transform | None:
        if transform is None:
            return None
        return Transform(
            center=(self._x.update(transform.center[0]), self._y.update(transform.center[1])),
            width=self._w.update(transform.width),
            height=self._h.update(transform.height),
            angle=self._a.update(transform.angle),
        )

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()
        self._w.reset()
        self._h.reset()
        self._a.reset()
