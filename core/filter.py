"""Filtering and Signal Processing Module.

Provides adaptive signal filtering algorithms including One Euro Filter (1€ Filter),
Dynamic Exponential Moving Average (EMA), Deadzone suppression, and 2D Kalman Filtering
to eliminate spatial hand tracking micro-jitter while maintaining instant responsiveness.
"""

import math
import time
from typing import Tuple, Optional
import cv2
import numpy as np


class OneEuroFilter1D:
    """1D One Euro Filter for adaptive low-pass signal filtering."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
    ):
        """Initialize OneEuroFilter1D.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz) for heavy smoothing at slow speeds.
            beta: Speed coefficient for increasing cutoff frequency during rapid movement.
            d_cutoff: Derivative cutoff frequency (Hz).
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """Filter 1D input value with timestamp t (seconds)."""
        if t is None:
            t = time.time()

        if self.x_prev is None or self.t_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return self.x_prev

        dt = max(1e-4, float(t - self.t_prev))
        self.t_prev = float(t)

        # Estimate rate of change (velocity)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adapt cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self) -> None:
        """Reset internal filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class OneEuroFilter2D:
    """2D One Euro Filter for smooth (x, y) spatial cursor tracking."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
    ):
        """Initialize 2D One Euro Filter."""
        self.filter_x = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        """Filter (x, y) coordinates smoothly."""
        if t is None:
            t = time.time()
        fx = self.filter_x.filter(x, t)
        fy = self.filter_y.filter(y, t)
        return fx, fy

    def update_params(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        """Dynamically update filter parameters."""
        if min_cutoff is not None:
            self.filter_x.min_cutoff = max(0.1, min_cutoff)
            self.filter_y.min_cutoff = max(0.1, min_cutoff)
        if beta is not None:
            self.filter_x.beta = max(0.0001, beta)
            self.filter_y.beta = max(0.0001, beta)

    def reset(self) -> None:
        """Reset internal filter state."""
        self.filter_x.reset()
        self.filter_y.reset()


class DeadzoneFilter:
    """Suppresses coordinate deltas below a specified spatial threshold."""

    def __init__(self, deadzone_px: float = 0.0):
        """Initialize DeadzoneFilter."""
        self.deadzone_px = deadzone_px
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """Apply deadzone filter to input coordinates."""
        if self._last_x is None or self._last_y is None:
            self._last_x = x
            self._last_y = y
            return x, y

        dist = math.hypot(x - self._last_x, y - self._last_y)
        if dist < self.deadzone_px:
            return self._last_x, self._last_y

        self._last_x = x
        self._last_y = y
        return x, y

    def reset(self) -> None:
        """Reset internal filter state."""
        self._last_x = None
        self._last_y = None


class DynamicEMAFilter:
    """Adaptive Exponential Moving Average (EMA) filter using One Euro Filter internally."""

    def __init__(
        self,
        min_alpha: float = 0.10,
        max_alpha: float = 0.85,
        deadzone_px: float = 0.0,
        velocity_scale: float = 35.0,
    ):
        """Initialize DynamicEMAFilter with One Euro Filter core."""
        # Map min_alpha -> min_cutoff (0.1 -> 0.6 Hz) and max_alpha -> beta
        min_cutoff = max(0.2, min_alpha * 5.0)
        beta = max(0.001, max_alpha * 0.01)
        self.one_euro = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)
        self.deadzone = DeadzoneFilter(deadzone_px=deadzone_px)
        self._last_alpha = min_alpha

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """Filter raw input coordinates."""
        dz_x, dz_y = self.deadzone.filter(x, y)
        return self.one_euro.filter(dz_x, dz_y)

    @property
    def current_alpha(self) -> float:
        return self._last_alpha

    def update_params(
        self,
        min_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
        deadzone_px: Optional[float] = None,
        velocity_scale: Optional[float] = None,
    ) -> None:
        """Dynamically update filter parameters."""
        if deadzone_px is not None:
            self.deadzone.deadzone_px = deadzone_px

        min_cutoff = None if min_alpha is None else max(0.2, min_alpha * 5.0)
        beta = None if max_alpha is None else max(0.001, max_alpha * 0.01)
        self.one_euro.update_params(min_cutoff=min_cutoff, beta=beta)

    def reset(self) -> None:
        """Reset internal filter state."""
        self.one_euro.reset()
        self.deadzone.reset()


class KalmanFilter2D:
    """2D Kalman Filter tracking position (x, y) and velocity (vx, vy)."""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        """Initialize 2D Kalman Filter using OpenCV."""
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """Predict and update Kalman filter state with new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._initialized:
            self.kf.statePre = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], dtype=np.float32)
            self._initialized = True
            return x, y

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return float(corrected[0, 0]), float(corrected[1, 0])

    def reset(self) -> None:
        """Reset Kalman Filter state."""
        self._initialized = False
