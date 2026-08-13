"""Mouse Controller Module.

Handles high-performance system-level mouse events using direct Win32 API and PyAutoGUI,
One Euro spatial filtering, jump suppression, single-thread 60 Hz smooth cursor interpolation,
and zero-flicker cursor movement.
"""

import sys
import math
import time
import ctypes
import threading
from typing import Tuple, Optional
import pyautogui
import numpy as np

from .filter import OneEuroFilter2D, DynamicEMAFilter
from .gesture_engine import GestureType


# Disable PyAutoGUI delay
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# Fast Win32 API cursor and mouse event controls on Windows
_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    _SetCursorPos = ctypes.windll.user32.SetCursorPos
    _mouse_event = ctypes.windll.user32.mouse_event

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    def _get_cursor_pos() -> Tuple[int, int]:
        pt = _POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
else:
    def _get_cursor_pos() -> Tuple[int, int]:
        pos = pyautogui.position()
        return pos.x, pos.y


class MouseController:
    """Controls OS mouse cursor, clicks, drag-and-drop, and scrolling with zero flickering."""

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        margin_percent: float = 0.02,
        min_alpha: float = 0.10,
        max_alpha: float = 0.85,
        deadzone_px: float = 0.0,
        click_cooldown: float = 0.15,
        scroll_sensitivity: float = 8.0,
    ):
        """Initialize MouseController."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.margin_percent = margin_percent
        self.click_cooldown = click_cooldown
        self.scroll_sensitivity = scroll_sensitivity

        # Fetch primary monitor screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # One Euro Filter 2D for smooth, zero-lag position estimation
        self.filter = DynamicEMAFilter(
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            deadzone_px=deadzone_px,
        )

        # State flags and locks
        self.enabled: bool = True
        self.is_dragging: bool = False
        self.hand_present: bool = False
        self._lock = threading.Lock()

        # Action cooldown timestamps
        self._last_left_click: float = 0.0
        self._last_right_click: float = 0.0
        self._last_double_click: float = 0.0

        # Jump suppression state
        self._last_raw_screen_x: Optional[float] = None
        self._last_raw_screen_y: Optional[float] = None

        # Physical Mouse Cooperation State
        init_x, init_y = _get_cursor_pos()
        self._last_set_x: int = init_x
        self._last_set_y: int = init_y
        self._manual_override_until: float = 0.0

        # Current & Target screen coordinates for 60Hz Single-Thread Interpolation Loop
        self.target_x: float = float(init_x)
        self.target_y: float = float(init_y)
        self.curr_x: float = float(init_x)
        self.curr_y: float = float(init_y)

        # Start 60 Hz Smooth Cursor Interpolation Background Thread
        self._running: bool = True
        self._interp_thread = threading.Thread(target=self._smooth_cursor_loop, daemon=True)
        self._interp_thread.start()

    def _set_pos_atomic(self, x: int, y: int) -> None:
        """Atomically update OS cursor position and internal tracking state."""
        clamped_x = max(0, min(self.screen_width - 1, x))
        clamped_y = max(0, min(self.screen_height - 1, y))

        if _IS_WINDOWS:
            try:
                _SetCursorPos(clamped_x, clamped_y)
            except Exception:
                pass
        else:
            pyautogui.moveTo(clamped_x, clamped_y)

        self._last_set_x = clamped_x
        self._last_set_y = clamped_y

    def _smooth_cursor_loop(self) -> None:
        """Dedicated 60 Hz thread for zero-flicker, ultra-smooth cursor movement."""
        while self._running:
            time.sleep(0.016)  # 60 Hz refresh rate (~16.6 ms)

            if not self.enabled:
                continue

            now = time.time()

            # 1. Fetch actual physical OS cursor position
            actual_x, actual_y = _get_cursor_pos()

            # 2. Check if user manually moved physical mouse/trackpad (threshold 35px)
            manual_dist = math.hypot(actual_x - self._last_set_x, actual_y - self._last_set_y)
            if manual_dist > 35.0:
                with self._lock:
                    self.curr_x = float(actual_x)
                    self.curr_y = float(actual_y)
                    self.target_x = float(actual_x)
                    self.target_y = float(actual_y)
                    self._last_set_x = actual_x
                    self._last_set_y = actual_y
                    self._manual_override_until = now + 0.20
                continue

            # 3. If manual override active or no hand present, yield to physical mouse
            if now < self._manual_override_until or not self.hand_present:
                with self._lock:
                    self.curr_x = float(actual_x)
                    self.curr_y = float(actual_y)
                    self._last_set_x = actual_x
                    self._last_set_y = actual_y
                continue

            # 4. Perform smooth lerp step towards target position (zero flickering)
            with self._lock:
                tx, ty = self.target_x, self.target_y

            # Dynamic lerp factor: smooth precision when close, fast tracking when far
            dist = math.hypot(tx - self.curr_x, ty - self.curr_y)
            lerp_factor = 0.40 if dist > 20.0 else 0.25

            self.curr_x += lerp_factor * (tx - self.curr_x)
            self.curr_y += lerp_factor * (ty - self.curr_y)

            # 5. Single thread atomic position move
            self._set_pos_atomic(int(self.curr_x), int(self.curr_y))

    def update_frame_dimensions(self, width: int, height: int) -> None:
        """Update camera frame dimensions."""
        with self._lock:
            self.frame_width = max(1, width)
            self.frame_height = max(1, height)

    def map_camera_to_screen(self, cam_x: float, cam_y: float) -> Tuple[float, float]:
        """Map camera frame coordinates to display screen resolution with jump suppression.

        Args:
            cam_x: X position in camera frame.
            cam_y: Y position in camera frame.

        Returns:
            Mapped (screen_x, screen_y) raw screen coordinates.
        """
        margin_x = self.frame_width * self.margin_percent
        margin_y = self.frame_height * self.margin_percent

        screen_x = np.interp(
            cam_x, (margin_x, self.frame_width - margin_x), (0, self.screen_width - 1)
        )
        screen_y = np.interp(
            cam_y, (margin_y, self.frame_height - margin_y), (0, self.screen_height - 1)
        )

        raw_x = float(np.clip(screen_x, 0, self.screen_width - 1))
        raw_y = float(np.clip(screen_y, 0, self.screen_height - 1))

        # Jump suppression: clamp single-frame outlier jumps (> 150px) to prevent teleporting/flickering
        if self._last_raw_screen_x is not None and self._last_raw_screen_y is not None:
            jump_dist = math.hypot(raw_x - self._last_raw_screen_x, raw_y - self._last_raw_screen_y)
            if jump_dist > 150.0:
                scale = 150.0 / jump_dist
                raw_x = self._last_raw_screen_x + (raw_x - self._last_raw_screen_x) * scale
                raw_y = self._last_raw_screen_y + (raw_y - self._last_raw_screen_y) * scale

        self._last_raw_screen_x = raw_x
        self._last_raw_screen_y = raw_y
        return raw_x, raw_y

    def reset(self) -> None:
        """Reset internal filter and jump suppression tracking."""
        self._last_raw_screen_x = None
        self._last_raw_screen_y = None
        self.filter.reset()

    def set_hand_present(self, present: bool) -> None:
        """Set whether hand landmarks are active in current frame."""
        self.hand_present = present
        if not present:
            self.reset()

    def execute_gesture(
        self,
        gesture: str,
        cam_cursor_pt: Tuple[int, int],
        scroll_delta_y: float = 0.0,
    ) -> Tuple[int, int]:
        """Process detected gesture and update target screen coordinates.

        Args:
            gesture: GestureType string.
            cam_cursor_pt: Raw camera coordinates of index finger tip.
            scroll_delta_y: Vertical scroll delta from gesture engine.

        Returns:
            Tuple[int, int] of final smoothed screen coordinates (screen_x, screen_y).
        """
        self.hand_present = True

        if not self.enabled:
            if self.is_dragging:
                self.release_drag()
            return int(self.curr_x), int(self.curr_y)

        # 1. Map raw camera point to screen coordinates with jump clamping
        raw_screen_x, raw_screen_y = self.map_camera_to_screen(
            cam_cursor_pt[0], cam_cursor_pt[1]
        )

        # 2. Apply One Euro Filter to eliminate micro-jitter
        smooth_x, smooth_y = self.filter.filter(raw_screen_x, raw_screen_y)

        with self._lock:
            self.target_x = smooth_x
            self.target_y = smooth_y

        int_x, int_y = int(self.curr_x), int(self.curr_y)
        now = time.time()

        try:
            # 3. Handle Drag & Select state transitions
            if gesture == GestureType.DRAG:
                if not self.is_dragging:
                    if _IS_WINDOWS:
                        _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    else:
                        pyautogui.mouseDown(int_x, int_y, button="left")
                    self.is_dragging = True
                return int_x, int_y
            elif self.is_dragging:
                self.release_drag()

            # 4. Handle Clicks (Click events execute without interrupting smooth lerp loop)
            if gesture == GestureType.LEFT_CLICK:
                if now - self._last_left_click > self.click_cooldown:
                    if _IS_WINDOWS:
                        _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    else:
                        pyautogui.click(int_x, int_y, button="left")
                    self._last_left_click = now

            elif gesture == GestureType.RIGHT_CLICK:
                if now - self._last_right_click > self.click_cooldown * 1.1:
                    if _IS_WINDOWS:
                        _mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        _mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                    else:
                        pyautogui.click(int_x, int_y, button="right")
                    self._last_right_click = now

            elif gesture == GestureType.DOUBLE_CLICK:
                if now - self._last_double_click > 0.55:
                    if _IS_WINDOWS:
                        _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        time.sleep(0.04)
                        _mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    else:
                        pyautogui.doubleClick(int_x, int_y)
                    self._last_double_click = now

            elif gesture == GestureType.SCROLL:
                if abs(scroll_delta_y) > 1.5:
                    scroll_clicks = int(-scroll_delta_y * self.scroll_sensitivity / 10.0)
                    if scroll_clicks != 0:
                        if _IS_WINDOWS:
                            _mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(scroll_clicks * 120), 0)
                        else:
                            pyautogui.scroll(scroll_clicks * 20)

        except Exception as e:
            print(f"[ERROR] MouseController action failed: {e}")

        return int_x, int_y

    def release_drag(self) -> None:
        """Safety release for active drag-and-drop state."""
        if self.is_dragging:
            try:
                if _IS_WINDOWS:
                    _mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                else:
                    pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self.is_dragging = False

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        min_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
        deadzone_px: Optional[float] = None,
        margin_percent: Optional[float] = None,
        scroll_sensitivity: Optional[float] = None,
        click_cooldown: Optional[float] = None,
    ) -> None:
        """Update controller parameters dynamically from UI controls."""
        with self._lock:
            if enabled is not None:
                self.enabled = enabled
                if not enabled:
                    self.release_drag()
            if margin_percent is not None:
                self.margin_percent = max(0.005, min(0.30, margin_percent))
            if scroll_sensitivity is not None:
                self.scroll_sensitivity = max(1.0, scroll_sensitivity)
            if click_cooldown is not None:
                self.click_cooldown = max(0.05, click_cooldown)

            self.filter.update_params(
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                deadzone_px=deadzone_px,
            )

    def close(self) -> None:
        """Stop background interpolation thread."""
        self._running = False
        self.release_drag()
