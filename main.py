"""Main Entry Point for AI Gesture-Controlled Virtual Mouse.

Orchestrates camera frame acquisition, MediaPipe hand tracking, gesture interpretation,
One Euro spatial filtering, 60 Hz smooth cursor interpolation, and CustomTkinter GUI dashboard execution.
"""

import os
import sys
import time
import urllib.request
import threading
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import mediapipe as mp

from core.gesture_engine import GestureEngine, GestureType
from core.mouse_controller import MouseController
from ui.app_window import AppWindow


# Hand skeleton bone connection pairs (MediaPipe 21 landmarks index schema)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class HandTracker:
    """Universal Hand Tracker supporting both MediaPipe Tasks API and Legacy Solutions API."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = os.path.join("assets", "hand_landmarker.task")

    def __init__(self):
        """Initialize Hand Tracker backend."""
        self.use_tasks_api = True
        self.landmarker = None
        self.legacy_hands = None

        # Check if legacy mp.solutions is available
        if hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions"), "hands"):
            try:
                self.legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,  # Fast model
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.use_tasks_api = False
                print("[INFO] Initialized MediaPipe Legacy Solutions Hands API (Fast Mode).")
                return
            except Exception as e:
                print(f"[INFO] Legacy Solutions fallback unavailable ({e}), using Tasks API...")

        # Initialize modern MediaPipe Tasks API
        self._init_tasks_api()

    def _init_tasks_api(self) -> None:
        """Download model if missing and initialize HandLandmarker."""
        os.makedirs("assets", exist_ok=True)
        if not os.path.exists(self.MODEL_PATH):
            print(f"[INFO] Downloading MediaPipe Hand Landmarker model to {self.MODEL_PATH}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("[INFO] Model download complete.")

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.15,
            min_hand_presence_confidence=0.15,
            min_tracking_confidence=0.15,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("[INFO] Initialized MediaPipe Tasks HandLandmarker API successfully.")

    def process(self, frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int]], str]:
        """Detect hand landmarks and return pixel coordinates and handedness label.

        Args:
            frame_bgr: OpenCV image matrix (BGR).

        Returns:
            Tuple of (list of 21 (x, y) pixel coordinates, hand_label string).
        """
        h, w, _ = frame_bgr.shape
        landmarks_px: List[Tuple[int, int]] = []
        hand_label: str = "Right"

        if not self.use_tasks_api and self.legacy_hands is not None:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.legacy_hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
                if results.multi_handedness:
                    hand_label = results.multi_handedness[0].classification[0].label
            return landmarks_px, hand_label

        # Pad frame to 1:1 square ROI for MediaPipe TFLite model compatibility
        size = max(h, w)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = frame_bgr

        rgb_sq = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        mp_img_sq = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_sq))
        result = self.landmarker.detect(mp_img_sq)

        if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_lms = result.hand_landmarks[0]
            landmarks_px = [(int(lm.x * size - x_off), int(lm.y * size - y_off)) for lm in hand_lms]
            if result.handedness and len(result.handedness) > 0:
                hand_label = result.handedness[0][0].category_name

        return landmarks_px, hand_label

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
        """Render hand joints and bone connections on OpenCV image frame."""
        if len(landmarks_px) < 21:
            return

        # Draw bone connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = landmarks_px[start_idx]
            pt2 = landmarks_px[end_idx]
            cv2.line(frame, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, (x, y) in enumerate(landmarks_px):
            color = (255, 0, 0) if idx in (4, 8, 12, 16, 20) else (0, 255, 255)
            radius = 5 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release underlying landmarker resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass


class VirtualMouseApp:
    """Application Controller managing thread coordination between CV and GUI."""

    def __init__(self, camera_id: int = 0):
        """Initialize VirtualMouseApp components.

        Args:
            camera_id: OpenCV video capture device index (default: 0).
        """
        self.camera_id = camera_id
        self.running: bool = False
        self._camera_thread: Optional[threading.Thread] = None

        # Core Engines
        self.gesture_engine = GestureEngine(
            click_threshold_px=45.0,
            drag_threshold_px=35.0,
            scroll_deadzone_px=6.0,
        )
        self.mouse_controller = MouseController(
            margin_percent=0.02,
            min_alpha=0.10,
            max_alpha=0.85,
            deadzone_px=0.0,
            scroll_sensitivity=8.0,
        )

        # Hand Tracker Abstraction Layer
        self.tracker = HandTracker()

        # GUI Window Dashboard
        self.gui = AppWindow(
            on_setting_change=self._on_gui_setting_changed,
            on_close_callback=self._stop_camera_loop,
        )

        # Telemetry
        self.fps: float = 0.0
        self._last_time: float = time.time()
        self._last_telemetry_update: float = 0.0

    def _on_gui_setting_changed(self, key: str, value: Any) -> None:
        """Callback triggered by CustomTkinter GUI controls to dynamically alter parameters."""
        if key == "enabled":
            self.mouse_controller.update_settings(enabled=bool(value))
        elif key in ("min_alpha", "max_alpha", "deadzone_px", "margin_percent", "scroll_sensitivity"):
            self.mouse_controller.update_settings(**{key: value})
        elif key == "click_threshold_px":
            self.gesture_engine.update_thresholds(click_threshold_px=value, drag_threshold_px=value)

    def start(self) -> None:
        """Start camera background processing loop and run GUI event loop."""
        self.running = True
        self._camera_thread = threading.Thread(target=self._run_camera_loop, daemon=True)
        self._camera_thread.start()

        # Run CustomTkinter mainloop on main thread
        self.gui.mainloop()

    def _stop_camera_loop(self) -> None:
        """Clean shutdown of camera capture loop."""
        self.running = False
        self.mouse_controller.close()
        if hasattr(self, "tracker"):
            self.tracker.close()

    def _configure_camera(self, camera_id: int) -> cv2.VideoCapture:
        """Configure OpenCV VideoCapture with natural lighting and MJPG streaming."""
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)

        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)

        # MJPG format for fast frame capture
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        return cap

    def _run_camera_loop(self) -> None:
        """Background thread loop for video capture, hand tracking, and mouse control."""
        cap = self._configure_camera(self.camera_id)

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Mirror frame horizontally so movements feel natural
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            self.mouse_controller.update_frame_dimensions(w, h)

            # Process hand landmarks via HandTracker
            landmarks_px, hand_label = self.tracker.process(frame)

            active_gesture = GestureType.NONE
            cursor_target = (w // 2, h // 2)

            if len(landmarks_px) == 21:
                # Evaluate gestures
                g_res = self.gesture_engine.process_gestures(landmarks_px, hand_label)
                active_gesture = g_res["gesture"]
                cursor_target = g_res["cursor_pt"]
                scroll_delta_y = g_res["scroll_delta_y"]

                # Execute mouse action via MouseController (One Euro Filter + 60Hz Smooth Lerp)
                screen_x, screen_y = self.mouse_controller.execute_gesture(
                    gesture=active_gesture,
                    cam_cursor_pt=cursor_target,
                    scroll_delta_y=scroll_delta_y,
                )

                # Draw hand skeleton overlay
                HandTracker.draw_skeleton(frame, landmarks_px)
            else:
                self.mouse_controller.set_hand_present(False)

            # Render Visual Debugging Overlays
            self._draw_debug_overlays(frame, landmarks_px, active_gesture, cursor_target, w, h)

            # FPS Telemetry Calculation
            now = time.time()
            dt = now - self._last_time
            self._last_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

            # Update CustomTkinter GUI non-blockingly
            self.gui.update_video_frame(frame)

            # Update text labels throttled to ~10 Hz to prevent UI overhead
            if now - self._last_telemetry_update > 0.10:
                self.gui.after(
                    0,
                    self.gui.update_telemetry,
                    self.fps,
                    active_gesture,
                    self.mouse_controller.enabled,
                )
                self._last_telemetry_update = now

        cap.release()

    def _draw_debug_overlays(
        self,
        frame: np.ndarray,
        landmarks_px: list,
        gesture: str,
        cursor_target: tuple,
        w: int,
        h: int,
    ) -> None:
        """Render HUD overlays including active margins, target reticle, and gesture badge."""
        # 1. Corner Boundary Margin Box
        margin_pct = self.mouse_controller.margin_percent
        mx, my = int(w * margin_pct), int(h * margin_pct)
        cv2.rectangle(frame, (mx, my), (w - mx, h - my), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "ACTIVE BOUNDARY AREA",
            (mx + 8, my + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Cursor Target Reticle on Index Tip
        if cursor_target and cursor_target != (0, 0):
            cx, cy = cursor_target
            cv2.circle(frame, (cx, cy), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Active Gesture Overlay Badge
        badge_color = (0, 255, 0)
        if gesture == GestureType.LEFT_CLICK:
            badge_color = (255, 100, 0)
        elif gesture == GestureType.RIGHT_CLICK:
            badge_color = (0, 100, 255)
        elif gesture == GestureType.DOUBLE_CLICK:
            badge_color = (255, 0, 255)
        elif gesture == GestureType.DRAG:
            badge_color = (0, 255, 255)
        elif gesture == GestureType.SCROLL:
            badge_color = (255, 255, 0)

        cv2.putText(
            frame,
            f"GESTURE: {gesture}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            badge_color,
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    app = VirtualMouseApp(camera_id=0)
    app.start()
