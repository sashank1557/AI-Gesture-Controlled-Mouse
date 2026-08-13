"""Unit tests for gesture engine math, filters, finger orientation, and new simpler gesture bindings."""

import math
import time
import unittest
from typing import List, Tuple

from core.filter import DynamicEMAFilter, DeadzoneFilter, OneEuroFilter2D
from core.gesture_engine import GestureEngine, FingerState, GestureType
from core.mouse_controller import MouseController


class TestFilters(unittest.TestCase):
    """Test One Euro Filter, dynamic EMA, and Deadzone filtering implementations."""

    def test_deadzone_filter_suppression(self):
        """Verify DeadzoneFilter suppresses coordinate changes below deadzone_px."""
        f = DeadzoneFilter(deadzone_px=5.0)

        # Initial point
        x1, y1 = f.filter(100.0, 100.0)
        self.assertEqual((x1, y1), (100.0, 100.0))

        # Delta = 3.0 (< 5.0) -> Should return original point
        x2, y2 = f.filter(103.0, 100.0)
        self.assertEqual((x2, y2), (100.0, 100.0))

        # Delta = 10.0 (> 5.0) -> Should update to new point
        x3, y3 = f.filter(110.0, 100.0)
        self.assertEqual((x3, y3), (110.0, 100.0))

    def test_one_euro_filter_responsiveness(self):
        """Verify OneEuroFilter2D smooth response and speed scaling."""
        f = OneEuroFilter2D(min_cutoff=0.8, beta=0.008)
        t0 = time.time()

        # Initial point
        x1, y1 = f.filter(100.0, 100.0, t0)
        self.assertEqual((x1, y1), (100.0, 100.0))

        # Small movement (heavy smoothing)
        x2, y2 = f.filter(101.0, 100.0, t0 + 0.033)
        self.assertLess(x2 - 100.0, 0.5)

        # Large rapid movement (high responsiveness)
        x3, y3 = f.filter(200.0, 100.0, t0 + 0.066)
        self.assertGreater(x3 - x2, 20.0)


class TestGestureEngine(unittest.TestCase):
    """Test distance geometry and updated simpler gesture classification state machine."""

    def setUp(self):
        self.engine = GestureEngine(click_threshold_px=35.0, drag_threshold_px=30.0)

    def _create_mock_landmarks(self) -> List[Tuple[int, int]]:
        """Helper to create baseline open hand landmark positions."""
        landmarks = [(0, 0)] * 21
        landmarks[0] = (200, 400)  # Wrist

        # Thumb
        landmarks[1] = (180, 360)
        landmarks[2] = (160, 320)
        landmarks[3] = (140, 280)
        landmarks[4] = (120, 240)  # Thumb tip

        # Index
        landmarks[5] = (200, 300)
        landmarks[6] = (200, 250)
        landmarks[7] = (200, 200)
        landmarks[8] = (200, 150)  # Index tip

        # Middle
        landmarks[9] = (240, 300)
        landmarks[10] = (240, 250)
        landmarks[11] = (240, 200)
        landmarks[12] = (240, 150)  # Middle tip

        # Ring
        landmarks[13] = (280, 300)
        landmarks[14] = (280, 250)
        landmarks[15] = (280, 200)
        landmarks[16] = (280, 150)  # Ring tip

        # Pinky
        landmarks[17] = (320, 300)
        landmarks[18] = (320, 260)
        landmarks[19] = (320, 220)
        landmarks[20] = (320, 180)  # Pinky tip

        return landmarks

    def test_euclidean_distance(self):
        """Verify 2D Euclidean distance calculation."""
        p1 = (0, 0)
        p2 = (3, 4)
        dist = GestureEngine.euclidean_distance_2d(p1, p2)
        self.assertAlmostEqual(dist, 5.0)

    def test_left_click_thumb_index_pinch(self):
        """Verify Left Click: Touch Thumb tip (4) to Index tip (8)."""
        landmarks = self._create_mock_landmarks()
        # Fold middle, ring, pinky down
        landmarks[12] = (240, 350)
        landmarks[16] = (280, 350)
        landmarks[20] = (320, 350)
        # Touch Thumb tip (4) to Index tip (8)
        landmarks[4] = (200, 155)
        landmarks[8] = (200, 150)

        res = self.engine.process_gestures(landmarks)
        self.assertEqual(res["gesture"], GestureType.LEFT_CLICK)

    def test_right_click_thumb_middle_pinch(self):
        """Verify Right Click: Touch Thumb tip (4) to Middle tip (12)."""
        landmarks = self._create_mock_landmarks()
        # Fold ring & pinky down
        landmarks[16] = (280, 350)
        landmarks[20] = (320, 350)
        # Move index tip far away
        landmarks[8] = (100, 150)
        # Touch Thumb tip (4) to Middle tip (12)
        landmarks[4] = (240, 155)
        landmarks[12] = (240, 150)

        res = self.engine.process_gestures(landmarks)
        self.assertEqual(res["gesture"], GestureType.RIGHT_CLICK)

    def test_double_click_index_middle_pinch(self):
        """Verify Double Click: Touch Index tip (8) to Middle tip (12)."""
        landmarks = self._create_mock_landmarks()
        # Fold ring & pinky down
        landmarks[16] = (280, 350)
        landmarks[20] = (320, 350)
        # Move thumb far away
        landmarks[4] = (50, 350)
        # Touch Index tip (8) to Middle tip (12)
        landmarks[8] = (220, 150)
        landmarks[12] = (225, 150)

        res = self.engine.process_gestures(landmarks)
        self.assertEqual(res["gesture"], GestureType.DOUBLE_CLICK)

    def test_drag_closed_fist(self):
        """Verify Drag & Select: Closed Fist (all fingers folded down)."""
        landmarks = self._create_mock_landmarks()
        # Fold all fingers down towards palm
        landmarks[4] = (150, 350)   # Thumb
        landmarks[8] = (200, 350)   # Index
        landmarks[12] = (240, 350)  # Middle
        landmarks[16] = (280, 350)  # Ring
        landmarks[20] = (320, 350)  # Pinky

        res = self.engine.process_gestures(landmarks)
        self.assertEqual(res["gesture"], GestureType.DRAG)

    def test_scroll_open_hand(self):
        """Verify Vertical Scroll: Open Hand (all main fingers extended UP)."""
        landmarks = self._create_mock_landmarks()
        # Open hand (Index, Middle, Ring, Pinky extended UP)
        res = self.engine.process_gestures(landmarks)
        self.assertEqual(res["gesture"], GestureType.SCROLL)


class TestMouseController(unittest.TestCase):
    """Test screen coordinate mapping logic."""

    def test_coordinate_mapping_margins(self):
        """Verify camera to screen coordinate interpolation with margin padding."""
        mc = MouseController(frame_width=640, frame_height=480, margin_percent=0.10)
        sx, sy = mc.map_camera_to_screen(320, 240)
        self.assertAlmostEqual(sx, mc.screen_width / 2.0, delta=2.0)
        self.assertAlmostEqual(sy, mc.screen_height / 2.0, delta=2.0)

        # Left-top margin boundary -> 0, 0
        mc.reset()
        sx_top, sy_top = mc.map_camera_to_screen(64, 48)
        self.assertEqual(sx_top, 0)
        self.assertEqual(sy_top, 0)
        mc.close()


if __name__ == "__main__":
    unittest.main()
