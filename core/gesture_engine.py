"""Gesture Engine Module.

Calculates 2D/3D hand landmark geometry, finger extension states, pinch distances,
and executes an ultra-accurate state machine for mapping human hand gestures to virtual mouse actions.

Updated High-Accuracy Gesture Priority:
1. Left Click: Touch Thumb tip (4) to Index tip (8)
2. Right Click: Touch Thumb tip (4) to Middle tip (12)
3. Double Click: Touch Index tip (8) to Middle tip (12)
4. Drag & Select: Closed Fist (all 4 main fingers folded) holds mouseDown
5. Scroll: Open Hand (3+ main fingers extended UP) vertical movement
6. Cursor Move: Default active hand tracking Index tip (8)
"""

from dataclasses import dataclass
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class FingerState:
    """Represents extension status of all 5 fingers."""
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    def to_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Convert finger state to tuple (thumb, index, middle, ring, pinky)."""
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def count_up(self) -> int:
        """Return total number of extended fingers."""
        return sum(self.to_tuple())


class GestureType:
    """Enumeration of supported hand gestures."""
    NONE = "NONE"
    MOVE = "MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAG = "DRAG"
    SCROLL = "SCROLL"


class GestureEngine:
    """Engine for landmark math and gesture recognition state machine."""

    # Landmark Indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(
        self,
        click_threshold_px: float = 50.0,  # 50px threshold for effortless & 100% reliable pinch clicks
        drag_threshold_px: float = 35.0,
        scroll_deadzone_px: float = 5.0,
    ):
        """Initialize GestureEngine."""
        self.click_threshold_px = click_threshold_px
        self.drag_threshold_px = drag_threshold_px
        self.scroll_deadzone_px = scroll_deadzone_px

        self._last_scroll_y: Optional[float] = None
        self._is_dragging: bool = False

    @staticmethod
    def euclidean_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate 2D Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def euclidean_distance_3d(
        p1: Tuple[float, float, float], p2: Tuple[float, float, float]
    ) -> float:
        """Calculate 3D Euclidean distance between two points."""
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )

    def fingers_up(
        self, landmarks_px: List[Tuple[int, int]], hand_label: str = "Right"
    ) -> FingerState:
        """Determine which fingers are extended based on landmark geometry.

        Args:
            landmarks_px: List of 21 (x, y) landmark pixel coordinates.
            hand_label: 'Right' or 'Left' handedness.

        Returns:
            FingerState dataclass.
        """
        if len(landmarks_px) < 21:
            return FingerState()

        wrist = landmarks_px[self.WRIST]

        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            dist_wrist_tip = self.euclidean_distance_2d(wrist, landmarks_px[tip_idx])
            dist_wrist_pip = self.euclidean_distance_2d(wrist, landmarks_px[pip_idx])
            y_extended = landmarks_px[tip_idx][1] < (landmarks_px[pip_idx][1] + 15)
            dist_extended = dist_wrist_tip > (dist_wrist_pip * 1.02)
            return y_extended or dist_extended

        index_up = is_extended(self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        middle_up = is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_up = is_extended(self.RING_TIP, self.RING_PIP, self.RING_MCP)
        pinky_up = is_extended(self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)

        pinky_mcp = landmarks_px[self.PINKY_MCP]
        dist_thumb_tip = self.euclidean_distance_2d(landmarks_px[self.THUMB_TIP], pinky_mcp)
        dist_thumb_ip = self.euclidean_distance_2d(landmarks_px[self.THUMB_IP], pinky_mcp)
        thumb_up = dist_thumb_tip > (dist_thumb_ip * 1.02)

        return FingerState(
            thumb=thumb_up,
            index=index_up,
            middle=middle_up,
            ring=ring_up,
            pinky=pinky_up,
        )

    def process_gestures(
        self, landmarks_px: List[Tuple[int, int]], hand_label: str = "Right"
    ) -> Dict[str, Any]:
        """Classify hand landmarks into system gesture action with high accuracy & top-priority pinches.

        Args:
            landmarks_px: List of 21 (x, y) pixel coordinates of hand landmarks.
            hand_label: Hand label ('Right' or 'Left').

        Returns:
            Dictionary containing gesture metadata.
        """
        if len(landmarks_px) < 21:
            return {
                "gesture": GestureType.NONE,
                "cursor_pt": (0, 0),
                "pinch_dist": 999.0,
                "scroll_delta_y": 0.0,
                "finger_state": FingerState(),
            }

        finger_state = self.fingers_up(landmarks_px, hand_label)
        thumb_tip = landmarks_px[self.THUMB_TIP]
        index_tip = landmarks_px[self.INDEX_TIP]
        middle_tip = landmarks_px[self.MIDDLE_TIP]
        ring_tip = landmarks_px[self.RING_TIP]
        pinky_tip = landmarks_px[self.PINKY_TIP]

        # Calculate 2D Euclidean distances between key fingertips
        dist_thumb_index = self.euclidean_distance_2d(thumb_tip, index_tip)
        dist_thumb_middle = self.euclidean_distance_2d(thumb_tip, middle_tip)
        dist_index_middle = self.euclidean_distance_2d(index_tip, middle_tip)

        active_gesture = GestureType.NONE
        scroll_delta_y = 0.0

        # Check main 4 fingers status (Index, Middle, Ring, Pinky)
        non_thumb_up_count = sum([finger_state.index, finger_state.middle, finger_state.ring, finger_state.pinky])

        # TOP PRIORITY 1: Pinch Clicks
        # 1A. LEFT CLICK: Touch Thumb tip (4) to Index tip (8)
        if dist_thumb_index < self.click_threshold_px and dist_thumb_index <= dist_thumb_middle:
            active_gesture = GestureType.LEFT_CLICK
            self._last_scroll_y = None

        # 1B. RIGHT CLICK: Touch Thumb tip (4) to Middle tip (12)
        elif dist_thumb_middle < self.click_threshold_px and dist_thumb_middle < dist_thumb_index:
            active_gesture = GestureType.RIGHT_CLICK
            self._last_scroll_y = None

        # 1C. DOUBLE CLICK: Tight pinch between Index tip (8) and Middle tip (12)
        # Requires Index & Middle tips touching closely (< 28.0px) while Thumb is separated (> 45.0px)
        elif dist_index_middle < 28.0 and dist_thumb_index > 45.0 and dist_thumb_middle > 45.0:
            active_gesture = GestureType.DOUBLE_CLICK
            self._last_scroll_y = None

        # PRIORITY 2: DRAG & SELECT (Closed Fist: all 4 main fingers folded down towards palm)
        elif non_thumb_up_count == 0:
            active_gesture = GestureType.DRAG
            self._is_dragging = True
            self._last_scroll_y = None

        # PRIORITY 3: SCROLL (Open Hand: 3+ main fingers extended UP)
        elif non_thumb_up_count >= 3 and finger_state.index and finger_state.middle and finger_state.ring:
            active_gesture = GestureType.SCROLL
            current_y = (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) / 4.0
            if self._last_scroll_y is not None:
                scroll_delta_y = current_y - self._last_scroll_y
            self._last_scroll_y = current_y
            self._is_dragging = False

        # PRIORITY 4: CURSOR MOVEMENT (Default Active Tracking for any hand in frame)
        else:
            active_gesture = GestureType.MOVE
            self._last_scroll_y = None
            self._is_dragging = False

        return {
            "gesture": active_gesture,
            "cursor_pt": index_tip,
            "pinch_dist": min(dist_thumb_index, dist_thumb_middle, dist_index_middle),
            "scroll_delta_y": scroll_delta_y,
            "finger_state": finger_state,
        }

    def update_thresholds(
        self,
        click_threshold_px: Optional[float] = None,
        drag_threshold_px: Optional[float] = None,
        scroll_deadzone_px: Optional[float] = None,
    ) -> None:
        """Dynamically adjust gesture detection thresholds."""
        if click_threshold_px is not None:
            self.click_threshold_px = max(10.0, click_threshold_px)
        if drag_threshold_px is not None:
            self.drag_threshold_px = max(10.0, drag_threshold_px)
        if scroll_deadzone_px is not None:
            self.scroll_deadzone_px = max(1.0, scroll_deadzone_px)
