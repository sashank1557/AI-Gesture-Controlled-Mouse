"""Dedicated Gestures Helper Module.

Provides gesture classification, 2D/3D Euclidean landmark distance math,
and finger orientation state checks for virtual mouse automation.
"""

from core.gesture_engine import GestureEngine, FingerState, GestureType

__all__ = ["GestureEngine", "FingerState", "GestureType"]
