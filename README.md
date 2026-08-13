# Production-Grade AI Gesture-Controlled Virtual Mouse 🖱️⚡

An advanced, cross-platform Python application that uses computer vision (OpenCV + MediaPipe) to track 21 3D hand landmarks in real time via a webcam, translating human hand gestures into low-latency, zero-jitter system mouse controls (cursor movement, left click, right click, double click, drag-and-drop, and vertical scrolling).

---

## 🌟 Architecture & Key Features

```
gesture-virtual-mouse/
├── assets/                  # Icons and hand_landmarker.task model
├── core/
│   ├── __init__.py          # Core package exports
│   ├── filter.py            # One Euro Filter (1€), Deadzone & 2D Kalman filter algorithms
│   ├── gesture_engine.py    # Finger state, 3D Euclidean math & simplified gesture state machine
│   └── mouse_controller.py  # PyAutoGUI wrappers, 60Hz smooth cursor interpolation & win32 safety
├── ui/
│   ├── __init__.py          # UI package exports
│   └── app_window.py        # CustomTkinter GUI dashboard & live camera preview
├── tests/
│   └── test_gestures.py     # Unit test suite for landmark math and filters
├── main.py                  # Project entry point & frame loop orchestration
├── gestures.py              # Dedicated gesture helper module alias
├── requirements.txt         # Production dependency lockfile
└── README.md                # System documentation
```

---

## 🖐️ Updated Simpler Gesture Mapping Guide

| Action | Gesture Description | Landmark Trigger Details |
| :--- | :--- | :--- |
| **Cursor Move** | Index Finger extended UP | Index Finger Tip (Landmark 8) mapped to screen resolution |
| **Left Click** | Touch Thumb tip to Index tip | Pinch distance $(4 \rightarrow 8) < \text{Threshold}$ (default $< 35\text{px}$) |
| **Right Click** | Touch Thumb tip to Middle tip | Pinch distance $(4 \rightarrow 12) < \text{Threshold}$ |
| **Double Click** | Touch Index tip to Middle tip | Pinch distance $(8 \rightarrow 12) < \text{Threshold}$ |
| **Drag & Select** | Closed Fist (holds `mouseDown`) / Open Hand (releases `mouseUp`) | All 4 main fingers folded holds left click down; Open hand releases |
| **Vertical Scroll** | Open Hand up / down movement | Relative vertical movement of open hand controls page scroll speed |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 - 3.13
- OpenCV compatible webcam

### 1. Clone & Set Up Virtual Environment
```powershell
# Create project folder and virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🎮 Execution Instructions

### Launch Main Application
```powershell
python main.py
```

### Run Automated Unit Tests
```powershell
python -m unittest discover -s tests -p "test_*.py"
```
