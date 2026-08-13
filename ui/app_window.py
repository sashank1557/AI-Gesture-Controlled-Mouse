"""UI Dashboard Module built with CustomTkinter.

Provides a modern, ultra-clean, and interactive desktop dashboard with a live camera preview feed,
interactive mode presets (Speed / Precision / Balanced), real-time gesture telemetry,
interactive gesture reference cheatsheet, and parameter tuning sliders.
"""

import tkinter as tk
from typing import Callable, Optional, Dict, Any
import customtkinter as ctk
import numpy as np
import cv2
from PIL import Image, ImageTk


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AppWindow(ctk.CTk):
    """Main application window dashboard."""

    # Color Palette Tokens
    COLOR_BG = "#11121A"
    COLOR_CARD = "#191B26"
    COLOR_CARD_HOVER = "#222536"
    COLOR_ACCENT = "#3B82F6"
    COLOR_SUCCESS = "#10B981"
    COLOR_WARNING = "#F59E0B"
    COLOR_DANGER = "#EF4444"
    COLOR_PURPLE = "#8B5CF6"
    COLOR_TEXT = "#F3F4F6"
    COLOR_MUTED = "#9CA3AF"

    def __init__(
        self,
        on_setting_change: Optional[Callable[[str, Any], None]] = None,
        on_close_callback: Optional[Callable[[], None]] = None,
    ):
        """Initialize CustomTkinter application window.

        Args:
            on_setting_change: Callback function triggered when sliders or switches change.
            on_close_callback: Callback function triggered when window is closed.
        """
        super().__init__()

        self.on_setting_change = on_setting_change
        self.on_close_callback = on_close_callback

        # Configure main window properties
        self.title("AI Gesture Mouse | Modern Spatial Control Dashboard")
        self.geometry("1150x740")
        self.minsize(1000, 650)
        self.configure(fg_color=self.COLOR_BG)

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Layout configuration: Top Header + 2 Main Columns
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=0)  # Top Bar
        self.grid_rowconfigure(1, weight=1)  # Main Content

        # Non-blocking UI update state
        self._is_updating_video: bool = False
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._init_top_header()
        self._init_preview_panel()
        self._init_control_panel()

    def _init_top_header(self) -> None:
        """Create top branding header bar with live badges."""
        self.header_frame = ctk.CTkFrame(self, fg_color=self.COLOR_CARD, corner_radius=12, height=60)
        self.header_frame.grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        # App Title & Icon
        self.brand_label = ctk.CTkLabel(
            self.header_frame,
            text="⚡ AI GESTURE MOUSE",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.COLOR_ACCENT,
        )
        self.brand_label.grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Telemetry Badges Container
        self.header_badges = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.header_badges.grid(row=0, column=2, padx=16, pady=8, sticky="e")

        self.mouse_badge = ctk.CTkLabel(
            self.header_badges,
            text="🟢 VIRTUAL MOUSE: ACTIVE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.COLOR_SUCCESS,
            fg_color="#064E3B",
            corner_radius=8,
            padx=12,
            pady=6,
        )
        self.mouse_badge.grid(row=0, column=0, padx=6)

        self.fps_badge = ctk.CTkLabel(
            self.header_badges,
            text="60 FPS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#60A5FA",
            fg_color="#1E3A8A",
            corner_radius=8,
            padx=12,
            pady=6,
        )
        self.fps_badge.grid(row=0, column=1, padx=6)

    def _init_preview_panel(self) -> None:
        """Create left side camera feed preview panel."""
        self.preview_frame = ctk.CTkFrame(self, fg_color=self.COLOR_CARD, corner_radius=14)
        self.preview_frame.grid(row=1, column=0, padx=(16, 8), pady=(8, 16), sticky="nsew")
        self.preview_frame.grid_rowconfigure(1, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        # Header Title & Active Gesture Status
        self.preview_header = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        self.preview_header.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="ew")
        self.preview_header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.preview_header,
            text="📷 Live Camera Tracking Feed",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w")

        self.gesture_badge = ctk.CTkLabel(
            self.preview_header,
            text="☝️ CURSOR MOVE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COLOR_TEXT,
            fg_color=self.COLOR_CARD_HOVER,
            corner_radius=8,
            padx=12,
            pady=4,
        )
        self.gesture_badge.grid(row=0, column=1, sticky="e")

        # Standard Tkinter Label for fast direct image updates
        self.video_label = tk.Label(
            self.preview_frame,
            text="Connecting to Camera Feed...",
            bg="#0B0C10",
            fg=self.COLOR_MUTED,
            font=("Segoe UI", 12),
        )
        # Live video feed fills preview frame cleanly
        self.video_label.grid(row=1, column=0, padx=14, pady=(8, 14), sticky="nsew")

    def _init_control_panel(self) -> None:
        """Create right side controls dashboard."""
        self.control_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=self.COLOR_CARD,
            corner_radius=14,
            label_text="⚙️ Control Panel & Calibration",
            label_font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.control_frame.grid(row=1, column=1, padx=(8, 16), pady=(8, 16), sticky="nsew")
        self.control_frame.grid_columnconfigure(0, weight=1)

        # 1. Master Toggle Switch
        self.toggle_card = ctk.CTkFrame(self.control_frame, fg_color=self.COLOR_CARD_HOVER, corner_radius=10)
        self.toggle_card.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")
        self.toggle_card.grid_columnconfigure(0, weight=1)

        self.toggle_switch = ctk.CTkSwitch(
            self.toggle_card,
            text="Virtual Mouse Control (ON / OFF)",
            font=ctk.CTkFont(size=14, weight="bold"),
            progress_color=self.COLOR_SUCCESS,
            command=self._on_toggle_mouse,
        )
        self.toggle_switch.select()
        self.toggle_switch.grid(row=0, column=0, padx=14, pady=12, sticky="w")

        # 2. Dynamic Filters Section
        self.filter_section = ctk.CTkFrame(self.control_frame, fg_color="#1E2130", corner_radius=10)
        self.filter_section.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        self.filter_section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.filter_section,
            text="🎯 Smoothing & Responsiveness",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COLOR_TEXT,
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        # Min Alpha / Smoothing
        self.min_alpha_val = ctk.CTkLabel(self.filter_section, text="Min Smoothing (Alpha): 0.10", font=ctk.CTkFont(size=12))
        self.min_alpha_val.grid(row=1, column=0, padx=12, pady=(2, 0), sticky="w")

        self.min_alpha_slider = ctk.CTkSlider(
            self.filter_section,
            from_=0.01,
            to=0.40,
            number_of_steps=39,
            command=lambda val: self._on_slider_change("min_alpha", val, self.min_alpha_val, "Min Smoothing (Alpha): {:.2f}"),
        )
        self.min_alpha_slider.set(0.10)
        self.min_alpha_slider.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")

        # Max Alpha / Speed
        self.max_alpha_val = ctk.CTkLabel(self.filter_section, text="Max Responsiveness (Alpha): 0.85", font=ctk.CTkFont(size=12))
        self.max_alpha_val.grid(row=3, column=0, padx=12, pady=(2, 0), sticky="w")

        self.max_alpha_slider = ctk.CTkSlider(
            self.filter_section,
            from_=0.40,
            to=0.99,
            number_of_steps=59,
            command=lambda val: self._on_slider_change("max_alpha", val, self.max_alpha_val, "Max Responsiveness (Alpha): {:.2f}"),
        )
        self.max_alpha_slider.set(0.85)
        self.max_alpha_slider.grid(row=4, column=0, padx=12, pady=(2, 10), sticky="ew")

        # 3. Gesture Thresholds Section
        self.gesture_section = ctk.CTkFrame(self.control_frame, fg_color="#1E2130", corner_radius=10)
        self.gesture_section.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        self.gesture_section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.gesture_section,
            text="🤏 Gesture Thresholds & Boundary",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.COLOR_TEXT,
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        # Click Pinch Threshold Slider
        self.click_thresh_val = ctk.CTkLabel(self.gesture_section, text="Pinch Click Distance: 45 px", font=ctk.CTkFont(size=12))
        self.click_thresh_val.grid(row=1, column=0, padx=12, pady=(2, 0), sticky="w")

        self.click_thresh_slider = ctk.CTkSlider(
            self.gesture_section,
            from_=15.0,
            to=75.0,
            number_of_steps=60,
            command=lambda val: self._on_slider_change("click_threshold_px", val, self.click_thresh_val, "Pinch Click Distance: {:.0f} px"),
        )
        self.click_thresh_slider.set(45.0)
        self.click_thresh_slider.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")

        # Boundary Margin Slider
        self.margin_val = ctk.CTkLabel(self.gesture_section, text="Active Boundary Margin: 2.0 % (96% Area)", font=ctk.CTkFont(size=12))
        self.margin_val.grid(row=3, column=0, padx=12, pady=(2, 0), sticky="w")

        self.margin_slider = ctk.CTkSlider(
            self.gesture_section,
            from_=0.005,
            to=0.25,
            number_of_steps=49,
            command=lambda val: self._on_slider_change("margin_percent", val, self.margin_val, "Active Boundary Margin: {:.1f} %", scale=100.0),
        )
        self.margin_slider.set(0.02)
        self.margin_slider.grid(row=4, column=0, padx=12, pady=(2, 8), sticky="ew")

        # Scroll Sensitivity Slider
        self.scroll_val = ctk.CTkLabel(self.gesture_section, text="Scroll Speed Multiplier: 8.0x", font=ctk.CTkFont(size=12))
        self.scroll_val.grid(row=5, column=0, padx=12, pady=(2, 0), sticky="w")

        self.scroll_slider = ctk.CTkSlider(
            self.gesture_section,
            from_=1.0,
            to=25.0,
            number_of_steps=24,
            command=lambda val: self._on_slider_change("scroll_sensitivity", val, self.scroll_val, "Scroll Speed Multiplier: {:.1f}x"),
        )
        self.scroll_slider.set(8.0)
        self.scroll_slider.grid(row=6, column=0, padx=12, pady=(2, 10), sticky="ew")

        # 4. Interactive Gesture Cheatsheet Card
        self.guide_card = ctk.CTkFrame(self.control_frame, fg_color="#182232", corner_radius=10)
        self.guide_card.grid(row=3, column=0, padx=12, pady=10, sticky="ew")
        self.guide_card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            self.guide_card,
            text="💡 Gesture Guide Cheatsheet",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#60A5FA",
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(8, 4), sticky="w")

        guides = [
            ("☝️ Move Cursor", "Index Finger UP"),
            ("🤏 Left Click", "Thumb + Index Tip Touch"),
            ("👉 Right Click", "Thumb + Middle Tip Touch"),
            ("✌️ Double Click", "Index + Middle Tip Touch"),
            ("✊ Drag & Select", "Closed Fist"),
            ("🖐️ Scroll", "Open Hand Up/Down"),
        ]

        for idx, (gesture_title, desc) in enumerate(guides):
            r = 1 + (idx // 2)
            c = idx % 2
            ctk.CTkLabel(
                self.guide_card,
                text=f"{gesture_title}\n{desc}",
                font=ctk.CTkFont(size=11),
                text_color=self.COLOR_MUTED,
                justify="left",
            ).grid(row=r, column=c, padx=10, pady=4, sticky="w")



    def _on_toggle_mouse(self) -> None:
        """Handle virtual mouse toggle switch."""
        enabled = bool(self.toggle_switch.get())
        if enabled:
            self.mouse_badge.configure(text="🟢 VIRTUAL MOUSE: ACTIVE", text_color=self.COLOR_SUCCESS, fg_color="#064E3B")
        else:
            self.mouse_badge.configure(text="🔴 VIRTUAL MOUSE: PAUSED", text_color=self.COLOR_DANGER, fg_color="#7F1D1D")

        if self.on_setting_change:
            self.on_setting_change("enabled", enabled)

    def _on_slider_change(
        self,
        key: str,
        value: float,
        label_widget: ctk.CTkLabel,
        format_str: str,
        scale: float = 1.0,
    ) -> None:
        """Generic handler for slider updates."""
        label_widget.configure(text=format_str.format(value * scale))
        if self.on_setting_change:
            self.on_setting_change(key, value)

    def update_video_frame(self, frame_bgr: np.ndarray) -> None:
        """Update live camera preview frame from OpenCV BGR image (non-blocking).

        Args:
            frame_bgr: OpenCV image matrix in BGR color space.
        """
        if self._is_updating_video:
            return

        self._is_updating_video = True
        try:
            lbl_w = max(320, self.video_label.winfo_width())
            lbl_h = max(240, self.video_label.winfo_height())

            h, w = frame_bgr.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

            resized_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            self._photo_image = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=self._photo_image, text="")
        except Exception:
            pass
        finally:
            self._is_updating_video = False

    def update_telemetry(self, fps: float, gesture: str, enabled: bool) -> None:
        """Update telemetry text labels."""
        self.fps_badge.configure(text=f"{fps:.0f} FPS")

        # Update gesture badge
        gesture_icons = {
            "MOVE": "☝️ CURSOR MOVE",
            "LEFT_CLICK": "🤏 LEFT CLICK",
            "RIGHT_CLICK": "👉 RIGHT CLICK",
            "DOUBLE_CLICK": "✌️ DOUBLE CLICK",
            "DRAG": "✊ DRAG & SELECT",
            "SCROLL": "🖐️ SCROLL",
            "NONE": "✋ SEARCHING HAND",
        }
        badge_text = gesture_icons.get(gesture, f"🖐️ {gesture}")
        self.gesture_badge.configure(text=badge_text)

        if enabled != bool(self.toggle_switch.get()):
            if enabled:
                self.toggle_switch.select()
                self.mouse_badge.configure(text="🟢 VIRTUAL MOUSE: ACTIVE", text_color=self.COLOR_SUCCESS, fg_color="#064E3B")
            else:
                self.toggle_switch.deselect()
                self.mouse_badge.configure(text="🔴 VIRTUAL MOUSE: PAUSED", text_color=self.COLOR_DANGER, fg_color="#7F1D1D")

    def _on_closing(self) -> None:
        """Handle window close event."""
        if self.on_close_callback:
            self.on_close_callback()
        self.destroy()
