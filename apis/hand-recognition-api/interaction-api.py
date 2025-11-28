#!/usr/bin/env python3
"""
Interaction API - self-contained hand recognition entrypoint (hyphenated).

Bundled here so vendored consumers can import this single file.
"""

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import sys
import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "modules"

# Ensure module imports work when loaded as a standalone file (no package context)
try:
    from .modules.gps_overlay import GPSOverlay  # type: ignore
except Exception:
    if str(MODULES_DIR) not in sys.path:
        sys.path.insert(0, str(MODULES_DIR))
    from gps_overlay import GPSOverlay  # type: ignore

# ============================
# 0) AXIS CAMERA STREAM
# ============================

STREAM_URL = "http://192.168.1.2/axis-cgi/mjpg/video.cgi?camera=1&resolution=2048x1536"

# ============================
# 1) Load calibration from gps_overlay.json
# ============================

# Initialize overlay - looks for gps_overlay.json in api/ directory
overlay = GPSOverlay()

camera_matrix = np.array(overlay.data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(overlay.data["distortion_coeffs"], dtype=np.float32)
calib_size    = tuple(overlay.calib_size)      # (2048, 1536)
margin        = overlay.margin_pixels          # e.g. 200
expanded_size = tuple(overlay.corrected_size)  # e.g. (2448, 1936)

new_camera_matrix = camera_matrix.copy()
new_camera_matrix[0, 2] += margin
new_camera_matrix[1, 2] += margin

scale_factor = 0.8
new_camera_matrix[0, 0] *= scale_factor
new_camera_matrix[1, 1] *= scale_factor

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    np.eye(3),
    new_camera_matrix,
    expanded_size,
    cv2.CV_16SC2
)

# ============================
# 2) MEDIAPIPE SETUP 
# ============================

# Explicit type ignores to satisfy static checkers (mediapipe stubs incomplete)
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ============================
# 3) GESTURE LOGIC
# ============================

def is_palm_open(lm):
    wrist = lm[0]
    fingers = [
        (8, 5),   # index
        (12, 9),  # middle
        (16, 13), # ring
        (20, 17)  # pinky
    ]
    extended = 0
    for tip_id, mcp_id in fingers:
        tip = lm[tip_id]
        mcp = lm[mcp_id]
        d_tip_wrist = dist(tip, wrist)
        d_mcp_wrist = dist(mcp, wrist)

        if d_tip_wrist > d_mcp_wrist * 1.05:
            extended += 1

    return extended >= 2


def is_fist(lm):
    wrist = lm[0]
    fingers = [
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]
    folded = 0
    for tip_id, mcp_id in fingers:
        tip = lm[tip_id]
        mcp = lm[mcp_id]
        d_tip_mcp = dist(tip, mcp)
        d_mcp_wrist = dist(mcp, wrist)
        d_tip_wrist = dist(tip, wrist)

        if d_tip_mcp < d_mcp_wrist * 0.8 and d_tip_wrist < d_mcp_wrist * 1.3:
            folded += 1

    return folded >= 2


def get_gesture_from_landmarks(lm):
    """
    Returnerar 'FIST', 'PALM' eller None
    """
    if is_fist(lm):
        return "FIST"
    if is_palm_open(lm):
        return "PALM"
    return None


def get_position(lm, frame_width, frame_height):
    """
    Returnerar (x, y) i pixlar som mitten av handen.
    lm = list of 21 landmarks med normaliserade koordinater [0,1]
    """
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    cx = int(sum(xs) / len(xs) * frame_width)
    cy = int(sum(ys) / len(ys) * frame_height)

    # Adjust for fisheye margin: undistorted image includes margin_pixels around
    # the calibrated area; subtract margin to return coordinates in the original
    # calibration/server space.
    adj_x = max(0, min(frame_width, cx - margin))
    adj_y = max(0, min(frame_height, cy - margin))

    return (adj_x, adj_y)


# ============================
# 4) CLASS-BASED API
# ============================

class HandRecognizer:
    def __init__(self, stream_url: str = STREAM_URL, show_window: bool = False):
        self.stream_url = stream_url
        self.show_window = show_window

        self.cap = None
        self.running = False
        self.thread = None

        self._lock = threading.Lock()
        self._gesture = None        # 'FIST' / 'PALM' / None
        self._position = None       # (x, y) eller None

    def _loop(self):
        # Open camera
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            print("Failed to open Axis camera stream...")
            self.running = False
            return

        # Create MediaPipe Hands locally in the loop (no global "hands")
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,             # we only need one hand
            model_complexity=0,          # faster model
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        ) as hands:

            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame from Axis camera...")
                    continue

                # Ensure correct size for undistortion
                h, w = frame.shape[:2]
                if (w, h) != calib_size:
                    frame_scaled = cv2.resize(frame, calib_size, interpolation=cv2.INTER_LINEAR)
                else:
                    frame_scaled = frame

                # Undistortion
                undistorted = cv2.remap(frame_scaled, map1, map2, cv2.INTER_LINEAR)

                # === SMALL IMAGE FOR MEDIAPIPE (FASTER) ===
                # We scale DOWN the image, but still use
                # undistorted-width/height for position calculation
                small = cv2.resize(undistorted, (640, 480))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                results = hands.process(small_rgb)

                gesture = None
                position = None
                text_color = (255, 255, 255)
                text_label = "NO HAND"

                if results.multi_hand_landmarks:
                    # Take first hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm = hand_landmarks.landmark

                    # Gesture
                    gesture = get_gesture_from_landmarks(lm)
                    if gesture == "FIST":
                        text_color = (0, 0, 255)
                        text_label = "FIST"
                    elif gesture == "PALM":
                        text_color = (0, 255, 0)
                        text_label = "PALM"
                    else:
                        text_color = (255, 255, 255)
                        text_label = "UNKNOWN"

                    # Position in PIXELS on the UNDISTORTED image
                    fh, fw = undistorted.shape[:2]
                    position = get_position(lm, fw, fh)

                    # Draw landmarks on undistorted (if you want to debug)
                    mp_drawing.draw_landmarks(
                        undistorted, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                # Uppdatera delat state
                with self._lock:
                    self._gesture = gesture
                    self._position = position

                # Rita text på bilden (som tidigare)
                cv2.putText(
                    undistorted,
                    text_label,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    text_color,
                    3
                )

                if self.show_window:
                    cv2.imshow('Hand Recognition (Axis Camera, undistorted)', undistorted)

                    # Close with 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.show_window:
            cv2.destroyAllWindows()

    def run(self):
        """
        Startar kameraloopen i en bakgrundstråd.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stoppar bakgrundstråden och stänger kameran/fönstret.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def get_gesture(self):
        """
        Returnerar senaste gest:
        'FIST', 'PALM' eller None om ingen hand/gest just nu.
        """
        with self._lock:
            return self._gesture

    def get_position(self):
        """
        Returnerar senaste handposition som (x, y) i pixlar,
        eller None om ingen hand hittats.
        """
        with self._lock:
            return self._position


# ============================
# 4) High-level API wrapper
# ============================


@dataclass
class HandState:
    """Current hand recognition state."""
    gesture: str  # 'FIST' or 'PALM'
    position: Tuple[int, int]  # (x, y)


class HandRecognitionAPI:
    """
    Simple API for hand recognition.
    Provides easy-to-use interface for detecting hand gestures and positions.
    """

    def __init__(self, stream_url: Optional[str] = None, show_window: bool = False):
        """
        Initialize hand recognition API.

        Args:
            stream_url: Camera stream URL (defaults to module's STREAM_URL)
        """
        self._recognizer = HandRecognizer(stream_url=stream_url or STREAM_URL, show_window=show_window)

    def start(self):
        """Start recognition in background thread."""
        self._recognizer.run()

    def stop(self):
        """Stop recognition and cleanup."""
        self._recognizer.stop()

    def get_state(self) -> Optional[HandState]:
        """
        Get current hand state (gesture + position) atomically.

        Returns:
            HandState if hand is detected, None otherwise
        """
        gesture = self._recognizer.get_gesture()
        position = self._recognizer.get_position()

        if gesture is not None and position is not None:
            return HandState(gesture=gesture, position=position)
        return None

    def is_running(self) -> bool:
        """Check if recognition is active."""
        return self._recognizer.running

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()


# Backward-compatible alias for vendored imports
GestureRecognizer = HandRecognizer

__all__ = ["HandRecognitionAPI", "HandState", "GestureRecognizer", "HandRecognizer", "STREAM_URL"]
