"""
Hand Recognition API - Standalone gesture recognition module

Detects hand gestures (Open_Palm, Closed_Fist) and provides hand position coordinates
from camera stream using MediaPipe.

This is a standalone API that can be exported and used independently.

Usage:
    from hand_recognition_api import GestureRecognizer
    
    recognizer = GestureRecognizer()
    recognizer.run()
    
    while recognizer.running:
        x, y = recognizer.get_position()
        gesture = recognizer.get_gesture()
        # Process gesture...
    
    recognizer.stop()
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading
import numpy as np
from pathlib import Path

# Configuration constants
STREAM_URL = "http://192.168.1.2/axis-cgi/mjpg/video.cgi?camera=1&resolution=2048x1536"
MODEL_PATH = str(Path(__file__).parent / "gesture_recognizer.task")
MAX_HANDS = 2


class GestureRecognizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_gestures = []
        # Senaste (x, y) i pixelkoordinater
        self.last_position = (None, None)
        # Senaste gest ("Closed_Fist" / "Open_Palm" / None)
        self.last_gesture = None

        # === Tråd–styrning ===
        self.running = False
        self.thread = None

    # =========================================================
    # Publikt: starta i bakgrundstråd
    # =========================================================
    def run(self):
        """Startar igenkänning i en separat tråd."""
        if self.running:
            print("GestureRecognizer already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # =========================================================
    # Publikt: stoppa tråd
    # =========================================================
    def stop(self):
        """Stoppar igenkänningen och stänger kameran."""
        if not self.running:
            return
        print("Stopping GestureRecognizer...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        print("GestureRecognizer stopped.")

    # =========================================================
    # Gamla start() – nu blockande variant som använder samma loop
    # =========================================================
    def start(self):
        """
        Blockerande start (som din gamla kod).
        Används t.ex. om du kör filen direkt.
        """
        if self.running:
            print("GestureRecognizer already running")
            return
        self.running = True
        self._loop()
        self.running = False

    # =========================================================
    # Intern huvud–loop – LÖSNING: samma som din gamla while True,
    # men bytt till while self.running. I övrigt orörd.
    # =========================================================
    def _loop(self):
        GestureRecognizerMP = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=MAX_HANDS,
            result_callback=self.__result_callback,
        )
        recognizer = GestureRecognizerMP.create_from_options(options)

        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Failed to open camera stream. Check STREAM_URL/auth and network.")
            self.running = False
            return

        real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Incoming stream resolution: {real_w} x {real_h}")

        timestamp = 0

        # ⚠️ Samma logik som innan, bara while self.running i stället för while True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed.")
                break

            # OBS: ingen resize / crop – exakt som din fungerande kod
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            recognizer.recognize_async(mp_image, timestamp)
            timestamp += 1

            self.put_gestures(frame)

            # (Du kan kommentera bort dessa prints om du vill)
            self.get_position()
            state = self.get_gesture()

            cv2.namedWindow("Hand gestures", cv2.WINDOW_NORMAL)
            cv2.imshow("Hand gestures", frame)
            # ESC → stoppa loopen
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")

    # =========================================================
    # Rita text
    # =========================================================
    def put_gestures(self, frame):
        self.lock.acquire()
        gestures = list(self.current_gestures)
        self.lock.release()

        y_pos = 50
        for name in gestures:
            cv2.putText(
                frame,
                name,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y_pos += 50

    # =========================================================
    # Mediapipe–callback
    # =========================================================
    def __result_callback(self, result, output_image, timestamp_ms):
        """
        Callback från MediaPipe:
        - filtrerar Closed_Fist / Open_Palm
        - beräknar center (x, y) i pixlar
        - uppdaterar last_position & last_gesture
        """
        self.lock.acquire()
        self.current_gestures = []
        self.last_position = (None, None)
        self.last_gesture = None
        self.lock.release()

        if result is None or not any(result.gestures):
            return

        img_w = output_image.width
        img_h = output_image.height

        for hand_index, single_hand in enumerate(result.gestures):
            gesture_name = single_hand[0].category_name

            if gesture_name not in ("Closed_Fist", "Open_Palm"):
                continue

            # Spara gest
            self.lock.acquire()
            self.current_gestures.append(gesture_name)
            self.last_gesture = gesture_name
            self.lock.release()

            # Beräkna center från landmarks
            if result.hand_landmarks and len(result.hand_landmarks) > hand_index:
                lm_list = result.hand_landmarks[hand_index]
                xs = [lm.x for lm in lm_list]
                ys = [lm.y for lm in lm_list]

                if xs and ys:
                    cx_norm = sum(xs) / len(xs)
                    cy_norm = sum(ys) / len(ys)
                    cx = int(cx_norm * img_w)
                    cy = int(cy_norm * img_h)

                    self.lock.acquire()
                    self.last_position = (cx, cy)
                    self.lock.release()
            break  # Ta bara första handen

    # =========================================================
    # Publika getters
    # =========================================================
    def get_position(self):
        self.lock.acquire()
        x, y = self.last_position
        self.lock.release()
        return x, y

    def get_gesture(self):
        self.lock.acquire()
        gesture = self.last_gesture
        self.lock.release()
        if gesture == "Open_Palm":
            return "FOOD"
        elif gesture == "Closed_Fist":
            return "THREAT"
        return None
