#!/usr/bin/env python3
"""
Basic runner for HandRecognizer (moved out of hand-recognition/api folder).
Run from repo root: python tests/test_hand_recognition.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_REC_DIR = REPO_ROOT / "hand-recognition"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HAND_REC_DIR) not in sys.path:
    sys.path.insert(0, str(HAND_REC_DIR))

import hand_recognition as hr


def main():
    detector = hr.HandRecognizer()
    detector.run()
    try:
        while True:
            gesture = detector.get_gesture()
            position = detector.get_position()

            if gesture in ("FIST", "PALM"):
                print("Gesture:", gesture)
                print("Position:", position)
                print("---------")

    except KeyboardInterrupt:
        detector.stop()


if __name__ == "__main__":
    main()
