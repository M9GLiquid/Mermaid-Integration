#!/usr/bin/env python3
"""
Integration-v1 - Main Entry Point

This is the main orchestrator that connects the APIs:
1. Hand Recognition API - Detects hand gestures and positions
2. Overlay API - Transforms coordinates (GPS → Grid cells)
3. Object Layout API - Manages grid map (walls, home, obstacles)
4. ROS2 Position API - Robot position tracking from GPS Server

Architecture follows Separation of Concerns (SoC):
- Each API is independent and can be replaced
- Main orchestrator coordinates between APIs
- Demo logic is separated from API logic

Usage:
    python3 main.py                    # Hand recognition demo
    python3 -m demo.robot_position_demo  # Robot position tracking (spiral 9)
"""

import sys
import os
from pathlib import Path

# Import utility functions
from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent

# IMPORTANT: Insert paths in order of priority to ensure we get the right modules
# APIs are now Git submodules in apis/ directory
sys.path.insert(0, str(integration_root / "apis" / "overlay-api"))
sys.path.insert(0, str(integration_root / "apis" / "layout-api"))
sys.path.insert(0, str(integration_root / "apis" / "hand-recognition-api"))
sys.path.insert(0, str(integration_root))

# Import all three APIs using simplified utility functions (from Git submodules)
hand_recognition_api = import_api(
    integration_root / "apis" / "hand-recognition-api" / "hand-recognition-api.py",
    "hand_recognition_api",
    "Make sure hand-recognition-api submodule is initialized: git submodule update --init"
)
GestureRecognizer = hand_recognition_api.GestureRecognizer

overlay_api = import_api(
    integration_root / "apis" / "overlay-api" / "overlay-api.py",
    "overlay_api",
    "Make sure overlay-api submodule is initialized: git submodule update --init"
)
GPSOverlay = overlay_api.GPSOverlay

layout_api = import_api(
    integration_root / "apis" / "layout-api" / "layout-api.py",
    "layout_api",
    "Make sure layout-api submodule is initialized: git submodule update --init"
)
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME'
)

# Import ROS2 API for robot position tracking
ros2_api = import_api(
    integration_root / "apis" / "ros2-api" / "ros2-api.py",
    "ros2_api",
    "Make sure ros2-api files are available: ros2-api.py and ros2.py"
)
RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow

# Import demo orchestrator
from demo.hand_grid_demo import HandGridDemo


def main():
    """
    Main entry point - orchestrates the three APIs.
    
    Architecture (SoC):
    - Hand Recognition API: Detects gestures and positions from camera
    - Overlay API: Transforms coordinates (GPS server → Grid cells)
    - Object Layout API: Manages grid map (walls, home, obstacles)
    - Demo: Orchestrates the three APIs together
    """
    print("=" * 60)
    print("Integration-v1 - Hand Recognition with Grid Layout")
    print("=" * 60)
    print("\nConnecting modules:")
    print("  1. Hand Recognition API - Detecting gestures")
    print("  2. Overlay API - Coordinate transformation")
    print("  3. Object Layout API - Grid map management")
    print("  4. ROS2 Position API - Robot position tracking")
    print()
    
    # Initialize demo (which uses all three APIs)
    demo = HandGridDemo()
    
    # Run the demo
    demo.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nIntegration stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
