#!/usr/bin/env python3
"""
Integration-v1 - Main Entry Point

This is the main orchestrator that connects the three APIs:
1. Hand Recognition API - Detects hand gestures and positions
2. Overlay API - Transforms coordinates (GPS → Grid cells)
3. Object Layout API - Manages grid map (walls, home, obstacles)

Architecture follows Separation of Concerns (SoC):
- Each API is independent and can be replaced
- Main orchestrator coordinates between APIs
- Demo logic is separated from API logic

Usage:
    python3 main.py
"""

import sys
import os
from pathlib import Path

# Import utility functions
from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent

# IMPORTANT: Insert paths in order of priority to ensure we get the right modules
# Put Integration-v1 paths FIRST to avoid importing from old demo/ folder
sys.path.insert(0, str(integration_root / "hand_recognition"))
sys.path.insert(0, str(integration_root / "overlay"))
sys.path.insert(0, str(integration_root / "object-layout" / "api"))
sys.path.insert(0, str(integration_root))

# Import APIs using simplified utility functions
hr = import_api(
    integration_root / "hand_recognition" / "hand_recognition.py",
    "hand_recognition",
    "Make sure hand_recognition.py is in Integration-v1/hand_recognition/"
)

overlay_api = import_api(
    integration_root / "overlay" / "overlay-api.py",
    "overlay_api",
    "Make sure overlay-api.py is in Integration-v1/overlay/"
)
GPSOverlay = overlay_api.GPSOverlay

layout_api = import_api(
    integration_root / "object-layout" / "api" / "layout-api.py",
    "layout_api",
    "Make sure layout-api.py is in Integration-v1/object-layout/api/"
)
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME'
)

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
    print("\nConnecting APIs:")
    print("  1. Hand Recognition API - Detecting gestures")
    print("  2. Overlay API - Coordinate transformation")
    print("  3. Object Layout API - Grid map management")
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
