#!/usr/bin/env python3
"""
Example usage of Hand Recognition API

Demonstrates how to use the HandRecognitionAPI to detect hand gestures
and positions from a camera stream.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path so we can import the api package
api_dir = Path(__file__).parent
parent_dir = api_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from api import HandRecognitionAPI, HandState


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    print("A cv2 window will open showing the camera feed with hand detection overlay.")
    print("Press 'q' in the cv2 window or Ctrl+C here to stop.\n")
    
    # Create API instance
    api = HandRecognitionAPI(show_window=True)
    
    # Start recognition (cv2 window will open automatically)
    api.start()
    print("Hand recognition started...")
    
    try:
        # Run for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10 and api.is_running():
            state = api.get_state()
            
            if state:
                print(f"Gesture: {state.gesture}, Position: {state.position}")
            else:
                print("No hand detected")
            
            time.sleep(0.5)  # Check every 500ms
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Always stop when done (closes cv2 window)
        api.stop()
        print("Hand recognition stopped")


def example_context_manager():
    """Example using context manager (recommended)"""
    print("\n=== Context Manager Example ===")
    print("A cv2 window will open showing the camera feed with hand detection overlay.")
    print("Press 'q' in the cv2 window to stop.\n")
    
    # Using 'with' statement automatically handles start/stop
    with HandRecognitionAPI(show_window=True) as api:
        print("Hand recognition started (context manager)...")
        print("Watch the cv2 window for visual feedback!")
        
        # Check for hand state
        for i in range(20):  # Check 20 times
            if not api.is_running():
                break  # User pressed 'q' in cv2 window
            
            state = api.get_state()
            
            if state:
                print(f"[{i}] {state.gesture} detected at ({state.position[0]}, {state.position[1]})")
            else:
                print(f"[{i}] No hand detected")
            
            time.sleep(0.5)
    
    # API automatically stopped when exiting 'with' block (cv2 window closed)
    print("Hand recognition stopped (context manager)")


def example_continuous_monitoring():
    """Example of continuous monitoring"""
    print("\n=== Continuous Monitoring Example ===")
    print("A cv2 window will open showing the camera feed with hand detection overlay.")
    print("Press 'q' in the cv2 window or Ctrl+C here to stop.\n")
    
    api = HandRecognitionAPI(show_window=True)
    api.start()
    
    try:
        last_gesture = None
        gesture_count = {"FIST": 0, "PALM": 0}
        
        print("Monitoring gestures...")
        print("Watch the cv2 window for real-time hand detection visualization!")
        
        while api.is_running():
            state = api.get_state()
            
            if state:
                # Detect gesture changes
                if state.gesture != last_gesture:
                    gesture_count[state.gesture] += 1
                    print(f"Gesture changed: {last_gesture} -> {state.gesture}")
                    print(f"  Position: {state.position}")
                    print(f"  Counts - FIST: {gesture_count['FIST']}, PALM: {gesture_count['PALM']}")
                    last_gesture = state.gesture
            
            time.sleep(0.1)  # Check every 100ms
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        api.stop()
        print(f"Final counts - FIST: {gesture_count['FIST']}, PALM: {gesture_count['PALM']}")


def example_custom_stream_url():
    """Example with custom camera stream URL"""
    print("\n=== Custom Stream URL Example ===")
    
    # Use a different camera stream URL
    custom_url = "http://192.168.1.2/axis-cgi/mjpg/video.cgi?camera=1&resolution=2048x1536"
    
    with HandRecognitionAPI(stream_url=custom_url, show_window=True) as api:
        print(f"Using custom stream URL: {custom_url}")
        
        for i in range(10):
            state = api.get_state()
            if state:
                print(f"Hand detected: {state.gesture} at {state.position}")
            time.sleep(1)


if __name__ == "__main__":
    print("Hand Recognition API - Example Usage")
    print("=" * 50)
    print("\nNote: A cv2 window will open showing the camera feed with:")
    print("  - Hand landmarks overlay")
    print("  - Gesture label (FIST/PALM/UNKNOWN)")
    print("  - Press 'q' in the cv2 window to stop\n")
    print("=" * 50)
    
    # Run examples
    try:
        # Uncomment the example you want to run:
        
        # example_basic_usage()
        # example_context_manager()
        example_continuous_monitoring()
        # example_custom_stream_url()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
