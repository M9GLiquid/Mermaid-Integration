#!/usr/bin/env python3
"""
Test script for Overlay API (overlay-api.py)

This script demonstrates how to use the overlay API and tests basic functionality.

Usage:
    # Run all tests
    python3 test_overlay_api.py
    
    # Run specific test
    python3 test_overlay_api.py test_coordinates
    python3 test_overlay_api.py test_grid_cells
    python3 test_overlay_api.py test_real_world
    python3 test_overlay_api.py test_image_transform
    python3 test_overlay_api.py test_stream_transform
"""

import sys
import os
from pathlib import Path

# Import utility functions
from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent
sys.path.insert(0, str(integration_root / "apis" / "overlay-api"))

# Import overlay-api.py using simplified utility
overlay_api = import_api(
    integration_root / "apis" / "overlay-api" / "overlay-api.py",
    "overlay_api"
)
GPSOverlay = overlay_api.GPSOverlay


def test_coordinates():
    """Test coordinate transformation from GPS server space to rectified space"""
    print("=" * 50)
    print("Test: Coordinate Transformation")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
        # Test with sample GPS coordinates (like from your server)
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]
        
        for gps_x, gps_y in test_coords:
            print(f"\nGPS Server ({gps_x}, {gps_y}):")
            
            # Get rectified coordinates
            x_rect, y_rect = overlay.map_coords(gps_x, gps_y)
            print(f"  -> Rectified: ({x_rect:.1f}, {y_rect:.1f})")
        
        print("\n[SUCCESS] Coordinate transformation test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Make sure gps_overlay.json is in Integration-v1/apis/overlay-api/")
        print("Initialize submodules: git submodule update --init")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_cells():
    """Test grid cell mapping from GPS coordinates"""
    print("=" * 50)
    print("Test: Grid Cell Mapping")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]
        
        for gps_x, gps_y in test_coords:
            print(f"\nGPS Server ({gps_x}, {gps_y}):")
            
            # Get grid cell
            cell = overlay.get_grid_cell(gps_x, gps_y)
            print(f"  -> Grid Cell: ({cell['col']}, {cell['row']})")
            print(f"  -> In bounds: {cell['in_bounds']}")
            print(f"  -> Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
        
        # Show grid info
        print("\nGrid Configuration:")
        print(f"  Size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"  Cell Size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"  Arena Bounds: {overlay.arena_bounds}")
        
        print("\n[SUCCESS] Grid cell mapping test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Make sure gps_overlay.json is in Integration-v1/apis/overlay-api/")
        print("Initialize submodules: git submodule update --init")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_world():
    """Test real-world coordinate conversion (requires Tool 6 calibration)"""
    print("=" * 50)
    print("Test: Real-World Coordinates")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
        if not overlay.real_world_available:
            print("[SKIP] Real-world calibration not available")
            print("Real-world coordinate conversion requires calibration data")
            return True
        
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]
        
        for gps_x, gps_y in test_coords:
            print(f"\nGPS Server ({gps_x}, {gps_y}):")
            
            # Get real-world coordinates
            real_pos = overlay.get_real_coords(gps_x, gps_y)
            print(f"  -> Real World: {real_pos['x_mm']:.1f}mm, {real_pos['y_mm']:.1f}mm")
            print(f"  -> Distance from origin: {real_pos['distance_from_origin_mm']:.1f}mm")
        
        print("\nReal-World Calibration:")
        print(f"  X: {overlay.mm_per_pixel_x:.3f} mm/pixel")
        print(f"  Y: {overlay.mm_per_pixel_y:.3f} mm/pixel")
        
        print("\n[SUCCESS] Real-world coordinate test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Make sure gps_overlay.json is in Integration-v1/apis/overlay-api/")
        print("Initialize submodules: git submodule update --init")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_transform():
    """Test image transformation from raw GPS image to rectified view"""
    print("=" * 50)
    print("Test: Image Transformation")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[SKIP] Image transformation requires OpenCV and NumPy")
        print("Install with: pip install opencv-python numpy")
        return True
    
    try:
        overlay = GPSOverlay()
        
        # Try to find a test image (optional - skip if not found)
        test_image_path = integration_root / "test_image.png"
        
        if not test_image_path.exists():
            print("[SKIP] Test image not found (test_image.png)")
            print("Image transformation function is available but no test image provided")
            print("To test: Place a GPS camera image as test_image.png in Integration-v1/")
            return True
        
        print(f"Test image: {test_image_path}")
        
        # Test with grid overlay
        print("\nTransforming image with grid overlay...")
        rectified, offset = overlay.transform_image(str(test_image_path), show_grid=True)
        print(f"  -> Rectified image size: {rectified.shape[1]}x{rectified.shape[0]}")
        print(f"  -> Offset: x={offset['offset_x']}, y={offset['offset_y']}")
        
        output_path = integration_root / "test_output_rectified_with_grid.png"
        cv2.imwrite(str(output_path), rectified)
        print(f"  -> Saved to: {output_path}")
        
        # Test without grid overlay
        print("\nTransforming image without grid overlay...")
        rectified_no_grid, offset_no_grid = overlay.transform_image(str(test_image_path), show_grid=False)
        print(f"  -> Rectified image size: {rectified_no_grid.shape[1]}x{rectified_no_grid.shape[0]}")
        print(f"  -> Offset: x={offset_no_grid['offset_x']}, y={offset_no_grid['offset_y']}")
        
        output_path_no_grid = integration_root / "test_output_rectified_no_grid.png"
        cv2.imwrite(str(output_path_no_grid), rectified_no_grid)
        print(f"  -> Saved to: {output_path_no_grid}")
        
        # Test grid cell conversion from click coordinates
        print("\nTesting grid cell conversion from click coordinates...")
        # Simulate a click at center of image
        click_x_img = rectified.shape[1] // 2
        click_y_img = rectified.shape[0] // 2
        x_canvas = click_x_img + offset["offset_x"]
        y_canvas = click_y_img + offset["offset_y"]
        cell = overlay.get_grid_cell_from_rectified(x_canvas, y_canvas)
        print(f"  -> Click at image pixel ({click_x_img}, {click_y_img})")
        print(f"  -> Canvas coordinate ({x_canvas:.1f}, {y_canvas:.1f})")
        if cell["in_bounds"]:
            print(f"  -> Grid cell: ({cell['col']}, {cell['row']})")
        else:
            print(f"  -> Click is outside arena bounds")
        
        print("\n[SUCCESS] Image transformation test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Make sure gps_overlay.json is in Integration-v1/apis/overlay-api/")
        print("Initialize submodules: git submodule update --init")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stream_transform():
    """Test stream transformation (transform_frame function)"""
    print("=" * 50)
    print("Test: Stream Transformation")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[SKIP] Stream transformation requires OpenCV and NumPy")
        print("Install with: pip install opencv-python numpy")
        return True
    
    try:
        overlay = GPSOverlay()
        
        # Create a dummy test frame (simulating camera frame)
        # Use server_size from overlay calibration
        test_frame = np.zeros((overlay.server_size[1], overlay.server_size[0], 3), dtype=np.uint8)
        # Add some test pattern
        cv2.rectangle(test_frame, (100, 100), (500, 400), (255, 255, 255), -1)
        cv2.circle(test_frame, (1024, 768), 200, (128, 128, 128), -1)
        
        print("Created test frame:")
        print(f"  -> Size: {test_frame.shape[1]}x{test_frame.shape[0]}")
        print(f"  -> Expected: {overlay.server_size[0]}x{overlay.server_size[1]}")
        
        # Test transform_frame with grid
        print("\nTransforming frame with grid overlay...")
        rectified, offset = overlay.transform_frame(test_frame, show_grid=True)
        print(f"  -> Rectified frame size: {rectified.shape[1]}x{rectified.shape[0]}")
        print(f"  -> Offset: x={offset['offset_x']}, y={offset['offset_y']}")
        
        # Save output
        output_path = integration_root / "test_output_stream_rectified.png"
        cv2.imwrite(str(output_path), rectified)
        print(f"  -> Saved to: {output_path}")
        
        # Test transform_frame without grid
        print("\nTransforming frame without grid overlay...")
        rectified_no_grid, offset_no_grid = overlay.transform_frame(test_frame, show_grid=False)
        print(f"  -> Rectified frame size: {rectified_no_grid.shape[1]}x{rectified_no_grid.shape[0]}")
        print(f"  -> Offset: x={offset_no_grid['offset_x']}, y={offset_no_grid['offset_y']}")
        
        # Test multiple frames (should reuse cached maps)
        print("\nTesting multiple frames (maps should be cached)...")
        import time
        start_time = time.time()
        for i in range(10):
            overlay.transform_frame(test_frame, show_grid=True)
        elapsed = time.time() - start_time
        print(f"  -> Transformed 10 frames in {elapsed:.3f} seconds")
        print(f"  -> Average: {elapsed/10*1000:.1f} ms per frame")
        
        print("\n[SUCCESS] Stream transformation test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Make sure gps_overlay.json is in Integration-v1/apis/overlay-api/")
        print("Initialize submodules: git submodule update --init")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all available tests"""
    print("\n" + "=" * 50)
    print("Overlay API - Running All Tests")
    print("=" * 50 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Coordinate Transformation", test_coordinates()))
    results.append(("Grid Cell Mapping", test_grid_cells()))
    results.append(("Real-World Coordinates", test_real_world()))
    results.append(("Image Transformation", test_image_transform()))
    results.append(("Stream Transformation", test_stream_transform()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"  ✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"  ✗ {test_name}: FAILED")
            failed += 1
        else:
            print(f"  ⊘ {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test(s) failed")
        return 1


def main():
    """Main entry point - run specific test or all tests"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "test_coordinates" or test_name == "coordinates":
            test_coordinates()
        elif test_name == "test_grid_cells" or test_name == "grid_cells":
            test_grid_cells()
        elif test_name == "test_real_world" or test_name == "real_world":
            test_real_world()
        elif test_name == "test_image_transform" or test_name == "image_transform":
            test_image_transform()
        elif test_name == "test_stream_transform" or test_name == "stream_transform":
            test_stream_transform()
        else:
            print(f"Unknown test: {test_name}")
            print("\nAvailable tests:")
            print("  test_coordinates    - Test coordinate transformation")
            print("  test_grid_cells     - Test grid cell mapping")
            print("  test_real_world     - Test real-world coordinates")
            print("  test_image_transform - Test image transformation")
            print("  test_stream_transform - Test stream transformation")
            print("\nOr run without arguments to run all tests")
            sys.exit(1)
    else:
        # Run all tests
        sys.exit(run_all_tests())


if __name__ == "__main__":
    main()
