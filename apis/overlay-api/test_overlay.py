#!/usr/bin/env python3
"""
Test script for GPSOverlay API

This script demonstrates how to use the overlay API and tests basic functionality.
"""

from overlay import GPSOverlay


def main():
    try:
        # Load the API
        overlay = GPSOverlay()

        print("GPSOverlay API - Standalone Test")
        print("=" * 40)

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

            # Get grid cell
            cell = overlay.get_grid_cell(gps_x, gps_y)
            print(f"  -> Grid Cell: {cell['col']}, {cell['row']} (in bounds: {cell['in_bounds']})")

            # Get real-world coordinates (if available)
            if overlay.real_world_available:
                real_pos = overlay.get_real_coords(gps_x, gps_y)
                print(f"  -> Real World: {real_pos['x_mm']:.1f}mm, {real_pos['y_mm']:.1f}mm")
            else:
                print("  -> Real World: Not available (run Tool 6 first)")
        
        # Show grid info
        print("\nGrid Configuration:")
        print(f"  Size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"  Cell Size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"  Arena: {overlay.arena_bounds}")

        if overlay.real_world_available:
            print(f"  Real-World: {overlay.mm_per_pixel_x:.3f} mm/pixel")
        else:
            print("  Real-World: Not calibrated")

        print("\n[SUCCESS] GPSOverlay API ready for use!")
        print("Copy overlay.py and gps_overlay.json to your project.")

    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure gps_overlay.json contains valid calibration data.")


if __name__ == "__main__":
    main()
