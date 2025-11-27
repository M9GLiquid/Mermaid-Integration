# GPSOverlay API - Standalone Package

This folder contains everything you need to use the GPSOverlay API in your project.

## Files

- `overlay.py` - The API code (no external dependencies)
- `gps_overlay.json` - Calibration data (created by Tool 8)
- `test_overlay.py` - Test script demonstrating API usage

## Quick Start

1. Copy both files to your project directory
2. Import and use:

```python
from overlay import GPSOverlay

# Initialize (auto-detects gps_overlay.json in same directory)
overlay = GPSOverlay()

# Or specify custom path
overlay = GPSOverlay("path/to/gps_overlay.json")
```

## Usage Examples

### Transform GPS Coordinates to Rectified Space

```python
# Transform GPS server coordinates (2048x1536) to rectified canvas coordinates
x_rect, y_rect = overlay.map_coords(1024, 768)
print(f"Rectified position: ({x_rect:.1f}, {y_rect:.1f})")
```

### Get Grid Cell Position

```python
# Get which grid cell a GPS coordinate belongs to
cell = overlay.get_grid_cell(1024, 768)

if cell["in_bounds"]:
    print(f"Robot is in cell ({cell['col']}, {cell['row']})")
    print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
else:
    print("Point is outside arena bounds")
```

### Get Real-World Coordinates (if calibrated)

```python
# Convert GPS coordinates to real-world millimeters
try:
    pos = overlay.get_real_coords(1024, 768)
    print(f"Real position: {pos['x_mm']:.1f}mm, {pos['y_mm']:.1f}mm")
    print(f"Distance from origin: {pos['distance_from_origin_mm']:.1f}mm")
except ValueError:
    print("Real-world calibration not available (run Tool 6 first)")
```

### Get Complete Grid Map

```python
# Get all grid cells as a 2D array
grid_map = overlay.get_grid_map()

# Access specific cell (row, col)
cell = grid_map[5][3]  # Row 5, Column 3
print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")

# Iterate through all cells
for row in grid_map:
    for cell in row:
        if cell['x_mm'] > 0:  # Only if real-world calibrated
            print(f"Cell ({cell['col']}, {cell['row']}): {cell['x_mm']:.1f}mm, {cell['y_mm']:.1f}mm")
```

## API Reference

### GPSOverlay Class

#### `__init__(json_path=None)`
Initialize the API by loading calibration data.

- `json_path`: Path to `gps_overlay.json`. If `None`, looks for it in the same directory.

#### `map_coords(x, y) -> (float, float)`
Transform GPS server coordinates to rectified canvas coordinates.

- `x`: GPS server X coordinate (typically 0-2048)
- `y`: GPS server Y coordinate (typically 0-1536)
- Returns: `(x_rect, y_rect)` tuple in rectified space

#### `get_grid_cell(x, y) -> dict`
Get grid cell information for GPS coordinates.

- `x`: GPS server X coordinate
- `y`: GPS server Y coordinate
- Returns: Dictionary with `col`, `row`, `in_bounds`, `center_x`, `center_y`

#### `get_real_coords(x, y) -> dict`
Get real-world coordinates in millimeters (requires Tool 6 calibration).

- `x`: GPS server X coordinate
- `y`: GPS server Y coordinate
- Returns: Dictionary with `x_mm`, `y_mm`, `distance_from_origin_mm`
- Raises: `ValueError` if real-world calibration not available

#### `get_grid_map() -> list`
Get complete grid mapping as 2D array.

- Returns: 2D list (rows x cols) of cell dictionaries

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Coordinate Systems

- **GPS Server Space**: Original camera image (typically 2048×1536)
- **Rectified Space**: Top-down view after fisheye correction and perspective transformation
- **Grid Space**: Discrete cell indices (col, row) within arena bounds
- **Real-World Space**: Millimeters from arena top-left corner (if calibrated)

## Notes

- The API automatically handles coordinate transformations
- Grid size is dynamically loaded from calibration data
- Real-world coordinates require Tool 6 calibration (optional)
- All coordinates use (0,0) as top-left origin

## Support

For issues or questions, refer to the main project README or calibration tools.
