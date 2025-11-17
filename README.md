# Integration-v1

Minimal integration package that connects three APIs for hand recognition with grid layout management.

## Architecture (Separation of Concerns)

This integration follows SoC principles with three independent APIs:

1. **Hand Recognition API** (`hand_recognition/`)
   - Detects hand gestures (Open_Palm, Closed_Fist)
   - Provides hand position coordinates from camera stream
   - Uses MediaPipe for gesture recognition

2. **Overlay API** (`overlay/overlay-api.py`)
   - Transforms coordinates: GPS server → Rectified → Grid cells
   - Handles perspective correction and grid overlay
   - Provides coordinate transformation functions

3. **Layout API** (`object-layout/api/layout-api.py`)
   - Manages grid map (walls, home positions, obstacles)
   - Provides colored symbol display
   - Handles grid persistence and access

## Structure

```
Integration-v1/
├── main.py                    # Main orchestrator (connects all APIs)
├── demo/
│   └── hand_grid_demo.py     # Demo implementation
├── hand_recognition/
│   ├── hand_recognition.py   # Hand recognition module
│   └── gesture_recognizer.task  # MediaPipe model
├── overlay/
│   ├── overlay-api.py        # GPSOverlay API
│   └── gps_overlay.json      # Calibration data
└── object-layout/
    └── api/
        ├── layout-api.py     # Layout API (standalone)
        └── grid.json          # Grid data
```

## Usage

Run the integration:

```bash
python3 main.py
```

The integration will:
1. Initialize all three APIs
2. Start hand recognition from camera stream
3. Transform hand coordinates to grid cells
4. Update grid based on gestures (in memory only)
5. Display grid with colored symbols in terminal

## Testing

Test the APIs individually:

```bash
# Test Layout API
python3 test_layout_api.py

# Test Overlay API
python3 test_overlay_api.py

# Test specific overlay functionality
python3 test_overlay_api.py test_coordinates
python3 test_overlay_api.py test_grid_cells
python3 test_overlay_api.py test_stream_transform
```

## Gestures

- **Open_Palm (FOOD)** → Marks cell as HOME (🍎)
- **Closed_Fist (THREAT)** → Marks cell as OBSTACLE (⚠️)

## Requirements

- Python 3.8+
- `mediapipe` - Hand gesture recognition
- `opencv-python` - Camera stream handling
- `numpy` - Numerical operations

## Standalone

This package is completely standalone:
- All APIs are self-contained
- No dependencies on other Mermaid projects
- Can be copied and used independently
- Each module can be replaced without affecting others

## API Independence

Each API is independent and can be replaced:
- **Hand Recognition**: Replace `hand_recognition.py` with different recognition system
- **Overlay**: Replace `overlay-api.py` with different coordinate transformation
- **Layout**: Replace `layout-api.py` with different grid management

The `main.py` orchestrator connects them together following SoC principles.
