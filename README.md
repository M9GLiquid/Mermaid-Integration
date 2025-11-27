# Integration-v1

Minimal integration package that connects three APIs for hand recognition with grid layout management.

## Architecture (Separation of Concerns)

This integration follows SoC principles with three independent APIs:

1. **Hand Recognition API** (`apis/hand-recognition-api/hand-recognition-api.py`)
   - Detects hand gestures (Open_Palm, Closed_Fist)
   - Provides hand position coordinates from camera stream
   - Uses MediaPipe for gesture recognition

2. **Overlay API** (`apis/overlay-api/overlay-api.py`)
   - Transforms coordinates: GPS server → Rectified → Grid cells
   - Handles perspective correction and grid overlay
   - Provides coordinate transformation functions

3. **Layout API** (`apis/layout-api/layout-api.py`)
   - Manages grid map (walls, home positions, obstacles)
   - Provides colored symbol display
   - Handles grid persistence and access

## Structure

```
Mermaid-Integration/
├── main.py                    # Main orchestrator (connects all APIs)
├── demo/
│   └── hand_grid_demo.py     # Demo implementation
├── apis/                      # API repositories
│   ├── overlay-api/          # Overlay API
│   │   ├── overlay-api.py
│   │   └── gps_overlay.json
│   ├── layout-api/           # Layout API
│   │   ├── layout-api.py
│   │   └── grid.json
│   ├── hand-recognition-api/ # Hand Recognition API
│   │   ├── hand-recognition-api.py
│   │   └── gesture_recognizer.task
│   ├── ros2-api/             # ROS2 Position API
│   │   ├── ros2-api.py
│   │   └── ros2.py
└── ...
```

**Note:** APIs are now vendored directly in `apis/`; no submodules required.

## Setup

Clone the repository:

```bash
git clone <repository-url> Integration-v1
cd Integration-v1
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
- ROS2 (Jazzy/Humble/Foxy) - For ROS2 Position API
- `pyyaml` - Required for ROS2 (install in venv: `pip install pyyaml`)

## Environment Setup

**VIKTIGT**: Aktivera environment innan du kör programmet!

```bash
cd Integration-v1
source ../setup_env.sh  # Aktiverar ROS2 Jazzy + venv automatiskt
python3 main.py
```

Scriptet gör:
- ✅ Aktiverar ROS2 Jazzy
- ✅ Aktiverar venv från `Python/.venv`
- ✅ Installerar requirements automatiskt om de saknas

## Standalone

This package is completely standalone:
- APIs are vendored in `apis/`
- No dependencies on other Mermaid projects
- Can be cloned and used independently
- Each module can be swapped without affecting others

## API Independence

Each API is independent and can be replaced:
- **Hand Recognition**: Replace `hand-recognition-api.py` with different recognition system
- **Overlay**: Replace `overlay-api.py` with different coordinate transformation
- **Layout**: Replace `layout-api.py` with different grid management

The `main.py` orchestrator connects all three APIs together following SoC principles.
