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

## Quick Start (from clone to run)

```bash
# 1) Clone
git clone <repository-url> Mermaid-Integration
cd Mermaid-Integration

# 2) (Optional but recommended) Activate shared env + ROS2 setup
# This repo expects the shared setup script one level up (../setup_env.sh)
source ../setup_env.sh

# 3) Install python deps (skip if setup_env already handled it)
pip install -r requirements.txt

# 4) Run the integration demo
python3 main.py

# 5) Run API tests
python3 run_tests.py
```

What main.py does:
1. Initializes all vendored APIs.
2. Starts hand recognition from camera stream.
3. Transforms hand coordinates to grid cells.
4. Updates grid based on gestures (in memory only).
5. Prints the grid with colored symbols in the terminal.

## Demos & Tests

- Full integration: `python3 main.py`
- Hand+grid demo: `python3 demo/hand_grid_demo.py`
- Robot position demo (needs ROS2 publisher): `python3 demo/robot_position_demo.py`
- All tests: `python3 run_tests.py`
- Individual tests:
  - `python3 test_layout_api.py`
  - `python3 test_overlay_api.py`
  - `python3 test_overlay_api.py test_coordinates`
  - `python3 test_overlay_api.py test_grid_cells`
  - `python3 test_overlay_api.py test_stream_transform`

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

`setup_env.sh` (one directory above this repo) activates:
- ROS2 Jazzy environment
- Python venv at `../Python/.venv`
- Installs `requirements.txt` on first run if needed

If you do not have ROS2, you can still run layout/overlay/hand tests; ROS2-dependent demos require a sourced ROS2 environment.

## Updating vendored APIs (git subtree)

APIs are vendored (no submodules). To pull upstream changes (from the local sibling repos used here), run:

```bash
# Layout API (from Mermaid-Layout/api)
git subtree pull --prefix=apis/layout-api /home/thomas/Dev/Python/Mermaid-Layout export-api --squash

# Overlay API (from Mermaid-Overlay/api)
git subtree pull --prefix=apis/overlay-api /home/thomas/Dev/Python/Mermaid-Overlay export-api --squash

# Hand Recognition API (from Mermaid-Interaction/hand-recognition)
git subtree pull --prefix=apis/hand-recognition-api /home/thomas/Dev/Python/Mermaid-Interaction export-hand --squash

# ROS2 Position API (from Mermaid-Ros2-Comm/api)
git subtree pull --prefix=apis/ros2-api /home/thomas/Dev/Python/Mermaid-Ros2-Comm export-api --squash
```

If you’re cloning this somewhere else, replace the `/home/thomas/Dev/...` paths with the appropriate remote URLs or local paths for your upstreams.

## Standalone & API Independence

- APIs are included in `apis/` — no submodules required.
- You can swap any API by replacing its files under `apis/<name>-api/`.
- Each API remains independent; `main.py` simply orchestrates them together.
