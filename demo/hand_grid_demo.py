#!/usr/bin/env python3
"""
Demo: Hand Recognition with Grid API Integration

Integrates hand recognition with grid API:
- Gets hand position from camera stream
- Converts GPS coordinates to grid cells using overlay API
- Updates grid based on gestures (in memory only):
  * Open_Palm (FOOD) -> Marks cell as HOME
  * Closed_Fist (THREAT) -> Marks cell as OBSTACLE
- Displays grid with hand position in real-time

Usage: python3 main.py
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import import_api, extract_attrs

integration_root = Path(__file__).parent.parent
sys.path.insert(0, str(integration_root / "hand_recognition"))
sys.path.insert(0, str(integration_root / "overlay"))
sys.path.insert(0, str(integration_root / "object-layout" / "api"))

# Import all three APIs
hand_recognition_api = import_api(
    integration_root / "hand_recognition" / "hand-recognition-api.py", 
    "hand_recognition_api"
)
GestureRecognizer = hand_recognition_api.GestureRecognizer

overlay_api = import_api(integration_root / "overlay" / "overlay-api.py", "overlay_api")
GPSOverlay = overlay_api.GPSOverlay

layout_api = import_api(integration_root / "object-layout" / "api" / "layout-api.py", "layout_api")
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME, SYMBOL_TO_CELL = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME', 'SYMBOL_TO_CELL'
)
Grid = List[List[int]]

# Gesture mapping
GESTURE_TO_CELL = {"FOOD": HOME, "THREAT": OBSTACLE}


class HandGridDemo:
    """Demo that integrates hand recognition with grid updates."""
    
    def __init__(self, overlay_json_path: str = None, grid_path: str = None):
        integration_root = Path(__file__).parent.parent
        
        # Initialize overlay
        overlay_json_path = overlay_json_path or str(integration_root / "overlay" / "gps_overlay.json")
        if not Path(overlay_json_path).exists():
            print(f"Error: {overlay_json_path} not found")
            sys.exit(1)
        
        self.overlay = GPSOverlay(overlay_json_path)
        self.grid_path = grid_path or integration_root / "object-layout" / "api" / "grid.json"
        self.recognizer = GestureRecognizer()
        self.grid = self._load_grid()
        
        # Tracking state
        self.hand_position = None
        self.hand_gesture = None
        self.last_gesture_cell = None
        self.last_gesture_type = None
        self.last_update_time = 0
        self.update_cooldown = 0.5
        
        print(f"Grid: {len(self.grid)}×{len(self.grid[0]) if self.grid else 0}")
        print(f"Gestures: Open_Palm→HOME ({get_symbol('HOME')}), Closed_Fist→OBSTACLE ({get_symbol('OBSTACLE')})")
        self._print_grid()
    
    def _load_grid(self) -> Grid:
        """Load grid from file or create empty grid."""
        if self.grid_path.exists():
            try:
                json_data = get_map_json(str(self.grid_path))
                if json_data:
                    grid = [[SYMBOL_TO_CELL.get(cell.upper(), FREE) if isinstance(cell, str) else int(cell) 
                            for cell in row] for row in json_data]
                    if len(grid) == self.overlay.grid_rows and len(grid[0]) == self.overlay.grid_cols:
                        return grid
            except Exception as e:
                print(f"Error loading grid: {e}")
        
        return [[FREE] * self.overlay.grid_cols for _ in range(self.overlay.grid_rows)]
    
    def _update_cell(self, row: int, col: int, value: int) -> bool:
        """Update grid cell if valid."""
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            self.grid[row][col] = value
            print(f"Updated ({row}, {col}): {value}")
            return True
        return False
    
    def update_hand_position(self, x: int, y: int, gesture: Optional[str] = None) -> bool:
        """Update hand position and apply gesture to grid."""
        # Get corrected cell position (with height offset)
        cell = self.overlay.get_grid_cell_with_height_offset(x, y, height_mm=1000.0)
        
        if not cell["in_bounds"]:
            changed = self.hand_position is not None
            self.hand_position = self.hand_gesture = None
            return changed
        
        row, col = cell["row"], cell["col"]
        new_pos = (row, col)
        changed = (self.hand_position != new_pos or self.hand_gesture != gesture)
        
        self.hand_position, self.hand_gesture = new_pos, gesture
        
        # Apply gesture to grid
        if gesture in GESTURE_TO_CELL:
            is_new = self.last_gesture_cell != new_pos
            gesture_changed = self.last_gesture_type != gesture
            current_value = self.grid[row][col] if (row < len(self.grid) and col < len(self.grid[0])) else FREE
            
            if (is_new and current_value == FREE) or (not is_new and gesture_changed):
                if not is_new and time.time() - self.last_update_time < self.update_cooldown:
                    return changed
                
                if self._update_cell(row, col, GESTURE_TO_CELL[gesture]):
                    self.last_gesture_cell, self.last_gesture_type = new_pos, gesture
                    self.last_update_time = time.time()
                    changed = True
        
        return changed
    
    def _print_grid(self):
        """Print grid with hand position overlay."""
        if not self.grid:
            return
        
        map_data = get_map(str(self.grid_path))
        if not map_data:
            return
        
        rows, cols = len(self.grid), len(self.grid[0])
        print("\n" + "=" * (cols * 2 + 1))
        print("Grid Map:")
        print("=" * (cols * 2 + 1))
        print("  " + " ".join(str(i % 10) for i in range(cols)))
        
        for row in range(rows):
            print(f"{row % 10} ", end="")
            for col in range(cols):
                if self.hand_position == (row, col) and self.hand_gesture:
                    symbol = get_symbol('FOOD' if self.hand_gesture == "FOOD" else 'THREAT')
                else:
                    symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                print(symbol, end=" ")
            print()
        
        print("=" * (cols * 2 + 1))
        symbols = {k: get_symbol(k) for k in ['FREE', 'HOME', 'OBSTACLE', 'FOOD', 'THREAT']}
        print(f"Legend: {symbols['FREE']}=FREE {symbols['HOME']}=HOME {symbols['OBSTACLE']}=OBSTACLE "
              f"{symbols['FOOD']}=Hand(Food) {symbols['THREAT']}=Hand(Threat)")
        if self.hand_position:
            print(f"Hand: ({self.hand_position[0]}, {self.hand_position[1]})")
        print()
    
    def run(self):
        """Main demo loop."""
        print("Starting hand recognition...")
        self.recognizer.run()
        
        try:
            print("Monitoring gestures... (Press ESC to stop)\n")
            while self.recognizer.running:
                x, y = self.recognizer.get_position()
                if x is not None and y is not None:
                    if self.update_hand_position(x, y, self.recognizer.get_gesture()):
                        self._print_grid()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.recognizer.stop()
            print("Demo finished!")
