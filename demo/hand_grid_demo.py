#!/usr/bin/env python3
"""
Demo: Hand Recognition with Grid API Integration

This demo integrates hand recognition coordinates with the grid API:
- Gets hand position coordinates from the camera stream
- Converts GPS server coordinates to grid cell positions using the overlay API
- Updates the grid based on gestures (in memory only):
  * Open_Palm (FOOD) -> Marks cell as HOME
  * Closed_Fist (THREAT) -> Marks cell as OBSTACLE
- Displays the grid with hand position in real-time
- Grid changes are only in memory and never saved to file

Architecture (SoC):
- Hand Recognition API: Detects gestures and positions
- Overlay API: Transforms coordinates (GPS → Grid cells)
- Object Layout API: Manages grid map (walls, home, obstacles)
- Demo: Orchestrates the three APIs

Usage:
    python3 main.py
"""

import sys
import os
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional

# Import utility functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import import_api, extract_attrs

# Add Integration-v1 root to path for imports
integration_root = Path(__file__).parent.parent

# IMPORTANT: Insert paths in order of priority to ensure we get the right modules
# Put Integration-v1 paths FIRST to avoid importing from old demo/ folder
sys.path.insert(0, str(integration_root / "hand_recognition"))
sys.path.insert(0, str(integration_root / "overlay"))
sys.path.insert(0, str(integration_root / "object-layout" / "api"))
sys.path.insert(0, str(integration_root))

# Import APIs (SoC - each API is independent) using simplified utility functions
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
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME, SYMBOL_TO_CELL = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME', 'SYMBOL_TO_CELL'
)
Grid = List[List[int]]


class HandGridDemo:
    """
    Demo class that integrates hand recognition with grid updates.
    
    Architecture (SoC):
    - Hand Recognition: Detects gestures and positions
    - Overlay: Transforms coordinates
    - Object Layout: Manages grid map
    - Demo: Orchestrates the three APIs
    
    This class:
    1. Initializes the hand recognizer and overlay API
    2. Loads the grid from file
    3. Continuously monitors hand gestures and positions
    4. Updates grid cells based on gestures (in memory only)
    5. Displays the grid with hand position in real-time
    """
    
    def __init__(self, overlay_json_path: str = None, grid_path: str = None):
        """
        Initialize the demo with overlay and grid paths.
        
        Args:
            overlay_json_path: Path to gps_overlay.json. If None, uses default location.
            grid_path: Path to grid.json. If None, uses default location.
        """
        # Paths relative to Integration-v1 root
        integration_root = Path(__file__).parent.parent
        
        # Initialize overlay API for coordinate conversion
        if overlay_json_path is None:
            default_overlay = integration_root / "overlay" / "gps_overlay.json"
            if default_overlay.exists():
                overlay_json_path = str(default_overlay)
            else:
                print(f"Warning: gps_overlay.json not found at {default_overlay}")
                print("Please specify overlay_json_path or ensure gps_overlay.json exists")
                sys.exit(1)
        
        print(f"Loading overlay from: {overlay_json_path}")
        self.overlay = GPSOverlay(overlay_json_path)
        print(f"Grid dimensions: {self.overlay.grid_cols} cols × {self.overlay.grid_rows} rows")
        
        # Grid path - use same path as map_api (object-layout/api/grid.json)
        if grid_path is None:
            self.grid_path = integration_root / "object-layout" / "api" / "grid.json"
        else:
            self.grid_path = Path(grid_path)
        
        # Initialize hand recognizer
        self.recognizer = hr.GestureRecognizer()
        
        # Load or create grid
        self.grid = self._load_or_create_grid()
        
        # Track last updated cell to avoid rapid updates
        self.last_cell = None
        self.last_update_time = 0
        self.update_cooldown = 0.5  # Minimum seconds between updates to same cell
        self.last_gesture_cell = None  # Track last cell where gesture was applied
        self.last_gesture_type = None  # Track last gesture type that was applied
        
        # Track current hand position on grid
        self.hand_position = None  # (row, col) tuple or None
        self.hand_gesture = None  # "FOOD" or "THREAT" or None
        
        # Calculate sample hand offset at center for display (hand is 1m above arena floor)
        center_row = self.overlay.grid_rows // 2
        center_col = self.overlay.grid_cols // 2
        sample_row_offset, sample_col_offset = self.calculate_hand_offset(center_row, center_col)
        
        print("\nDemo initialized!")
        print(f"Grid size: {len(self.grid)} rows × {len(self.grid[0]) if self.grid else 0} cols")
        print(f"Grid path: {self.grid_path}")
        print(f"Hand offset: position-dependent (sample at center: {sample_row_offset:.2f} rows, {sample_col_offset:.2f} cols)")
        print("\nGestures:")
        food_sym = get_symbol('FOOD')
        threat_sym = get_symbol('THREAT')
        home_sym = get_symbol('HOME')
        obstacle_sym = get_symbol('OBSTACLE')
        print(f"  - Open_Palm (FOOD) -> Marks cell as HOME ({home_sym}), shows as {food_sym}")
        print(f"  - Closed_Fist (THREAT) -> Marks cell as OBSTACLE ({obstacle_sym}), shows as {threat_sym}")
        print("  - Press ESC in the camera window to stop\n")
        
        # Show initial grid
        self._print_grid()
    
    def calculate_hand_offset(self, row_apparent: int, col_apparent: int) -> Tuple[float, float]:
        """
        Calculate position-dependent offset for hand 1m above arena floor.
        
        Uses perspective projection geometry:
        - Camera height: 2410 mm (187cm + 54cm)
        - Hand height: 1000 mm (1 meter)
        - Homography perspective component provides camera viewing angle
        - Offset varies with distance from center (perspective effect)
        
        Args:
            row_apparent: Apparent row position from overlay API
            col_apparent: Apparent column position from overlay API
        
        Returns:
            Tuple of (row_offset, col_offset) in grid cells.
            - Row offset: positive if below center (add), negative if above center (subtract)
            - Col offset: negative if right of center (subtract), positive if left of center (add)
        """
        # Physical measurements
        HAND_HEIGHT_MM = 1000     # 1 meter above arena floor
        
        # Extract camera viewing angle from homography
        h = self.overlay.homography
        perspective_scale = h[2][2]  # ~0.976 in your calibration
        
        # Calculate viewing angle
        camera_angle_rad = math.acos(perspective_scale)
        
        # Calculate base offset using perspective projection
        base_offset_mm = HAND_HEIGHT_MM * math.tan(camera_angle_rad)
        
        # Convert base offset to pixels
        base_offset_x_px = base_offset_mm / self.overlay.mm_per_pixel_x
        base_offset_y_px = base_offset_mm / self.overlay.mm_per_pixel_y
        
        # Calculate grid cell dimensions
        cell_width = (self.overlay.arena_bounds["right"] - 
                     self.overlay.arena_bounds["left"]) / self.overlay.grid_cols
        cell_height = (self.overlay.arena_bounds["bottom"] - 
                      self.overlay.arena_bounds["top"]) / self.overlay.grid_rows
        
        # Calculate center of grid
        center_row = self.overlay.grid_rows / 2.0
        center_col = self.overlay.grid_cols / 2.0
        
        # Calculate distance from center (normalized)
        row_distance = (row_apparent - center_row) / center_row  # -1 to +1 (negative = above, positive = below)
        col_distance = (col_apparent - center_col) / center_col  # -1 to +1 (negative = left, positive = right)
        
        # Convert base offset to grid cells
        base_row_offset_cells = base_offset_y_px / cell_height
        base_col_offset_cells = base_offset_x_px / cell_width
        
        # Apply position-dependent offset
        # Row: add if below center (positive distance), subtract if above center (negative distance)
        row_offset = base_row_offset_cells * row_distance
        
        # Col: subtract if right of center (positive distance), add if left of center (negative distance)
        col_offset = -base_col_offset_cells * col_distance  # Negative: subtract right, add left
        
        return (row_offset, col_offset)
    
    def _load_or_create_grid(self) -> Grid:
        """
        Load existing grid or create a new empty grid.
        
        Validates that loaded grid dimensions match overlay grid dimensions.
        If dimensions don't match, creates a new grid.
        
        Returns:
            Grid matrix (list of rows, each row is a list of cell values)
        """
        if self.grid_path.exists():
            print(f"Loading existing grid from {self.grid_path}")
            try:
                # Use map_api to get grid JSON, then convert to integer grid
                json_data = get_map_json(str(self.grid_path))
                if json_data and len(json_data) > 0:
                    # Convert JSON (symbols) to integer grid
                    grid = []
                    for row in json_data:
                        grid_row = []
                        for cell in row:
                            if isinstance(cell, str):
                                cell_value = SYMBOL_TO_CELL.get(cell.upper(), FREE)
                            else:
                                cell_value = int(cell)
                            grid_row.append(cell_value)
                        grid.append(grid_row)
                    
                    grid_rows = len(grid)
                    grid_cols = len(grid[0]) if grid_rows > 0 else 0
                    
                    # Validate dimensions match overlay
                    if grid_rows == self.overlay.grid_rows and grid_cols == self.overlay.grid_cols:
                        print(f"Loaded grid: {grid_rows} rows × {grid_cols} cols")
                        return grid
                    else:
                        print("Grid dimensions mismatch!")
                        print(f"  Loaded: {grid_rows} rows × {grid_cols} cols")
                        print(f"  Expected: {self.overlay.grid_rows} rows × {self.overlay.grid_cols} cols")
                        print("Creating new grid with correct dimensions...")
            except Exception as e:
                print(f"Error loading grid: {e}")
                print("Creating new grid...")
        
        # Create new empty grid (all FREE cells)
        print(f"Creating new grid: {self.overlay.grid_rows} rows × {self.overlay.grid_cols} cols")
        grid = [[FREE for _ in range(self.overlay.grid_cols)] 
                for _ in range(self.overlay.grid_rows)]
        
        # Grid is only created in memory for demo
        print("Created new grid in memory")
        
        return grid
    
    def _update_grid_cell(self, row: int, col: int, cell_value: int) -> bool:
        """
        Update a specific grid cell if valid.
        
        Args:
            row: Grid row index (0-based)
            col: Grid column index (0-based)
            cell_value: New cell value (FREE, OBSTACLE, or HOME)
        
        Returns:
            True if cell was updated, False otherwise
        """
        # Check bounds
        if row < 0 or row >= len(self.grid):
            return False
        if col < 0 or col >= len(self.grid[0]):
            return False
        
        # Update cell
        old_value = self.grid[row][col]
        self.grid[row][col] = cell_value
        
        # Print update info
        value_names = {FREE: "FREE", OBSTACLE: "OBSTACLE", HOME: "HOME"}
        print(f"Updated grid cell ({row}, {col}): {value_names.get(old_value, '?')} -> {value_names.get(cell_value, '?')}")
        
        return True
    
    def update_hand_position(self, x: int, y: int, gesture: Optional[str] = None) -> bool:
        """
        Update the current hand position on the grid.
        
        Args:
            x: GPS server X coordinate (pixel position)
            y: GPS server Y coordinate (pixel position)
            gesture: Gesture type ("FOOD", "THREAT", or None)
        
        Returns:
            True if position changed, False otherwise
        """
        # Transform coordinates through Overlay API
        cell_info = self.overlay.get_grid_cell(float(x), float(y))
        
        if not cell_info["in_bounds"]:
            # Hand is outside arena bounds
            position_changed = (self.hand_position is not None)
            self.hand_position = None
            self.hand_gesture = None
            return position_changed
        
        # Get apparent position
        row_apparent = cell_info["row"]
        col_apparent = cell_info["col"]
        
        # Calculate position-dependent offset for hand 1m above arena floor
        row_offset, col_offset = self.calculate_hand_offset(row_apparent, col_apparent)
        
        # Apply offsets
        row = int(round(row_apparent + row_offset))
        col = int(round(col_apparent + col_offset))
        
        # Verify bounds (safety check)
        if row < 0 or row >= len(self.grid) or col < 0 or col >= len(self.grid[0]):
            row = max(0, min(row, len(self.grid) - 1))
            col = max(0, min(col, len(self.grid[0]) - 1))
        
        # Check if position changed
        new_position = (row, col)
        position_changed = (self.hand_position != new_position or self.hand_gesture != gesture)
        
        # Update hand position tracking
        self.hand_position = new_position
        self.hand_gesture = gesture
        
        # If we have a gesture, update the grid cell
        if gesture == "FOOD":
            is_new_cell = self.last_gesture_cell != (row, col)
            gesture_changed = self.last_gesture_type != "FOOD"
            current_cell_value = self.grid[row][col] if (row < len(self.grid) and col < len(self.grid[0])) else FREE
            
            should_update = False
            if is_new_cell:
                should_update = (current_cell_value == FREE)
            else:
                should_update = gesture_changed
            
            if should_update:
                current_time = time.time()
                if not is_new_cell:
                    if current_time - self.last_update_time < self.update_cooldown:
                        return position_changed
                
                if self._update_grid_cell(row, col, HOME):
                    self.last_gesture_cell = (row, col)
                    self.last_gesture_type = "FOOD"
                    self.last_update_time = current_time
                    position_changed = True
        
        elif gesture == "THREAT":
            is_new_cell = self.last_gesture_cell != (row, col)
            gesture_changed = self.last_gesture_type != "THREAT"
            current_cell_value = self.grid[row][col] if (row < len(self.grid) and col < len(self.grid[0])) else FREE
            
            should_update = False
            if is_new_cell:
                should_update = (current_cell_value == FREE)
            else:
                should_update = gesture_changed
            
            if should_update:
                current_time = time.time()
                if not is_new_cell:
                    if current_time - self.last_update_time < self.update_cooldown:
                        return position_changed
                
                if self._update_grid_cell(row, col, OBSTACLE):
                    self.last_gesture_cell = (row, col)
                    self.last_gesture_type = "THREAT"
                    self.last_update_time = current_time
                    position_changed = True
        
        return position_changed
    
    def _print_grid(self):
        """Print the grid to terminal with colored symbols, showing hand position."""
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0
        
        if rows == 0 or cols == 0:
            print("Empty grid")
            return
        
        # Use Object Layout API to get grid with colored symbols
        map_data = get_map(str(self.grid_path))
        
        if not map_data:
            print("Could not load grid from map_api")
            return
        
        print("\n" + "=" * (cols * 2 + 1))
        print("Grid Map:")
        print("=" * (cols * 2 + 1))
        
        # Print column numbers
        print("  ", end="")
        for col in range(cols):
            print(f"{col % 10}", end=" ")
        print()
        
        # Print grid rows
        for row in range(rows):
            print(f"{row % 10} ", end="")
            for col in range(cols):
                # Check if this is the hand position with a gesture
                if self.hand_position == (row, col) and self.hand_gesture:
                    # Show hand symbol based on gesture
                    if self.hand_gesture == "FOOD":
                        symbol = get_symbol('FOOD')
                    elif self.hand_gesture == "THREAT":
                        symbol = get_symbol('THREAT')
                    else:
                        symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                else:
                    # Use colored symbol from Object Layout API
                    symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                print(symbol, end=" ")
            print()
        
        print("=" * (cols * 2 + 1))
        # Get symbols for legend
        free_sym = get_symbol('FREE')
        obstacle_sym = get_symbol('OBSTACLE')
        home_sym = get_symbol('HOME')
        food_sym = get_symbol('FOOD')
        threat_sym = get_symbol('THREAT')
        print(f"Legend: {free_sym} = FREE, {home_sym} = HOME, {obstacle_sym} = OBSTACLE, {food_sym} = Hand (Food), {threat_sym} = Hand (Threat)")
        if self.hand_position:
            print(f"Hand position: ({self.hand_position[0]}, {self.hand_position[1]})")
        print()
    
    def run(self):
        """
        Main demo loop: start hand recognition and process gestures.
        
        This runs continuously until the user presses ESC in the camera window.
        """
        # Start hand recognition in background thread
        print("Starting hand recognition...")
        self.recognizer.run()
        
        try:
            # Main processing loop
            print("Monitoring hand gestures...")
            print("(Press ESC in the camera window to stop)\n")
            
            while self.recognizer.running:
                # Get current hand position and gesture
                x, y = self.recognizer.get_position()
                gesture = self.recognizer.get_gesture()
                
                # Update hand position on grid
                if x is not None and y is not None:
                    position_changed = self.update_hand_position(x, y, gesture)
                    
                    # Print grid only if position changed
                    if position_changed:
                        self._print_grid()
                
                # Small sleep to avoid busy-waiting
                time.sleep(0.2)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Stop hand recognition
            print("\nStopping hand recognition...")
            self.recognizer.stop()
            
            print("Demo finished!")
