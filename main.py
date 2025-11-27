#!/usr/bin/env python3
"""
Integration-v1 - Main Entry Point

This orchestrates the APIs:
1. Hand Recognition API - Detects hand gestures and positions
2. Overlay API - Transforms coordinates (GPS → Grid cells)
3. Object Layout API - Manages grid map (walls, home, obstacles)
4. ROS2 Position API - Robot position tracking from GPS Server

Usage:
    python3 main.py                      # Hand recognition + grid + robots
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent

# Insert paths in order of priority to ensure we get the right modules (vendored in apis/)
sys.path.insert(0, str(integration_root / "apis" / "overlay-api"))
sys.path.insert(0, str(integration_root / "apis" / "layout-api"))
sys.path.insert(0, str(integration_root / "apis" / "hand-recognition-api"))
sys.path.insert(0, str(integration_root))

# Import APIs using simplified utility functions (vendored)
hand_recognition_api = import_api(
    integration_root / "apis" / "hand-recognition-api" / "hand-recognition-api.py",
    "hand_recognition_api",
    "hand-recognition-api not found; ensure apis/hand-recognition-api exists"
)
GestureRecognizer = hand_recognition_api.GestureRecognizer

overlay_api = import_api(
    integration_root / "apis" / "overlay-api" / "overlay-api.py",
    "overlay_api",
    "overlay-api not found; ensure apis/overlay-api exists"
)
GPSOverlay = overlay_api.GPSOverlay

layout_api = import_api(
    integration_root / "apis" / "layout-api" / "layout-api.py",
    "layout_api",
    "layout-api not found; ensure apis/layout-api exists"
)
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME, SYMBOL_TO_CELL, Colors = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME', 'SYMBOL_TO_CELL', 'Colors'
)

# Import ROS2 API for robot position tracking
ros2_api = import_api(
    integration_root / "apis" / "ros2-api" / "ros2-api.py",
    "ros2_api",
    "Make sure ros2-api files are available: ros2-api.py and ros2.py"
)
RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow

Grid = List[List[int]]
# Gesture mapping
GESTURE_TO_CELL = {"FOOD": HOME, "THREAT": OBSTACLE}


class HandGridDemo:
    """Demo that integrates hand recognition with grid updates."""
    
    def __init__(self, overlay_json_path: Optional[Path] = None, grid_path: Optional[Path] = None):
        # Initialize overlay
        # Use config files from vendored APIs
        overlay_path = Path(overlay_json_path) if overlay_json_path is not None else integration_root / "apis" / "overlay-api" / "gps_overlay.json"
        if not overlay_path.exists():
            print(f"Error: {overlay_path} not found")
            print("Ensure apis/overlay-api exists and contains gps_overlay.json")
            sys.exit(1)
        
        self.overlay = GPSOverlay(str(overlay_path))
        self.grid_path: Path = Path(grid_path) if grid_path is not None else integration_root / "apis" / "layout-api" / "grid.json"
        self.recognizer = GestureRecognizer()
        self.grid = self._load_grid()
        
        # Initialize ROS2 API for robot tracking
        self.robot_api = RobotPositionAPI(
            topic='robotPositions',
            msg_type='string',
            min_certainty=0.25,
            max_speed=500.0
        )
        self.robot_api.start()
        
        # Tracking state
        self.hand_position = None
        self.hand_gesture = None
        self.last_gesture_cell = None
        self.last_gesture_type = None
        self.last_update_time = 0
        self.update_cooldown = 0.5
        
        # Robot tracking state - trackar alla robotar
        self.robot_positions: Dict[int, Tuple[int, int]] = {}  # spiral_id -> (row, col)
        self.last_robot_update = 0
        self.robot_update_interval = 0.5  # Uppdatera robotar var 0.5 sekund
        
        # Position history tracking (last 2 positions per spiral for color fading)
        self.position_history: Dict[int, List[Tuple[int, int]]] = {}  # spiral_id -> [(row, col), ...]
        self.max_history_size = 2  # Keep 2 for color fading (darker → darkest)
        
        # Predicted paths tracking (2-4 grid cells ahead)
        self.predicted_paths: Dict[int, List[Tuple[int, int]]] = {}  # spiral_id -> [(row, col), ...]
        self.prediction_steps = 4  # Predict 4 grid cells ahead
        
        # Context-aware display settings
        self.show_history_default = True  # Show history by default
        self.show_predictions_default = True  # Show predictions by default
        self.proximity_threshold = 5  # Show full details when spirals are within 5 grid cells
        self.min_history_display = 1  # Minimum history to show (even when not in proximity)
        self.min_prediction_display = 2  # Minimum predictions to show (even when not in proximity)
        
        # All robots use white color - number differentiates them
        # History: darker white, Future: brighter white
        
        # Temporary symbols overlay (for robots, history, predictions, etc.)
        # Maps (row, col) -> (symbol, original_symbol, type)
        # type: 'robot', 'history_0', 'history_1', 'history_2', 'history_3', 'prediction_0', 'prediction_1', 'prediction_2', 'prediction_3'
        self.temporary_symbols: Dict[Tuple[int, int], Tuple[str, str, str]] = {}
        
        print(f"Grid: {len(self.grid)}×{len(self.grid[0]) if self.grid else 0}")
        print(f"Gestures: Open_Palm→HOME ({get_symbol('HOME')}), Closed_Fist→OBSTACLE ({get_symbol('OBSTACLE')})")
        print("Robots: Tracking all active robots (0-9)")
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
    
    def _add_temporary_symbol(self, row: int, col: int, symbol: str, symbol_type: str = 'robot') -> bool:
        """
        Add a temporary symbol to the grid, replacing the underlying symbol.
        Stores the original symbol so it can be restored.
        
        Args:
            row: Grid row
            col: Grid column
            symbol: Symbol to display (e.g., robot number)
            symbol_type: Type of symbol (robot, history, prediction)
        
        Returns:
            bool: True if added, False if out of bounds
        """
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            # Get original symbol from grid map (not overlay)
            original_symbol = get_symbol('FREE')
            map_data = get_map(str(self.grid_path))
            if map_data and row < len(map_data) and col < len(map_data[0]):
                original_symbol = map_data[row][col]
            
            # Store temporary symbol with original
            self.temporary_symbols[(row, col)] = (symbol, original_symbol, symbol_type)
            return True
        return False
    
    def _remove_temporary_symbol(self, row: int, col: int, symbol_type: Optional[str] = None) -> bool:
        """Remove temporary symbol and restore original."""
        key = (row, col)
        if key not in self.temporary_symbols:
            return False
        
        if symbol_type is not None and not self.temporary_symbols[key][2].startswith(symbol_type):
            return False
        
        original_symbol = self.temporary_symbols[key][1]
        map_data = get_map(str(self.grid_path))
        if map_data and row < len(map_data) and col < len(map_data[0]):
            map_data[row][col] = original_symbol
        
        del self.temporary_symbols[key]
        return True
    
    def _move_temporary_symbol(self, old_row: int, old_col: int, new_row: int, new_col: int) -> bool:
        """
        Move a temporary symbol from one cell to another.
        Restores original symbol at old position and preserves it at new position.
        """
        old_key = (old_row, old_col)
        if old_key not in self.temporary_symbols:
            return False
        
        symbol, _, symbol_type = self.temporary_symbols[old_key]
        
        if not self._add_temporary_symbol(new_row, new_col, symbol, symbol_type):
            return False
        
        self._remove_temporary_symbol(old_row, old_col, symbol_type)
        return True
    
    def _clear_symbols_by_type(self, symbol_type_prefix: str):
        """Remove temporary symbols matching the given type prefix."""
        to_remove = [pos for pos, (_, _, sym_type) in self.temporary_symbols.items()
                    if sym_type.startswith(symbol_type_prefix)]
        for pos in to_remove:
            self._remove_temporary_symbol(pos[0], pos[1], symbol_type_prefix)
    
    def _get_robot_symbol(self, robot_id: int) -> str:
        """Get display symbol for a robot (white number)."""
        robot_char = str(robot_id % 10)
        return f"{Colors.WHITE}{robot_char}{Colors.RESET}"
    
    def _get_history_symbol(self, robot_id: int, age: int) -> str:
        """Get display symbol for history (darker white)."""
        robot_char = str(robot_id % 10)
        # Darker white for history (using bright black or gray)
        if age == 0:
            color = '\033[37m'  # Light gray
        elif age == 1:
            color = '\033[90m'  # Dark gray
        else:
            color = '\033[90m'  # Default to dark gray for older history
        
        return f"{color}{robot_char}{Colors.RESET}"
    
    def _get_prediction_symbol(self, robot_id: int, step: int) -> str:
        """Get display symbol for prediction (brighter white)."""
        robot_char = str(robot_id % 10)
        # Brighter white for predictions
        color = '\033[97;1m'  # Bright white with bold
        
        return f"{color}{robot_char}{Colors.RESET}"
    
    def _update_position_history(self, spiral_id: int, new_pos: Tuple[int, int]):
        """
        Update position history for a spiral (keep last 4 positions).
        
        Args:
            spiral_id: Spiral ID
            new_pos: New position (row, col)
        """
        if spiral_id not in self.position_history:
            self.position_history[spiral_id] = []
        
        history = self.position_history[spiral_id]
        
        # Add new position if it's different from the last one
        if not history or history[-1] != new_pos:
            history.append(new_pos)
            # Keep only last max_history_size positions
            if len(history) > self.max_history_size:
                history.pop(0)
    
    def _update_predicted_paths(self):
        """
        Update predicted paths for all spirals.
        Simple prediction: current position + movement direction = future positions.
        Uses grid cell coordinates directly.
        """
        # Clear old predictions
        self._clear_symbols_by_type('prediction_')
        self.predicted_paths.clear()
        
        try:
            # For each spiral with history, predict future positions
            for spiral_id, history in self.position_history.items():
                if len(history) < 2:
                    continue  # Need at least 2 positions for prediction
                
                # Get current and previous positions (grid cell coordinates)
                current_pos = history[-1]  # (row, col)
                prev_pos = history[-2]     # (row, col)
                
                # Calculate movement direction (simple: current - previous)
                dr = current_pos[0] - prev_pos[0]  # Row change
                dc = current_pos[1] - prev_pos[1]  # Col change
                
                # If no movement, don't predict
                if dr == 0 and dc == 0:
                    continue
                
                # Predict future positions by continuing in the same direction
                predicted = []
                for step in range(1, self.prediction_steps + 1):
                    pred_row = current_pos[0] + dr * step
                    pred_col = current_pos[1] + dc * step
                    
                    # Check bounds
                    if 0 <= pred_row < len(self.grid) and 0 <= pred_col < len(self.grid[0]):
                        predicted.append((int(pred_row), int(pred_col)))
                    else:
                        break  # Stop if we hit a boundary
                
                if predicted:
                    self.predicted_paths[spiral_id] = predicted
        except Exception as e:
            # Debug: print error if prediction fails
            print(f"Prediction error: {e}")
    
    def _is_spiral_near_others(self, spiral_id: int) -> bool:
        """
        Check if a spiral is near other spirals (within proximity threshold).
        """
        if spiral_id not in self.robot_positions:
            return False
        
        current_pos = self.robot_positions[spiral_id]
        
        # Check proximity to other spirals
        for other_id, other_pos in self.robot_positions.items():
            if other_id == spiral_id:
                continue
            
            # Calculate Manhattan distance
            distance = abs(current_pos[0] - other_pos[0]) + abs(current_pos[1] - other_pos[1])
            if distance <= self.proximity_threshold:
                return True
        
        return False
    
    def _display_history_and_predictions(self):
        """
        Display position history and predicted paths on the grid.
        Context-aware: Shows more details when spirals are near others.
        Shows same robot number with different colors: darker for history, lighter for predictions.
        """
        # Clear old history and prediction symbols
        self._clear_symbols_by_type('history_')
        
        # Display history positions (same number as robot, darker color)
        for spiral_id, history in self.position_history.items():
            if len(history) == 0:
                continue
            
            # Determine how much history to show based on context
            is_near = self._is_spiral_near_others(spiral_id)
            history_to_show = self.max_history_size if is_near else self.min_history_display
            
            # Show history positions
            for age, pos in enumerate(reversed(history[-history_to_show:])):
                if age < history_to_show:
                    history_symbol = self._get_history_symbol(spiral_id, age)
                    self._add_temporary_symbol(pos[0], pos[1], history_symbol, f'history_{age}')
        
        # Display predicted paths (same number as robot, lighter color)
        for spiral_id, predicted in self.predicted_paths.items():
            if len(predicted) == 0:
                continue
            
            # Determine how many predictions to show based on context
            is_near = self._is_spiral_near_others(spiral_id)
            predictions_to_show = self.prediction_steps if is_near else self.min_prediction_display
            
            for step, pos in enumerate(predicted):
                if step < predictions_to_show:
                    prediction_symbol = self._get_prediction_symbol(spiral_id, step)
                    self._add_temporary_symbol(pos[0], pos[1], prediction_symbol, f'prediction_{step}')
    
    def update_robot_positions(self) -> bool:
        """Update all robot positions from ROS2 API and convert to grid cells."""
        current_time = time.time()
        if current_time - self.last_robot_update < self.robot_update_interval:
            return False
        
        # Hämta alla robotpositioner från ROS2 API (alla spirals individuellt)
        all_positions = self.robot_api.getPosition()  # None = alla robotar/spirals
        self.last_robot_update = current_time
        
        # Skapa set med nuvarande aktiva spirals
        current_spiral_ids = {pos.id for pos in all_positions}
        old_robot_positions = self.robot_positions.copy()
        
        changed = False
        
        # Uppdatera eller lägg till spirals
        for position in all_positions:
            spiral_id = position.id
            old_pos = old_robot_positions.get(spiral_id)
            
            # Transformera GPS-koordinater till grid cell (robotar är på golvet, ingen höjdoffset)
            try:
                cell = self.overlay.get_grid_cell(
                    int(position.col),  # X-koordinat från GPS
                    int(position.row)   # Y-koordinat från GPS
                )
                
                if not cell["in_bounds"]:
                    # Spiral utanför grid - ta bort från grid
                    if old_pos is not None:
                        self._remove_temporary_symbol(old_pos[0], old_pos[1], 'robot')
                        del self.robot_positions[spiral_id]
                        if spiral_id in self.position_history:
                            del self.position_history[spiral_id]
                        changed = True
                    continue
                
                new_pos = (cell["row"], cell["col"])
                
                # Update position history
                self._update_position_history(spiral_id, new_pos)
                
                if old_pos == new_pos:
                    # Samma position - ingen ändring för robot symbol
                    continue
                
                # Spiral har flyttat sig eller är ny
                robot_symbol = self._get_robot_symbol(spiral_id)
                
                if old_pos is None:
                    # Ny spiral - lägg till symbol
                    self._add_temporary_symbol(new_pos[0], new_pos[1], robot_symbol, 'robot')
                else:
                    # Spiral flyttat - flytta symbol
                    self._move_temporary_symbol(old_pos[0], old_pos[1], new_pos[0], new_pos[1])
                
                self.robot_positions[spiral_id] = new_pos
                changed = True
            except Exception:
                # Om transformation misslyckas, hoppa över denna spiral
                continue
        
        # Ta bort spirals som inte längre är aktiva
        for spiral_id, old_pos in old_robot_positions.items():
            if spiral_id not in current_spiral_ids:
                self._remove_temporary_symbol(old_pos[0], old_pos[1], 'robot')
                del self.robot_positions[spiral_id]
                if spiral_id in self.position_history:
                    del self.position_history[spiral_id]
                changed = True
        
        # Update predicted paths
        self._update_predicted_paths()
        
        # Display history and predictions
        self._display_history_and_predictions()
        
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
                pos = (row, col)
                # Prioritera: Hand > Temporary symbols (robots) > Grid
                if self.hand_position == pos and self.hand_gesture:
                    symbol = get_symbol('FOOD' if self.hand_gesture == "FOOD" else 'THREAT')
                elif pos in self.temporary_symbols:
                    # Temporary symbol (robot, history, prediction) replaces underlying symbol
                    symbol, _, _ = self.temporary_symbols[pos]
                else:
                    symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                print(symbol, end=" ")
            print()
        
        print("=" * (cols * 2 + 1))
        symbols = {k: get_symbol(k) for k in ['FREE', 'HOME', 'OBSTACLE', 'FOOD', 'THREAT']}
        print(f"Legend: {symbols['FREE']}=FREE {symbols['HOME']}=HOME {symbols['OBSTACLE']}=OBSTACLE "
              f"{symbols['FOOD']}=Hand(Food) {symbols['THREAT']}=Hand(Threat)")
        print(f"Robots: {Colors.WHITE}0-9{Colors.RESET}=Current (white) | "
              f"\033[37m0-9\033[0m=History (darker white) | "
              f"\033[97;1m0-9\033[0m=Prediction (brighter white)")
        if self.hand_position:
            print(f"Hand: ({self.hand_position[0]}, {self.hand_position[1]})")
        if self.robot_positions:
            robot_info = ", ".join([f"S{rid}({pos[0]},{pos[1]})" for rid, pos in sorted(self.robot_positions.items())])
            print(f"Spirals: {robot_info}")
        print()
    
    def run(self):
        """Main demo loop."""
        print("Starting hand recognition...")
        self.recognizer.run()
        
        try:
            print("Monitoring gestures and robot positions... (Press ESC to stop)\n")
            while self.recognizer.running:
                # Uppdatera alla robotpositioner
                robot_changed = self.update_robot_positions()
                
                # Uppdatera handposition
                x, y = self.recognizer.get_position()
                hand_changed = False
                if x is not None and y is not None:
                    hand_changed = self.update_hand_position(x, y, self.recognizer.get_gesture())
                
                # Uppdatera grid om något ändrats
                if robot_changed or hand_changed:
                    self._print_grid()
                
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup: remove all temporary symbols
            while self.temporary_symbols:
                pos = next(iter(self.temporary_symbols))
                self._remove_temporary_symbol(pos[0], pos[1])
            self.recognizer.stop()
            self.robot_api.stop()
            print("Demo finished!")


def run_hand_grid_demo(overlay_json_path: Optional[Path] = None, grid_path: Optional[Path] = None):
    """Helper to run the hand + grid + robot tracking demo."""
    demo = HandGridDemo(overlay_json_path=overlay_json_path, grid_path=grid_path)
    demo.run()


def main():
    run_hand_grid_demo()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
