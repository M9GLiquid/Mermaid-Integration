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
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import import_api, extract_attrs

integration_root = Path(__file__).parent.parent
# APIs are now Git submodules in apis/ directory
sys.path.insert(0, str(integration_root / "apis" / "overlay-api"))
sys.path.insert(0, str(integration_root / "apis" / "layout-api"))
sys.path.insert(0, str(integration_root / "apis" / "hand-recognition-api"))
sys.path.insert(0, str(integration_root / "apis" / "ros2-api"))

# Import all three APIs (from Git submodules)
hand_recognition_api = import_api(
    integration_root / "apis" / "hand-recognition-api" / "hand-recognition-api.py", 
    "hand_recognition_api"
)
GestureRecognizer = hand_recognition_api.GestureRecognizer

overlay_api = import_api(
    integration_root / "apis" / "overlay-api" / "overlay-api.py", 
    "overlay_api"
)
GPSOverlay = overlay_api.GPSOverlay

layout_api = import_api(
    integration_root / "apis" / "layout-api" / "layout-api.py", 
    "layout_api"
)
get_map, get_map_json, get_symbol, FREE, OBSTACLE, HOME, SYMBOL_TO_CELL = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_symbol', 'FREE', 'OBSTACLE', 'HOME', 'SYMBOL_TO_CELL'
)
# Import Colors for robot display
Colors = layout_api.Colors
Grid = List[List[int]]

# Import ROS2 API
ros2_api = import_api(
    integration_root / "apis" / "ros2-api" / "ros2-api.py",
    "ros2_api",
    "Make sure ros2-api files are available: ros2-api.py and ros2.py"
)
RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow

# Gesture mapping
GESTURE_TO_CELL = {"FOOD": HOME, "THREAT": OBSTACLE}


class HandGridDemo:
    """Demo that integrates hand recognition with grid updates."""
    
    def __init__(self, overlay_json_path: str = None, grid_path: str = None):
        integration_root = Path(__file__).parent.parent
        
        # Initialize overlay
        # Use config files from API submodules
        overlay_json_path = overlay_json_path or str(integration_root / "apis" / "overlay-api" / "gps_overlay.json")
        if not Path(overlay_json_path).exists():
            print(f"Error: {overlay_json_path} not found")
            print("Make sure submodules are initialized: git submodule update --init")
            sys.exit(1)
        
        self.overlay = GPSOverlay(overlay_json_path)
        self.grid_path = grid_path or integration_root / "apis" / "layout-api" / "grid.json"
        self.recognizer = GestureRecognizer()
        self.grid = self._load_grid()
        
        # Initialize ROS2 API for robot tracking (spiral 9)
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
        
        # Temporary symbols overlay (for robots, hands, etc.)
        # Maps (row, col) -> (symbol, original_symbol)
        # These symbols temporarily replace underlying grid symbols
        self.temporary_symbols: Dict[Tuple[int, int], Tuple[str, str]] = {}
        
        print(f"Grid: {len(self.grid)}×{len(self.grid[0]) if self.grid else 0}")
        print(f"Gestures: Open_Palm→HOME ({get_symbol('HOME')}), Closed_Fist→OBSTACLE ({get_symbol('OBSTACLE')})")
        print(f"Robots: Tracking all active robots (0-9)")
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
    
    def _add_temporary_symbol(self, row: int, col: int, symbol: str) -> bool:
        """
        Add a temporary symbol to the grid, replacing the underlying symbol.
        Stores the original symbol so it can be restored.
        
        Args:
            row: Grid row
            col: Grid column
            symbol: Symbol to display (will replace underlying symbol)
        
        Returns:
            True if symbol was added/changed, False otherwise
        """
        if not (0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])):
            return False
        
        # Get current symbol at this position
        map_data = get_map(str(self.grid_path))
        if not map_data or row >= len(map_data) or col >= len(map_data[row]):
            return False
        
        current_symbol = map_data[row][col]
        
        # Check if there's already a temporary symbol here
        pos = (row, col)
        if pos in self.temporary_symbols:
            existing_symbol, original = self.temporary_symbols[pos]
            if existing_symbol == symbol:
                return False  # Same symbol, no change
            # Update symbol but keep original
            self.temporary_symbols[pos] = (symbol, original)
            return True
        
        # Add new temporary symbol
        self.temporary_symbols[pos] = (symbol, current_symbol)
        return True
    
    def _remove_temporary_symbol(self, row: int, col: int) -> bool:
        """
        Remove a temporary symbol from the grid, restoring the original symbol.
        
        Args:
            row: Grid row
            col: Grid column
        
        Returns:
            True if symbol was removed, False if no symbol was there
        """
        pos = (row, col)
        if pos in self.temporary_symbols:
            del self.temporary_symbols[pos]
            return True
        return False
    
    def _move_temporary_symbol(self, old_row: int, old_col: int, new_row: int, new_col: int) -> bool:
        """
        Move a temporary symbol from one position to another.
        
        Args:
            old_row: Old grid row
            old_col: Old grid column
            new_row: New grid row
            new_col: New grid column
        
        Returns:
            True if symbol was moved, False otherwise
        """
        old_pos = (old_row, old_col)
        if old_pos not in self.temporary_symbols:
            return False
        
        # Get symbol and original from old position
        symbol, original = self.temporary_symbols[old_pos]
        
        # Remove from old position
        del self.temporary_symbols[old_pos]
        
        # Add to new position (with new original)
        map_data = get_map(str(self.grid_path))
        if map_data and new_row < len(map_data) and new_col < len(map_data[new_row]):
            new_original = map_data[new_row][new_col]
            self.temporary_symbols[(new_row, new_col)] = (symbol, new_original)
            return True
        
        return False
    
    def _get_robot_symbol(self, robot_id: int) -> str:
        """
        Get colored robot symbol showing robot ID.
        
        Args:
            robot_id: Robot ID (0-9)
        
        Returns:
            Colored symbol string (yellow robot ID)
        """
        robot_char = str(robot_id % 10)  # Ensure 0-9
        return f"{Colors.YELLOW}{robot_char}{Colors.RESET}"
    
    def update_robot_positions(self) -> bool:
        """Update all robot positions from ROS2 API and convert to grid cells."""
        current_time = time.time()
        if current_time - self.last_robot_update < self.robot_update_interval:
            return False
        
        # Hämta alla robotpositioner från ROS2 API
        all_positions = self.robot_api.getPosition()  # None = alla robotar
        self.last_robot_update = current_time
        
        # Skapa set med nuvarande aktiva robotar
        current_robot_ids = {pos.id for pos in all_positions}
        old_robot_positions = self.robot_positions.copy()
        
        changed = False
        
        # Uppdatera eller lägg till robotar
        for position in all_positions:
            robot_id = position.id
            old_pos = old_robot_positions.get(robot_id)
            
            # Transformera GPS-koordinater till grid cell (robotar är på golvet, ingen höjdoffset)
            # OBS: SpiralRow.row och SpiralRow.col kan vara ombytta eller i annat koordinatsystem
            try:
                # Testa olika kombinationer för att hitta rätt mapping
                # Först: använd col som X och row som Y (GPS koordinatsystem)
                cell = self.overlay.get_grid_cell(
                    int(position.col),  # X-koordinat från GPS
                    int(position.row)   # Y-koordinat från GPS
                )
                
                if not cell["in_bounds"]:
                    # Robot utanför grid - ta bort från grid
                    if old_pos is not None:
                        self._remove_temporary_symbol(old_pos[0], old_pos[1])
                        del self.robot_positions[robot_id]
                        changed = True
                    continue
                
                new_pos = (cell["row"], cell["col"])
                
                if old_pos == new_pos:
                    # Samma position - ingen ändring
                    continue
                
                # Robot har flyttat sig eller är ny
                robot_symbol = self._get_robot_symbol(robot_id)
                
                if old_pos is None:
                    # Ny robot - lägg till symbol
                    self._add_temporary_symbol(new_pos[0], new_pos[1], robot_symbol)
                else:
                    # Robot flyttat - flytta symbol
                    self._move_temporary_symbol(old_pos[0], old_pos[1], new_pos[0], new_pos[1])
                
                self.robot_positions[robot_id] = new_pos
                changed = True
            except Exception:
                # Om transformation misslyckas, hoppa över denna robot
                continue
        
        # Ta bort robotar som inte längre är aktiva
        for robot_id, old_pos in old_robot_positions.items():
            if robot_id not in current_robot_ids:
                self._remove_temporary_symbol(old_pos[0], old_pos[1])
                del self.robot_positions[robot_id]
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
                pos = (row, col)
                # Prioritera: Hand > Temporary symbols (robots) > Grid
                if self.hand_position == pos and self.hand_gesture:
                    symbol = get_symbol('FOOD' if self.hand_gesture == "FOOD" else 'THREAT')
                elif pos in self.temporary_symbols:
                    # Temporary symbol (robot) replaces underlying symbol
                    symbol, _ = self.temporary_symbols[pos]
                else:
                    symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                print(symbol, end=" ")
            print()
        
        print("=" * (cols * 2 + 1))
        symbols = {k: get_symbol(k) for k in ['FREE', 'HOME', 'OBSTACLE', 'FOOD', 'THREAT']}
        print(f"Legend: {symbols['FREE']}=FREE {symbols['HOME']}=HOME {symbols['OBSTACLE']}=OBSTACLE "
              f"{symbols['FOOD']}=Hand(Food) {symbols['THREAT']}=Hand(Threat) Yellow=Robot(ID)")
        if self.hand_position:
            print(f"Hand: ({self.hand_position[0]}, {self.hand_position[1]})")
        if self.robot_positions:
            robot_info = ", ".join([f"R{rid}({pos[0]},{pos[1]})" for rid, pos in sorted(self.robot_positions.items())])
            print(f"Robots: {robot_info}")
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
            for pos in list(self.temporary_symbols.keys()):
                self._remove_temporary_symbol(pos[0], pos[1])
            self.recognizer.stop()
            self.robot_api.stop()
            print("Demo finished!")
