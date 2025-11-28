#!/usr/bin/env python3
"""
Integration-v1 - Main Entry Point

This orchestrates the APIs:
1. Hand Recognition API (from Interaction code) - Detects hand gestures and positions
2. Overlay API - Transforms coordinates (GPS → Grid cells)
3. Object Layout API - Manages grid map (walls, home, obstacles)
4. ROS2 Position API - Robot position tracking from GPS Server

Usage:
    python3 main.py                      # Hand + robot positions + grid
"""

import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent

# Insert paths in order of priority to ensure we get the right modules (vendored in apis/)
sys.path.insert(0, str(integration_root / "apis" / "overlay-api"))
sys.path.insert(0, str(integration_root / "apis" / "layout-api"))
sys.path.insert(0, str(integration_root / "apis" / "hand-recognition-api"))
sys.path.insert(0, str(integration_root / "apis" / "ros2-api"))
sys.path.insert(0, str(integration_root / "apis" / "astar-api"))
sys.path.insert(0, str(integration_root))

# Import APIs using simplified utility functions (vendored)

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
# Map gesture strings to grid values (support both semantic and raw gesture names)
GESTURE_TO_CELL = {
    "FOOD": HOME,
    "PALM": HOME,
    "OPEN_PALM": HOME,
    "THREAT": OBSTACLE,
    "FIST": OBSTACLE,
    "CLOSED_FIST": OBSTACLE,
}

# Hand recognition (Interaction code vendored locally)
hand_recognition_api = import_api(
    integration_root / "apis" / "hand-recognition-api" / "interaction-api.py",
    "hand_recognition_api",
    "hand-recognition-api not found; ensure apis/hand-recognition-api exists"
)
GestureRecognizer = hand_recognition_api.GestureRecognizer

astar_api = import_api(
    integration_root / "apis" / "astar-api" / "astar-api.py",
    "astar_api",
    "astar-api not found; ensure apis/astar-api exists"
)
astar_search, next_action_from_path, encode_action_ascii = extract_attrs(
    astar_api,
    'search',
    'next_action_from_path',
    'encode_action_ascii'
)


class HandGridDemo:
    """Demo that integrates hand recognition and robot tracking with grid updates."""
    
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
            min_certainty=0.35,
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
        self.robot_paths: Dict[int, Dict[str, Any]] = {}  # spiral_id -> path info
        self.pathfinder_interval = 10.0  # seconds between pathfinding attempts
        self._pathfinder_stop = threading.Event()
        self._pathfinder_thread: Optional[threading.Thread] = None
        
        # Temporary symbols overlay (for robots)
        # Maps (row, col) -> (symbol, original_symbol, type)
        # type: 'robot'
        self.temporary_symbols: Dict[Tuple[int, int], Tuple[str, str, str]] = {}
        
        print(f"Grid: {len(self.grid)}×{len(self.grid[0]) if self.grid else 0}")
        print(f"Gestures: Open_Palm→HOME ({get_symbol('HOME')}), Closed_Fist→OBSTACLE ({get_symbol('OBSTACLE')})")
        print("Robots: Tracking all active robots (0-9)")
        self._print_grid()
        self._start_pathfinder()
    
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
    
    def _add_temporary_symbol(self, row: int, col: int, symbol: str, symbol_type: str = 'robot') -> bool:
        """
        Add a temporary symbol to the grid, replacing the underlying symbol.
        Stores the original symbol so it can be restored.
        
        Args:
            row: Grid row
            col: Grid column
            symbol: Symbol to display (e.g., robot number)
            symbol_type: Type of symbol (robot, history)
        
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
        gesture_key = (gesture or "").upper()
        self.hand_position, self.hand_gesture = new_pos, gesture_key or None

        if gesture_key in GESTURE_TO_CELL:
            is_new = self.last_gesture_cell != new_pos
            gesture_changed = self.last_gesture_type != gesture_key
            current_value = self.grid[row][col] if (row < len(self.grid) and col < len(self.grid[0])) else FREE
            
            if (is_new and current_value == FREE) or (not is_new and gesture_changed):
                if not is_new and time.time() - self.last_update_time < self.update_cooldown:
                    return changed
                
                if self._update_cell(row, col, GESTURE_TO_CELL[gesture_key]):
                    self.last_gesture_cell, self.last_gesture_type = new_pos, gesture_key
                    self.last_update_time = time.time()
                    changed = True
        
        return changed
    
    def _get_robot_symbol(self, robot_id: int) -> str:
        """Get display symbol for a robot (white number)."""
        robot_char = str(robot_id % 10)
        return f"{Colors.WHITE}{robot_char}{Colors.RESET}"
    
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
                        changed = True
                    continue
                
                new_pos = (cell["row"], cell["col"])
                
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
                changed = True
        
        return changed

    def _start_pathfinder(self) -> None:
        """Start background pathfinder thread that mocks routes to nearest home."""
        if self._pathfinder_thread and self._pathfinder_thread.is_alive():
            return
        self._pathfinder_stop.clear()
        self._pathfinder_thread = threading.Thread(
            target=self._pathfinder_loop,
            name="PathfinderLoop",
            daemon=True,
        )
        self._pathfinder_thread.start()

    def _stop_pathfinder(self) -> None:
        """Signal pathfinder thread to stop and wait briefly."""
        self._pathfinder_stop.set()
        if self._pathfinder_thread:
            self._pathfinder_thread.join(timeout=1.0)

    def _pathfinder_loop(self) -> None:
        """Periodically recompute mock paths for all detected robots."""
        while not self._pathfinder_stop.is_set():
            try:
                self._update_robot_paths()
            except Exception as exc:
                # Keep loop alive even if path computation fails once
                print(f"[Pathfinder] Error computing paths: {exc}")
            self._pathfinder_stop.wait(self.pathfinder_interval)

    def _update_robot_paths(self) -> None:
        """Find mock paths for each robot to the nearest home tile."""
        positions = dict(self.robot_positions)
        if not positions:
            if self.robot_paths:
                self.robot_paths = {}
                self._clear_path_symbols()
            return

        grid_snapshot = [row[:] for row in self.grid] if self.grid else []
        if not grid_snapshot:
            return

        home_cells = self._collect_home_cells(grid_snapshot)
        if not home_cells:
            if self.robot_paths:
                self.robot_paths = {}
                self._clear_path_symbols()
            return

        new_paths: Dict[int, Dict[str, Any]] = {}
        for robot_id, start_rc in positions.items():
            plan = self._plan_robot_path(start_rc, grid_snapshot, home_cells)
            if plan:
                new_paths[robot_id] = plan

        if new_paths != self.robot_paths:
            self.robot_paths = new_paths
            self._log_robot_paths(new_paths)
            self._render_paths(new_paths)

    def _collect_home_cells(self, grid_snapshot: Grid) -> List[Tuple[int, int]]:
        """Return list of (row, col) cells marked as HOME."""
        homes: List[Tuple[int, int]] = []
        for row_idx, row in enumerate(grid_snapshot):
            for col_idx, value in enumerate(row):
                if value == HOME:
                    homes.append((row_idx, col_idx))
        return homes

    def _prepare_astar_grid(
        self,
        grid_snapshot: Grid,
        start_rc: Tuple[int, int],
        goal_rc: Tuple[int, int]
    ) -> Grid:
        """Build a grid tailored for the astar API (start=2, goal=4)."""
        prepared: Grid = []
        for row_idx, row in enumerate(grid_snapshot):
            new_row: List[int] = []
            for col_idx, value in enumerate(row):
                if (row_idx, col_idx) == start_rc:
                    new_row.append(2)
                elif (row_idx, col_idx) == goal_rc:
                    new_row.append(4)
                elif value == HOME:
                    new_row.append(FREE)
                else:
                    new_row.append(value)
            prepared.append(new_row)
        return prepared

    def _run_astar_search(self, astar_grid: Grid):
        """
        Run astar.search with tolerant unpacking (API returns 2 or 4 tuple values).
        Returns (nodes, path, start, goal)
        """
        result = astar_search(astar_grid, goaltype="HOME")
        nodes = 0
        path = None
        start_raw = None
        goal_raw = None
        if isinstance(result, tuple):
            if len(result) >= 2:
                nodes, path = result[0], result[1]
            if len(result) >= 3:
                start_raw = result[2]
            if len(result) >= 4:
                goal_raw = result[3]
        return nodes, path, start_raw, goal_raw

    def _plan_robot_path(
        self,
        start_rc: Tuple[int, int],
        grid_snapshot: Grid,
        home_cells: List[Tuple[int, int]]
    ) -> Optional[Dict[str, Any]]:
        """Select nearest home and compute a mock path + next action."""
        sr, sc = start_rc
        ordered_homes = sorted(
            home_cells,
            key=lambda h: abs(h[0] - sr) + abs(h[1] - sc)
        )

        for home_rc in ordered_homes:
            astar_grid = self._prepare_astar_grid(grid_snapshot, start_rc, home_rc)
            nodes, path, start_raw, goal_raw = self._run_astar_search(astar_grid)
            if not path:
                continue

            path_list = list(path)
            path_rc = [(y, x) for (x, y) in path_list]
            next_action, _ = next_action_from_path(path_list, init_heading="N", allow_back=False)
            return {
                "target_home": home_rc,
                "start": start_raw or (sc, sr),
                "goal": goal_raw or (home_rc[1], home_rc[0]),
                "path": path_rc,
                "steps": max(len(path_rc) - 1, 0),
                "nodes": nodes,
                "next_action": next_action,
                "next_action_ascii": encode_action_ascii(next_action)
            }
        return None

    def _log_robot_paths(self, paths: Dict[int, Dict[str, Any]]) -> None:
        """Print a lightweight summary of current mock robot paths."""
        if not paths:
            print("[Pathfinder] No robots to plan for.")
            self._clear_path_symbols()
            return

        print("[Pathfinder] Updated robot routes:")
        for rid, info in sorted(paths.items()):
            target = info.get("target_home")
            steps = info.get("steps", 0)
            action = info.get("next_action_ascii", "STOP")
            if target:
                print(f"  Robot {rid} -> home ({target[0]},{target[1]}): steps={steps} next={action}")
            else:
                print(f"  Robot {rid}: no reachable home")
        print()
    
    def _clear_path_symbols(self) -> None:
        """Remove temporary path overlays."""
        to_remove = [key for key, (_, _, typ) in self.temporary_symbols.items() if typ.startswith('path')]
        for row, col in to_remove:
            self._remove_temporary_symbol(row, col, 'path')

    def _render_paths(self, paths: Dict[int, Dict[str, Any]]) -> None:
        """Overlay current paths onto the printed grid."""
        self._clear_path_symbols()
        if not paths:
            return

        path_symbol = f"{Colors.CYAN}*{Colors.RESET}"
        for info in paths.values():
            path_cells = info.get("path", [])
            for row, col in path_cells:
                # Do not override robot markers
                existing = self.temporary_symbols.get((row, col))
                if existing and existing[2].startswith('robot'):
                    continue
                self._add_temporary_symbol(row, col, path_symbol, 'path')
    
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
        
        hand_symbol_key = None
        if self.hand_gesture:
            mapped = GESTURE_TO_CELL.get(self.hand_gesture)
            if mapped == HOME:
                hand_symbol_key = 'FOOD'
            elif mapped == OBSTACLE:
                hand_symbol_key = 'THREAT'

        for row in range(rows):
            print(f"{row % 10} ", end="")
            for col in range(cols):
                pos = (row, col)
                # Prioritera: Hand > Temporary symbols (robots, history) > Grid
                if self.hand_position == pos and hand_symbol_key:
                    symbol = get_symbol(hand_symbol_key)
                elif pos in self.temporary_symbols:
                    # Temporary symbol (robot, history) replaces underlying symbol
                    symbol, _, _ = self.temporary_symbols[pos]
                else:
                    symbol = map_data[row][col] if row < len(map_data) and col < len(map_data[row]) else "?"
                print(symbol, end=" ")
            print()
        
        print("=" * (cols * 2 + 1))
        symbols = {k: get_symbol(k) for k in ['FREE', 'HOME', 'OBSTACLE', 'FOOD', 'THREAT']}
        print(f"Legend: {symbols['FREE']}=FREE {symbols['HOME']}=HOME {symbols['OBSTACLE']}=OBSTACLE "
              f"{symbols['FOOD']}=Hand(Food) {symbols['THREAT']}=Hand(Threat)")
        print(f"Robots: {Colors.WHITE}0-9{Colors.RESET}=Current (white)")
        if self.hand_position:
            print(f"Hand: ({self.hand_position[0]}, {self.hand_position[1]})")
        if self.robot_positions:
            robot_info = ", ".join([f"S{rid}({pos[0]},{pos[1]})" for rid, pos in sorted(self.robot_positions.items())])
            print(f"Spirals: {robot_info}")
        if self.robot_paths:
            path_summaries = []
            for rid, info in sorted(self.robot_paths.items()):
                target = info.get("target_home")
                steps = info.get("steps", 0)
                action = info.get("next_action_ascii", "STOP")
                if target:
                    path_summaries.append(f"S{rid}->H({target[0]},{target[1]}): steps={steps} next={action}")
                else:
                    path_summaries.append(f"S{rid}: no path")
            print("Paths: " + "; ".join(path_summaries))
        print()
    
    def run(self):
        """Main demo loop."""
        try:
            print("Monitoring hand gestures and robot positions... (Press Ctrl+C to stop)\n")
            self.recognizer.run()
            while self.recognizer.running:
                # Uppdatera alla robotpositioner
                robot_changed = self.update_robot_positions()
                
                # Uppdatera handposition
                position = self.recognizer.get_position()
                hand_changed = False
                if position is not None:
                    x, y = position
                    if x is not None and y is not None:
                        hand_changed = self.update_hand_position(x, y, self.recognizer.get_gesture())
                
                # Uppdatera grid om något ändrats
                if robot_changed or hand_changed:
                    self._print_grid()
                
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_pathfinder()
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
