"""
Simple API for accessing the occupancy grid map with customizable symbols and colors.

This API provides easy access to the grid map data stored in grid.json (located
in the same directory as this API), allowing users to retrieve the map with 
custom symbol representations, colors, and additional symbols for entities.

The grid.json file is included in the api folder for easy distribution - just
copy the entire api folder to use this API.

Example:
    from layout_api import get_map, get_symbol, get_symbols
    
    # Get map with default colored symbols
    map_data = get_map()
    
    # Get map with custom symbols
    map_data = get_map(symbols={'FREE': '.', 'OBSTACLE': '#', 'HOME': '1'})
    
    # Get specific symbols programmatically
    food_symbol = get_symbol('FOOD')  # Returns: 🍎
    threat_symbol = get_symbol('THREAT')  # Returns: ⚠️
    robot_symbol = get_symbol('ROBOT')  # Returns: 🤖
    
    # Get all available symbols
    all_symbols = get_symbols()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Iterable
import json

# Grid cell constants (standalone - no external dependencies)
FREE = 0
OBSTACLE = 1
HOME = 2

SYMBOL_TO_CELL = {"O": FREE, "X": OBSTACLE, "H": HOME}

# Default path to grid.json (in the same directory as this API)
DEFAULT_MAP_PATH = Path(__file__).parent / "grid.json"

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    GREEN = '\033[92m'      # Bright green
    RED = '\033[91m'        # Bright red
    YELLOW = '\033[93m'     # Bright yellow
    BLUE = '\033[94m'       # Bright blue
    MAGENTA = '\033[95m'    # Bright magenta
    CYAN = '\033[96m'       # Bright cyan
    WHITE = '\033[97m'      # Bright white
    GRAY = '\033[90m'       # Gray

# Symbol registry - all available symbols (with colors by default)
SYMBOL_REGISTRY = {
    # Grid cell types (with colors)
    'FREE': f'{Colors.GRAY}.{Colors.RESET}',      # Gray dot
    'OBSTACLE': f'{Colors.RED}#{Colors.RESET}',   # Red hash
    'HOME': f'{Colors.GREEN}H{Colors.RESET}',     # Green H
    
    # Entity symbols
    'FOOD': '🍎',      # Apple - Food gesture
    'THREAT': '⚠️',    # Warning - Threat gesture
    'ROBOT': '🤖',     # Robot - Spiral detection position
}

# Default symbol mapping (always colored)
DEFAULT_SYMBOLS = {
    'FREE': SYMBOL_REGISTRY['FREE'],
    'OBSTACLE': SYMBOL_REGISTRY['OBSTACLE'],
    'HOME': SYMBOL_REGISTRY['HOME']
}


def _load_grid_internal(path: Path | str | None = None) -> List[List[int]]:
    """
    Internal function to load grid from JSON (standalone implementation).
    
    This is a self-contained version of grid loading to make api module standalone.
    """
    grid_path = Path(path) if path else DEFAULT_MAP_PATH
    
    if not grid_path.exists():
        return []
    
    with grid_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    # Ensure the JSON structure is a list of rows
    if not isinstance(data, list):
        raise ValueError(f"Invalid grid format in {grid_path}: expected outer list.")
    
    grid = []
    for row_idx, row in enumerate(data):
        if not isinstance(row, Iterable):
            raise ValueError(f"Invalid grid row at index {row_idx}: must be iterable.")
        grid_row = []
        for col_idx, cell in enumerate(row):
            try:
                if isinstance(cell, str):
                    cell_value = SYMBOL_TO_CELL[cell.upper()]
                else:
                    cell_value = int(cell)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid cell value at ({row_idx}, {col_idx}): {cell!r}"
                ) from exc
            grid_row.append(cell_value)
        grid.append(grid_row)
    
    return grid


def get_map_json(path: Path | str | None = None) -> Dict:
    """
    Get the raw JSON data from the grid file.
    
    Args:
        path: Optional path to grid.json. Defaults to "grid.json" in the api folder.
        
    Returns:
        Dictionary containing the raw JSON data, or empty dict if file doesn't exist.
        
    Example:
        json_data = get_map_json()
        print(json_data)  # Raw JSON structure
    """
    grid_path = Path(path) if path else DEFAULT_MAP_PATH
    
    if not grid_path.exists():
        return {}
    
    with grid_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_symbol(symbol_name: str) -> str:
    """
    Get a symbol by name from the symbol registry.
    
    All symbols are returned with colors by default (grid cells) or as-is (entity symbols).
    
    Args:
        symbol_name: Name of the symbol ('FREE', 'OBSTACLE', 'HOME', 'FOOD', 'THREAT', 'ROBOT')
    
    Returns:
        Symbol string (colored for grid cells, plain for entity symbols)
    
    Example:
        food = get_symbol('FOOD')  # Returns: 🍎
        threat = get_symbol('THREAT')  # Returns: ⚠️
        robot = get_symbol('ROBOT')  # Returns: 🤖
        free = get_symbol('FREE')  # Returns colored: '\033[90m.\033[0m'
    """
    return SYMBOL_REGISTRY.get(symbol_name, '?')


def get_symbols() -> Dict[str, str]:
    """
    Get all available symbols as a dictionary.
    
    Returns:
        Dictionary mapping symbol names to symbol strings (all colored by default).
    
    Example:
        all_symbols = get_symbols()
        print(all_symbols['FOOD'])  # 🍎
        print(all_symbols['ROBOT'])  # 🤖
        print(all_symbols['FREE'])   # Colored version
    """
    return SYMBOL_REGISTRY.copy()


def get_map(

    path: Path | str | None = None,
    symbols: Optional[Dict[str, str]] = None
) -> List[List[str]]:
    """
    Get the grid map as a 2D list with customizable symbols.
    
    Default symbols are colored: gray '.', red '#', green 'H'
    
    Args:
        path: Optional path to grid.json. Defaults to "grid.json" in the api folder.
        symbols: Optional dictionary to customize symbols. Keys: 'FREE', 'OBSTACLE', 'HOME'
                 Defaults to colored symbols: {'FREE': gray '.', 'OBSTACLE': red '#', 'HOME': green 'H'}
                 
    Returns:
        2D list (rows x cols) where each cell is represented by the specified symbol.
        Returns empty list if grid file doesn't exist.
        
    Example:
        # Default colored symbols
        map_data = get_map()
        # Returns: [['\033[90m.\033[0m', ...], ...] (colored)
        
        # Custom symbols
        map_data = get_map(symbols={'FREE': '.', 'OBSTACLE': '#', 'HOME': '1'})
        # Returns: [['.', '.', '#'], ['.', '1', '.'], ...]
    """
    grid = _load_grid_internal(path)
    
    if not grid:
        return []
    
    # Use custom symbols or defaults (always colored)
    symbol_map = symbols if symbols else DEFAULT_SYMBOLS
    
    # Map cell values to symbols
    cell_to_symbol = {
        FREE: symbol_map.get('FREE', get_symbol('FREE')),
        OBSTACLE: symbol_map.get('OBSTACLE', get_symbol('OBSTACLE')),
        HOME: symbol_map.get('HOME', get_symbol('HOME'))
    }
    
    # Convert grid to symbol representation
    return [
        [cell_to_symbol.get(cell, '?') for cell in row]
        for row in grid
    ]


def get_map_as_string(
    path: Path | str | None = None,
    symbols: Optional[Dict[str, str]] = None,
    separator: str = " "
) -> str:
    """
    Get the grid map as a formatted string.
    
    Default symbols are colored: gray '.', red '#', green 'H'
    
    Args:
        path: Optional path to grid.json. Defaults to "grid.json" in the api folder.
        symbols: Optional dictionary to customize symbols. Keys: 'FREE', 'OBSTACLE', 'HOME'
        separator: String to separate cells in each row. Defaults to " ".
        
    Returns:
        Formatted string representation of the grid, one row per line (colored by default).
        
    Example:
        map_str = get_map_as_string()
        print(map_str)
        # Output: (colored symbols)
        
        map_str = get_map_as_string(symbols={'HOME': '1'}, separator='')
        # Output:
        # ..#.
        # .1..
        # ##..
    """
    map_data = get_map(path, symbols)
    
    return "\n".join(
        separator.join(row) for row in map_data
    )


def get_map_info(path: Path | str | None = None) -> Dict:
    """
    Get information about the grid map (dimensions, cell counts, etc.).
    
    Args:
        path: Optional path to grid.json. Defaults to "grid.json" in the api folder.
        
    Returns:
        Dictionary with map information:
        {
            'rows': int,
            'cols': int,
            'total_cells': int,
            'free_count': int,
            'obstacle_count': int,
            'home_count': int,
            'exists': bool
        }
        
    Example:
        info = get_map_info()
        print(f"Map size: {info['rows']}x{info['cols']}")
        print(f"Obstacles: {info['obstacle_count']}")
    """
    grid = _load_grid_internal(path)
    
    if not grid:
        return {
            'rows': 0,
            'cols': 0,
            'total_cells': 0,
            'free_count': 0,
            'obstacle_count': 0,
            'home_count': 0,
            'exists': False
        }
    
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    
    free_count = sum(1 for row in grid for cell in row if cell == FREE)
    obstacle_count = sum(1 for row in grid for cell in row if cell == OBSTACLE)
    home_count = sum(1 for row in grid for cell in row if cell == HOME)
    
    return {
        'rows': rows,
        'cols': cols,
        'total_cells': rows * cols,
        'free_count': free_count,
        'obstacle_count': obstacle_count,
        'home_count': home_count,
        'exists': True
    }
