#!/usr/bin/env python3
"""
Example usage of layout-api.py

This script demonstrates how to use the layout API to access grid map data.

Usage:
    python3 example_usage.py
"""

from pathlib import Path
import sys
import importlib.util

# Import layout-api.py (handles hyphen in filename)
api_path = Path(__file__).parent / "layout-api.py"
spec = importlib.util.spec_from_file_location("layout_api", api_path)
layout_api = importlib.util.module_from_spec(spec)
sys.modules["layout_api"] = layout_api
spec.loader.exec_module(layout_api)

# Import functions and constants
get_map = layout_api.get_map
get_map_json = layout_api.get_map_json
get_map_as_string = layout_api.get_map_as_string
get_map_info = layout_api.get_map_info
get_symbol = layout_api.get_symbol
get_symbols = layout_api.get_symbols
FREE = layout_api.FREE
OBSTACLE = layout_api.OBSTACLE
HOME = layout_api.HOME
Colors = layout_api.Colors


def example_basic_usage():
    """Basic usage: Get map with default colored symbols."""
    print("=" * 60)
    print("Example 1: Basic Usage - Get map with default symbols")
    print("=" * 60)
    
    # Get map with default colored symbols
    map_data = get_map()
    
    if map_data:
        print(f"✓ Map loaded: {len(map_data)} rows × {len(map_data[0])} cols")
        print("\nFirst 5 rows:")
        for i, row in enumerate(map_data[:5]):
            print(f"  Row {i}: {row[:30]}...")
    else:
        print("✗ No map data found")
    
    print()


def example_custom_symbols():
    """Use custom symbols instead of default colored ones."""
    print("=" * 60)
    print("Example 2: Custom Symbols")
    print("=" * 60)
    
    # Define custom symbols
    custom_symbols = {
        'FREE': '.',
        'OBSTACLE': '#',
        'HOME': 'H'
    }
    
    # Get map with custom symbols
    map_data = get_map(symbols=custom_symbols)
    
    if map_data:
        print(f"Custom symbols: {custom_symbols}")
        print("\nFirst 3 rows:")
        for i, row in enumerate(map_data[:3]):
            print(f"  Row {i}: {row[:40]}...")
    
    print()


def example_get_symbols():
    """Get individual symbols programmatically."""
    print("=" * 60)
    print("Example 3: Get Individual Symbols")
    print("=" * 60)
    
    # Get specific symbols
    free_symbol = get_symbol('FREE')
    obstacle_symbol = get_symbol('OBSTACLE')
    home_symbol = get_symbol('HOME')
    food_symbol = get_symbol('FOOD')
    threat_symbol = get_symbol('THREAT')
    
    print("Available symbols:")
    print(f"  FREE: {free_symbol!r}")
    print(f"  OBSTACLE: {obstacle_symbol!r}")
    print(f"  HOME: {home_symbol!r}")
    print(f"  FOOD: {food_symbol!r}")
    print(f"  THREAT: {threat_symbol!r}")
    
    # Get all symbols
    all_symbols = get_symbols()
    print(f"\nTotal symbols available: {len(all_symbols)}")
    
    print()


def example_map_as_string():
    """Get map as formatted string."""
    print("=" * 60)
    print("Example 4: Get Map as String")
    print("=" * 60)
    
    # Get map as string with default colored symbols
    map_str = get_map_as_string()
    
    if map_str:
        lines = map_str.split('\n')
        print(f"✓ Map string generated ({len(lines)} lines)")
        print("\nFirst 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"  {line[:50]}...")
    
    print()


def example_map_info():
    """Get map statistics and information."""
    print("=" * 60)
    print("Example 5: Map Information")
    print("=" * 60)
    
    info = get_map_info()
    
    print("Map Statistics:")
    print(f"  Dimensions: {info['rows']} rows × {info['cols']} cols")
    print(f"  Total cells: {info['total_cells']}")
    print(f"  Free cells: {info['free_count']}")
    print(f"  Obstacle cells: {info['obstacle_count']}")
    print(f"  Home cells: {info['home_count']}")
    
    if info['total_cells'] > 0:
        free_pct = (info['free_count'] / info['total_cells']) * 100
        obstacle_pct = (info['obstacle_count'] / info['total_cells']) * 100
        home_pct = (info['home_count'] / info['total_cells']) * 100
        print(f"\nPercentages:")
        print(f"  Free: {free_pct:.1f}%")
        print(f"  Obstacle: {obstacle_pct:.1f}%")
        print(f"  Home: {home_pct:.1f}%")
    
    print()


def example_raw_json():
    """Access raw JSON data."""
    print("=" * 60)
    print("Example 6: Raw JSON Data")
    print("=" * 60)
    
    # Get raw JSON data
    json_data = get_map_json()
    
    if json_data:
        print(f"✓ JSON loaded: {len(json_data)} rows")
        print(f"  First row: {json_data[0][:20]}...")
        print(f"  Data type: List of {type(json_data[0]).__name__}")
    else:
        print("✗ No JSON data found")
    
    print()


def example_constants():
    """Use API constants."""
    print("=" * 60)
    print("Example 7: API Constants")
    print("=" * 60)
    
    print("Grid cell constants:")
    print(f"  FREE = {FREE}")
    print(f"  OBSTACLE = {OBSTACLE}")
    print(f"  HOME = {HOME}")
    
    print("\nColor constants:")
    print(f"  Colors.GREEN = {Colors.GREEN}GREEN{Colors.RESET}")
    print(f"  Colors.RED = {Colors.RED}RED{Colors.RESET}")
    print(f"  Colors.YELLOW = {Colors.YELLOW}YELLOW{Colors.RESET}")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Layout API - Example Usage")
    print("=" * 60)
    print("\nThis script demonstrates how to use the layout-api.py module.")
    print("The API provides access to grid map data stored in grid.json\n")
    
    try:
        example_basic_usage()
        example_custom_symbols()
        example_get_symbols()
        example_map_as_string()
        example_map_info()
        example_raw_json()
        example_constants()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print("\nFor more information, see the docstrings in layout-api.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
