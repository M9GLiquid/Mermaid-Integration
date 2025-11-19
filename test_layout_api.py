#!/usr/bin/env python3
"""
Test script for Layout API (layout-api.py)

This script demonstrates all the functions in layout-api.py and verifies they work correctly.

Usage:
    python3 test_layout_api.py
"""

import sys
from pathlib import Path

# Import utility functions
from utils import import_api, extract_attrs

# Add Integration-v1 root to path
integration_root = Path(__file__).parent
sys.path.insert(0, str(integration_root / "apis" / "layout-api"))

# Import layout-api.py using simplified utility
layout_api = import_api(
    integration_root / "apis" / "layout-api" / "layout-api.py",
    "layout_api"
)

# Extract functions and constants
get_map, get_map_json, get_map_as_string, get_map_info, get_symbol, get_symbols = extract_attrs(
    layout_api, 'get_map', 'get_map_json', 'get_map_as_string', 'get_map_info', 'get_symbol', 'get_symbols'
)


def test_get_map_json():
    """Test getting raw JSON data."""
    print("=" * 60)
    print("Test 1: get_map_json()")
    print("=" * 60)
    
    json_data = get_map_json()
    
    if json_data:
        print("✓ Successfully loaded JSON data")
        print(f"  - Number of rows: {len(json_data)}")
        if json_data:
            print(f"  - Number of cols: {len(json_data[0])}")
        print(f"  - First row preview: {json_data[0][:10]}...")
    else:
        print("✗ No JSON data found (grid.json might not exist)")
    
    print()


def test_get_map_default():
    """Test getting map with default colored symbols."""
    print("=" * 60)
    print("Test 2: get_map() with default colored symbols")
    print("=" * 60)
    
    map_data = get_map()
    
    if map_data:
        print("✓ Successfully loaded map with colored symbols")
        print(f"  - Map dimensions: {len(map_data)} rows × {len(map_data[0])} cols")
        print(f"  - Symbols are colored by default (gray '.', red '#', green 'H')")
        print(f"  - First row: {map_data[0][:20]}...")
        print("  - Sample cells from first 3 rows:")
        for i, row in enumerate(map_data[:3]):
            print(f"    Row {i}: {row[:15]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_custom_symbols():
    """Test getting map with custom symbols."""
    print("=" * 60)
    print("Test 3: get_map() with custom symbols")
    print("=" * 60)
    
    custom_symbols = {
        'FREE': '.',
        'OBSTACLE': '#',
        'HOME': 'H'
    }
    
    map_data = get_map(symbols=custom_symbols)
    
    if map_data:
        print("✓ Successfully loaded map with custom symbols")
        print(f"  - Custom symbols: {custom_symbols}")
        print(f"  - First row: {map_data[0][:61]}...")
        print("  - Sample cells from first 3 rows:")
        for i, row in enumerate(map_data[:3]):
            print(f"    Row {i}: {row[:15]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_as_string():
    """Test getting map as formatted string with colored symbols."""
    print("=" * 60)
    print("Test 4: get_map_as_string() with colored symbols")
    print("=" * 60)
    
    map_str = get_map_as_string()
    
    if map_str:
        lines = map_str.split('\n')
        print("✓ Successfully generated string representation with colored symbols")
        print(f"  - Total lines: {len(lines)}")
        print(f"  - Symbols are colored (gray '.', red '#', green 'H')")
        print(f"  - First {min(5, len(lines))} lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    {line[:60]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_as_string_custom():
    """Test getting map as string with custom symbols and separator."""
    print("=" * 60)
    print("Test 5: get_map_as_string() with custom symbols")
    print("=" * 60)
    
    custom_symbols = {
        'FREE': '.',
        'OBSTACLE': '#',
        'HOME': '1'
    }
    
    map_str = get_map_as_string(symbols=custom_symbols, separator='')
    
    if map_str:
        lines = map_str.split('\n')
        print("✓ Successfully generated string with custom symbols")
        print(f"  - Custom symbols: {custom_symbols}")
        print("  - Separator: '' (no spaces)")
        print("  - First 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    {line[:60]}...")
    else:
        print("✗ No map data found")
    
    print()


def test_get_map_info():
    """Test getting map information."""
    print("=" * 60)
    print("Test 6: get_map_info()")
    print("=" * 60)
    
    info = get_map_info()
    
    print("✓ Map information:")
    print(f"  - File exists: {info['exists']}")
    print(f"  - Dimensions: {info['rows']} rows × {info['cols']} cols")
    print(f"  - Total cells: {info['total_cells']}")
    print(f"  - Free cells: {info['free_count']}")
    print(f"  - Obstacle cells: {info['obstacle_count']}")
    print(f"  - Home cells: {info['home_count']}")
    
    if info['total_cells'] > 0:
        free_pct = (info['free_count'] / info['total_cells']) * 100
        obstacle_pct = (info['obstacle_count'] / info['total_cells']) * 100
        home_pct = (info['home_count'] / info['total_cells']) * 100
        print(f"  - Free percentage: {free_pct:.1f}%")
        print(f"  - Obstacle percentage: {obstacle_pct:.1f}%")
        print(f"  - Home percentage: {home_pct:.1f}%")
    
    print()


def test_get_symbol():
    """Test getting individual symbols."""
    print("=" * 60)
    print("Test 7: get_symbol() - Get individual symbols")
    print("=" * 60)
    
    symbols_to_test = ['FREE', 'OBSTACLE', 'HOME', 'FOOD', 'THREAT', 'ROBOT']
    
    print("✓ Available symbols:")
    for sym_name in symbols_to_test:
        symbol = get_symbol(sym_name)
        print(f"  - {sym_name}: {symbol!r}")
    
    print()


def test_get_symbols():
    """Test getting all symbols."""
    print("=" * 60)
    print("Test 8: get_symbols() - Get all symbols")
    print("=" * 60)
    
    all_symbols = get_symbols()
    
    print("✓ All available symbols:")
    for name, symbol in all_symbols.items():
        print(f"  - {name}: {symbol!r}")
    
    print()


def test_custom_symbol_examples():
    """Test various custom symbol configurations."""
    print("=" * 60)
    print("Test 9: Custom symbol examples")
    print("=" * 60)
    
    examples = [
        {'FREE': '0', 'OBSTACLE': '1', 'HOME': '2'},
        {'FREE': '.', 'OBSTACLE': 'X', 'HOME': 'H'},
        {'FREE': ' ', 'OBSTACLE': '█', 'HOME': '★'},
    ]
    
    for i, symbols in enumerate(examples, 1):
        print(f"\nExample {i}: {symbols}")
        map_data = get_map(symbols=symbols)
        if map_data:
            print(f"  First row preview: {map_data[0][:30]}...")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Layout API Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_get_map_json()
        test_get_map_default()
        test_get_map_custom_symbols()
        test_get_map_as_string()
        test_get_map_as_string_custom()
        test_get_map_info()
        test_get_symbol()
        test_get_symbols()
        test_custom_symbol_examples()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
