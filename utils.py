"""
Utility functions for Integration-v1

Simplifies common operations like importing hyphenated module names.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_module(file_path: Path, module_name: str) -> Any:
    """
    Load a Python module from a file path (handles hyphenated filenames).
    
    Args:
        file_path: Path to the Python file
        module_name: Name to assign to the module (e.g., 'overlay_api')
    
    Returns:
        The loaded module
    
    Raises:
        ImportError: If the module cannot be loaded
    """
    if not file_path.exists():
        raise ImportError(f"Module file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for module: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_api(api_path: Path, module_name: str, error_msg: str = None) -> Any:
    """
    Import an API module with error handling.
    
    Args:
        api_path: Path to the API file
        module_name: Name for the module
        error_msg: Optional custom error message
    
    Returns:
        The loaded module
    
    Exits:
        sys.exit(1) if import fails
    """
    try:
        return load_module(api_path, module_name)
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        if error_msg:
            print(error_msg)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def extract_attrs(module: Any, *attr_names: str) -> tuple:
    """
    Extract multiple attributes from a module.
    
    Args:
        module: The module to extract from
        *attr_names: Names of attributes to extract
    
    Returns:
        Tuple of extracted attributes in order
    
    Example:
        get_map, get_symbol = extract_attrs(layout_api, 'get_map', 'get_symbol')
    """
    return tuple(getattr(module, name) for name in attr_names)
