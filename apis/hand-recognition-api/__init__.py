"""
Interaction API - portable package (hyphenated entrypoint).
Loads the hyphenated interaction-api.py so standard imports work.
"""

import importlib.util
from pathlib import Path

_INTERACTION_PATH = Path(__file__).parent / "interaction-api.py"
_spec = importlib.util.spec_from_file_location("interaction_api", _INTERACTION_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load interaction_api from {_INTERACTION_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

HandRecognitionAPI = _module.HandRecognitionAPI
HandState = _module.HandState
GestureRecognizer = _module.GestureRecognizer

__all__ = ['HandRecognitionAPI', 'HandState', 'GestureRecognizer']
