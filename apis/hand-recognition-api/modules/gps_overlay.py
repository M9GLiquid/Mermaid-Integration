"""
GPSOverlay - Minimal version for hand recognition calibration
"""

import json
import os
from pathlib import Path


class GPSOverlay:
    """
    Minimal GPSOverlay for camera calibration data.
    Only loads the fields needed for hand recognition.
    """

    def __init__(self, json_path: str = None):
        """
        Initialize GPSOverlay by loading calibration data.
        
        Args:
            json_path: Path to gps_overlay.json file. If None, looks for 
                      gps_overlay.json in the api/ directory.
        """
        if json_path is None:
            # Look for gps_overlay.json in the api/ directory (parent of modules/)
            api_dir = Path(__file__).parent.parent
            json_path = str(api_dir / "gps_overlay.json")

        # Load the calibration JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["gps_overlay"]

        # Image sizes
        self.calib_size = tuple(self.data["calibration_size"])
        self.server_size = tuple(self.data["server_size"])
        self.margin_pixels = self.data.get("margin_pixels", 200)
        
        # Get corrected_size (size of fisheye-corrected image)
        corrected_size = self.data.get("corrected_size", None)
        if corrected_size is None:
            # Calculate from calib_size + margins if not present
            self.corrected_size = (
                self.calib_size[0] + 2 * self.margin_pixels,
                self.calib_size[1] + 2 * self.margin_pixels
            )
        else:
            self.corrected_size = tuple(corrected_size)
