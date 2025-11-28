#!/usr/bin/env python3
"""
GPSOverlay - Standalone coordinate transformation API

Transforms GPS server coordinates (2048x1536) to:
- Rectified canvas coordinates
- Grid cell positions (dynamically configured)
- Real-world coordinates (mm)

Also provides image transformation:
- Transform raw GPS images to rectified top-down view with optional grid overlay

This is a standalone API that can be exported and used independently.

Usage:
    from overlay import GPSOverlay
    overlay = GPSOverlay("gps_overlay.json")

    # Transform GPS coordinates
    cell = overlay.get_grid_cell(50, 50)
    real_pos = overlay.get_real_coords(50, 50)
    grid_map = overlay.get_grid_map()
    
    # Transform images (requires opencv-python and numpy)
    import cv2
    rectified = overlay.transform_image("GPS-Real.png", show_grid=True)
    cv2.imwrite("rectified.png", rectified)
"""

import json
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np


class GPSOverlay:
    """
    Standalone GPS coordinate transformation API

    Loads calibration data from a single JSON file and provides
    simple functions to transform GPS server coordinates.
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize the GPSOverlay API by loading calibration data.
        
        Args:
            json_path: Path to gps_overlay.json file. If None, looks for 
                      gps_overlay.json in the same directory as this script.
        
        Example:
            # Load from default location (same directory)
            overlay = GPSOverlay()
            
            # Load from custom path
            overlay = GPSOverlay("path/to/gps_overlay.json")
        """
        if json_path is None:
            # Auto-detect: look for gps_overlay.json in the same directory as this script
            # This makes it easy to copy both files together
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "gps_overlay.json")

        # Load the calibration JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["gps_overlay"]

        # Image sizes:
        # - calib_size: Original calibration image size (e.g., 1206x906)
        # - server_size: GPS server image size (typically 2048x1536)
        # - corrected_size: Size of fisheye-corrected image (with margins)
        # - margin_pixels: Margin added around image during fisheye correction
        self.calib_size = tuple(self.data["calibration_size"])
        self.server_size = tuple(self.data["server_size"])
        self.margin_pixels = self.data.get("margin_pixels", 200)  # Default to 200 if not present
        corrected_size = self.data.get("corrected_size", None)
        if corrected_size is None:
            # Calculate from calib_size + margins if not present (backward compatibility)
            self.corrected_size = (self.calib_size[0] + 2 * self.margin_pixels,
                                   self.calib_size[1] + 2 * self.margin_pixels)
        else:
            self.corrected_size = tuple(corrected_size)

        # ===== Transformation Matrix =====
        # Homography matrix that transforms from corrected (fisheye-removed) space
        # to rectified (top-down) canvas space
        # This is a 3x3 matrix that performs perspective transformation
        self.homography = self.data["homography"]

        # ===== Arena Bounds =====
        # The rectangular bounds of the arena in rectified canvas coordinates
        # Format: {"left": float, "top": float, "right": float, "bottom": float}
        # These define the area where the grid is placed
        self.arena_bounds = self.data["arena_bounds"]

        # ===== Grid Configuration =====
        # Grid dimensions and cell size (dynamically loaded from Tool 4)
        # - grid_cols: Number of columns in the grid (e.g., 14)
        # - grid_rows: Number of rows in the grid (e.g., 9)
        # - cell_size_px: Size of each cell in pixels {"x": float, "y": float}
        self.grid_cols = self.data["grid"]["cols"]
        self.grid_rows = self.data["grid"]["rows"]
        self.cell_size_px = self.data["grid"]["cell_size_px"]

        # ===== Real-World Calibration =====
        # Optional: Converts pixels to millimeters (only if Tool 6 was run)
        # - mm_per_pixel_x: Millimeters per pixel in X direction
        # - mm_per_pixel_y: Millimeters per pixel in Y direction
        self.real_world_available = "real_world" in self.data
        self.mm_per_pixel_x: Optional[float]
        self.mm_per_pixel_y: Optional[float]
        if self.real_world_available:
            self.mm_per_pixel_x = self.data["real_world"]["mm_per_pixel_x"]
            self.mm_per_pixel_y = self.data["real_world"]["mm_per_pixel_y"]
        else:
            # Real-world calibration not available - set to None
            self.mm_per_pixel_x = self.mm_per_pixel_y = None

        # Optional: camera pose assumptions for height correction
        camera_cfg = self.data.get("camera", {})
        self.camera_height_mm: Optional[float] = camera_cfg.get("height_mm")
        # Assumed tilt (pitch) in degrees; positive = camera looking toward grid top (negative row direction)
        self.camera_tilt_pitch_deg: float = float(camera_cfg.get("tilt_pitch_deg", 0.0))
        # Optional attenuation factor for tilt compensation to avoid over-shift
        self.camera_tilt_scale: float = float(camera_cfg.get("tilt_scale", 0.25))
        # Cache for rectified principal point (camera center projected to rectified plane)
        self._principal_point_rect: Optional[Tuple[float, float]] = None

    def _get_principal_point_rectified(self) -> Optional[Tuple[float, float]]:
        """
        Project the camera principal point to rectified space (cached).
        Returns None if unavailable.
        """
        if self._principal_point_rect is not None:
            return self._principal_point_rect

        try:
            cx = float(self.data["camera_matrix"][0][2])
            cy = float(self.data["camera_matrix"][1][2])
        except Exception:
            return None

        x_rect, y_rect = self.map_coords(cx, cy)
        if math.isnan(x_rect) or math.isnan(y_rect):
            return None

        self._principal_point_rect = (x_rect, y_rect)
        return self._principal_point_rect

    def map_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform GPS server coordinates to rectified canvas coordinates.
        
        This is the core transformation function that converts coordinates from
        the original GPS server image space (with fisheye distortion) to the
        rectified top-down view space.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Tuple of (x_rect, y_rect) in rectified canvas space.
            Returns (nan, nan) if transformation fails (e.g., point outside valid area).
        
        Example:
            # Transform a GPS coordinate to rectified space
            x_rect, y_rect = overlay.map_coords(1024, 768)
            print(f"Rectified position: ({x_rect:.1f}, {y_rect:.1f})")
        """
        # Step 1: Scale from GPS server resolution to calibration resolution
        # The calibration was done at a different resolution than the server image,
        # so we need to scale coordinates to match the calibration space
        scale_x = self.calib_size[0] / self.server_size[0]
        scale_y = self.calib_size[1] / self.server_size[1]

        x_cal = float(x) * scale_x
        y_cal = float(y) * scale_y

        # Step 2: Apply fisheye undistortion to get corrected coordinates
        # The homography matrix maps from corrected (fisheye-removed) space to rectified space
        # We need to undistort the calibration coordinates first
        try:
            import cv2
            import numpy as np
            
            # Build new camera matrix (includes margin shift and scale factor)
            camera_matrix = np.array(self.data["camera_matrix"], dtype=np.float64)
            margin = self.margin_pixels
            scale_factor = self.data.get("scale_factor", 0.8)  # Default to 0.8 if not in JSON (backward compatibility)
            
            new_camera_matrix = camera_matrix.copy().astype(np.float64)
            new_camera_matrix[0, 2] += float(margin)  # cx offset
            new_camera_matrix[1, 2] += float(margin)  # cy offset
            new_camera_matrix[0, 0] *= float(scale_factor)  # fx scale
            new_camera_matrix[1, 1] *= float(scale_factor)  # fy scale
            
            # Apply fisheye undistortion
            dist_coeffs = np.array(self.data["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
            pts = np.array([[[x_cal, y_cal]]], dtype=np.float64)
            undist = cv2.fisheye.undistortPoints(pts, camera_matrix, dist_coeffs, R=np.eye(3), P=new_camera_matrix)
            x_corr = float(undist[0, 0, 0])
            y_corr = float(undist[0, 0, 1])
        except ImportError:
            # Fallback if OpenCV not available (shouldn't happen for coordinate transformation)
            # This is a simplified approximation - not accurate but prevents crashes
            x_corr, y_corr = x_cal, y_cal

        # Step 3: Apply homography transformation (corrected → rectified canvas)
        # Homography is a 3x3 matrix that performs perspective transformation
        # Formula: [x', y', w'] = H * [x, y, 1]
        # Final coordinates: x_rect = x'/w', y_rect = y'/w'
        h = self.homography
        x_rect = h[0][0] * x_corr + h[0][1] * y_corr + h[0][2]
        y_rect = h[1][0] * x_corr + h[1][1] * y_corr + h[1][2]
        w = h[2][0] * x_corr + h[2][1] * y_corr + h[2][2]

        # Check for division by zero (invalid transformation)
        if abs(w) < 1e-10:
            return float('nan'), float('nan')

        # Return normalized coordinates
        return x_rect / w, y_rect / w

    def _apply_soft_clamp_x(self, x_rect: float) -> float:
        """
        Compress the right-most 40% of the arena so X advances slower past 60%.

        Keeps 0–60% untouched, then eases the remaining span to delay reaching
        the far-right columns unless the hand truly approaches the edge.
        """
        left = self.arena_bounds["left"]
        right = self.arena_bounds["right"]
        width = right - left
        if width <= 0:
            return x_rect

        x_norm = (x_rect - left) / width  # 0..1 across arena width
        clamp_start = 0.6
        if x_norm <= clamp_start:
            return x_rect

        # Ease-out compression on the final 40%
        remaining = 1.0 - clamp_start
        t = min(1.0, (x_norm - clamp_start) / remaining)  # 0..1
        k = 2.0  # larger k = stronger slowdown
        eased = 1.0 - math.exp(-k * t)
        eased /= 1.0 - math.exp(-k)  # normalize to 0..1
        x_remapped = clamp_start + eased * remaining
        return left + x_remapped * width

    def get_grid_cell(self, x: float, y: float) -> Dict:
        """
        Transform GPS server coordinates to grid cell position.
        
        This function maps GPS coordinates to a specific grid cell in the arena.
        Useful for navigation systems that need to know which cell a robot is in.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Dictionary with grid cell information:
            {
                "col": int,           # Grid column index (0 to grid_cols-1)
                "row": int,           # Grid row index (0 to grid_rows-1)
                "in_bounds": bool,    # True if point is within arena bounds
                "center_x": float,    # Cell center X coordinate in rectified space
                "center_y": float     # Cell center Y coordinate in rectified space
            }
        
        Example:
            # Get grid cell for a GPS coordinate
            cell = overlay.get_grid_cell(1024, 768)
            if cell["in_bounds"]:
                print(f"Robot is in cell ({cell['col']}, {cell['row']})")
                print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
        """
        x_rect, y_rect = self.map_coords(x, y)
        x_rect = self._apply_soft_clamp_x(x_rect)

        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"col": 0, "row": 0, "in_bounds": False, "center_x": 0, "center_y": 0}

        # Calculate grid cell position
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]

        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        col = int((x_rect - left) // cell_width)
        row = int((y_rect - top) // cell_height)

        # Check bounds
        in_bounds = (0 <= col < self.grid_cols and
                    0 <= row < self.grid_rows)

        # Calculate cell center
        center_x = left + (col + 0.5) * cell_width
        center_y = top + (row + 0.5) * cell_height

        return {
            "col": col,
            "row": row,
            "in_bounds": in_bounds,
            "center_x": center_x,
            "center_y": center_y
        }

    def get_grid_cell_with_height_offset(self, x: float, y: float, height_mm: float = 1000.0) -> Dict:
        """
        Get grid cell position with height offset correction.
        
        Corrects for objects above the arena floor (e.g., hand 1m above floor).
        Uses either a calibrated camera height (preferred) or a heuristic fallback.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
            height_mm: Height of object above arena floor in millimeters (default: 1000mm = 1m)
        
        Returns:
            Dictionary with corrected grid cell information (same format as get_grid_cell).
            If real-world calibration is not available, falls back to get_grid_cell() without offset.
        
        Example:
            # Get grid cell for hand position (1m above floor)
            cell = overlay.get_grid_cell_with_height_offset(1024, 768, height_mm=1000.0)
            if cell["in_bounds"]:
                print(f"Hand is in cell ({cell['col']}, {cell['row']})")
        """
        # Map to rectified space first
        x_rect, y_rect = self.map_coords(x, y)
        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"col": 0, "row": 0, "in_bounds": False, "center_x": 0, "center_y": 0}

        # Apparent grid cell without correction
        cell_info = self.get_grid_cell_from_rectified(x_rect, y_rect)
        if not cell_info["in_bounds"]:
            return cell_info

        # If we don't have a camera height, fall back to the old heuristic
        if self.camera_height_mm is None or height_mm <= 0 or height_mm >= self.camera_height_mm:
            return self._get_grid_cell_with_height_heuristic(cell_info, height_mm)

        # Compute principal point in rectified space; if unavailable, fall back
        pp_rect = self._get_principal_point_rectified()
        if pp_rect is None:
            return self._get_grid_cell_with_height_heuristic(cell_info, height_mm)

        # Simple pinhole model assuming nadir camera:
        # Elevated point appears radially farther from principal point.
        # Scale vector toward principal point by (H - h) / H to drop to floor.
        cam_h = self.camera_height_mm
        scale = (cam_h - height_mm) / cam_h
        px, py = pp_rect
        x_floor = px + scale * (x_rect - px)
        y_floor = py + scale * (y_rect - py)

        # Apply small tilt compensation along grid columns (toward right side = positive X)
        if self.camera_tilt_pitch_deg != 0 and self.real_world_available and self.mm_per_pixel_x not in (None, 0):
            pitch_rad = math.radians(self.camera_tilt_pitch_deg)
            arena_width = self.arena_bounds["right"] - self.arena_bounds["left"]
            if arena_width != 0:
                norm_x = (x_rect - px) / arena_width
                # Only apply on the camera-away side (positive norm_x = right side); leave left side untouched
                if norm_x > 0:
                    norm_x = min(1.0, norm_x)
                    tilt_offset_px = norm_x * (height_mm * math.tan(pitch_rad)) / self.mm_per_pixel_x
                    tilt_offset_px *= self.camera_tilt_scale
                    x_floor -= tilt_offset_px

        corrected_cell = self.get_grid_cell_from_rectified(x_floor, y_floor)
        return corrected_cell

    def _get_grid_cell_with_height_heuristic(self, cell_info: Dict, height_mm: float) -> Dict:
        """
        Legacy heuristic for height correction (kept as fallback when camera height is unknown).
        """
        # If real-world calibration is not available, return uncorrected position
        if not self.real_world_available:
            return cell_info

        row_apparent = cell_info["row"]
        col_apparent = cell_info["col"]

        # Calculate offset using perspective projection
        # The homography matrix's [2][2] element relates to perspective scale
        h = self.homography
        perspective_scale = h[2][2]

        # Estimate camera angle from perspective scale
        # This is a simplified model - assumes camera is looking down at an angle
        try:
            camera_angle_rad = math.acos(min(1.0, max(0.0, perspective_scale)))
        except (ValueError, TypeError):
            # Fallback if calculation fails
            return cell_info

        # Calculate base offset in millimeters
        base_offset_mm = height_mm * math.tan(camera_angle_rad)

        # Convert to pixels using real-world calibration
        mm_x = self.mm_per_pixel_x
        mm_y = self.mm_per_pixel_y
        if mm_x is None or mm_y is None or mm_x == 0 or mm_y == 0:
            return cell_info
        base_offset_x_px = base_offset_mm / mm_x
        base_offset_y_px = base_offset_mm / mm_y

        # Grid cell dimensions
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]
        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        # Distance from center (normalized)
        center_row, center_col = self.grid_rows / 2.0, self.grid_cols / 2.0
        row_distance = (row_apparent - center_row) / max(center_row, 1.0)
        col_distance = (col_apparent - center_col) / max(center_col, 1.0)

        # Apply position-dependent offset
        # Objects further from center appear more offset due to perspective
        row_offset = (base_offset_y_px / cell_height) * row_distance
        col_offset = -(base_offset_x_px / cell_width) * col_distance

        # Calculate corrected cell position
        row_corrected = row_apparent - row_offset
        col_corrected = col_apparent - col_offset

        # Clamp to valid grid bounds
        row_corrected = max(0, min(self.grid_rows - 1, row_corrected))
        col_corrected = max(0, min(self.grid_cols - 1, col_corrected))

        # Calculate corrected cell center
        center_x = left + (int(col_corrected) + 0.5) * cell_width
        center_y = top + (int(row_corrected) + 0.5) * cell_height

        return {
            "col": int(col_corrected),
            "row": int(row_corrected),
            "in_bounds": True,
            "center_x": center_x,
            "center_y": center_y
        }

    def get_grid_cell_from_rectified(self, x_rect: float, y_rect: float) -> Dict:
        """
        Get grid cell from rectified canvas coordinates.
        
        This function is useful when you have coordinates in rectified canvas space
        (e.g., from clicking on a transformed image) and want to know which grid cell
        they correspond to.
        
        Args:
            x_rect: X coordinate in rectified canvas space
            y_rect: Y coordinate in rectified canvas space
        
        Returns:
            Dictionary with grid cell information:
            {
                "col": int,           # Grid column index (0 to grid_cols-1)
                "row": int,           # Grid row index (0 to grid_rows-1)
                "in_bounds": bool,    # True if point is within arena bounds
                "center_x": float,    # Cell center X coordinate in rectified space
                "center_y": float     # Cell center Y coordinate in rectified space
            }
        
        Example:
            # After transforming an image and getting click coordinates
            # Click is at pixel (100, 200) in the transformed image
            # First convert to canvas space: x_canvas = 100 + offset_x
            cell = overlay.get_grid_cell_from_rectified(x_canvas, y_canvas)
            if cell["in_bounds"]:
                print(f"Clicked cell: ({cell['col']}, {cell['row']})")
        """
        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"col": 0, "row": 0, "in_bounds": False, "center_x": 0, "center_y": 0}

        # Calculate grid cell position
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]

        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        col = int((x_rect - left) // cell_width)
        row = int((y_rect - top) // cell_height)

        # Check bounds
        in_bounds = (0 <= col < self.grid_cols and
                    0 <= row < self.grid_rows)

        # Calculate cell center
        center_x = left + (col + 0.5) * cell_width
        center_y = top + (row + 0.5) * cell_height

        return {
            "col": col,
            "row": row,
            "in_bounds": in_bounds,
            "center_x": center_x,
            "center_y": center_y
        }

    def get_real_coords(self, x: float, y: float) -> Dict:
        """
        Transform GPS server coordinates to real-world coordinates in millimeters.
        
        This function converts pixel coordinates to real-world measurements.
        Requires that Tool 6 (Real-World Calibrator) was run to provide calibration data.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Dictionary with real-world position:
            {
                "x_mm": float,              # X position in millimeters from origin (top-left)
                "y_mm": float,              # Y position in millimeters from origin (top-left)
                "distance_from_origin_mm": float  # Euclidean distance from origin in mm
            }
        
        Raises:
            ValueError: If real-world calibration is not available.
                       Run Tool 6 and recreate gps_overlay.json with Tool 8.
        
        Example:
            # Get real-world position
            try:
                pos = overlay.get_real_coords(1024, 768)
                print(f"Real position: {pos['x_mm']:.1f}mm, {pos['y_mm']:.1f}mm")
                print(f"Distance from origin: {pos['distance_from_origin_mm']:.1f}mm")
            except ValueError as e:
                print("Real-world calibration not available:", e)
        """
        if not self.real_world_available:
            raise ValueError(
                "Real-world calibration not available. "
                "Run Tool 6 (Real-World Calibrator) and recreate gps_overlay.json with Tool 8."
            )

        x_rect, y_rect = self.map_coords(x, y)

        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"x_mm": 0, "y_mm": 0, "distance_from_origin_mm": 0}

        # Convert from rectified space to real-world mm
        # Origin is at arena top-left corner
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]

        x_mm = (x_rect - left) * self.mm_per_pixel_x
        y_mm = (y_rect - top) * self.mm_per_pixel_y

        distance_mm = math.hypot(x_mm, y_mm)

        return {
            "x_mm": x_mm,
            "y_mm": y_mm,
            "distance_from_origin_mm": distance_mm
        }

    def get_grid_map(self) -> List[List[Dict]]:
        """
        Get complete grid mapping as a 2D array.
        
        This function returns a complete representation of all grid cells,
        useful for path planning, visualization, or creating lookup tables.
        
        Returns:
            2D list (rows x cols) where each element is a dictionary with cell info:
            [
                [cell(0,0), cell(1,0), ..., cell(cols-1,0)],  # Row 0
                [cell(0,1), cell(1,1), ..., cell(cols-1,1)],  # Row 1
                ...
            ]
            
            Each cell dictionary contains:
            {
                "col": int,           # Grid column index (0 to cols-1)
                "row": int,           # Grid row index (0 to rows-1)
                "x_mm": float,        # X position in mm (0 if calibration not available)
                "y_mm": float,        # Y position in mm (0 if calibration not available)
                "center_x": float,    # Cell center X in rectified space
                "center_y": float     # Cell center Y in rectified space
            }
        
        Example:
            # Get complete grid map
            grid_map = overlay.get_grid_map()
            
            # Access a specific cell (row, col)
            cell = grid_map[5][3]  # Row 5, Column 3
            print(f"Cell ({cell['col']}, {cell['row']}) center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
            
            # Iterate through all cells
            for row in grid_map:
                for cell in row:
                    if cell['x_mm'] > 0:  # Only if real-world calibrated
                        print(f"Cell ({cell['col']}, {cell['row']}): {cell['x_mm']:.1f}mm, {cell['y_mm']:.1f}mm")
        """
        grid_map = []

        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]

        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        for row in range(self.grid_rows):
            grid_row = []

            for col in range(self.grid_cols):
                center_x = left + (col + 0.5) * cell_width
                center_y = top + (row + 0.5) * cell_height

                # Convert to real-world coordinates if available
                if self.real_world_available:
                    x_mm = (center_x - left) * self.mm_per_pixel_x
                    y_mm = (center_y - top) * self.mm_per_pixel_y
                else:
                    x_mm = y_mm = 0

                grid_row.append({
                    "col": col,
                    "row": row,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "center_x": center_x,
                    "center_y": center_y
                })

            grid_map.append(grid_row)

        return grid_map

    def transform_image(self, image_path: str, show_grid: bool = True) -> Tuple['np.ndarray', Dict[str, int]]:
        """
        Transform a raw GPS image to rectified view with optional grid overlay.
        
        This function takes a raw GPS camera image (like GPS-Real.png) and applies:
        1. Fisheye distortion correction
        2. Perspective transformation (homography) to get top-down rectified view
        3. Optional grid overlay matching the calibration settings
        
        Requires: OpenCV (cv2) and NumPy (numpy) - install with: pip install opencv-python numpy
        
        Args:
            image_path: Path to the input GPS image file
            show_grid: If True, draws the grid overlay on the rectified image
        
        Returns:
            Tuple of:
            - NumPy array (BGR image) of the rectified image, ready to save or display.
              Shape: (height, width, 3) with dtype uint8
            - Dictionary with offset information:
              {
                  "offset_x": int,  # X offset to convert image pixel to canvas coordinate
                  "offset_y": int   # Y offset to convert image pixel to canvas coordinate
              }
              To convert a click coordinate (x_img, y_img) to canvas space:
              x_canvas = x_img + offset_x
              y_canvas = y_img + offset_y
        
        Raises:
            ImportError: If OpenCV or NumPy are not installed
            FileNotFoundError: If image_path doesn't exist
            ValueError: If image cannot be loaded
        
        Example:
            import cv2
            from overlay import GPSOverlay
            
            overlay = GPSOverlay()
            
            # Transform image with grid
            rectified, offset = overlay.transform_image("GPS-Real.png", show_grid=True)
            
            # Handle click on transformed image
            def on_click(event, x_img, y_img, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Convert image pixel to canvas coordinate
                    x_canvas = x_img + offset["offset_x"]
                    y_canvas = y_img + offset["offset_y"]
                    
                    # Get grid cell
                    cell = overlay.get_grid_cell_from_rectified(x_canvas, y_canvas)
                    if cell["in_bounds"]:
                        print(f"Clicked cell: ({cell['col']}, {cell['row']})")
            
            cv2.imshow("Rectified", rectified)
            cv2.setMouseCallback("Rectified", on_click)
            cv2.waitKey(0)
        """
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "transform_image() requires OpenCV and NumPy. "
                "Install with: pip install opencv-python numpy"
            ) from e
        
        # Load input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Step 1: Apply fisheye correction
        # Convert calibration data to numpy arrays
        camera_matrix = np.array(self.data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(self.data["distortion_coeffs"], dtype=np.float32)
        
        # Scale image to calibration resolution if needed
        if (w, h) != self.calib_size:
            scale_x = self.calib_size[0] / w
            scale_y = self.calib_size[1] / h
            image_scaled = cv2.resize(image, self.calib_size, interpolation=cv2.INTER_LINEAR)
        else:
            image_scaled = image
            scale_x = scale_y = 1.0
        
        # Create new camera matrix for expanded output (with margins)
        margin = self.margin_pixels
        expanded_size = self.corrected_size
        new_camera_matrix = camera_matrix.copy()
        new_camera_matrix[0, 2] += margin  # cx offset
        new_camera_matrix[1, 2] += margin  # cy offset
        
        # Reduce focal length slightly to show more area (same as Tool 1/7/9)
        scale_factor = self.data.get("scale_factor", 0.8)  # Read from JSON (same as map_coords)
        new_camera_matrix[0, 0] *= scale_factor  # fx
        new_camera_matrix[1, 1] *= scale_factor  # fy
        
        # Create undistortion maps for expanded size
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix,
            expanded_size, cv2.CV_16SC2
        )
        
        # Apply fisheye correction
        corrected_image = cv2.remap(image_scaled, map1, map2, cv2.INTER_LINEAR)
        
        # Step 2: Apply homography transformation to get rectified view
        # The stored homography (homography_image_to_world_canvas) already includes
        # the translation to canvas space, so we use it directly without applying
        # another translation. This ensures coordinates match map_coords().
        h_matrix = np.array(self.homography, dtype=np.float32)
        
        # Calculate warp bounds to determine output canvas size
        # The homography produces coordinates in canvas space (may include negative values)
        corr_h, corr_w = corrected_image.shape[:2]
        corners = np.array([
            [0, 0],
            [corr_w - 1, 0],
            [corr_w - 1, corr_h - 1],
            [0, corr_h - 1]
        ], dtype=np.float32)
        
        # Transform corners to rectified canvas space (homography already has translation)
        corners_h = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h_matrix).reshape(-1, 2)
        min_x = int(np.floor(corners_h[:, 0].min()))
        max_x = int(np.ceil(corners_h[:, 0].max()))
        min_y = int(np.floor(corners_h[:, 1].min()))
        max_y = int(np.ceil(corners_h[:, 1].max()))
        
        # Add padding
        padding = 50
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Translate homography so output starts at (0,0) for display
        # This translation is ONLY for the output image, not for coordinate calculations
        translation = np.array([[1.0, 0.0, -min_x],
                                [0.0, 1.0, -min_y],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
        h_matrix_canvas = translation @ h_matrix
        
        # Calculate output canvas size
        out_w = int(max_x - min_x)
        out_h = int(max_y - min_y)
        
        # Warp the corrected image to rectified space
        rectified_image = cv2.warpPerspective(
            corrected_image, h_matrix_canvas, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Step 3: Optionally add grid overlay
        if show_grid:
            # Arena bounds are in the original canvas coordinate space (from map_coords)
            # We need to translate them to the output image coordinates
            left, top = self.arena_bounds["left"], self.arena_bounds["top"]
            right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]
            
            # Translate bounds to output canvas coordinates (accounting for the display translation)
            grid_left = int(left - min_x)
            grid_top = int(top - min_y)
            grid_right = int(right - min_x)
            grid_bottom = int(bottom - min_y)
            
            # Draw grid lines
            cols = self.grid_cols
            rows = self.grid_rows
            
            # Vertical lines
            for i in range(cols + 1):
                x = int(grid_left + (i * (grid_right - grid_left) / cols))
                x = max(0, min(out_w - 1, x))
                cv2.line(rectified_image, (x, grid_top), (x, grid_bottom),
                        (0, 0, 255), 1, cv2.LINE_AA)
            
            # Horizontal lines
            for j in range(rows + 1):
                y = int(grid_top + (j * (grid_bottom - grid_top) / rows))
                y = max(0, min(out_h - 1, y))
                cv2.line(rectified_image, (grid_left, y), (grid_right, y),
                        (0, 0, 255), 1, cv2.LINE_AA)
        
        # Return image and offset information for coordinate conversion
        offset_info = {
            "offset_x": min_x,
            "offset_y": min_y
        }
        
        return rectified_image, offset_info
