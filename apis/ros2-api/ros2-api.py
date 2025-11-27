from __future__ import annotations
import json
import re
import time
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

import rclpy
from rclpy import executors
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray

import sys
from pathlib import Path

# Lägg till nuvarande katalog så ros2.py hittas när skriptet körs direkt
_ros2_api_dir = Path(__file__).parent
if str(_ros2_api_dir) not in sys.path:
    sys.path.insert(0, str(_ros2_api_dir))
from ros2 import SpiralRow


class MessageParser:
    """Hanterar parsing av ROS2-meddelanden till SpiralRow-objekt"""
    
    @staticmethod
    def _as_spiral_rows_from_list(vals: List[float]) -> List[SpiralRow]:
        """Konvertera flat list till SpiralRow-objekt"""
        n = len(vals)
        rows: List[SpiralRow] = []
        if n % 5 != 0:
            return rows
        for i in range(0, n, 5):
            row, col, angle, rid, cert = (
                float(vals[i]),
                float(vals[i + 1]),
                float(vals[i + 2]),
                int(vals[i + 3]),
                float(vals[i + 4]),
            )
            if int(row) == -1 or rid == -1:
                continue
            rows.append(SpiralRow(rid, row, col, angle, cert))
        return rows
    
    @staticmethod
    def _parse_dict(obj: dict) -> Optional[SpiralRow]:
        """Parsa ett dict-objekt till SpiralRow"""
        row = float(obj.get('row', obj.get('r', obj.get('Row', -1))))
        col = float(obj.get('col', obj.get('c', obj.get('Col', -1))))
        angle = float(obj.get('angle', obj.get('Angle', 0.0)))
        rid = int(obj.get('id', obj.get('ID', obj.get('identity', -1))))
        cert = float(obj.get('certainty', obj.get('conf', obj.get('Conf', 1.0))))
        if int(row) != -1 and rid != -1:
            return SpiralRow(rid, row, col, angle, cert)
        return None
    
    @staticmethod
    def _parse_json_string(s: str) -> List[SpiralRow]:
        """Parsa JSON-sträng till SpiralRow-objekt"""
        rows: List[SpiralRow] = []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                if obj and isinstance(obj[0], (list, tuple)):
                    # Nested list: [[row, col, angle, id, cert], ...]
                    flat: List[float] = []
                    for r in obj:
                        flat.extend([float(v) for v in r])
                    rows.extend(MessageParser._as_spiral_rows_from_list(flat))
                elif obj and isinstance(obj[0], dict):
                    # List of dicts: [{"row": ..., "col": ..., ...}, ...]
                    for r in obj:
                        parsed = MessageParser._parse_dict(r)
                        if parsed:
                            rows.append(parsed)
                else:
                    # Flat list: [row, col, angle, id, cert, ...]
                    vals = [float(v) for v in obj]
                    rows.extend(MessageParser._as_spiral_rows_from_list(vals))
            elif isinstance(obj, dict):
                # Single dict: {"row": ..., "col": ..., ...}
                parsed = MessageParser._parse_dict(obj)
                if parsed:
                    rows.append(parsed)
            else:
                # Fallback: regex parsing
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                vals = [float(v) for v in nums]
                rows.extend(MessageParser._as_spiral_rows_from_list(vals))
        except Exception:
            # Fallback: regex parsing om JSON misslyckas
            try:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                vals = [float(v) for v in nums]
                rows.extend(MessageParser._as_spiral_rows_from_list(vals))
            except Exception:
                pass
        return rows
    
    @staticmethod
    def parse_string(msg: String) -> List[SpiralRow]:
        """Parsa String-meddelande till SpiralRow-objekt"""
        return MessageParser._parse_json_string(msg.data)
    
    @staticmethod
    def parse_array(msg: Float32MultiArray) -> List[SpiralRow]:
        """Parsa Float32MultiArray-meddelande till SpiralRow-objekt"""
        try:
            vals = list(msg.data)
            return MessageParser._as_spiral_rows_from_list(vals)
        except Exception:
            return []


class PositionFilter:
    """Filtrerar robotpositioner baserat på certainty och outlier detection"""
    
    @staticmethod
    def should_accept(row: SpiralRow, min_certainty: float) -> bool:
        """Kolla om position ska accepteras baserat på certainty"""
        return row.certainty >= min_certainty
    
    @staticmethod
    def is_outlier(row: SpiralRow, prev_row: Optional[SpiralRow], 
                   prev_timestamp: Optional[float], max_speed: float) -> bool:
        """
        Kolla om position är en outlier (för långt från föregående).
        
        Args:
            row: Nuvarande position
            prev_row: Föregående giltiga position (None om första gången)
            prev_timestamp: Timestamp för föregående position
            max_speed: Max hastighet (enheter per sekund)
        
        Returns:
            True om position är outlier och ska ignoreras
        """
        if prev_row is None:
            return False  # Första positionen accepteras alltid
        
        # Beräkna avstånd till föregående position
        dx = row.row - prev_row.row
        dy = row.col - prev_row.col
        distance = math.hypot(dx, dy)
        
        # Beräkna tid sedan senaste uppdatering
        if prev_timestamp is None:
            return False  # Ingen timestamp, acceptera
        
        current_time = time.time()
        delta_time = max(0.001, current_time - prev_timestamp)
        
        # Max avstånd baserat på max hastighet
        max_distance = max_speed * delta_time
        
        # Om avståndet är för stort, det är en outlier
        return distance > max_distance


class PositionStore:
    """Thread-safe storage för robotpositioner"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._rows: Dict[int, SpiralRow] = {}
        self._last_seen: Dict[int, float] = {}
        self._prev_rows: Dict[int, SpiralRow] = {}  # Föregående giltiga positioner
        self._prev_timestamps: Dict[int, float] = {}  # Timestamps för föregående positioner
    
    def get(self, spiral_id: int) -> Optional[SpiralRow]:
        """Hämta position för specifik spiral (thread-safe)"""
        with self._lock:
            return self._rows.get(spiral_id)
    
    def get_all(self) -> List[SpiralRow]:
        """Hämta alla positioner (thread-safe, kopierar data)"""
        with self._lock:
            return list(self._rows.values())
    
    def set(self, row: SpiralRow):
        """
        Uppdatera position (thread-safe).
        Sparar automatiskt föregående position för outlier detection.
        """
        with self._lock:
            # Spara föregående position om den finns
            prev = self._rows.get(row.id)
            if prev is not None:
                self._prev_rows[row.id] = prev
                self._prev_timestamps[row.id] = self._last_seen.get(row.id, time.time())
            
            # Uppdatera position
            self._rows[row.id] = row
            self._last_seen[row.id] = time.time()
    
    def get_prev(self, spiral_id: int) -> Tuple[Optional[SpiralRow], Optional[float]]:
        """
        Hämta föregående position och timestamp för outlier detection.
        
        Returns:
            Tuple av (prev_row, prev_timestamp) eller (None, None)
        """
        with self._lock:
            prev_row = self._prev_rows.get(spiral_id)
            prev_timestamp = self._prev_timestamps.get(spiral_id)
            return prev_row, prev_timestamp


class MatrixCollisionNode(Node):
    """ROS2-nod som tar emot robotpositioner från GPS Server"""
    
    def __init__(self, topic: str, msg_type: str, min_certainty: float, 
                 max_speed: float, tick_hz: float):
        super().__init__('matrix_gps_ros2')
        self.min_certainty = min_certainty
        self.max_speed = max_speed
        
        # Komponenter
        self._parser = MessageParser()
        self._filter = PositionFilter()
        self._store = PositionStore()
        
        # Callback för position updates
        self._position_callback: Optional[Callable[[List[SpiralRow]], None]] = None
        self._callback_lock = threading.RLock()
        
        # ROS2 subscription och timer
        if msg_type == 'string':
            self.sub = self.create_subscription(String, topic, self.cb_string, 10)
        else:
            self.sub = self.create_subscription(Float32MultiArray, topic, self.cb_array, 10)
        
        tick_interval = 1.0 / max(1e-3, tick_hz)
        self.timer = self.create_timer(tick_interval, self.on_tick)
    
    def cb_string(self, msg: String):
        """Callback för String-meddelanden"""
        rows = self._parser.parse_string(msg)
        for row in rows:
            self._upsert(row)
    
    def cb_array(self, msg: Float32MultiArray):
        """Callback för Float32MultiArray-meddelanden"""
        rows = self._parser.parse_array(msg)
        for row in rows:
            self._upsert(row)
    
    def _upsert(self, row: SpiralRow):
        """Uppdatera position med filtrering"""
        # Filtrera på certainty
        if not self._filter.should_accept(row, self.min_certainty):
            return
        
        # Hämta föregående position för outlier detection
        prev_row, prev_timestamp = self._store.get_prev(row.id)
        
        # Filtrera outliers
        if self._filter.is_outlier(row, prev_row, prev_timestamp, self.max_speed):
            return
        
        # Spara position
        self._store.set(row)
        
        # Anropa callback utanför store-låset
        with self._callback_lock:
            callback = self._position_callback
        
        if callback is not None:
            try:
                callback([row])
            except Exception:
                # Ignorera fel i callback för att inte krascha ROS2-noden
                pass
    
    def on_tick(self):
        """Körs regelbundet av timern - kan användas för kollisionsdetektering senare"""
        pass
    
    def set_position_callback(self, callback: Optional[Callable[[List[SpiralRow]], None]]):
        """Sätt callback som anropas när positioner uppdateras"""
        with self._callback_lock:
            self._position_callback = callback
    
    def get_position(self, spiral_id: int) -> Optional[SpiralRow]:
        """Hämta position för en specifik spiral (thread-safe)"""
        return self._store.get(spiral_id)
    
    def get_all_positions(self) -> List[SpiralRow]:
        """Hämta alla aktiva robotpositioner (thread-safe)"""
        return self._store.get_all()
    
    def destroy_node(self):
        """Städa upp vid shutdown"""
        super().destroy_node()


@dataclass
class NodeHandle:
    """Hanterar ROS2-nod och executor"""
    node: MatrixCollisionNode
    executor: executors.Executor
    thread: threading.Thread
    
    def stop(self):
        """Stoppa noden och executor"""
        try:
            self.executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def start_async(topic: str = 'robotPositions',
                msg_type: str = 'string',
                min_cert: float = 0.25,
                max_speed: float = 500.0,
                tick_hz: float = 5.0,
                multithread: bool = True) -> NodeHandle:
    """Starta ROS2-noden asynkront i bakgrunden"""
    try:
        if not rclpy.ok():
            rclpy.init()
    except RuntimeError:
        # rclpy redan initierad, ignorera
        pass
    node = MatrixCollisionNode(topic, msg_type, min_cert, max_speed, tick_hz)
    executor = executors.MultiThreadedExecutor() if multithread else executors.SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return NodeHandle(node=node, executor=executor, thread=t)


class RobotPositionAPI:
    """
    Enkelt API för att hämta robotpositioner från GPS Server via ROS2.
    
    Användning:
        api = RobotPositionAPI(topic='robotPositions', min_certainty=0.25, max_speed=500.0)
        api.start()
        
        # Hämta position för specifik spiral
        position = api.getPosition(5)
        
        # Sätt callback för uppdateringar
        api.setPositionCallback(lambda robots: print(f"Uppdaterade: {len(robots)}"))
        
        api.stop()
    """
    
    def __init__(self,
                 topic: str = 'robotPositions',
                 msg_type: str = 'string',
                 min_certainty: float = 0.25,
                 max_speed: float = 500.0,
                 tick_hz: float = 5.0):
        """
        Initiera API:et för robotpositioner.
        
        Args:
            topic: ROS2 topic att lyssna på
            msg_type: Meddelandetyp ('string' eller 'float32multiarray')
            min_certainty: Minsta certainty för att acceptera position (0.0-1.0)
            max_speed: Max hastighet för outlier detection (enheter per sekund)
            tick_hz: Frekvens för timer (används för kollisionsdetektering senare)
        """
        self.topic = topic
        self.msg_type = msg_type
        self.min_certainty = min_certainty
        self.max_speed = max_speed
        self.tick_hz = tick_hz
        
        # ROS2-noden och executor
        self._node_handle: Optional[NodeHandle] = None
        self._node: Optional[MatrixCollisionNode] = None
    
    def start(self):
        """Starta att lyssna på robotpositioner från GPS Server"""
        if self._node_handle is not None:
            return  # Redan startad
        
        self._node_handle = start_async(
            topic=self.topic,
            msg_type=self.msg_type,
            min_cert=self.min_certainty,
            max_speed=self.max_speed,
            tick_hz=self.tick_hz,
            multithread=True
        )
        self._node = self._node_handle.node
    
    def stop(self):
        """Stoppa att lyssna på positioner"""
        if self._node_handle:
            self._node_handle.stop()
            self._node_handle = None
            self._node = None
    
    def getPosition(self, spiralID=None):
        """
        Hämta position för spiral(ar) - enkel och flexibel.
        
        Args:
            spiralID: 
                - None: Returnerar alla robotar (List[SpiralRow])
                - int: Returnerar en specifik robot (SpiralRow eller None)
                - List[int]: Returnerar specifika robotar (t.ex. [5, 9])
        
        Returns:
            - Om spiralID är None: Lista med alla SpiralRow
            - Om spiralID är int: SpiralRow eller None
            - Om spiralID är List[int]: Lista med SpiralRow
        
        Exempel:
            api.getPosition()        # Alla robotar
            api.getPosition(9)       # Bara spiral 9
            api.getPosition([5, 9])  # Spiral 5 och 9
        """
        if self._node is None:
            if spiralID is None or isinstance(spiralID, list):
                return []
            return None
        
        # Alla robotar
        if spiralID is None:
            return self._node.get_all_positions()
        
        # Flera specifika robotar
        if isinstance(spiralID, list):
            result = []
            for rid in spiralID:
                pos = self._node.get_position(rid)
                if pos is not None:
                    result.append(pos)
            return result
        
        # En specifik robot
        return self._node.get_position(spiralID)
    
    def setPositionCallback(self, callback: Optional[Callable[[List[SpiralRow]], None]]):
        """
        Sätt callback som anropas när positioner uppdateras.
        
        Args:
            callback: Funktion som tar en lista med SpiralRow som argument
                     Anropas i ROS2-tråden, så håll den snabb!
        """
        if self._node is None:
            raise RuntimeError("API måste startas först (anropa start())")
        self._node.set_position_callback(callback)
    
    def __enter__(self):
        """Context manager support - startar automatiskt"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - stoppar automatiskt"""
        self.stop()
