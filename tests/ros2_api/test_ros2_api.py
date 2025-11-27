#!/usr/bin/env python3
"""
Testprogram för ros2-api.py (flyttad ut från api/).

Kör med: python tests/test_ros2_api.py
"""

import os
import threading
import time
import sys
from pathlib import Path

# Lägg till repo-roten till path för att hitta moduler
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "api"))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

# Importera ros2-api via importlib (hanterar bindestreck)
import importlib.util
ros2_api_path = REPO_ROOT / "api" / "ros2-api.py"
spec = importlib.util.spec_from_file_location("ros2_api", ros2_api_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Kunde inte läsa in ros2_api-modulen från {ros2_api_path}")
ros2_api = importlib.util.module_from_spec(spec)
# Registrera modulen i sys.modules för att undvika problem med dataclass
sys.modules["ros2_api"] = ros2_api
spec.loader.exec_module(ros2_api)

RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow


class TestPublisher(Node):
    """ROS2 publisher som skickar testmeddelanden"""
    
    def __init__(self, topic: str, msg_type: str):
        super().__init__('test_publisher')
        self.msg_type = msg_type
        if msg_type == 'string':
            self.publisher = self.create_publisher(String, topic, 10)
        else:
            self.publisher = self.create_publisher(Float32MultiArray, topic, 10)
        self.timer = self.create_timer(0.1, self.publish_test_data)  # 10 Hz
        self.counter = 0
    
    def publish_test_data(self):
        """Publicera testdata"""
        self.counter += 1
        
        if self.msg_type == 'string':
            # Testdata som JSON-sträng
            if self.counter % 3 == 0:
                # Format: lista med dicts
                data = [
                    {"id": 0, "row": 100.0 + self.counter, "col": 200.0, "angle": 0.0, "certainty": 0.8},
                    {"id": 1, "row": 150.0, "col": 250.0 + self.counter, "angle": 1.57, "certainty": 0.9},
                    {"id": 2, "row": 200.0, "col": 300.0, "angle": 3.14, "certainty": 0.15},  # Låg certainty
                ]
            elif self.counter % 3 == 1:
                # Format: flat list
                data = [
                    50.0 + self.counter, 10.0, 0.0, 3, 0.7,  # Spiral 3
                    60.0, 20.0 + self.counter, 1.57, 4, 0.85,  # Spiral 4
                ]
            else:
                # Format: single dict
                data = {
                    "id": 5,
                    "row": 300.0 + self.counter * 0.1,
                    "col": 400.0,
                    "angle": 0.0,
                    "certainty": 0.75
                }
            
            import json
            msg = String()
            msg.data = json.dumps(data)
            self.publisher.publish(msg)
        else:
            # Float32MultiArray format
            msg = Float32MultiArray()
            # Format: [row, col, angle, id, cert, ...]
            msg.data = [
                70.0 + self.counter, 30.0, 0.0, 6, 0.8,
                80.0, 40.0 + self.counter, 1.57, 7, 0.9,
            ]
            self.publisher.publish(msg)
        
        if self.counter % 10 == 0:
            self.get_logger().info(f'Publicerat {self.counter} meddelanden')


def test_basic_functionality():
    """Test grundläggande funktionalitet"""
    print("\n" + "="*60)
    print("TEST 1: Grundläggande funktionalitet")
    print("="*60)
    
    api = RobotPositionAPI(
        topic='test_robot_positions',
        msg_type='string',
        min_certainty=0.25,
        max_speed=500.0
    )
    
    api.start()
    print("✓ API startat")
    
    # Vänta lite för att få meddelanden
    time.sleep(2)
    
    # Testa getPosition
    print("\nHämtar positioner:")
    found_any = False
    for spiral_id in range(10):
        position = api.getPosition(spiral_id)
        if position:
            found_any = True
            print(f"  Spiral {spiral_id}: ({position.row:.2f}, {position.col:.2f}) angle={position.angle:.2f} cert={position.certainty:.2f}")
    if not found_any:
        print("  Inga positioner hittades (kontrollera publisher)")
    
    api.stop()
    print("✓ API stoppat\n")


def test_context_manager():
    """Test context manager stöd"""
    print("\n" + "="*60)
    print("TEST 5: Context manager")
    print("="*60)
    
    print("Använder 'with'-statement...")
    with RobotPositionAPI(topic='test_robot_positions', min_certainty=0.25) as api:
        print("✓ API startat automatiskt")
        time.sleep(1)
        position = api.getPosition(0)
        if position:
            print(f"  Hittade position: Spiral {position.id}")
        print("✓ API kommer stoppas automatiskt när vi går ut")
    
    print("✓ Test 5 klar\n")


def main():
    """Huvudfunktion som kör valda tester"""
    print("\n" + "="*60)
    print("TESTPROGRAM FÖR ROS2-API")
    print("="*60)
    print("\nStartar ROS2...")
    
    # Initiera ROS2 (om inte redan initierad)
    try:
        if not rclpy.ok():
            rclpy.init()
    except RuntimeError:
        # rclpy redan initierad, ignorera
        pass
    
    # Skapa publisher i bakgrunden
    publisher = TestPublisher('test_robot_positions', 'string')
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(publisher)
    
    def run_publisher():
        """Kör publisher i bakgrunden"""
        executor.spin()
    
    pub_thread = threading.Thread(target=run_publisher, daemon=True)
    pub_thread.start()
    print("✓ Test publisher startad")
    
    time.sleep(1)  # Ge publisher tid att starta
    
    try:
        # Kör utvalda tester
        test_basic_functionality()
        test_context_manager()
    except KeyboardInterrupt:
        print("\n\nTest avbrutet av användaren")
    except Exception as e:
        print(f"\n\nFel under testning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Städa upp
        executor.shutdown()
        publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
