#!/usr/bin/env python3
"""
Testprogram för RobotPositionAPI med RIKTIG GPS Server (flyttad ut från api/).

Kör med: python tests/test_real_gps.py
"""

import time
import sys
import os
from pathlib import Path

# Lägg till repo-root i path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "api"))

import rclpy

# Importera från ros2-api modulen (filnamnet har bindestreck så vi använder importlib)
import importlib.util
ros2_api_path = REPO_ROOT / "api" / "ros2-api.py"
spec = importlib.util.spec_from_file_location("ros2_api", ros2_api_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Kunde inte läsa in ros2_api-modulen från {ros2_api_path}")
ros2_api = importlib.util.module_from_spec(spec)
sys.modules["ros2_api"] = ros2_api
spec.loader.exec_module(ros2_api)

RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow


def test_real_gps_basic():
    """Test grundläggande funktionalitet med riktig GPS Server"""
    print("\n" + "="*60)
    print("TEST: Grundläggande funktionalitet med RIKTIG GPS Server")
    print("="*60)
    print("\nFörutsätter att GPS-servern körs på topic 'robotPositions'")
    print("Väntar 3 sekunder för att få meddelanden...\n")
    
    api = RobotPositionAPI(
        topic='robotPositions',  # Riktig topic från GPS-servern
        msg_type='string',  # Ändra till 'float32multiarray' om servern använder det
        min_certainty=0.25,
        max_speed=500.0
    )
    
    api.start()
    print("✓ API startat")
    
    # Vänta lite för att få meddelanden från GPS-servern
    time.sleep(3)
    
    # Hämta positioner
    print("\nHämtar positioner från GPS-servern:")
    found_any = False
    for spiral_id in range(10):
        position = api.getPosition(spiral_id)
        if position:
            found_any = True
            print(f"  Spiral {spiral_id}: ({position.row:.2f}, {position.col:.2f}) "
                  f"angle={position.angle:.2f} cert={position.certainty:.2f}")
    
    if not found_any:
        print("  ⚠ Inga positioner hittades")
        print("\nMöjliga orsaker:")
        print("  - GPS-servern körs inte")
        print("  - Topic-namnet är fel (kontrollera med: ros2 topic list)")
        print("  - Meddelandeformatet är fel (prova ändra msg_type)")
        print("  - Positioner filtreras bort (certainty för låg eller outliers)")
    
    api.stop()
    print("\n✓ API stoppat")
    print("✓ Test klar\n")


def main():
    """Huvudfunktion"""
    print("\n" + "="*60)
    print("TESTPROGRAM FÖR RIKTIG GPS SERVER")
    print("="*60)
    
    # Initiera ROS2
    try:
        if not rclpy.ok():
            rclpy.init()
    except RuntimeError:
        pass
    
    try:
        test_real_gps_basic()
    except KeyboardInterrupt:
        print("\n\nAvbruten av användaren")
    except Exception as e:
        print(f"\n\nFel: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
