#!/usr/bin/env python3
"""
Enkel demo av RobotPositionAPI

Detta visar grundläggande användning av API:et.
För att köra detta behöver du ha en ROS2 publisher som skickar meddelanden
på topic 'robotPositions'.

Kör med: python3 demo_ros2_api.py
"""

import time
import sys
import os
import importlib.util

# Lägg till parent directory till path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importera från ros2-api modulen (filnamnet har bindestreck)
ros2_api_path = os.path.join(os.path.dirname(__file__), "ros2-api.py")
spec = importlib.util.spec_from_file_location("ros2_api", ros2_api_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Kunde inte läsa in ros2_api-modulen från {ros2_api_path}")
ros2_api = importlib.util.module_from_spec(spec)
# Registrera modulen i sys.modules för att undvika problem med dataclass
sys.modules["ros2_api"] = ros2_api
spec.loader.exec_module(ros2_api)

RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow


def simple_demo():
    """Enkel demo av API:et"""
    print("="*60)
    print("DEMO: RobotPositionAPI")
    print("="*60)
    
    # Skapa API-instans
    api = RobotPositionAPI(
        topic='robotPositions',
        msg_type='string',  # eller 'float32multiarray'
        min_certainty=0.25,
        max_speed=500.0
    )
    
    # Starta API:et
    print("\n1. Startar API...")
    api.start()
    print("   ✓ API startat")
    
    # Vänta lite för att få meddelanden
    print("\n2. Väntar på positioner (5 sekunder)...")
    time.sleep(5)
    
    # Hämta positioner
    print("\n3. Hämtar positioner:")
    found_any = False
    for spiral_id in range(10):
        position = api.getPosition(spiral_id)
        if position:
            found_any = True
            print(f"   Spiral {spiral_id}: "
                  f"({position.row:.2f}, {position.col:.2f}) "
                  f"angle={position.angle:.2f} "
                  f"cert={position.certainty:.2f}")
    
    if not found_any:
        print("   ⚠ Inga positioner hittades")
        print("   Kontrollera att ROS2 publisher körs på topic 'robotPositions'")
    
    # Stoppa API:et
    print("\n4. Stoppar API...")
    api.stop()
    print("   ✓ API stoppat")
    
    print("\n" + "="*60)
    print("Demo klar!")
    print("="*60)


def callback_demo():
    """Demo med callback"""
    print("\n" + "="*60)
    print("DEMO: Callback-funktionalitet")
    print("="*60)
    
    update_count = [0]
    
    def on_update(robots: list[SpiralRow]):
        """Callback som anropas när positioner uppdateras"""
        update_count[0] += 1
        for robot in robots:
            print(f"   Uppdatering #{update_count[0]}: "
                  f"Spiral {robot.id} på ({robot.row:.2f}, {robot.col:.2f})")
    
    # Använd context manager för automatisk start/stop
    print("\nAnvänder 'with'-statement för automatisk hantering...")
    with RobotPositionAPI(topic='robotPositions', min_certainty=0.25) as api:
        # Sätt callback
        api.setPositionCallback(on_update)
        print("   ✓ Callback satt")
        
        # Vänta på uppdateringar
        print("\n   Väntar på uppdateringar (10 sekunder)...")
        time.sleep(10)
        
        print(f"\n   Totalt antal uppdateringar: {update_count[0]}")
    
    print("\n   ✓ API stoppat automatiskt")
    print("\n" + "="*60)


def main():
    """Huvudfunktion"""
    print("\nVälkommen till RobotPositionAPI demo!")
    print("\nDetta program visar hur man använder API:et.")
    print("För att detta ska fungera behöver du ha en ROS2 publisher")
    print("som skickar meddelanden på topic 'robotPositions'.\n")
    
    try:
        # Kör enkel demo
        simple_demo()
        
        # Fråga om callback-demo
        print("\n" + "-"*60)
        response = input("\nVill du se callback-demo? (j/n): ").strip().lower()
        if response == 'j':
            callback_demo()
        
    except KeyboardInterrupt:
        print("\n\nDemo avbruten av användaren")
    except Exception as e:
        print(f"\n\nFel: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
