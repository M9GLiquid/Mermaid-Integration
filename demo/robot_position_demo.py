#!/usr/bin/env python3
"""
Demo: Robot Position Tracking (Spiral 9)

Enkel demo som visar hur man använder RobotPositionAPI.getPosition(spiralID)
för att hämta position för en specifik spiral.

API:et lyssnar på alla spiraler från GPS-servern, men du kan hämta
position för vilken spiral som helst med getPosition(spiralID).

Usage: python3 -m demo.robot_position_demo
"""

import sys
import time
from pathlib import Path

# Lägg till root till path
integration_root = Path(__file__).parent.parent
sys.path.insert(0, str(integration_root))

from main import RobotPositionAPI, SpiralRow


def main():
    """Huvudfunktion - trackar spiral 9"""
    print("=" * 60)
    print("Robot Position Demo - Tracking Spiral 9")
    print("=" * 60)
    print("\nAnvänder RobotPositionAPI.getPosition(9) för att hämta spiral 9.")
    print("API:et lyssnar på alla spiraler, men vi hämtar bara spiral 9.")
    print("\nTryck Ctrl+C för att avsluta.\n")
    
    # Skapa och starta API
    api = RobotPositionAPI(
        topic='robotPositions',
        msg_type='string',
        min_certainty=0.45,
        max_speed=500.0
    )
    
    print("✓ Startar ROS2 API...")
    api.start()
    print("✓ API startat - lyssnar på alla spiraler från GPS-servern\n")
    
    # Vänta lite för att få första meddelanden
    print("Väntar på positioner från GPS-servern (3 sekunder)...")
    time.sleep(3)
    
    try:
        # Hämta position för spiral 9
        position = api.getPosition(9)  # <-- Detta är allt du behöver!
        
        if position:
            print("\n✓ Spiral 9 hittad!")
            print(f"  Position: ({position.row:.2f}, {position.col:.2f})")
            print(f"  Vinkel: {position.angle:.2f}")
            print(f"  Certainty: {position.certainty:.2f}")
        else:
            print("\n⚠ Spiral 9 hittades inte ännu.")
            print("  Kontrollera att GPS-servern körs och skickar data.")
        
        # Kontinuerlig tracking
        print("\n--- Kontinuerlig tracking av Spiral 9 ---")
        print("Uppdaterar position var 2:e sekund. Tryck Ctrl+C för att stoppa.\n")
        
        update_count = 0
        while True:
            time.sleep(2)
            update_count += 1
            
            # Hämta position för spiral 9 - enkelt!
            position = api.getPosition(9)
            if position:
                print(f"[#{update_count}] Spiral 9: "
                      f"({position.row:.2f}, {position.col:.2f}) "
                      f"angle={position.angle:.2f} cert={position.certainty:.2f}")
            else:
                print(f"[#{update_count}] Spiral 9: Ingen position hittad")
                
    except KeyboardInterrupt:
        print("\n\nAvbruten av användaren")
    finally:
        # Städa upp
        print("\nStoppar API...")
        api.stop()
        print("✓ API stoppat")
        print("\nDemo avslutad.")


if __name__ == "__main__":
    main()
