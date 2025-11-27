#!/usr/bin/env python3
"""
Testprogram för RobotPositionAPI med RIKTIG GPS Server

Detta program använder den riktiga GPS-servern istället för en simulerad publisher.
Det förutsätter att GPS-servern redan körs och skickar meddelanden på topic 'robotPositions'.

Kör med: python3 test_real_gps.py
"""

import time
import sys
import os
import importlib.util

# Lägg till parent directory till path för att hitta moduler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy

# Importera från ros2-api modulen (filnamnet har bindestreck så vi använder importlib)
ros2_api_path = os.path.join(os.path.dirname(__file__), "ros2-api.py")
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


def test_real_gps_continuous():
    """Kontinuerlig övervakning av positioner från GPS-servern"""
    print("\n" + "="*60)
    print("TEST: Kontinuerlig övervakning")
    print("="*60)
    print("\nVisar positioner var 2:e sekund i 20 sekunder")
    print("Tryck Ctrl+C för att avbryta tidigare\n")
    
    update_count = [0]
    
    def on_position_update(robots: list[SpiralRow]):
        """Callback som anropas när positioner uppdateras"""
        update_count[0] += 1
        for robot in robots:
            print(f"  [Uppdatering #{update_count[0]}] Spiral {robot.id}: "
                  f"({robot.row:.2f}, {robot.col:.2f}) cert={robot.certainty:.2f}")
    
    api = RobotPositionAPI(
        topic='robotPositions',
        msg_type='string',
        min_certainty=0.25,
        max_speed=500.0
    )
    
    api.start()
    api.setPositionCallback(on_position_update)
    print("✓ API startat med callback")
    print("Väntar på uppdateringar...\n")
    
    try:
        # Visa positioner var 2:e sekund
        for i in range(10):
            time.sleep(2)
            print(f"\n--- Snapshot {i+1} (efter {2*(i+1)} sekunder) ---")
            found_any = False
            for spiral_id in range(10):
                position = api.getPosition(spiral_id)
                if position:
                    found_any = True
                    print(f"  Spiral {spiral_id}: ({position.row:.2f}, {position.col:.2f}) "
                          f"cert={position.certainty:.2f}")
            if not found_any:
                print("  Inga positioner hittades")
            print(f"Totalt antal uppdateringar via callback: {update_count[0]}")
    except KeyboardInterrupt:
        print("\n\nAvbruten av användaren")
    
    api.stop()
    print("\n✓ API stoppat")
    print("✓ Test klar\n")


def check_ros2_topics():
    """Hjälpfunktion för att lista tillgängliga topics"""
    print("\n" + "="*60)
    print("KONTROLLERA ROS2 TOPICS")
    print("="*60)
    print("\nKör detta kommando i en annan terminal för att se tillgängliga topics:")
    print("  ros2 topic list")
    print("\nFör att se meddelanden på topic 'robotPositions':")
    print("  ros2 topic echo /robotPositions")
    print("\nFör att se topic-typ:")
    print("  ros2 topic info /robotPositions")
    print("\nFör att se topic-typ och format:")
    print("  ros2 topic type /robotPositions")
    print("\n")


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
        # Visa hjälp om topics
        check_ros2_topics()
        
        # Fråga användaren vad de vill göra
        print("Vad vill du göra?")
        print("1. Grundläggande test (hämtar positioner en gång)")
        print("2. Kontinuerlig övervakning (visar positioner var 2:e sekund)")
        print("3. Avsluta")
        
        choice = input("\nVälj (1-3): ").strip()
        
        if choice == '1':
            test_real_gps_basic()
        elif choice == '2':
            test_real_gps_continuous()
        elif choice == '3':
            print("\nAvslutar...")
        else:
            print("\nOgiltigt val, kör grundläggande test...")
            test_real_gps_basic()
        
    except KeyboardInterrupt:
        print("\n\nAvbruten av användaren")
    except Exception as e:
        print(f"\n\nFel: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Städa upp
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass
        print("\nStädat upp och avslutat.")


if __name__ == '__main__':
    main()
