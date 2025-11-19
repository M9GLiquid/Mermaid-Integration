# ROS2 Position API

Detta är ROS2 API:et för att hämta robotpositioner från GPS Server.

## Filstruktur

- `ros2-api.py` - Huvudfilen med RobotPositionAPI
- `ros2.py` - Innehåller SpiralRow dataclass

## Användning i Integration-v1

API:et importeras automatiskt i `main.py`:

```python
from main import RobotPositionAPI, SpiralRow

# Använd API:et
api = RobotPositionAPI(topic='robotPositions', min_certainty=0.25)
api.start()

position = api.getPosition(5)
if position:
    print(f"Spiral 5: ({position.row}, {position.col})")

api.stop()
```

## Krav

- ROS2 måste vara installerat och aktiverat:
  ```bash
  source /opt/ros/jazzy/setup.bash
  ```
- GPS-servern måste köra och skicka meddelanden på topic `robotPositions`

## Dokumentation

Se huvuddokumentationen i `Ros2/api/API_OVERVIEW.md` för fullständig dokumentation.
