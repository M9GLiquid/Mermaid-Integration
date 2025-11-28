import sys
from pathlib import Path
import importlib.util

# Ensure local api modules resolve when run from repo root or api/
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Import astar-api.py despite hyphen
api_path = THIS_DIR / "astar-api.py"
spec = importlib.util.spec_from_file_location("astar_api", api_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load astar_api from {api_path}")
astar_api = importlib.util.module_from_spec(spec)
sys.modules["astar_api"] = astar_api
spec.loader.exec_module(astar_api)

search = astar_api.search
draw_grid = astar_api.draw_grid
next_action_from_path = astar_api.next_action_from_path
encode_action_ascii = astar_api.encode_action_ascii

def main():
    # Inline sample grid (3x3 robot fits; outer bounds are implicit walls)
    # 0 = free, 1 = wall, 2 = crab/start, 3 = food, 4 = home, 5 = threat
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    print("Loaded inline grid\n")
    # ---- 2. Choose which goal you want to search for ----
    # Options: "FOOD" or "HOME"
    GOAL_TYPE = "HOME"

    # ---- 4. Run A* ----
    nodes, path, start_raw, goal_raw = search(grid, GOAL_TYPE)
    print(f"Nodes expanded: {nodes}")

     # ---- 5. For Visualisation A* ----
    print(f"Start position (Crab): {start_raw}")
    print(f"Goal position: {goal_raw}\n")

    if path:
        print("Path found!\n")
        draw_grid(grid, path, start_raw, goal_raw)
    else:
        print("No path found.")
        draw_grid(grid, path, start_raw, goal_raw)
        exit()

    # ---- EXTRA 6. Calculate next action based on path ----
    init_heading = 'N'  # Initial heading

    action, init_heading = next_action_from_path(
        path,
        init_heading=init_heading,
        allow_back=False
    )

    print("Next action:", action)

    # ---- 7. Convert to ASCII command ----
    tx = encode_action_ascii(action)
    print("TX string:", tx)


if __name__ == "__main__":
    main()
