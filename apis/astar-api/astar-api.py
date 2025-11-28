#!/usr/bin/env python3
"""
Standalone A* pathfinding API for Mermaid-Astar.

Provides:
- search(grid, goaltype): A* over grid values (2=start, 3=food, 4=home, 1/5 obstacles)
- next_action_from_path(path, init_heading, allow_back): step-wise action from path
- encode_action_ascii(action): ASCII command mapping
- identify_start/identify_goal helpers
- draw_grid for simple CLI visualization
"""

import heapq
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# Grid cell values
OBSTACLE_VALUES = {1, 5}  # walls, threats
START_VALUES = {2, "2", "CRAB", "ROBOT"}
FOOD_VALUES = {3, "3", "FOOD", "Food", "F"}
HOME_VALUES = {4, "4", "HOME", "Home", "H"}
AVOID_VALUES = {5, "5", "THREAT", "Threat"}

# Cardinal directions for action generation
DIR_VECT = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
VECT_DIR = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}
TURN_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}
TURN_LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, Tuple[int, int]]] = []

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def add(self, item: Tuple[int, int], priority: float) -> None:
        heapq.heappush(self.elements, (priority, item))

    def remove(self) -> Tuple[int, int]:
        return heapq.heappop(self.elements)[1]


def _is_free_for_robot(center: Tuple[int, int], grid: List[List[int]], robot_radius: int = 1) -> bool:
    """Check if a (2*radius+1)x(2*radius+1) robot fits with its center at 'center'."""
    x, y = center
    rows, cols = len(grid), len(grid[0])

    for dy in range(-robot_radius, robot_radius + 1):
        for dx in range(-robot_radius, robot_radius + 1):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= cols or ny >= rows:
                return False
            if grid[ny][nx] in OBSTACLE_VALUES:
                return False
    return True


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance in (x, y)."""
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)


def _get_neighbors(pos: Tuple[int, int], grid: List[List[int]]) -> Iterable[Tuple[int, int]]:
    """4-neighbors where the robot fits (3x3)."""
    x, y = pos
    rows, cols = len(grid), len(grid[0])
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows and _is_free_for_robot((nx, ny), grid, robot_radius=1):
            yield (nx, ny)


def _find_nearest_valid(center: Optional[Tuple[int, int]], grid: List[List[int]], robot_radius: int = 1,
                       max_search_radius: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Find nearest valid center where robot fits; returns None if none found."""
    if center is None:
        return None
    if _is_free_for_robot(center, grid, robot_radius=robot_radius):
        return center

    rows, cols = len(grid), len(grid[0])
    cx, cy = center
    q: Deque[Tuple[int, int]] = deque([center])
    visited = {center}

    while q:
        x, y = q.popleft()
        if _is_free_for_robot((x, y), grid, robot_radius=robot_radius):
            return (x, y)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if (nx, ny) in visited:
                continue
            if max_search_radius is not None and abs(nx - cx) + abs(ny - cy) > max_search_radius:
                continue
            visited.add((nx, ny))
            q.append((nx, ny))
    return None


def _identify_start(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] in START_VALUES:
                return (x, y)
    return None


def _identify_goal(grid: List[List[int]], goal_type: str = "FOOD") -> Optional[Tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    target_values = HOME_VALUES if goal_type.upper() == "HOME" else FOOD_VALUES
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] in target_values:
                return (x, y)
    return None


def search(grid: List[List[int]], goaltype: Optional[str] = None):
    """
    A* with 3x3 robot and auto-adjusted start/goal if too close to walls.

    Returns: (nodes_expanded, path_deque or None, start, goal)
    """
    start_raw = _identify_start(grid)
    goal_raw = _identify_goal(grid, goal_type=goaltype or "")

    if start_raw is None or goal_raw is None:
        return 0, None

    start = _find_nearest_valid(start_raw, grid, robot_radius=1)
    goal = _find_nearest_valid(goal_raw, grid, robot_radius=1)
    if start is None or goal is None:
        return 0, None, start_raw, goal_raw

    pq = PriorityQueue()
    pq.add(start, 0)
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0}
    nodes = 0

    while not pq.is_empty():
        current = pq.remove()
        nodes += 1

        if current == goal:
            path: List[Tuple[int, int]] = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return nodes, deque(path), start_raw, goal_raw

        for nxt in _get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                f_score = tentative_g + _heuristic(nxt, goal)
                came_from[nxt] = current
                pq.add(nxt, f_score)

    return nodes, None, start_raw, goal_raw


# ---------------- Actions ----------------

def _desired_direction(a: Tuple[int, int], b: Tuple[int, int]) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = VECT_DIR.get((dx, dy))
    if d is None:
        raise ValueError(f"Non-cardinal step in path: {(dx, dy)}")
    return d


def _turns_needed(curr: str, target: str):
    if curr == target:
        return []
    if TURN_RIGHT[curr] == target:
        return ["RIGHT"]
    if TURN_LEFT[curr] == target:
        return ["LEFT"]
    return ["AROUND"]


def next_action_from_path(path: List[Tuple[int, int]], init_heading: str = "N", allow_back: bool = False):
    """
    Given a path [(x,y), ...] and current heading, return ONE action.
    Returns: (action, new_heading)
    action: ('TURN', 'LEFT'|'RIGHT'|'AROUND') or ('FORWARD',1) or ('BACK',1) or ('STOP',None)
    """
    if not path or len(path) < 2:
        return ("STOP", None), init_heading

    curr, nxt = path[0], path[1]
    need_dir = _desired_direction(curr, nxt)
    need = _turns_needed(init_heading, need_dir)

    if need == ["AROUND"] and allow_back:
        return ("BACK", 1), init_heading
    elif need:
        turn = need[0]
        new_heading = (
            TURN_RIGHT[init_heading]
            if turn == "RIGHT"
            else TURN_LEFT[init_heading]
            if turn == "LEFT"
            else TURN_RIGHT[TURN_RIGHT[init_heading]]
        )
        return ("TURN", turn), new_heading
    else:
        return ("FORWARD", 1), init_heading


def encode_action_ascii(action):
    kind, val = action
    if kind == "FORWARD":
        return f"F;{val}"
    if kind == "BACK":
        return f"B;{val}"
    if kind == "TURN":
        deg = 180 if val == "AROUND" else 90
        return f"{'TR' if val != 'LEFT' else 'TL'};{deg}"
    return "STOP"


# ---------------- Utility visualization ----------------

def draw_grid(grid: List[List[int]], path, start, goal):
    """Draw grid with (x, y) coordinates."""
    path_set = set(path) if path else set()
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    for y in range(rows):
        row = []
        for x in range(cols):
            cell = grid[y][x]
            pos = (x, y)
            if pos == start:
                row.append("C")
            elif pos == goal:
                if cell in HOME_VALUES:
                    row.append("H")
                elif cell in FOOD_VALUES:
                    row.append("F")
            elif cell in OBSTACLE_VALUES:
                row.append("#")
            elif cell in AVOID_VALUES:
                row.append("T")
            elif pos in path_set:
                row.append("*")
            else:
                row.append(".")
        print(" ".join(row))
    print()


__all__ = ["search", "next_action_from_path", "encode_action_ascii", "draw_grid"]
