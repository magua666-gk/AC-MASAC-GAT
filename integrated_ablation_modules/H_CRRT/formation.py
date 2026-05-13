from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]


def get_slot_ref(leader_xy: Point, leader_theta: float, dist_back: float = 45.0) -> Point:
    """Compute a single slot reference point behind leader along -heading.

    Args:
        leader_xy: (x, y) of leader
        leader_theta: heading angle (rad)
        dist_back: longitudinal distance behind leader (meters)
    Returns:
        (x_slot, y_slot)
    """
    x, y = leader_xy
    # Unit vector along leader forward direction
    ux = math.cos(leader_theta)
    uy = math.sin(leader_theta)
    # Slot behind leader
    return (x - ux * dist_back, y - uy * dist_back)


def get_multi_slots(
    leader_xy: Point,
    leader_theta: float,
    n_followers: int,
    dist_back: float = 45.0,
    lateral: float = 0.0,
) -> List[Point]:
    """Minimal multi-slot generator.

    Current minimal implementation places all followers at the same
    longitudinal distance behind leader without lateral offsets to keep
    implementation simple and comparable.
    """
    slots: List[Point] = []
    base = get_slot_ref(leader_xy, leader_theta, dist_back)
    if n_followers <= 0:
        return slots
    # No lateral staggering for simplicity; replicate same target.
    for _ in range(n_followers):
        slots.append(base)
    return slots


def get_formation_rate(distances: List[float], window=(40.0, 50.0)) -> float:
    """Compute proportion of distances within formation window.

    Args:
        distances: list of follower-leader distances sampled over time
        window: (min_d, max_d)
    Returns:
        ratio in [0,1]
    """
    if not distances:
        return 0.0
    low, high = window
    ok = sum(1 for d in distances if low <= d <= high)
    return ok / float(len(distances))

