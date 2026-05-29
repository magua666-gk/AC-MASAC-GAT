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
    """Generate follower slots behind the leader with optional lateral staggering."""
    slots: List[Point] = []
    if n_followers <= 0:
        return slots

    x, y = leader_xy
    ux = math.cos(leader_theta)
    uy = math.sin(leader_theta)
    # Perpendicular unit vector.
    px = -uy
    py = ux

    if lateral <= 0.0:
        base = get_slot_ref(leader_xy, leader_theta, dist_back)
        return [base for _ in range(n_followers)]

    center = (n_followers - 1) / 2.0
    for idx in range(n_followers):
        offset = (idx - center) * lateral
        slots.append((x - ux * dist_back + px * offset, y - uy * dist_back + py * offset))
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
