from __future__ import annotations

import math
from typing import List, Tuple, Dict


def _angle_normalize(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class LeaderTracker:
    """Minimal tracker mapping path to environment actions for leader.

    - Pure pursuit like heading control
    - Simple speed tracking towards a reference (use current speed as reference)
    - Output normalized actions per environment scaling (a/0.3, omega/0.6)
    """

    def __init__(self, v_min=10.0, v_max=20.0, lookahead=25.0, a_gain=0.25, w_gain=0.02):
        self.v_min = v_min
        self.v_max = v_max
        self.lookahead = lookahead
        self.a_gain = a_gain
        self.w_gain = w_gain

    def step(self, state: Dict, path_xy: List[Tuple[float, float]], dt: float) -> Tuple[float, float]:
        x = float(state["pos_x"])
        y = float(state["pos_y"])
        theta = float(state["theta"])
        v = float(state["speed"])

        # pick a lookahead target on the path: nearest then forward by lookahead distance
        if not path_xy:
            return 0.0, 0.0

        # find nearest path vertex to current position
        best_d = float("inf")
        best_i = 0
        for i, (px, py) in enumerate(path_xy):
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d = d
                best_i = i

        # march forward along the polyline by lookahead distance
        remain = max(0.0, float(self.lookahead))
        i = best_i
        tx, ty = path_xy[best_i]
        while i < len(path_xy) - 1 and remain > 0.0:
            x1, y1 = path_xy[i]
            x2, y2 = path_xy[i + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len <= 1e-6:
                i += 1
                continue
            if remain <= seg_len:
                ratio = remain / seg_len
                tx = x1 + ratio * (x2 - x1)
                ty = y1 + ratio * (y2 - y1)
                remain = 0.0
                break
            else:
                remain -= seg_len
                i += 1
                tx, ty = x2, y2
        desired_heading = math.atan2(ty - y, tx - x)
        heading_error = _angle_normalize(desired_heading - theta)

        omega_cmd = self.w_gain * heading_error / max(dt, 1e-3)

        # simple speed control: prefer mid of allowed range if moving towards target
        v_ref = min(max(self.v_min, v), self.v_max)
        a_cmd = self.a_gain * (v_ref - v) / max(dt, 1e-3)

        # normalize to environment action
        a_norm = max(-1.0, min(1.0, a_cmd / 0.3))
        phi_norm = max(-1.0, min(1.0, omega_cmd / 0.6))
        return a_norm, phi_norm


class FollowerTracker:
    """Minimal follower tracker aiming at slot reference with speed sync to leader.

    Output normalized actions per environment scaling (a/0.6, omega/1.2)
    """

    def __init__(self, v_min=10.0, v_max=40.0, a_gain=0.28, w_gain=0.015):
        self.v_min = v_min
        self.v_max = v_max
        self.a_gain = a_gain
        self.w_gain = w_gain

    def step(
        self,
        state: Dict,
        slot_xy: Tuple[float, float],
        leader_speed: float,
        dt: float,
    ) -> Tuple[float, float]:
        x = float(state["pos_x"])
        y = float(state["pos_y"])
        theta = float(state["theta"])
        v = float(state["speed"])

        sx, sy = slot_xy
        desired_heading = math.atan2(sy - y, sx - x)
        heading_error = _angle_normalize(desired_heading - theta)
        omega_cmd = self.w_gain * heading_error / max(dt, 1e-3)

        # speed sync to leader (keep within follower range)
        v_ref = max(self.v_min, min(self.v_max, leader_speed))
        a_cmd = self.a_gain * (v_ref - v) / max(dt, 1e-3)

        a_norm = max(-1.0, min(1.0, a_cmd / 0.6))
        phi_norm = max(-1.0, min(1.0, omega_cmd / 1.2))
        return a_norm, phi_norm

