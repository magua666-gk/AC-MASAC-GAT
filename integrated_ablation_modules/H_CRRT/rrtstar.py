from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional


Point = Tuple[float, float]


class CollisionChecker:
    """Minimal collision checker using circle obstacles.

    Obstacles are provided as tuples (x, y, R). If obstacle size is unknown,
    a conservative default radius can be passed in by the caller.
    """

    def __init__(self, obstacles: List[Tuple[float, float, float]], margin: float = 0.0):
        self.obstacles = [(x, y, max(0.0, r + margin)) for x, y, r in obstacles]

    def point_free(self, p: Point) -> bool:
        x, y = p
        for ox, oy, r in self.obstacles:
            if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= r * r:
                return False
        return True

    def segment_free(self, p1: Point, p2: Point) -> bool:
        """Check if a segment intersects any obstacle (circle-line distance)."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        for ox, oy, r in self.obstacles:
            # project obstacle center onto the segment
            if denom == 0:
                # degenerate segment
                if (x1 - ox) * (x1 - ox) + (y1 - oy) * (y1 - oy) <= r * r:
                    return False
                continue
            t = ((ox - x1) * dx + (oy - y1) * dy) / denom
            t = max(0.0, min(1.0, t))
            cx = x1 + t * dx
            cy = y1 + t * dy
            if (cx - ox) * (cx - ox) + (cy - oy) * (cy - oy) <= r * r:
                return False
        return True


class RRTStarPlanner:
    """Minimal RRT* planner in 2D continuous space.

    - Uniform sampling with goal bias
    - Euclidean cost, neighbor rewiring by radius
    - Simple stopping condition when node is within step_size to goal
    - Optional path shortcutting
    """

    class _Node:
        __slots__ = ("x", "y", "parent", "cost")

        def __init__(self, x: float, y: float, parent: Optional[int], cost: float):
            self.x = x
            self.y = y
            self.parent = parent
            self.cost = cost

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        obstacles: List[Tuple[float, float, float]],
        step_size: float = 22.0,
        goal_bias: float = 0.12,
        max_nodes: int = 1200,
        obstacle_margin: float = 0.0,
    ):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.cc = CollisionChecker(obstacles, margin=obstacle_margin)
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_nodes = max_nodes
        # typical radius ~ step_size*2 for rewiring
        self.neighbor_radius = step_size * 2.0

    def _sample(self, goal: Point) -> Point:
        if random.random() < self.goal_bias:
            return goal
        return (
            random.uniform(self.xmin, self.xmax),
            random.uniform(self.ymin, self.ymax),
        )

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _dist2(a: Point, b: Point) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _steer(self, from_p: Point, to_p: Point) -> Point:
        d = self._dist(from_p, to_p)
        if d <= self.step_size:
            return to_p
        ux = (to_p[0] - from_p[0]) / d
        uy = (to_p[1] - from_p[1]) / d
        return (from_p[0] + ux * self.step_size, from_p[1] + uy * self.step_size)

    def plan(self, start_xy: Point, goal_xy: Point) -> List[Point]:
        # if start/goal inside obstacle, early return empty
        if not self.cc.point_free(start_xy) or not self.cc.point_free(goal_xy):
            return []
        if self.cc.segment_free(start_xy, goal_xy):
            return [start_xy, goal_xy]

        nodes: List[RRTStarPlanner._Node] = [self._Node(start_xy[0], start_xy[1], None, 0.0)]
        neighbor_radius2 = self.neighbor_radius * self.neighbor_radius

        for _ in range(self.max_nodes):
            sample = self._sample(goal_xy)

            # find nearest
            nearest_idx = 0
            best_d = float("inf")
            for i, nd in enumerate(nodes):
                d = self._dist2((nd.x, nd.y), sample)
                if d < best_d:
                    best_d = d
                    nearest_idx = i

            nearest = nodes[nearest_idx]
            new_pt = self._steer((nearest.x, nearest.y), sample)
            # keep inside bounds
            if not (self.xmin <= new_pt[0] <= self.xmax and self.ymin <= new_pt[1] <= self.ymax):
                continue
            # collision check
            if not self.cc.segment_free((nearest.x, nearest.y), new_pt):
                continue

            # choose parent among neighbors to minimize cost
            best_parent = nearest_idx
            best_cost = nearest.cost + self._dist((nearest.x, nearest.y), new_pt)
            for i, nd in enumerate(nodes):
                if self._dist2((nd.x, nd.y), new_pt) <= neighbor_radius2:
                    if self.cc.segment_free((nd.x, nd.y), new_pt):
                        c = nd.cost + self._dist((nd.x, nd.y), new_pt)
                        if c < best_cost:
                            best_cost = c
                            best_parent = i

            nodes.append(self._Node(new_pt[0], new_pt[1], best_parent, best_cost))
            new_idx = len(nodes) - 1

            # rewire neighbors
            for i, nd in enumerate(nodes[:-1]):
                if self._dist2((nd.x, nd.y), new_pt) <= neighbor_radius2:
                    new_cost = best_cost + self._dist(new_pt, (nd.x, nd.y))
                    if new_cost + 1e-6 < nd.cost and self.cc.segment_free(new_pt, (nd.x, nd.y)):
                        nd.parent = new_idx
                        nd.cost = new_cost

            # check goal proximity
            if self._dist(new_pt, goal_xy) <= self.step_size and self.cc.segment_free(new_pt, goal_xy):
                # add goal node
                goal_parent = new_idx
                goal_cost = nodes[new_idx].cost + self._dist(new_pt, goal_xy)
                nodes.append(self._Node(goal_xy[0], goal_xy[1], goal_parent, goal_cost))
                return self._backtrack_path(nodes, len(nodes) - 1)

        # fallback: connect to the closest node to goal if possible
        closest_idx = min(range(len(nodes)), key=lambda i: self._dist((nodes[i].x, nodes[i].y), goal_xy))
        if self.cc.segment_free((nodes[closest_idx].x, nodes[closest_idx].y), goal_xy):
            nodes.append(self._Node(goal_xy[0], goal_xy[1], closest_idx, nodes[closest_idx].cost + self._dist((nodes[closest_idx].x, nodes[closest_idx].y), goal_xy)))
            return self._backtrack_path(nodes, len(nodes) - 1)
        return []

    def _backtrack_path(self, nodes: List[_Node], goal_idx: int) -> List[Point]:
        path: List[Point] = []
        i = goal_idx
        while i is not None:
            nd = nodes[i]
            path.append((nd.x, nd.y))
            i = nd.parent
        path.reverse()
        return path

    def shortcut(self, path: List[Point], iters: int = 100) -> List[Point]:
        if len(path) <= 2:
            return path
        pts = list(path)

        # Greedy shortcut first: repeatedly keep the farthest collision-free jump.
        greedy: List[Point] = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.cc.segment_free(pts[i], pts[j]):
                j -= 1
            greedy.append(pts[j])
            i = j
        pts = greedy
        
        for _ in range(iters):
            if len(pts) <= 2:
                break
            i = random.randrange(0, len(pts) - 2)
            j = random.randrange(i + 2, len(pts))
            if self.cc.segment_free(pts[i], pts[j]):
                # 检查是否真正缩短路径
                segment_length = self._dist(pts[i], pts[j])
                removed_length = sum(self._dist(pts[k], pts[k+1]) for k in range(i, j))
                if segment_length < removed_length * 0.8:  # 至少20%改善
                    pts = pts[: i + 1] + pts[j:]
        return pts
    
    def _path_length(self, path: List[Point]) -> float:
        """计算路径总长度"""
        if len(path) < 2:
            return 0.0
        return sum(self._dist(path[i], path[i+1]) for i in range(len(path)-1))
