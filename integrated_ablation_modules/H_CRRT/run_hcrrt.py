from __future__ import annotations

import argparse
import math
import os
import random
import time
import sys
from typing import List, Tuple

import numpy as np

# Allow running as script from within H_CRRT directory by adding project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Optional pygame import for event pumping when rendering
try:
    import pygame  # type: ignore

    _PYGAME_AVAILABLE = True
except Exception:
    _PYGAME_AVAILABLE = False

from rl_env.path_env import RlGame

from integrated_ablation_modules.H_CRRT.rrtstar import RRTStarPlanner
from integrated_ablation_modules.H_CRRT.formation import get_multi_slots
from integrated_ablation_modules.H_CRRT.tracking import LeaderTracker, FollowerTracker
from integrated_ablation_modules.H_CRRT.io_utils import save_results, ensure_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except Exception:
        pass


def _collect_state(agent) -> dict:
    return {
        "pos_x": float(agent.pos_x),
        "pos_y": float(agent.pos_y),
        "theta": float(agent.theta),
        "speed": float(agent.speed),
    }


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pump_pygame_events(render_enabled: bool) -> None:
    """Pump pygame events to prevent window from becoming unresponsive.

    Only active if rendering is enabled and pygame is available.
    """
    if not render_enabled:
        return
    if _PYGAME_AVAILABLE:
        try:
            pygame.event.pump()
        except Exception:
            # Do not fail the run if pygame has issues
            pass


def main():
    ap = argparse.ArgumentParser("H-CRRT baseline runner")
    ap.add_argument("--hero_count", type=int, default=1)
    ap.add_argument("--enemy_count", type=int, default=3)
    ap.add_argument("--obstacle_count", type=int, default=3)
    ap.add_argument("--test_episodes", type=int, default=20)
    # default True with explicit disabling flags
    ap.add_argument("--use_formation", dest="use_formation", action="store_true")
    ap.add_argument("--no_use_formation", dest="use_formation", action="store_false")
    ap.set_defaults(use_formation=True)
    ap.add_argument("--replan_horizon", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--render", dest="render", action="store_true")
    ap.add_argument("--no_render", dest="render", action="store_false")
    ap.set_defaults(render=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # Create environment
    env = RlGame(
        leader_count=args.hero_count,
        follower_count=args.enemy_count,
        obstacle_num=args.obstacle_count,
        render=args.render,
    ).unwrapped

    env.set_time_step(args.dt)

    # Extract entities
    leader = env.entity_manager.leaders[0]
    goal = env.entity_manager.goals[0]

    # Build obstacle list (x, y, R). If radius not explicit, use conservative 20.0
    obstacles = []
    default_r = 20.0
    for ob in env.entity_manager.obstacles:
        r = getattr(ob, "radius", default_r)
        obstacles.append((float(ob.pos_x), float(ob.pos_y), float(r)))

    bounds = (100.0, 100.0, 600.0, 500.0)
    planner = RRTStarPlanner(bounds, obstacles, step_size=22.0, goal_bias=0.12, max_nodes=1200)

    # Metrics accumulators
    wins = 0
    rewards_all = []
    steps_all = []
    formation_rates_all = []
    final_dist_all = []
    path_len_all = []
    energy_all = []
    avg_abs_omega_all = []

    for ep in range(args.test_episodes):
        obs = env.reset()
        # re-pull references
        leader = env.entity_manager.leaders[0]
        goal = env.entity_manager.goals[0]
        # Render initial frame if enabled
        if args.render:
            try:
                env.render()
            except Exception:
                pass

        start = (float(leader.pos_x), float(leader.pos_y))
        target = (float(goal.pos_x), float(goal.pos_y))

        path = planner.plan(start, target)
        if not path:
            path = [start, target]
        else:
            path = planner.shortcut(path)

        leader_tracker = LeaderTracker()
        follower_tracker = FollowerTracker()

        done = False
        total_reward = 0.0
        total_steps = 0
        total_path_len = 0.0
        total_energy = 0.0
        omega_abs_sum = 0.0
        omega_count = 0
        formation_window_hits = 0
        formation_samples = 0
        win = False

        last_leader_pos = (float(leader.pos_x), float(leader.pos_y))

        while not done and total_steps < 1000:
            # Pump render events when visualization is enabled
            _pump_pygame_events(args.render)
            # Leader action
            st_leader = _collect_state(leader)
            aL, phiL = leader_tracker.step(st_leader, path, args.dt)

            actions = {"leader": [aL, phiL], "followers": []}

            # Followers actions
            if args.use_formation and env.entity_manager.followers:
                slots = get_multi_slots(
                    (float(leader.pos_x), float(leader.pos_y)), float(leader.theta), len(env.entity_manager.followers)
                )
                for f, slot in zip(env.entity_manager.followers, slots):
                    st_f = _collect_state(f)
                    aF, phiF = follower_tracker.step(st_f, slot, float(leader.speed), args.dt)
                    actions["followers"].append([aF, phiF])

                # formation metric
                for f in env.entity_manager.followers:
                    d = _euclid((f.pos_x, f.pos_y), (leader.pos_x, leader.pos_y))
                    formation_samples += 1
                    if 40.0 <= d <= 50.0:
                        formation_window_hits += 1

            # step
            obs, reward, done, win, team_counter, dis = env.step(actions)

            # accumulate metrics
            total_steps += 1
            # reward can be dict or list; accumulate robustly
            r_sum = 0.0
            if isinstance(reward, dict):
                r_sum += float(reward.get("leader", 0.0))
                for rv in reward.get("followers", []):
                    try:
                        r_sum += float(rv)
                    except Exception:
                        pass
            elif isinstance(reward, (list, np.ndarray)):
                r_sum += float(np.sum(reward))
            elif isinstance(reward, (int, float, np.number)):
                r_sum += float(reward)
            total_reward += r_sum

            # path length increment via integral of speed
            total_path_len += float(st_leader["speed"]) * float(args.dt)

            # energy integral with de-normalized commands and dt
            # leader
            a_cmd_L = float(aL) * 0.3
            omega_cmd_L = float(phiL) * 0.6
            total_energy += (abs(a_cmd_L) + abs(omega_cmd_L)) * float(args.dt)
            omega_abs_sum += abs(omega_cmd_L)
            omega_count += 1
            # followers
            for a_norm, phi_norm in actions["followers"]:
                a_cmd_F = float(a_norm) * 0.6
                omega_cmd_F = float(phi_norm) * 1.2
                total_energy += (abs(a_cmd_F) + abs(omega_cmd_F)) * float(args.dt)

            # simple periodic replan if enabled
            if args.replan_horizon > 0 and (total_steps % args.replan_horizon == 0):
                start = (float(leader.pos_x), float(leader.pos_y))
                path_new = planner.plan(start, target)
                if path_new:
                    path = planner.shortcut(path_new)

            # render this step if enabled (align with main_SAC_curriculum behavior)
            if args.render:
                try:
                    env.render()
                except Exception:
                    pass

        wins += 1 if win else 0
        rewards_all.append(total_reward)
        steps_all.append(total_steps)
        final_dist_all.append(float(dis) if isinstance(dis, (float, int, np.number)) else 0.0)
        if args.use_formation and formation_samples > 0:
            formation_rates_all.append(formation_window_hits / float(formation_samples))
        else:
            formation_rates_all.append(float("nan"))
        path_len_all.append(total_path_len)
        energy_all.append(total_energy)
        avg_abs_omega_all.append(omega_abs_sum / float(max(1, omega_count)))

    # aggregate
    success_rate = wins / float(args.test_episodes)
    results = {
        "test_episodes": args.test_episodes,
        "success_rate": success_rate,
        "rewards": {"mean": float(np.mean(rewards_all)), "std": float(np.std(rewards_all))},
        "steps": {"mean": float(np.mean(steps_all)), "std": float(np.std(steps_all))},
        "formation_rates": {
            "mean": float(np.nanmean(formation_rates_all)) if any(not np.isnan(x) for x in formation_rates_all) else 0.0,
            "std": float(np.nanstd(formation_rates_all)) if any(not np.isnan(x) for x in formation_rates_all) else 0.0,
        },
        "distances": {"mean": float(np.mean(final_dist_all)), "std": float(np.std(final_dist_all))},
        "path_length": {"mean": float(np.mean(path_len_all)), "std": float(np.std(path_len_all))},
        "energy": {"mean": float(np.mean(energy_all)), "std": float(np.std(energy_all))},
        "avg_abs_omega": {"mean": float(np.mean(avg_abs_omega_all)), "std": float(np.std(avg_abs_omega_all))},
        "test_config": {
            "hero_count": args.hero_count,
            "enemy_count": args.enemy_count,
            "obstacle_count": args.obstacle_count,
            "uav_speed": None,
        },
        "timestamp": time.time(),
    }

    # calculate and print five metrics
    five_metrics = calculate_five_metrics(results)
    print_five_metrics(five_metrics)
    
    # add five metrics to results
    results["five_metrics"] = five_metrics
    
    # save
    base = os.path.join(args.results_dir, "test_results")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(base, f"test_{ts}_h{args.hero_count}_e{args.enemy_count}_o{args.obstacle_count}"))
    save_results(results, out_dir)
    print(f"H-CRRT results saved to: {out_dir}")


def calculate_five_metrics(results):
    """计算并格式化五个性能指标"""
    
    # 1. 任务完成率(MCR)
    mcr = results["success_rate"]
    
    # 2. 编队保持率(FKR)
    fkr_data = results["formation_rates"]
    fkr_mean = fkr_data["mean"] if not np.isnan(fkr_data["mean"]) else 0.0
    fkr_std = fkr_data["std"] if not np.isnan(fkr_data["std"]) else 0.0
    
    # 3. 成功率加权探索时间(SET)
    set_score = mcr * results["steps"]["mean"]
    
    # 4. 飞行轨迹(J_S) 
    js_mean = results["path_length"]["mean"]
    js_std = results["path_length"]["std"]
    
    # 5. 能量消耗(J_C)
    jc_mean = results["energy"]["mean"] 
    jc_std = results["energy"]["std"]
    
    return {
        "mcr": mcr,
        "fkr": {"mean": fkr_mean, "std": fkr_std},
        "set": set_score,
        "js": {"mean": js_mean, "std": js_std},
        "jc": {"mean": jc_mean, "std": jc_std}
    }

def print_five_metrics(metrics):
    """按照标准格式输出五指标"""
    print("\n五指标性能评估:")
    print(f"1. 任务完成率(MCR): {metrics['mcr']:.2f}")
    print(f"2. 编队保持率(FKR): {metrics['fkr']['mean']:.2f}±{metrics['fkr']['std']:.2f}")
    print(f"3. 成功率加权探索时间(SET): {metrics['set']:.2f} (SR: {metrics['mcr']:.2f} × 平均时间: {metrics['set']/max(metrics['mcr'], 0.01):.2f})")
    print(f"4. 飞行轨迹(J_S): {metrics['js']['mean']:.2f}±{metrics['js']['std']:.2f}")
    print(f"5. 能量消耗(J_C): {metrics['jc']['mean']:.2f}±{metrics['jc']['std']:.2f}")


if __name__ == "__main__":
    main()

