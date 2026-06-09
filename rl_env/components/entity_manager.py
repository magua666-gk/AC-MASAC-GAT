import random
import math
import pygame
import numpy as np
from rl_env.components.entities import LeaderAgent, FollowerAgent, Obstacle, Goal, Constants, get_dt
from rl_env.components.position_generator import PositionGenerator

class EntityManager:
    """Manage all entities in the environment"""
    
    def __init__(self, leader_count=1, follower_count=1, obstacle_count=1, goal_count=1, predefined_positions=None):
        """Initialize entity manager
        
        Args:
            leader_count: Number of leaders
            follower_count: Number of followers
            obstacle_count: Number of obstacles
            goal_count: Number of goals
            predefined_positions: Predefined position dictionary, format:
                {
                    'leaders': [(x1,y1), (x2,y2), ...],
                    'followers': [(x1,y1), (x2,y2), ...],
                    'obstacles': [(x1,y1), (x2,y2), ...],
                    'goals': [(x1,y1), (x2,y2), ...]
                }
        """
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_count = obstacle_count
        self.goal_count = goal_count
        self.total_agents = leader_count + follower_count
        
        # Save predefined positions
        self.predefined_positions = predefined_positions
        
        # Create entity containers
        self.leaders = []
        self.followers = []
        self.obstacles = []
        self.goals = []
        
        # State flags
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
        self.images_loaded = False 
        # Leader safety shield parameters
        # 只对主无人机动作做安全兜底，不改变观测维度、奖励函数和终止条件
        self.enable_leader_safety_shield = False
        # 撞障碍物的真实判定阈值是 20；测试默认采用灵敏度实验选定的折中参数
        self.leader_safe_distance = 45.0
        # 进入预警区后才启动动作筛选，避免一直干预正常策略
        self.leader_warning_distance = 150.0
        # 向前预测多少步，越大越保守，但会稍微影响路径效率
        self.leader_safety_horizon = 10
        self.reset_leader_safety_stats()
        
        # Create entities
        self._create_entities()
        print("<<<<< EntityManager VERSION XYZ RUNNING >>>>>") 
    
    def _create_entities(self):
        """Create all entities"""
        # ---- BEGIN DEBUG PRINT ----
        print(f"[EM._create_entities] Creating entities with: leader_count={self.leader_count}, follower_count={self.follower_count}")
        # ---- END DEBUG PRINT ----
        
        # Clear existing entities
        self.leaders = []
        self.followers = []
        self.obstacles = []
        self.goals = []
        
        # Create leaders
        for i in range(self.leader_count):
            leader = LeaderAgent()
            self.leaders.append(leader)
        
        # Create followers
        for i in range(self.follower_count):
            follower = FollowerAgent()
            self.followers.append(follower)
        
        # Create obstacles
        for i in range(self.obstacle_count):
            obstacle = Obstacle()
            self.obstacles.append(obstacle)
        
        # Create goals
        for i in range(self.goal_count):
            goal = Goal()
            self.goals.append(goal)


        # Note: Image loading will be handled by Renderer, after environment initialization
        # Don't load images here to avoid image_dict undefined error
            
        # Randomly place entities
        self._randomize_positions()
    
    def _randomize_positions(self):
        """Randomly place entity initial positions"""
        # Use PositionGenerator to generate all positions
        positions = PositionGenerator.generate_all_positions(
            leader_count=self.leader_count,
            follower_count=self.follower_count,
            obstacle_count=self.obstacle_count,
            goal_count=self.goal_count
        )
        
        # If there are predefined positions, override corresponding positions
        if self.predefined_positions is not None:
            for entity_type, pos_list in self.predefined_positions.items():
                if entity_type in positions and pos_list:
                    positions[entity_type] = pos_list
        
        # Set leader positions
        for i, leader in enumerate(self.leaders):
            if i < len(positions['leaders']):
                leader.set_position(*positions['leaders'][i])
        
        # Set follower positions
        for i, follower in enumerate(self.followers):
            if i < len(positions['followers']):
                follower.set_position(*positions['followers'][i])
        
        # Set obstacle positions
        for i, obstacle in enumerate(self.obstacles):
            if i < len(positions['obstacles']):
                obstacle.set_position(*positions['obstacles'][i])
        
        # Set goal positions
        for i, goal in enumerate(self.goals):
            if i < len(positions['goals']):
                goal.set_position(*positions['goals'][i])
    
    def apply_actions(self, leader_action, follower_actions):
        """Apply structured actions to entities
        
        Args:
            leader_action: Leader action array
            follower_actions: Followers action list
        """
        # Apply Leader action with safety shield
        if len(self.leaders) > 0:
            leader = self.leaders[0]
    
            if self.enable_leader_safety_shield:
                raw_leader_action = np.asarray(leader_action, dtype=np.float32).reshape(-1)
                if raw_leader_action.size < 2:
                    raw_leader_action = np.zeros(2, dtype=np.float32)
                raw_leader_action = np.clip(raw_leader_action[:2], -1.0, 1.0)
                leader_action = self._leader_action_safety_filter(leader, raw_leader_action)
                action_delta = float(np.linalg.norm(np.asarray(leader_action, dtype=np.float32)[:2] - raw_leader_action))
                self.leader_safety_filter_calls += 1
                self.leader_safety_action_delta_sum += action_delta
                self.leader_safety_action_delta_values.append(action_delta)
                if action_delta > 1e-6:
                    self.leader_safety_interventions += 1
                    self.leader_safety_intervention_delta_sum += action_delta
    
            self.leaders[0].apply_action(leader_action)
        
        # Apply Followers actions
        for i, action in enumerate(follower_actions):
            if i < len(self.followers):
                self.followers[i].apply_action(action)
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


    def reset_leader_safety_stats(self):
        """Reset accumulated Leader safety shield diagnostics."""
        self.leader_safety_filter_calls = 0
        self.leader_safety_interventions = 0
        self.leader_safety_action_delta_sum = 0.0
        self.leader_safety_intervention_delta_sum = 0.0
        self.leader_safety_action_delta_values = []


    def get_leader_safety_stats(self):
        """Return accumulated Leader safety shield diagnostics."""
        calls = int(self.leader_safety_filter_calls)
        interventions = int(self.leader_safety_interventions)
        return {
            "filter_calls": calls,
            "interventions": interventions,
            "intervention_rate": float(interventions / calls) if calls else 0.0,
            "mean_action_delta": float(self.leader_safety_action_delta_sum / calls) if calls else 0.0,
            "mean_intervention_action_delta": float(self.leader_safety_intervention_delta_sum / interventions) if interventions else 0.0,
            "action_delta_values": [float(value) for value in self.leader_safety_action_delta_values],
        }
    
    
    def _predict_leader_trajectory(self, leader, action, horizon=None):
        """
        根据当前Leader状态和候选动作，预测未来若干步轨迹。
        注意：这里只做安全筛选预测，不会真正改变环境状态。
        """
        if horizon is None:
            horizon = self.leader_safety_horizon
    
        dt = get_dt()
    
        # 当前状态
        x = float(leader.pos_x)
        y = float(leader.pos_y)
        speed = float(leader.speed)
        theta = float(leader.theta)
    
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 2:
            action = np.zeros(2, dtype=np.float32)
    
        a = float(np.clip(action[0], -1.0, 1.0))
        phi = float(np.clip(action[1], -1.0, 1.0))
    
        traj = []
    
        for _ in range(horizon):
            # 与 LeaderAgent.apply_action 保持一致
            speed = speed + 0.3 * a * dt
            speed = float(np.clip(speed, 10.0, 20.0))
    
            theta = theta + 0.6 * phi * dt
            theta = theta % (2.0 * math.pi)
    
            # 与 Agent.update 修正后的运动模型保持一致
            x = x + speed * math.cos(theta) * dt
            y = y + speed * math.sin(theta) * dt
    
            # 与 Agent.update 中的边界裁剪保持一致
            x = float(np.clip(x, Constants.AREA_X, Constants.AREA_WITH))
            y = float(np.clip(y, Constants.AREA_Y, Constants.AREA_HEIGHT))
    
            traj.append((x, y, speed, theta))
    
        return traj
    
    
    def _min_distance_to_obstacles_on_traj(self, traj):
        """计算预测轨迹到所有障碍物的最小距离。"""
        if not self.obstacles:
            return float("inf")
    
        min_dist = float("inf")
    
        for x, y, _, _ in traj:
            for obstacle in self.obstacles:
                d = math.hypot(x - obstacle.pos_x, y - obstacle.pos_y)
                if d < min_dist:
                    min_dist = d
    
        return min_dist
    
    
    def _leader_is_heading_to_obstacle(self, leader, obstacle):
        """
        判断Leader是否正在朝障碍物方向运动。
        用于减少安全层对正常动作的过度干预。
        """
        dx = obstacle.pos_x - leader.pos_x
        dy = obstacle.pos_y - leader.pos_y
    
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
    
        heading_x = math.cos(leader.theta)
        heading_y = math.sin(leader.theta)
    
        # dot > 0 表示障碍物大致在机头前方
        dot = (heading_x * dx + heading_y * dy) / dist
    
        # 夹角小于约 72 度时，认为存在朝向风险
        return dot > 0.3
    
    
    def _nearest_obstacle(self, leader):
        """返回距离Leader最近的障碍物及距离。"""
        if not self.obstacles:
            return None, float("inf")
    
        nearest = None
        min_dist = float("inf")
    
        for obstacle in self.obstacles:
            d = leader.distance_to(obstacle)
            if d < min_dist:
                min_dist = d
                nearest = obstacle
    
        return nearest, min_dist
    
    
    def _leader_action_safety_filter(self, leader, raw_action):
        """
        Leader动作安全屏蔽层。
    
        逻辑：
        1. 如果附近没有障碍物，直接返回Actor原动作；
        2. 如果预测未来几步不会进入危险区，也返回原动作；
        3. 如果原动作有碰撞风险，则从候选动作中选择一个更安全、且尽量接近原动作的动作。
        """
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    
        if raw_action.size < 2:
            raw_action = np.zeros(2, dtype=np.float32)
    
        raw_action = np.clip(raw_action[:2], -1.0, 1.0)
    
        if not leader.is_alive() or not self.obstacles:
            return raw_action
    
        nearest_obstacle, nearest_dist = self._nearest_obstacle(leader)
    
        # 障碍物距离较远时，不干预策略动作
        if nearest_obstacle is None or nearest_dist > self.leader_warning_distance:
            return raw_action
    
        # 如果障碍物不在前方，且当前距离仍比较安全，则不干预
        if nearest_dist > self.leader_safe_distance and not self._leader_is_heading_to_obstacle(leader, nearest_obstacle):
            return raw_action
    
        # 先检查原动作未来轨迹是否安全
        raw_traj = self._predict_leader_trajectory(leader, raw_action)
        raw_min_dist = self._min_distance_to_obstacles_on_traj(raw_traj)
    
        if raw_min_dist >= self.leader_safe_distance:
            return raw_action
    
        # 原动作有风险，构造候选动作集
        # 第一维：加速度，优先减速
        # 第二维：角速度，尝试左右转向
        candidate_actions = []
    
        accel_candidates = [-1.0, -0.5, 0.0]
        turn_candidates = [-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0]
    
        for a in accel_candidates:
            for phi in turn_candidates:
                candidate_actions.append(np.array([a, phi], dtype=np.float32))
    
        # 同时保留原动作的减速版本，避免动作变化过猛
        candidate_actions.append(np.array([-1.0, raw_action[1]], dtype=np.float32))
        candidate_actions.append(np.array([-0.5, raw_action[1]], dtype=np.float32))
    
        best_action = raw_action.copy()
        best_score = -float("inf")
    
        # 如果有目标点，安全动作也尽量保持向目标前进
        goal = self.goals[0] if self.goals else None
    
        for cand in candidate_actions:
            traj = self._predict_leader_trajectory(leader, cand)
            min_obs_dist = self._min_distance_to_obstacles_on_traj(traj)
    
            # 预测终点
            end_x, end_y, _, _ = traj[-1]
    
            # 安全性得分：距离越大越好
            safety_score = min_obs_dist
    
            # 原动作接近性：避免安全层让动作跳变太剧烈
            action_change_penalty = 3.0 * float(np.linalg.norm(cand - raw_action))
    
            # 目标推进得分：尽量不要因为避障完全背离目标
            goal_progress_score = 0.0
            if goal is not None:
                current_goal_dist = math.hypot(leader.pos_x - goal.pos_x, leader.pos_y - goal.pos_y)
                next_goal_dist = math.hypot(end_x - goal.pos_x, end_y - goal.pos_y)
                goal_progress_score = 0.05 * (current_goal_dist - next_goal_dist)
    
            # 硬安全优先：低于安全距离的候选动作强烈惩罚
            unsafe_penalty = 0.0
            if min_obs_dist < self.leader_safe_distance:
                unsafe_penalty = 100.0 * (self.leader_safe_distance - min_obs_dist)
    
            score = safety_score + goal_progress_score - action_change_penalty - unsafe_penalty
    
            if score > best_score:
                best_score = score
                best_action = cand
    
        return np.clip(best_action, -1.0, 1.0)
    
    def update(self):
        """Update all entity states"""
        # Update leaders
        for leader in self.leaders:
            leader.update()
            # Rotate image to match heading
            leader.rotate()
        
        # Update followers
        for follower in self.followers:
            follower.update()
            # Rotate image to match heading
            follower.rotate()
        
        # Check collisions and goal achievement
        self._check_collisions()
        self._check_goals()
        
        # Check formation status
        self._check_formation()
        
        # Update counters
        self.time_counter += 1
    
    def _check_collisions(self):
        """Check collisions"""
        for leader in self.leaders:
            if not leader.is_alive():
                continue
                
            # Check collisions with obstacles
            for obstacle in self.obstacles:
                if leader.distance_to(obstacle) < 20.0:
                    leader.kill(won=False)
                    self.done = True
                    return
            
            # Boundary collision detection removed
            # Position constraints implemented by np.clip in Agent.update() method
    
    def _check_goals(self):
        """Check goal achievement"""
        for leader in self.leaders:
            if not leader.is_alive():
                continue
                
            for goal in self.goals:
                if leader.distance_to(goal) < 40.0:
                    leader.kill(won=True)
                    self.done = True
                    return
    
    def _check_formation(self):
        """Check formation maintenance status"""
        if not self.leaders or not self.followers:
            return
        
        # Check if all followers maintain formation with leader
        all_in_formation = True
        leader = self.leaders[0]  # Assume first leader is the leader
        
        # Only keep distance check, remove position relationship condition
        for follower in self.followers:
            # Calculate follower to leader distance
            distance = follower.distance_to(leader)
            
            # Formation condition: only check distance < 50
            if distance >= 50:
                all_in_formation = False
                break
        
        # Only increase counter when all followers meet the condition
        if all_in_formation:
            self.team_counter += 1
    
    def reset(self):
        """Reset entity manager state"""
        # Reset entities
        self._create_entities()
        
        # Reset state
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
        self.images_loaded = False  # Reset image loading status
    
    def is_episode_done(self):
        """Check if episode is done
        
        Returns:
            Whether done
        """
        return self.done
    
    def is_hero_win(self):
        """Check if leader wins
        
        Returns:
            Whether wins
        """
        if not self.leaders:
            return False
        
        return any(leader.has_won for leader in self.leaders)
    
    def get_formation_rate(self):
        """Get formation maintenance rate
        
        Returns:
            Formation maintenance rate
        """
        if self.time_counter == 0:
            return 0.0
        
        return self.team_counter / self.time_counter
    
    def get_agent_distances(self):
        """Get distance matrix between agents
        
        Returns:
            Distance dictionary, keys are agent pair names (e.g., "leader_to_goal"), values are corresponding distances
        """
        distances = {}
        
        # Leader to goal distance
        if self.leaders and self.goals:
            leader = self.leaders[0]
            goal = self.goals[0]
            distances['leader_to_goal'] = leader.distance_to(goal)
        
        # Leader to follower distance
        if self.leaders and self.followers:
            leader = self.leaders[0]
            for i, follower in enumerate(self.followers):
                distances[f"leader_to_follower_{i}"] = leader.distance_to(follower)
        
        # Leader to obstacle minimum distance
        if self.leaders and self.obstacles:
            leader = self.leaders[0]
            min_obstacle_dist = min([leader.distance_to(obs) for obs in self.obstacles], default=float('inf'))
            distances["leader_to_obstacle"] = min_obstacle_dist
        
        return distances
    
    def reconfigure(self, leader_count, follower_count, obstacle_count, goal_count=1):
        """Reconfigure entity counts
        
        Args:
            leader_count: New leader count
            follower_count: New follower count
            obstacle_count: New obstacle count
            goal_count: New goal count
        """
        # Update counts
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_count = obstacle_count
        self.goal_count = goal_count
        self.total_agents = leader_count + follower_count
        
        # Recreate entities
        self._create_entities()
        
        # Reset state
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
    
    def render(self, screen):
        """Render all entities
        
        Args:
            screen: pygame screen object
        """
        # Render obstacles
        for obstacle in self.obstacles:
            obstacle.render(screen)
        
        # Render goals
        for goal in self.goals:
            goal.render(screen)
        
        # Render followers
        for follower in self.followers:
            follower.render(screen)
        
        # Render leaders
        for leader in self.leaders:
            leader.render(screen)
    
    def load_images(self, image_dict):
        """Load entity images
        
        Args:
            image_dict: Image dictionary, format: {"leader": img, "follower": img, "obstacle": img, "goal": img}
        """
        # Load leader images
        if "leader" in image_dict and self.leaders:
            for leader in self.leaders:
                leader.load_image(image_dict["leader"], (30, 30))
        
        # Load follower images
        if "follower" in image_dict and self.followers:
            for follower in self.followers:
                follower.load_image(image_dict["follower"], (30, 30))
        
        # Load obstacle images
        if "obstacle" in image_dict and self.obstacles:
            for obstacle in self.obstacles:
                obstacle.load_image(image_dict["obstacle"], (40, 40))
        
        # Load goal images
        if "goal" in image_dict and self.goals:
            for goal in self.goals:
                goal.load_image(image_dict["goal"], (30, 30)) 
