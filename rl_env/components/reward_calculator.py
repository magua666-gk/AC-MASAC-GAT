import numpy as np
import math
from rl_env.components.entities import Constants

class RewardCalculator:
    """Calculate environment rewards"""
    
    def __init__(self, entity_manager, weights=None):
        """Initialize reward calculator
        
        Args:
            entity_manager: Entity manager
            weights: Weights for different reward components
        """
        self.entity_manager = entity_manager
        
        # Set default weights
        default_weights = {
            'edge': 1.0,
            'collision': 1.0,
            'goal': 1.0,
            'formation': 1.0,
            'speed': 1.0,
        }
        
        self.weights = weights if weights is not None else default_weights
    
    def compute_rewards(self):
        """Compute rewards for all agents
        
        Returns:
            Reward array, one reward value per agent
        """
        rewards = np.zeros(self.entity_manager.total_agents)
        
        for i, leader in enumerate(self.entity_manager.leaders):
            leader_reward = self._compute_leader_reward(leader)
            rewards[i] = leader_reward
        
        for i, follower in enumerate(self.entity_manager.followers):
            follower_reward = self._compute_follower_reward(follower)
            rewards[i + len(self.entity_manager.leaders)] = follower_reward
        
        return rewards
    
    def _compute_leader_reward(self, leader):
        """Compute leader reward
        
        Args:
            leader: Leader
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        edge_reward = 0.0
        if leader.is_near_boundary():
            edge_reward = -1.0
        
        obstacle_reward = 0.0
        for obstacle in self.entity_manager.obstacles:
            distance = leader.distance_to(obstacle)
            if distance < 20.0:
                obstacle_reward = -500.0
                break
            elif distance < 40.0:
                obstacle_reward = -2.0
        
        goal_reward = 0.0
        if self.entity_manager.goals:
            goal = self.entity_manager.goals[0]
            distance = leader.distance_to(goal)
            if distance < 40.0:
                goal_reward = 1000.0
                if not hasattr(self, '_goal_reached_printed') or not self._goal_reached_printed:
                    print(f"\n********************************************")
                    print(f"* Agent successfully reached goal! Distance: {distance:.2f} *")
                    print(f"********************************************\n")
                    self._goal_reached_printed = True
            else:
                goal_reward = -0.001 * distance
                self._goal_reached_printed = False
        
        formation_reward = 0.0
        speed_reward = 0.0
        position_reward = 0.0
        
        if self.entity_manager.followers:
            followers_distances = []
            for follower in self.entity_manager.followers:
                distance = leader.distance_to(follower)
                followers_distances.append(distance)
            
            # 1. Formation maintenance reward
            # Check if all followers are within 50 units
            all_in_formation = True
            for distance in followers_distances:
                if distance >= 50:
                    all_in_formation = False
                    break
            
            if all_in_formation:
                formation_reward = 0  # All followers in ideal distance, no penalty
            else:
                # Use farthest follower distance as penalty
                max_distance = max(followers_distances) if followers_distances else 0
                formation_reward = -0.001 * max_distance  # Greater distance, greater penalty
            
            # 2. Speed matching reward (with formation prerequisite)
            for follower in self.entity_manager.followers:
                # Check formation condition with current follower (ideal formation distance is 50)
                is_in_formation_with_current_follower = leader.distance_to(follower) < 50 
                
                if is_in_formation_with_current_follower:  # Only consider speed matching when in formation
                    speed_diff = abs(leader.speed - follower.speed)
                    if speed_diff < 1.0:
                        speed_reward += 1.0  # Reward if speed difference < 1
            
            # 3. Position relationship reward
            if self.entity_manager.goals:
                goal = self.entity_manager.goals[0]
                leader_to_goal = leader.distance_to(goal)
                
                # Check if leader is closer to goal than all followers
                all_correct_position = True
                for follower in self.entity_manager.followers:
                    follower_to_goal = follower.distance_to(goal)
                    if leader_to_goal >= follower_to_goal:  # Leader is not closer to goal
                        all_correct_position = False
                        break
                
                if all_correct_position:
                    position_reward = 0.1  # Correct position: +0.1
                else:
                    position_reward = -0.1  # Incorrect position: -0.1
        
        # Combined reward
        reward = (
            edge_reward * self.weights['edge'] +
            obstacle_reward * self.weights['collision'] +
            goal_reward * self.weights['goal'] +
            formation_reward * self.weights['formation'] +
            speed_reward * self.weights['speed']  # Position reward with default weight 1.0
        )
        
        return reward
    
    def _compute_follower_reward(self, follower):
        """Compute follower reward
        
        Args:
            follower: Follower
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # If agent is dead, return zero reward
        if not follower.is_alive():
            return reward
        
        # Boundary reward - removed
        
        # Formation reward - new modified logic
        formation_reward = 0.0
        
        # Follower's main task is to maintain formation with leader
        if self.entity_manager.leaders:
            leader = self.entity_manager.leaders[0]  # Follow first leader
            distance = follower.distance_to(leader)
            
            # 1. Formation maintenance reward
            position_correct = False
            if self.entity_manager.goals:
                goal = self.entity_manager.goals[0]
                leader_to_goal = leader.distance_to(goal)
                follower_to_goal = follower.distance_to(goal)
                position_correct = leader_to_goal < follower_to_goal  # Leader is closer to goal
            
            if 40 < distance < 50:
                formation_reward = 10  # Within ideal distance and leader closer to goal, no penalty
            elif 0 < distance < 40:  # Ideal range [20, 40)
                formation_reward = 100.0
            else:
                formation_reward = -0.01 * distance  # Greater distance, greater penalty
                
            # New: Follower speed matching reward
            speed_reward_follower = 0.0
            # leader, distance, position_correct variables are reused from earlier formation_reward calculation
            speed_diff_to_leader = abs(leader.speed - follower.speed)
            if 0 < distance < 80 and speed_diff_to_leader < 1.0:
                speed_reward_follower = 100.0
        
        # Combined reward
        reward = (
            formation_reward * self.weights['formation'] + 
            (speed_reward_follower if 'speed_reward_follower' in locals() else 0.0)  # Add speed reward (if defined)
        )
        
        return reward
    
    def compute_shaped_rewards(self):
        """Compute shaped rewards for curriculum learning
        
        Returns:
            Shaped reward array
        """
        basic_rewards = self.compute_rewards()
        
        # Adjust rewards based on current task difficulty
        # Can adjust rewards based on current task difficulty, currently returning basic rewards
        # Can be extended in the future based on actual needs
        
        return basic_rewards
    
    def compute_team_reward(self):
        """Compute overall team reward
        
        Returns:
            Team reward value
        """
        individual_rewards = self.compute_rewards()
        
        # Simple average team reward
        return np.mean(individual_rewards)
    
    def update_weights(self, new_weights):
        """Update reward weights
        
        Args:
            new_weights: New weight dictionary
        """
        self.weights.update(new_weights)
    
    def compute_structured_rewards(self):
        """Compute structured rewards
        
        Returns:
            rewards: Structured reward dictionary {"leader": reward_leader, "followers": [reward_f1, reward_f2, ...]}
        """
        leader_reward = 0.0
        # Calculate Leader reward (assume Leader is the first leader)
        if self.entity_manager.leaders:
            leader = self.entity_manager.leaders[0]
            leader_reward = self._compute_leader_reward(leader)
        
        # Calculate Followers rewards
        follower_rewards = []
        for follower in self.entity_manager.followers:
            follower_reward = self._compute_follower_reward(follower)
            follower_rewards.append(follower_reward)
        
        # Return structured rewards
        return {
            "leader": leader_reward,
            "followers": follower_rewards
        } 