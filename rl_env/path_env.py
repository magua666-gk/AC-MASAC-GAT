import numpy as np
import gym
from gym import spaces
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from components package
from rl_env.components.entities import Constants
from rl_env.components.entity_manager import EntityManager
from rl_env.components.state_manager import StateManager
from rl_env.components.reward_calculator import RewardCalculator
from rl_env.components.renderer import Renderer
from rl_env.components.entities import set_dt, get_dt

class RlGame(gym.Env):
    """Refactored RlGame environment class, implements OpenAI Gym interface"""
    
    def __init__(
        self,
        leader_count=1,
        follower_count=4,
        obstacle_num=1,
        render=False,
        dt=None,
        predefined_positions=None,
        n=None,
        m=None,
        legacy_api=False,
    ):
        """Initialize environment
        
        Args:
            leader_count: Number of leaders
            follower_count: Number of followers
            obstacle_num: Number of obstacles
            render: Whether to render
            dt: Time step, default None (does not modify current value)
            predefined_positions: Predefined position dictionary
            n: Backward-compatible alias for leader_count
            m: Backward-compatible alias for follower_count
            legacy_api: Return legacy ndarray observations/rewards instead of structured dicts
        """
        if n is not None:
            leader_count = n
            legacy_api = True
        if m is not None:
            follower_count = m
            legacy_api = True

        # Save parameters
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_num = obstacle_num
        self.goal_num = 1
        self.Render = render
        self.legacy_api = legacy_api
        self.step_count = 0
        
        # Set time step if provided
        if dt is not None:
            self.set_time_step(dt)
        
        # Game information
        self.game_info = {
            'epsoide': 0,
            'leader_win': 0,
            'follower_win': 0,
            'win': 'Unknown',
        }
        
        # Create components
        self.entity_manager = EntityManager(
            leader_count=leader_count,
            follower_count=follower_count,
            obstacle_count=obstacle_num,
            goal_count=self.goal_num,
            predefined_positions=predefined_positions
        )
        
        self.state_manager = StateManager(self.entity_manager)
        
        reward_weights = {
            'edge': 1.0,
            'collision': 1.0,
            'goal': 1.0,
            'formation': 1.0,
            'speed': 1.0,
        }
        self.reward_calculator = RewardCalculator(self.entity_manager, reward_weights)
        
        # Initialize renderer if needed
        self.renderer = None
        if self.Render:
            self.renderer = Renderer(self.entity_manager, mode='human')

        # Define action space
        low = np.array([-1, -1])
        high = np.array([1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Define observation space
        # Note: Gym interface still uses flattened Box space definition, but we return structured dict observations
        # This avoids modifying Gym interface and encapsulates complexity in Agent/Controller
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.state_manager.get_observation_space_shape(),
            dtype=np.float32
        )
        
        # Other member variables
        self.done = False
        self.team_counter = 0

    def _get_observation(self, legacy_format=None):
        """Return observation in the configured API format."""
        use_legacy = self.legacy_api if legacy_format is None else legacy_format
        if use_legacy:
            return self.state_manager.get_full_state()
        return self.state_manager.get_structured_state()

    def _get_reward(self, legacy_format=None):
        """Return reward in the configured API format."""
        use_legacy = self.legacy_api if legacy_format is None else legacy_format
        if use_legacy:
            return self.reward_calculator.compute_rewards()
        return self.reward_calculator.compute_structured_rewards()
    
    def reset(self):
        """Reset environment
        
        Returns:
            Initial observation (structured dict)
        """
        # Reset entity manager
        self.entity_manager.reset()
        
        # Reset state
        self.done = False
        self.team_counter = 0
        self.step_count = 0
        # Reset trajectories if renderer exists
        if self.renderer:
            self.renderer.reset_trajectories()
            # Reload images to reset entities
            if hasattr(self.renderer, 'graphics') and self.renderer.graphics:
                self.entity_manager.load_images(self.renderer.graphics)
                self.entity_manager.images_loaded = True
                print("Reloaded images to entities after environment reset")
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one environment step update
        
        Args:
            action: Structured action dict {"leader": action_leader, "followers": [action_f1, action_f2, ...]}
                or legacy flattened action array (n_agents, action_dim)
        
        Returns:
            observation: Structured observation dict {"leader": obs_leader, "followers": [obs_f1, obs_f2, ...]}
            reward: Structured reward dict {"leader": reward_leader, "followers": [reward_f1, reward_f2, ...]}
            done: Whether episode is finished
            info: Additional information
        """
        # Check if input is structured format
        if self.done or self.entity_manager.is_episode_done():
            use_legacy_output = self.legacy_api or not (
                isinstance(action, dict) and "leader" in action and "followers" in action
            )
            observation = self._get_observation(legacy_format=use_legacy_output)
            rewards = self._get_reward(legacy_format=use_legacy_output)
            return observation, rewards, True, self.entity_manager.is_hero_win(), self.entity_manager.get_formation_rate(), self.entity_manager.get_agent_distances()

        # Check if input is structured format
        is_structured_action = isinstance(action, dict) and "leader" in action and "followers" in action
        if is_structured_action:
            # Handle structured action
            leader_action = action["leader"]
            follower_actions = action["followers"]
            
            # Ensure numpy arrays
            leader_action = np.array(leader_action, dtype=np.float32)
            follower_actions = [np.array(a, dtype=np.float32) for a in follower_actions]
            
            # Apply actions
            self.entity_manager.apply_actions(leader_action, follower_actions)
            
        else:
            # Legacy flattened input, maintain compatibility
            # Ensure action is numpy array
            action = np.array(action, dtype=np.float32)
            
            # Separate leader and followers actions
            leader_action = action[0]
            follower_actions = action[1:] if len(action) > 1 else []
            
            # Apply actions
            self.entity_manager.apply_actions(leader_action, follower_actions)
        
        # Update environment
        self.entity_manager.update()
        
        # Calculate rewards first, then get observations.
        # Array actions are treated as legacy calls even when the environment was
        # constructed with the newer named parameters.
        use_legacy_output = self.legacy_api or not is_structured_action
        rewards = self._get_reward(legacy_format=use_legacy_output)
        observation = self._get_observation(legacy_format=use_legacy_output)
        
        # Check if episode is done
        done = self.entity_manager.is_episode_done()
        self.done = done
        
        # Get win information
        win = self.entity_manager.is_hero_win()
        
        # Get formation count
        team_counter = self.entity_manager.get_formation_rate()
        
        # Get distance information
        dis = self.entity_manager.get_agent_distances()
        
        # Update step count
        self.step_count += 1
        
        # Return all 6 values
        return observation, rewards, done, win, team_counter, dis
    
    def render(self, mode='human'):
        """Render environment
        
        Args:
            mode: Render mode ('human', 'rgb_array')
            
        Returns:
            RGB array if mode is 'rgb_array'
        """
        try:
            if not self.Render:
                return None
                
            if self.renderer is None:
                try:
                    self.renderer = Renderer(self.entity_manager, mode=mode)
                except Exception as e:
                    print(f"Error creating renderer: {e}")
                    import traceback
                    traceback.print_exc()
                    self.Render = False  # Disable rendering to avoid retry
                    return None
            
            result = self.renderer.render()
            if self.renderer.is_closed():
                self.Render = False
                self.done = True
                self.entity_manager.done = True
                print("Render window closed; episode stopped automatically.")
            return result
        except Exception as e:
            print(f"Error rendering environment: {e}")
            import traceback
            traceback.print_exc()
            self.Render = False  # Disable rendering to avoid retry
            self.done = True
            self.entity_manager.done = True
            return None
    
    def close(self):
        """Close environment"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        self.Render = False
    
    def reconfigure(self, leader_count=None, follower_count=None, obstacle_count=None):
        """Reconfigure environment parameters
        
        Args:
            leader_count: New number of leaders
            follower_count: New number of followers
            obstacle_count: New number of obstacles
        """
        # Keep unspecified parameters
        leader_count = leader_count if leader_count is not None else self.leader_count
        follower_count = follower_count if follower_count is not None else self.follower_count
        obstacle_count = obstacle_count if obstacle_count is not None else self.obstacle_num
        
        # Update environment parameters
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_num = obstacle_count
        
        # Reconfigure entity manager
        self.entity_manager.reconfigure(
            leader_count=leader_count,
            follower_count=follower_count,
            obstacle_count=obstacle_count
        )
        
        # Reset environment
        self.reset()

    def set_time_step(self, dt_value):
        """Set environment time step
        
        Args:
            dt_value: New time step value
        """
        # Use function from entities module to set dt value
        set_dt(dt_value)
        print(f"Environment time step set to: {dt_value}")
        return get_dt()

    def get_formation_state(self):
        """Get detailed formation state information
        
        Returns:
            dict: Dictionary containing detailed status of all UAVs
        """
        leaders_data = []
        followers_data = []
        
        # Get leader states
        for i, leader in enumerate(self.entity_manager.leaders):
            if leader.alive:
                leader_data = {
                    'agent_id': i,
                    'pos_x': float(leader.pos_x),
                    'pos_y': float(leader.pos_y),
                    'speed': float(leader.speed),
                    'heading_angle': float(leader.theta),
                    'target_distance': float(leader.distance_to(self.entity_manager.goals[0]) if self.entity_manager.goals else 0.0)
                }
                leaders_data.append(leader_data)
        
        # Get follower states
        for i, follower in enumerate(self.entity_manager.followers):
            if follower.alive:
                # Calculate distance to nearest leader
                leader_distance = float('inf')
                if self.entity_manager.leaders:
                    leader_distance = min(follower.distance_to(leader) for leader in self.entity_manager.leaders if leader.alive)
                
                # Calculate formation distance error (relative to expected formation distance)
                expected_formation_distance = 50.0  # Expected formation distance
                formation_distance_error = abs(leader_distance - expected_formation_distance)
                
                follower_data = {
                    'agent_id': i,
                    'pos_x': float(follower.pos_x),
                    'pos_y': float(follower.pos_y),
                    'speed': float(follower.speed),
                    'heading_angle': float(follower.theta),
                    'leader_distance': float(leader_distance),
                    'formation_distance_error': float(formation_distance_error)
                }
                followers_data.append(follower_data)
        
        return {
            'step': self.step_count,
            'leaders': leaders_data,
            'followers': followers_data
        }
