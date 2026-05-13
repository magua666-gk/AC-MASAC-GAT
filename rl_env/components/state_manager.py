import numpy as np

class StateManager:
    """Manage environment state representation"""
    
    def __init__(self, entity_manager):
        """Initialize state manager
        
        Args:
            entity_manager: Entity manager
        """
        self.entity_manager = entity_manager
    
    def get_full_state(self):
        """Get complete state representation
        
        Returns:
            Complete state array containing all agent states
        """
        states = []
        
        # Add leader states
        for leader in self.entity_manager.leaders:
            leader_state = self._get_agent_state(leader)
            states.append(leader_state)
        
        # Add follower states
        for follower in self.entity_manager.followers:
            follower_state = self._get_agent_state(follower)
            states.append(follower_state)
        
        # Convert states to NumPy array
        return np.array(states, dtype=np.float32)
    
    def _get_min_obstacle_distance(self, agent, obstacles):
        if not obstacles:
            return float('inf')
        min_dist = float('inf')
        for obs in obstacles:
            dist = agent.distance_to(obs)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def get_structured_state(self):
        """Get structured state representation
        
        Returns a dictionary containing structured observations for Leader and Followers
        
        Returns:
            Structured state dictionary {"leader": leader_state, "followers": followers_states}
        """
        # Get Leader (first leader) state
        if self.entity_manager.leaders:
            leader_state = self._get_agent_state(self.entity_manager.leaders[0])
        else:
            # If no leader, create a zero state
            leader_state = np.zeros(7, dtype=np.float32)
        
        # Get all Followers states
        followers_states = []
        for follower in self.entity_manager.followers:
            follower_state = self._get_agent_state(follower)
            followers_states.append(follower_state)
        
        # Return structured state dictionary
        return {
            "leader": leader_state,
            "followers": followers_states
        }
    
    def _get_agent_state(self, agent):
        """Get single agent state
        
        Args:
            agent: Agent object
            
        Returns:
            Agent state array (np.float32)
        """
        # Basic state: position, velocity, heading (normalized)
        state = [
            agent.pos_x / 1000.0,  # Normalized position x
            agent.pos_y / 1000.0,  # Normalized position y
            agent.speed / 30.0,    # Normalized speed
            agent.theta * 57.3 / 360.0  # Normalized heading (convert to degrees then normalize)
        ]
        
        # For leaders, add goal information
        if agent.agent_type == 'leader' and self.entity_manager.goals:
            goal = self.entity_manager.goals[0]
            min_obs_dist = self._get_min_obstacle_distance(agent, self.entity_manager.obstacles)
            o_flag = 1.0 if min_obs_dist < 40.0 else 0.0
            state.extend([
                goal.pos_x / 1000.0,  # Normalized goal position x
                goal.pos_y / 1000.0,  # Normalized goal position y
                o_flag  # Obstacle avoidance flag
            ])
        # For followers, add target leader information
        elif agent.agent_type == 'follower' and self.entity_manager.leaders:
            leader = self.entity_manager.leaders[0]  # Assume following first leader
            state.extend([
                leader.pos_x / 1000.0,  # Normalized leader position x
                leader.pos_y / 1000.0,  # Normalized leader position y
                leader.speed / 30.0     # Normalized leader speed
            ])
        
        return np.array(state, dtype=np.float32)
    
    def get_observation_space_shape(self):
        """Get observation space shape
        
        Returns:
            Observation space shape tuple
        """
        # Assume all agents have same state dimension
        state_dim = 7  # State dimension per agent
        n_agents = len(self.entity_manager.leaders) + len(self.entity_manager.followers)
        
        # Observation space shape is (number of agents, state dimension per agent)
        return (n_agents, state_dim)
    
    def get_action_space_shape(self):
        """Get action space shape
        
        Returns:
            Action space shape tuple
        """
        # Each agent has 2D action: acceleration and angular velocity
        action_dim = 2
        n_agents = len(self.entity_manager.leaders) + len(self.entity_manager.followers)
        
        # Action space shape is (number of agents, action dimension per agent)
        return (n_agents, action_dim)
    
    def get_partial_state(self, agent_idx):
        """Get partial state representation for specified agent
        
        Args:
            agent_idx: Agent index
            
        Returns:
            Partial state array
        """
        all_agents = self.entity_manager.leaders + self.entity_manager.followers
        
        if agent_idx >= len(all_agents):
            raise ValueError(f"Agent index {agent_idx} out of range")
        
        agent = all_agents[agent_idx]
        return self._get_agent_state(agent)
    
    def get_obstacle_states(self):
        """Get obstacle states
        
        Returns:
            Obstacle state array
        """
        obstacle_states = []
        
        for obstacle in self.entity_manager.obstacles:
            state = [
                obstacle.pos_x / 1000.0,  # Normalized position x
                obstacle.pos_y / 1000.0   # Normalized position y
            ]
            obstacle_states.append(state)
        
        return np.array(obstacle_states, dtype=np.float32)
    
    def get_goal_states(self):
        """Get goal states
        
        Returns:
            Goal state array
        """
        goal_states = []
        
        for goal in self.entity_manager.goals:
            state = [
                goal.pos_x / 1000.0,  # Normalized position x
                goal.pos_y / 1000.0   # Normalized position y
            ]
            goal_states.append(state)
        
        return np.array(goal_states, dtype=np.float32) 
