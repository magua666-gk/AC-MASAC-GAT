import random
import numpy as np
from rl_env.components.entities import Constants

class PositionGenerator:
    """Unified position generator that ensures curriculum learning and non-curriculum learning use the same map boundaries and entity generation logic"""
    
    @staticmethod
    def get_map_bounds():
        """Get the boundary values of the map
        
        Returns:
            (area_x, area_y, area_width, area_height, margin) tuple
        """
        area_x = Constants.AREA_X
        area_y = Constants.AREA_Y
        area_width = Constants.AREA_WITH - Constants.AREA_X  # Actual width
        area_height = Constants.AREA_HEIGHT - Constants.AREA_Y  # Actual height
        margin = 50  # Safety margin
        
        return area_x, area_y, area_width, area_height, margin
    
    @staticmethod
    def generate_leader_position():
        """Generate leader position - bottom left area
        
        Returns:
            (x, y) coordinate tuple
        """
        area_x, area_y, area_width, area_height, margin = PositionGenerator.get_map_bounds()
        
        x = random.randint(area_x + margin, area_x + area_width//3)
        y = random.randint(area_y + 2*area_height//3, area_y + area_height - margin)
        
        return x, y
    
    @staticmethod
    def generate_follower_position():
        """Generate follower position - bottom right area
        
        Returns:
            (x, y) coordinate tuple
        """
        area_x, area_y, area_width, area_height, margin = PositionGenerator.get_map_bounds()
        
        x = random.randint(area_x + 2*area_width//3, area_x + area_width - margin)
        y = random.randint(area_y + 2*area_height//3, area_y + area_height - margin)
        
        return x, y
    
    @staticmethod
    def generate_obstacle_position():
        """Generate obstacle position - middle area
        
        Returns:
            (x, y) coordinate tuple
        """
        area_x, area_y, area_width, area_height, margin = PositionGenerator.get_map_bounds()
        
        x = random.randint(area_x + area_width//3, area_x + 2*area_width//3)
        y = random.randint(area_y + area_height//3, area_y + 2*area_height//3)
        
        return x, y
    
    @staticmethod
    def generate_goal_position():
        """Generate goal position - top right area
        
        Returns:
            (x, y) coordinate tuple
        """
        area_x, area_y, area_width, area_height, margin = PositionGenerator.get_map_bounds()
        
        x = random.randint(area_x + 2*area_width//3, area_x + area_width - margin)
        y = random.randint(area_y + margin, area_y + area_height//3)
        
        return x, y
    
    @staticmethod
    def generate_all_positions(leader_count, follower_count, obstacle_count, goal_count):
        """Generate positions for all entities
        
        Args:
            leader_count: Number of leaders
            follower_count: Number of followers
            obstacle_count: Number of obstacles
            goal_count: Number of goals
            
        Returns:
            Dictionary containing all entity positions
        """
        positions = {
            'leaders': [PositionGenerator.generate_leader_position() for _ in range(leader_count)],
            'followers': [PositionGenerator.generate_follower_position() for _ in range(follower_count)],
            'obstacles': [PositionGenerator.generate_obstacle_position() for _ in range(obstacle_count)],
            'goals': [PositionGenerator.generate_goal_position() for _ in range(goal_count)]
        }
        
        return positions 