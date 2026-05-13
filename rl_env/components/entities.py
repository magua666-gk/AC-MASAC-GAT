import numpy as np
import math
import random
import pygame
import os

# Set constants
class Constants:
    """Environment constants definition"""
    # Screen dimensions
    SCREEN_W = 800
    SCREEN_H = 600
    SCREEN_SIZE = (SCREEN_W, SCREEN_H)
    
    # Area bounds
    AREA_X = 100
    AREA_Y = 100
    AREA_WITH = 600
    AREA_HEIGHT = 500
    
    # Game settings
    FPS = 60
    FOLLOWER_MAKE_TIME = 1000
    CREATE_FOLLOWER_EVENT = pygame.USEREVENT
    FOLLOWER_FLAG = False

# Time step
# dt = 0.1
# Use a variable to store current dt value
_current_dt = 1.0

def set_dt(value):
    """Set time step dt value
    
    Args:
        value: New dt value
    """
    global _current_dt
    _current_dt = value
    print(f"Time step dt set to: {_current_dt}")

def get_dt():
    """Get current time step dt value
    
    Returns:
        Current dt value
    """
    return _current_dt

class Entity:
    """Base class for all entities"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(20, 20)):
        """Initialize entity
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.size = size
        self.image_path = image_path
        self.image = None
        self.rect = pygame.Rect(0, 0, size[0], size[1])
        
        # Load image if available
        if image_path and os.path.exists(image_path):
            self.load_image(image_path, size)
    
    def load_image(self, image_path_or_img, size=(20, 20)):
        """Load and set entity image
        
        Args:
            image_path_or_img: Image path or pygame image object
            size: Image size
        """
        try:
            if isinstance(image_path_or_img, str):
                self.image_path = image_path_or_img
                if not os.path.exists(image_path_or_img):
                    print(f"Image file does not exist: {image_path_or_img}")
                    return
                    
                self.image = pygame.image.load(image_path_or_img).convert_alpha()
                self.image = pygame.transform.scale(self.image, size)
            else:
                self.image = image_path_or_img
                if self.image:
                    if size != self.image.get_size():
                        self.image = pygame.transform.scale(self.image, size)
            
            if self.image:
                self.rect = self.image.get_rect()
                self.rect.center = (self.pos_x, self.pos_y)
                self.orig_image = self.image
                return True
            return False
            
        except Exception as e:
            print(f"Failed to load image: {image_path_or_img if isinstance(image_path_or_img, str) else 'image object'}, error: {e}")
            return False
    
    def set_position(self, x, y):
        """Set entity position
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.pos_x = x
        self.pos_y = y
        self.rect.center = (x, y)
    
    def distance_to(self, other_entity):
        """Calculate distance to another entity
        
        Args:
            other_entity: Another entity
            
        Returns:
            Euclidean distance to other entity
        """
        return math.hypot(self.pos_x - other_entity.pos_x, self.pos_y - other_entity.pos_y)
    
    def collides_with(self, other_entity):
        """Check if collides with another entity
        
        Args:
            other_entity: Another entity
            
        Returns:
            Whether collision occurred
        """
        collision_threshold = (self.size[0] + other_entity.size[0]) / 2
        return self.distance_to(other_entity) < collision_threshold
    
    def is_out_of_bounds(self):
        """Check if out of bounds
        
        Returns:
            Whether out of bounds
        """
        out_x = self.pos_x < Constants.AREA_X or self.pos_x > Constants.AREA_WITH
        out_y = self.pos_y < Constants.AREA_Y or self.pos_y > Constants.AREA_HEIGHT
        return out_x or out_y
    
    def is_near_boundary(self, threshold=50):
        """Check if near boundary
        
        Args:
            threshold: Boundary proximity threshold
            
        Returns:
            Whether near boundary
        """
        near_x = (self.pos_x - Constants.AREA_X < threshold or 
                 Constants.AREA_WITH - self.pos_x < threshold)
        near_y = (self.pos_y - Constants.AREA_Y < threshold or 
                 Constants.AREA_HEIGHT - self.pos_y < threshold)
        return near_x or near_y
    
    def update(self):
        """Update entity state"""
        pass
    
    def render(self, screen):
        """Render entity
        
        Args:
            screen: Pygame screen object
        """
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            # If no image, draw filled rect to represent entity
            # Use different default colors for different entity types
            color = (200, 200, 200)  # Default gray
            
            # Set color based on entity type
            if hasattr(self, 'agent_type'):
                if self.agent_type == 'leader':
                    color = (0, 0, 255)  # Blue
                elif self.agent_type == 'follower':
                    color = (255, 0, 0)  # Red
            
            # Check if goal or obstacle
            if isinstance(self, Goal):
                color = (0, 255, 0)  # Green
            elif isinstance(self, Obstacle):
                color = (100, 100, 100)  # Dark gray
                
            # Draw filled rect
            pygame.draw.rect(screen, color, self.rect)
            
            # Draw border
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 1)


class Agent(Entity):
    """Agent base class, represents movable entities"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(20, 20), speed=10):
        """Initialize agent
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
            speed: Initial speed
        """
        super().__init__(pos_x, pos_y, image_path, size)
        
        # Physical properties
        self.speed = speed
        self.theta = random.uniform(0, 2 * math.pi)  # Heading angle (radians)
        
        # State flags
        self.alive = True
        self.has_won = False
        
        # Other properties
        self.agent_type = "agent"  # Agent type for distinguishing different agents
    
    def apply_action(self, action):
        """Apply action to agent
        
        Args:
            action: Action array [acceleration, angular_velocity]
        """
        if not self.alive:
            return
            
        a = action[0]  # Acceleration
        phi = action[1]  # Angular velocity
        
        # Update speed and heading
        self.speed = self.speed + 0.3 * a * get_dt()
        self.theta = self.theta + 0.6 * phi * get_dt()
        
        # Limit speed range
        self.speed = np.clip(self.speed, 10, 20)
    
        # Normalize heading angle self.theta
        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        elif self.theta < 0:
            self.theta += 2 * math.pi
    
    def update(self):
        """Update agent state"""
        if not self.alive:
            return
            
        # Update position based on speed and heading
        dt = get_dt()  # Get current dt value
        self.pos_x = self.pos_x + self.speed * np.cos(self.theta) * dt
        self.pos_y = self.pos_y + self.speed * np.sin(self.theta) * dt
        
        # Keep within boundaries
        self.pos_x = np.clip(self.pos_x, Constants.AREA_X, Constants.AREA_WITH)
        self.pos_y = np.clip(self.pos_y, Constants.AREA_Y, Constants.AREA_HEIGHT)
        
        # Update rect position
        self.rect.center = (self.pos_x, self.pos_y)
    
    def rotate(self):
        """Rotate image based on heading angle"""
        if self.image:
            # Convert radians to degrees, negate for pygame rotation direction
            angle = -self.theta * (180 / math.pi)
            # Rotate original image
            self.image = pygame.transform.rotate(self.orig_image, angle)
            # Keep position unchanged after rotation
            self.rect = self.image.get_rect(center=(self.pos_x, self.pos_y))
    
    def kill(self, won=False):
        """Kill agent
        
        Args:
            won: Whether ended in victory
        """
        self.alive = False
        self.has_won = won
        
        # Can add death effects or sounds here
        
        # Modify image appearance to show death state
        if self.image:
            # Convert image to grayscale to represent death
            array = pygame.surfarray.array3d(self.orig_image)
            # Calculate grayscale value
            grayscale = ((array[:,:,0] + array[:,:,1] + array[:,:,2]) / 3)
            array[:,:,0] = grayscale
            array[:,:,1] = grayscale
            array[:,:,2] = grayscale
            self.image = pygame.surfarray.make_surface(array)
    
    def reached_goal(self, goal):
        """Check if reached goal
        
        Args:
            goal: Goal entity
            
        Returns:
            Whether reached goal
        """
        distance = self.distance_to(goal)
        return distance < (self.size[0] + goal.size[0]) / 2
    
    def is_alive(self):
        """Check if agent is alive
        
        Returns:
            Whether alive
        """
        return self.alive
    
    def has_won(self):
        """Check if agent has won
        
        Returns:
            Whether won
        """
        return self.has_won
    
    def is_in_formation_with(self, other_agent, distance_threshold=50):
        """Check if maintaining formation with another agent
        
        Args:
            other_agent: Another agent
            distance_threshold: Distance threshold
            
        Returns:
            Whether in formation
        """
        if not self.alive or not other_agent.alive:
            return False
            
        # Check distance
        distance = self.distance_to(other_agent)
        if distance > distance_threshold:
            return False
            
        # Formation condition met
        return True
    
    def is_leader_for(self, other_agent):
        """Check if this agent is leader for another agent
        
        Args:
            other_agent: Another agent
            
        Returns:
            Whether is leader
        """
        if not self.alive or not other_agent.alive:
            return False
            
        # Default implementation: if this agent is leader type and other is follower type, consider as leader
        return self.agent_type == 'leader' and other_agent.agent_type == 'follower'


class LeaderAgent(Agent):
    """Leader agent, represents primary control role"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(20, 20), speed=None):
        """Initialize leader
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
            speed: Initial speed
        """
        super().__init__(pos_x, pos_y, image_path, size, speed=0)
        self.speed = random.randint(10, 20)  # Leader initial speed [10, 20]
        self.agent_type = "leader"  # Set as leader type

    def apply_action(self, action):
        """Apply action to leader
        
        Args:
            action: Action array [acceleration, angular_velocity]
        """
        if not self.alive:
            return
            
        dt = get_dt()
        a = action[0]  # Acceleration component
        phi = action[1]  # Angular velocity component
        
        # Leader-specific speed update
        self.speed = self.speed + 0.3 * a * dt
        self.speed = np.clip(self.speed, 10, 20) # Leader speed clipping range [10, 20]
        
        # Leader-specific heading update
        self.theta = self.theta + 0.6 * phi * dt
        
        # Normalize heading angle self.theta
        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        elif self.theta < 0:
            self.theta += 2 * math.pi


class FollowerAgent(Agent):
    """Follower agent, represents follower role"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(20, 20), speed=None):
        """Initialize follower
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
            speed: Initial speed
        """
        super().__init__(pos_x, pos_y, image_path, size, speed=0)
        self.speed = random.randint(20, 30)  # Follower initial speed
        self.agent_type = "follower"  # Set as follower type
    
    def apply_action(self, action):
        """Apply action to follower
        
        Args:
            action: Action array [acceleration, angular_velocity]
        """
        if not self.alive:
            return
            
        dt = get_dt()
        a = action[0]  # Acceleration component
        phi = action[1]  # Angular velocity component
        
        # Follower-specific speed update
        self.speed = self.speed + 0.6 * a * dt
        self.speed = np.clip(self.speed, 10, 40) # Follower speed clipping range
        
        # Follower-specific heading update
        self.theta = self.theta + 1.2 * phi * dt
        
        # Normalize heading angle self.theta
        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        elif self.theta < 0:
            self.theta += 2 * math.pi


class Obstacle(Entity):
    """Obstacle entity"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(40, 40)):
        """Initialize obstacle
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
        """
        super().__init__(pos_x, pos_y, image_path, size)


class Goal(Entity):
    """Goal entity"""
    
    def __init__(self, pos_x=0, pos_y=0, image_path=None, size=(20, 20)):
        """Initialize goal
        
        Args:
            pos_x: Initial X coordinate
            pos_y: Initial Y coordinate
            image_path: Entity image path
            size: Entity size
        """
        super().__init__(pos_x, pos_y, image_path, size) 