import pygame
import numpy as np
import os
from rl_env.components.entities import Constants

class Renderer:
    """Environment renderer"""
    
    def __init__(self, entity_manager, mode='human', record_trajectory=True):
        """Initialize renderer
        
        Args:
            entity_manager: Entity manager
            mode: Render mode ('human', 'rgb_array')
            record_trajectory: Whether to record trajectory
        """
        self.entity_manager = entity_manager
        self.mode = mode
        self.record_trajectory = record_trajectory
        self.closed = False
        self.stop_requested = False
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except pygame.error:
            pass
        
        # Create screen
        self.screen_width = Constants.SCREEN_W
        self.screen_height = Constants.SCREEN_H
        self.screen = None
        
        if self.mode == 'human':
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Agent Reinforcement Learning Environment")
        else:
            # Create a hidden screen for rendering to array
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        # Initialize image resource dictionary
        self.graphics = {}
        
        # Load resources
        self.load_resources()
        
        # Trajectory recording
        self.leader_trajectories = [[] for _ in range(entity_manager.leader_count)]
        self.follower_trajectories = [[] for _ in range(entity_manager.follower_count)]
        
        # Create clock
        self.clock = pygame.time.Clock()
        
        # Background
        self.background = pygame.Surface((self.screen_width, self.screen_height))
        self.background.fill((255, 255, 255))  # White background

    def is_closed(self):
        """Return whether the render window has been closed."""
        return self.closed or self.stop_requested

    def _request_stop(self):
        """Mark rendering as closed and release pygame resources."""
        self.closed = True
        self.stop_requested = True
        self.screen = None
        try:
            if pygame.display.get_init():
                pygame.display.quit()
        except pygame.error:
            pass
    
    def load_resources(self):
        """Load rendering resources"""
        # Color definitions
        self.colors = {
            'leader': (0, 0, 255),      # Blue
            'follower': (255, 0, 0),     # Red
            'obstacle': (0, 0, 0),    # Black
            'goal': (0, 255, 0),      # Green
            'boundary': (200, 200, 200),  # Gray
            'trajectory_leader': (0, 0, 200),  # Dark blue
            'trajectory_follower': (200, 0, 0)  # Dark red
        }
        
        # Load images using absolute path
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            image_path = os.path.join(project_root, 'assignment', 'source', 'image')
            print(f"Attempting to load images from path: {image_path}")
            
            # Check if image directory exists
            if not os.path.exists(image_path):
                print(f"Image directory does not exist: {image_path}")
                print("Using default rectangle rendering")
                return
            
            # List possible image alternatives for debugging
            try:
                files = os.listdir(image_path)
                print(f"Files in directory: {files}")
            except Exception as e:
                print(f"Unable to list directory contents: {e}")
                files = []
            
            # Try to load images, use fallback or return None if failed
            leader_img = self._load_image(os.path.join(image_path, 'leader.png'), (30, 30))
            if not leader_img and files and 'hero1.png' in files:
                leader_img = self._load_image(os.path.join(image_path, 'leader.png'), (30, 30))
            
            follower_img = self._load_image(os.path.join(image_path, 'follower.png'), (30, 30)) 
            if not follower_img and files and 'enemy1.png' in files:
                follower_img = self._load_image(os.path.join(image_path, 'follower.png'), (30, 30))
            
            obstacle_img = self._load_image(os.path.join(image_path, 'hole.png'), (40, 40))
            if not obstacle_img and files and 'enemy0.png' in files:
                obstacle_img = self._load_image(os.path.join(image_path, 'enemy0.png'), (40, 40))
            
            goal_img = self._load_image(os.path.join(image_path, 'goal.png'), (30, 30))
            
            background_img = self._load_image(os.path.join(image_path, 'background.png'), 
                                            (self.screen_width, self.screen_height))
            if not background_img and files and 'background3.png' in files:
                background_img = self._load_image(os.path.join(image_path, 'background3.png'), 
                                                (self.screen_width, self.screen_height))
            
            # Store successfully loaded image objects
            if leader_img:
                self.graphics['leader'] = leader_img
            if follower_img:
                self.graphics['follower'] = follower_img
            if obstacle_img:
                self.graphics['obstacle'] = obstacle_img
            if goal_img:
                self.graphics['goal'] = goal_img
            if background_img:
                self.graphics['background'] = background_img
            
            # Update entity images, only use successfully loaded images
            if self.graphics:
                self.entity_manager.load_images(self.graphics)
                print(f"Successfully loaded {len(self.graphics)} images and applied to entities")
            else:
                print("All image loading failed, using default rectangle rendering")
        except Exception as e:
            print(f"Error loading image resources: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_image(self, path, size=None):
        """Load and scale image
        
        Args:
            path: Image path
            size: Image size
            
        Returns:
            Loaded pygame image, returns None if failed
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                print(f"Image file does not exist: {path}")
                return None
            
            print(f"Loading image: {path}")    
            image = pygame.image.load(path)
            
            # Check if image has alpha channel
            if image.get_alpha():
                image = image.convert_alpha()
            else:
                image = image.convert()
                
            # Resize
            if size:
                orig_size = image.get_size()
                image = pygame.transform.scale(image, size)
                print(f"Image size adjusted from {orig_size} to {size}")
                
            print(f"Successfully loaded image: {path}, size: {image.get_size()}")
            return image
        except Exception as e:
            print(f"Unable to load image: {path}, error: {e}")
            # Print more detailed error stack
            import traceback
            traceback.print_exc()
            return None
    
    def reset_trajectories(self):
        """Reset trajectory recording"""
        self.leader_trajectories = [[] for _ in range(self.entity_manager.leader_count)]
        self.follower_trajectories = [[] for _ in range(self.entity_manager.follower_count)]
    
    def record_positions(self):
        """Record entity positions for trajectory drawing"""
        if not self.record_trajectory:
            return
        
        # Record leader positions
        for i, leader in enumerate(self.entity_manager.leaders):
            if i < len(self.leader_trajectories):
                self.leader_trajectories[i].append((leader.pos_x, leader.pos_y))
        
        # Record follower positions
        for i, follower in enumerate(self.entity_manager.followers):
            if i < len(self.follower_trajectories):
                self.follower_trajectories[i].append((follower.pos_x, follower.pos_y))
    
    def render(self):
        """Render current environment state
        
        Returns:
            If mode is 'rgb_array', return rendered RGB array
        """
        try:
            if self.is_closed():
                return None

            if not pygame.get_init():
                self._request_stop()
                return None

            # Handle events
            if self.mode == 'human':
                if not pygame.display.get_init():
                    self._request_stop()
                    return None
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self._request_stop()
                            return None
                except pygame.error as e:
                    print(f"Rendering stopped: {e}")
                    self._request_stop()
                    return None

            if self.screen is None:
                self._request_stop()
                return None
            
            # Record positions
            self.record_positions()
            
            # Clear screen
            if 'background' in self.graphics and self.graphics['background']:
                self.screen.blit(self.graphics['background'], (0, 0))
            else:
                self.screen.fill((255, 255, 255))
            
            # Draw boundary
            pygame.draw.rect(
                self.screen, 
                self.colors['boundary'], 
                (Constants.AREA_X, Constants.AREA_Y, 
                Constants.AREA_WITH - Constants.AREA_X, 
                Constants.AREA_HEIGHT - Constants.AREA_Y), 
                2
            )
            
            # Draw trajectories
            self._draw_trajectories()
            
            # Render all entities
            try:
                self.entity_manager.render(self.screen)
            except Exception as e:
                print(f"Error rendering entities: {e}")
            
            # Draw game information
            try:
                self._draw_game_info()
            except Exception as e:
                print(f"Error drawing game information: {e}")
            
            # Update display
            if self.mode == 'human':
                try:
                    pygame.display.update()
                    self.clock.tick(Constants.FPS)
                except pygame.error as e:
                    print(f"Rendering stopped: {e}")
                    self._request_stop()
                    return None
                except Exception as e:
                    print(f"Error updating display: {e}")
            
            # If mode is 'rgb_array', return screen pixel array
            if self.mode == 'rgb_array':
                try:
                    return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
                except Exception as e:
                    print(f"Error getting RGB array: {e}")
                    return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            return None
            
        except Exception as e:
            print(f"Error during rendering: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _draw_trajectories(self):
        """Draw entity trajectories"""
        if not self.record_trajectory:
            return
        
        # Draw leader trajectories
        for traj in self.leader_trajectories:
            if len(traj) > 1:
                pygame.draw.lines(
                    self.screen, 
                    self.colors['trajectory_leader'],
                    False,
                    traj,
                    2
                )
        
        # Draw follower trajectories
        for traj in self.follower_trajectories:
            if len(traj) > 1:
                pygame.draw.lines(
                    self.screen, 
                    self.colors['trajectory_follower'], 
                    False,
                    traj,
                    2
                )
    
    def _draw_game_info(self):
        """Draw game information"""
        # Create font object
        try:
            font = pygame.font.SysFont('arial', 20)
        except:
            try:
                # Try to use default font
                font = pygame.font.Font(None, 20)
            except Exception as e:
                print(f"Failed to create font object: {e}")
                return
        
        # Render game information text
        info_text = f"Leaders: {self.entity_manager.leader_count} | Followers: {self.entity_manager.follower_count}"
        
        # Add formation rate information
        formation_rate = self.entity_manager.get_formation_rate()
        info_text += f" | Formation Rate: {formation_rate:.2f}"
        
        # Add timestep information
        info_text += f" | Timestep: {self.entity_manager.time_counter}"
        
        try:
            text_surface = font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))
        except Exception as e:
            print(f"Error rendering text: {e}")
    
    def close(self):
        """Close renderer"""
        self._request_stop()
        try:
            if pygame.get_init():
                pygame.quit()
        except pygame.error:
            pass
