import pygame
import os
from assignment import  constants as C
from assignment import tools
pygame.init()
try:
    pygame.mixer.init()
except pygame.error:
    pass
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPHICS=tools.load_graphics(os.path.join(BASE_DIR, 'source', 'image'))

SOUND=tools.load_sound(os.path.join(BASE_DIR, 'source', 'music'))
