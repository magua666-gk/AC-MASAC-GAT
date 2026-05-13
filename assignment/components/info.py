import pygame
from assignment import constants as C
pygame.font.init()

class Info():
    def __init__(self,state,game_info):  # state represents game stage, different stages show different information
        self.game_info=game_info
        self.state=state
        self.create_state_labels()  # Create stage-specific labels
        self.create_info_labels()  # Create common labels

    def create_state_labels(self):
        self.state_labels=[]
        if self.state=='main_menu':
            self.state_labels.append((self.create_label('Settings', size=30, flag='E'), (700, 0)))
            self.menu_info_rect=self.create_label('Settings', size=30, flag='E').get_rect()
            self.menu_info_rect.x=self.menu_info_rect.x+700
        elif self.state=='load_screen':
            self.state_labels.append((self.create_label('Multi-Agent Path Planning', size=60, flag='E'), (150, 0)))
        elif self.state=='battle_screen':
            self.state_labels.append((self.create_label('Battle Simulation', size=30, flag='E'), (300, 0)))
            self.state_labels.append((self.create_label('Bullets', size=15, flag='E'), (600, 0)))
        elif self.state=='game_over':
            self.state_labels.append((self.create_label('Game Over', size=60,flag='E',color=C.RED), (200, 300)))
            self.state_labels.append((self.create_label('{} Wins!'.format(self.game_info['win']), size=60,flag='E',color=C.RED), (500, 305)))
            self.state_labels.append((self.create_label('Episode {}'.format(self.game_info['epsoide']),size=30,flag='E'),(300,150)))
            self.state_labels.append((self.create_label('Failures: {}'.format(self.game_info['enemy_win']),size=30,flag='E'),(300,190)))
            self.state_labels.append((self.create_label('Successes: {}'.format(self.game_info['hero_win']), size=30, flag='E'), (300, 230)))

    def create_info_labels(self):
        self.info_labels=[]
        self.info_labels.append((self.create_label('Info',size=20,flag='E'),(0,0)))
        self.info_rect=self.create_label('Info',size=20,flag='E').get_rect()
        # self.info_rect.x=self.info_rect.x+0
        # self.info_rect.y = self.info_rect.y + 0

    def create_label(self,label,size=40,flag='Chinese',color=C.WHITE):  # Render text as image
        if flag=='Chinese':
            font=pygame.font.SysFont(C.FONT_CHINESE,size)
        else:
            font = pygame.font.SysFont(C.FONT_ENGLISH, size)
        label_image=font.render(label,1,color)
        return label_image

    def update(self,mouse_pos):
        if self.info_rect.collidepoint(mouse_pos):
            self.info_labels[0]=(self.create_label('Info',size=20,flag='E',color=C.GREEN),(0,0))
            self.info_labels.append((self.create_label('Episode {}'.format(self.game_info['epsoide']),size=20,flag='E'),(0,20)))
            self.info_labels.append((self.create_label('Failures: {}'.format(self.game_info['enemy_win']), size=20, flag='E'), (0, 40)))
            self.info_labels.append((self.create_label('Successes: {}'.format(self.game_info['hero_win']), size=20, flag='E'), (0, 60)))
        else:
            self.info_labels.clear()
            self.info_labels.append((self.create_label('Info', size=20, flag='E'), (0, 0)))
        if not C.OPEN_MENU and self.state=='main_menu':
            if self.menu_info_rect.collidepoint(mouse_pos):
                self.state_labels[0]=(self.create_label('Settings', size=30, flag='E', color=C.GREEN), (700, 0))
                if C.CLICK:
                    C.OPEN_MENU=True
            else:
                self.state_labels[0]=(self.create_label('Settings', size=30, flag='E'), (700, 0))

    def draw(self,surface):
        for label in self.state_labels:
            surface.blit(label[0],label[1])
        for label in self.info_labels:
            surface.blit(label[0], label[1])