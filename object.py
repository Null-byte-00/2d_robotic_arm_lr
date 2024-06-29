import pygame
from hook import HookTip
import math

pygame.init()

WIDTH  = 500
HEIGHT = 500
GRAVITY = 0.1
screen = pygame.display.set_mode((WIDTH, HEIGHT))

fps = 60
timer = pygame.time.Clock()


class Stand:
    def __init__(self, screen, x_pos, height=100, width=100, color='red') -> None:
        self.screen = screen
        self.color = color
        self.x_pos = x_pos
        self.height = height
        self.width = width
        self.pos = (x_pos, 0)
    
    def draw(self):
        pygame.draw.rect(
            self.screen,
            self.color,
            pygame.Rect(self.pos[0], HEIGHT - self.height, self.width, self.height)
        )


class TargetObject:
    def __init__(self, screen, pos, length=30, thicness=100, y_speed=0.0,x_speed=0.0, retention=0.99, color='green') -> None:
        self.color = color
        self.retention = retention
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.screen = screen
        self.pos = pos
        self.length = length
        self.thicness = thicness
        self.top = pygame.Surface((thicness, length))
        self.bottom = pygame.Surface((thicness, length))
        self.left = pygame.Surface((length, thicness))
        self.stands = []

        self.top.fill(color)
        self.bottom.fill(color)
        self.left.fill(color)
        
        self.update_state()
    
    def update_state(self):
        self.top_pos = (self.pos[0], self.pos[1] - 30)
        self.bottom_pos = (self.pos[0], self.pos[1] + 60)
        self.left_pos = (self.pos[0], self.pos[1] - 30)
        self.check_gravity()
        self.check_x_force()

    def check_gravity(self):
        for stand in self.stands:
            print(self.pos[1],HEIGHT, stand.height )
            if self.bottom_pos[1] + 32.7 > HEIGHT - stand.height and stand.x_pos < self.pos[0] < stand.x_pos + stand.width:
                return
        if self.pos[1] < HEIGHT - 90:
            self.y_speed += GRAVITY
            self.pos = (self.pos[0], self.pos[1] + self.y_speed)

    def check_x_force(self):
        self.x_speed = self.x_speed * self.retention
        self.pos = (
            self.pos[0] + self.x_speed,
            self.pos[1] 
            )

    def x_force(self, force):
        self.x_speed += force
    

    def draw(self):
        self.screen.blit(self.top, self.top_pos)
        self.screen.blit(self.bottom, self.bottom_pos)
        self.screen.blit(self.left, self.left_pos)

    def colides_with_hook(self, hook: HookTip):
        overlabs = None
        top_mask = pygame.mask.from_surface(self.top)
        bottom_mask = pygame.mask.from_surface(self.bottom)
        left_mask = pygame.mask.from_surface(self.left)

        if hook.hook_overlapls(top_mask, self.top_pos):
            overlabs = hook.hook_overlapls(top_mask, self.top_pos)
        elif hook.hook_overlapls(bottom_mask, self.bottom_pos):
            overlabs = hook.hook_overlapls(bottom_mask, self.bottom_pos)
        elif hook.hook_overlapls(left_mask, self.left_pos):
            overlabs = hook.hook_overlapls(left_mask, self.left_pos)
        return overlabs
    
    def add_stand(self, stand):
        self.stands.append(stand)


target = TargetObject(screen, (100, 100), x_speed=1.5)
hook_tip = HookTip(screen, (250, 50), 3,  speed=0.1)
stand =  Stand(screen, 100)
target.add_stand(stand)

def main():
    run = True
    while run:
        timer.tick(fps)
        if target.colides_with_hook(hook_tip):
            screen.fill('blue')
        else:
            screen.fill('black')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        stand.draw()
        target.draw()
        target.update_state()

        mouse_pos = pygame.mouse.get_pos()
        hook_tip.pos = mouse_pos

        hook_tip.draw()

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()