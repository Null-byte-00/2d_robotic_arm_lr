import pygame
import math
from joints import ConstantJoint, MovingJoint


class HookTip(MovingJoint):
    def __init__(self, screen, pos, id, speed=0, retention=0.99, bounce_stop=0.00005) -> None:
        super().__init__(screen, pos, id, speed, retention, bounce_stop)
        self.hook_img = pygame.image.load("hook.png").convert_alpha()
        self.hook_mask = pygame.mask.from_surface(self.hook_img)

    def draw(self):
        super().draw()
        self.screen.blit(self.hook_img, (
            self.pos[0] - 100,
            self.pos[1]
        ))
    
    def hook_overlapls(self,mask, mask_pos):
        relative_pos = (
            mask_pos[0] - self.pos[0] +100,
            mask_pos[1] - self.pos[1],
        )
        print(relative_pos)
        return self.hook_mask.overlap(mask, relative_pos)



def main():
    pygame.init()
    WIDTH  = 500
    HEIGHT = 500
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    const_joint = ConstantJoint(screen, (250, 50), 1)
    moving_joint = MovingJoint(screen, (310, 110), 2)
    moving_joint2 = HookTip(screen, (250, 50), 3,  speed=0.1)
    const_joint.connect(moving_joint)
    moving_joint.connect(const_joint)
    moving_joint2.connect(moving_joint)

    cursor_color = 'blue'
    cursor = pygame.Surface((10, 10))
    cursor.fill(cursor_color)

    fps = 60
    timer = pygame.time.Clock()
    run = True
    while run:
        timer.tick(fps)
        screen.fill('black')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        moving_joint.update_position()
        moving_joint2.connections[0] = moving_joint
        moving_joint2.update_position()

        moving_joint2.draw()
        moving_joint.draw()
        const_joint.draw()

        mouse_pos = pygame.mouse.get_pos()
        cursor.fill(cursor_color)
        screen.blit(cursor, mouse_pos)
        if moving_joint2.hook_overlapls(pygame.mask.from_surface(cursor), mouse_pos):
            cursor_color = 'red'
        else:
            cursor_color = 'blue'

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()