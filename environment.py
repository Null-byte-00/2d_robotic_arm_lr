from joints import ConstantJoint, MovingJoint
from hook import HookTip
from target_object import TargetObject, Stand
import pygame


class Environment:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.timer = pygame.time.Clock()
        self.fps = 60

        self.constant_joint = ConstantJoint(screen, (250, 50), 1)
        self.moving_joint = MovingJoint(screen, (310, 110), 2)
        self.hook_tip = HookTip(screen, (300, 50), 3,  speed=0.1)
        self.stand = Stand(screen, 100, self.hook_tip)
        self.target_object = TargetObject(screen, (300, 10), self.hook_tip)
        self.target_object.add_stand(self.stand)

        self.hook_tip.connect(self.moving_joint)
        self.moving_joint.connect(self.constant_joint)
    

    def run(self):
        run = True
        while run:
            self.timer.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            self.screen.fill('black')
            self.constant_joint.draw()
            self.moving_joint.update_position()
            self.moving_joint.draw()
            self.hook_tip.connection = self.moving_joint
            self.hook_tip.update_position()
            self.hook_tip.draw()

            self.stand.draw()
            self.target_object.draw()
            self.target_object.update_state()

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    env = Environment(screen)
    env.run()