from joints import ConstantJoint, MovingJoint
from hook import HookTip
from target_object import TargetObject, Stand
import pygame


class Environment:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.timer = pygame.time.Clock()
        self.fps = 60

        self.constant_joint = ConstantJoint(screen, (330, 50), 1)
        self.moving_joint = MovingJoint(screen, (350, 150), 2, bounce_stop=0)
        self.hook_tip = HookTip(screen, (350, 50), 3,  speed=0.1, bounce_stop=0)
        self.stand = Stand(screen, 20, self.hook_tip, height=70, width=70)
        self.target_object = TargetObject(screen, (300, 10), self.hook_tip)
        self.target_object.add_stand(self.stand)

        self.hook_tip.connect(self.moving_joint)
        self.moving_joint.connect(self.constant_joint)
    

    def run(self, user_control=False):
        run = True
        while run:
            self.timer.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        if user_control:
                            self.moving_joint.speed -= 0.03
                    elif event.key == pygame.K_LEFT:
                        if user_control:
                            self.moving_joint.speed += 0.03
                    elif event.key == pygame.K_d:
                        if user_control:
                            self.hook_tip.speed -= 0.03
                    elif event.key == pygame.K_a:
                        if user_control:
                            self.hook_tip.speed += 0.03

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
    
    def moving_joint_push(self, force):
        self.moving_joint.speed += force


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    env = Environment(screen)
    env.run(user_control=True)