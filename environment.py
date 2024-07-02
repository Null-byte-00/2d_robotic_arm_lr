from joints import ConstantJoint, MovingJoint
from hook import HookTip
from target_object import TargetObject, Stand
import pygame
from agent import Agent
from rewards import get_environ_reward


class Environment:
    def __init__(self, screen, step_every=10) -> None:
        self.screen = screen
        self.agent = Agent(self)
        self.timer = pygame.time.Clock()
        self.timestep = 0
        self.step_every = step_every
        self.fps = 60

        self.constant_joint = ConstantJoint(screen, (330, 50), 1)
        self.moving_joint = MovingJoint(screen, (350, 150), 2, bounce_stop=0)
        self.hook_tip = HookTip(screen, (350, 50), 3,  speed=0.1, bounce_stop=0)
        self.stand = Stand(screen, 20, self.hook_tip, height=85, width=85)
        self.target_object = TargetObject(screen, (300, 10), self.hook_tip)
        self.target_object.add_stand(self.stand)

        self.hook_tip.connect(self.moving_joint)
        self.moving_joint.connect(self.constant_joint)
        self.previous_state = self.agent.get_state()
    

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
            state = self.agent.get_state()
            action = self.agent.get_action()

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

            #reward = get_environ_reward(self)
            #print(reward)
            if not user_control and self.timestep % self.step_every == 0:
                self.agent.update_environment(self)
                next_state = self.agent.get_state()
                reward = get_environ_reward(self)
                print(reward)
                self.agent.model.train_critic(action, self.previous_state, next_state, reward)
                self.agent.model.train_actor(self.previous_state)
                self.agent.model.soft_update()
                self.previous_state = next_state

                joint_speed, hook_speed =  self.agent.get_action()
                self.moving_joint.speed += joint_speed.item()
                self.hook_tip.speed += hook_speed.item()
                #print(self.moving_joint.speed, self.hook_tip.speed)
            
            self.timestep += 1
            pygame.display.flip()

        pygame.quit()
    
    def moving_joint_push(self, force):
        self.moving_joint.speed += force


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    env = Environment(screen)
    env.run(user_control=False)