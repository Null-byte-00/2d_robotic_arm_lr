from model import Model
import torch
import pygame
import math


class Agent:
    def __init__(self, environ) -> None:
        self.model = Model()
        self.screen = pygame.display.set_mode((500, 500))
        self.environ = environ
        self.time = 0
        self.stablizer = 0.001
    
    def update_environment(self,env):
        self.environ = env

    def get_state(self):
        joint_x, joint_y = self.environ.moving_joint.pos 
        hook_x, hook_y = self.environ.hook_tip.pos 
        joint_speed = self.environ.moving_joint.speed
        hook_speed = self.environ.hook_tip.speed
        object_x = self.environ.target_object.pos[0]
        object_y = self.environ.target_object.pos[1]
        return torch.tensor([
            joint_x, joint_y, hook_x, hook_y, joint_speed, hook_speed, object_x, object_y
        ]) 
    
    def get_action(self):
        return self.model.actor(self.get_state()) * self.stablizer

    def get_hook_distance(self):
        return math.sqrt(
            (self.environ.hook_tip.pos[0] + self.environ.target_object.pos[0])**2 +
            (self.environ.hook_tip.pos[1] + self.environ.target_object.pos[1])**2 
        )

    def run(self):
        self.environ.run()