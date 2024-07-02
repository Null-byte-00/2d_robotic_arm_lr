import math
import torch

def get_environ_reward(env):
    #return (400 - env.target_object.pos[1]) / 10
    stand_pos = env.stand.pos
    hook_pos = env.hook_tip.pos
    #print(math.cos(env.hook_tip.angle))
    distance = math.sqrt(
        (200 - hook_pos[0]) ** 2 +
        (200 - hook_pos[1]) ** 2 
    ) 
    return -(env.moving_joint.pos[1] - 120) / 10 #+ math.cos(env.hook_tip.angle)