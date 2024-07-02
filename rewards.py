import math
import torch

def get_environ_reward(env):
    #return (400 - env.target_object.pos[1]) / 10
    stand_pos = env.stand.pos
    target_pos = env.target_object.pos
    hook_pos = env.hook_tip.pos
    #print(math.cos(env.hook_tip.angle))
    hook_distance = math.sqrt(
        (target_pos[0] - hook_pos[0]) ** 2 +
        (target_pos[0] - hook_pos[1]) ** 2 
    ) 

    stand_distance = math.sqrt(
        (target_pos[0] - stand_pos[0]) ** 2 +
        (target_pos[0] - stand_pos[1]) ** 2 
    ) 
    return -hook_distance #+ math.cos(env.hook_tip.angle)