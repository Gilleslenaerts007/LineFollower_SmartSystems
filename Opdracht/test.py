import os
import pickle
import json
import warnings
from time import time, sleep
import random

from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_line_follower.track import Track
from gym_line_follower.track_plane_builder import build_track_plane
from gym_line_follower.bullet_client import BulletClient
from gym_line_follower.line_follower_bot import LineFollowerBot
from gym_line_follower.randomizer_dict import RandomizerDict
from gym_line_follower.envs.line_follower_env import LineFollowerEnv, LineFollowerCameraEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym_line_follower


LR = 1e-2
goal_steps = 1000
score_requirement = 84
initial_games = 30000
scores = []
choices = []

def plotCameraPOV():
    plt.plot(env.get_pov_image())
    plt.ylabel('LineCameraPOV')
    plt.show()


env = gym.make("LineFollower-v0")
env.reset()
# env.render('rgb_array') # 'rgb_array' or 'human'
for _ in range(100):
    score = 0
    prev_obs = []
    env.reset()
    action = random.randrange(1,3)
    for i in range(goal_steps):
        env.render()      
		#Actions 0: RIGHT, 1:straight, 2:Left 
        obsv, rew, done, info = env.step((1, action))
        #print(info)
        if (rew >= -100):
            print(rew)
            action = 0
        else:
            action = 2
        #print(obsv[1], obsv[3], obsv[5], obsv[7], obsv[10], obsv[12], obsv[14])
        choices.append(action)
        score=rew
        scores.append(score)
        print(scores)
        print(env._get_info())
        #plotCameraPOV()
        sleep(10)
        #if done:
           #break
        env.reset()
env.close()

Average = sum(scores)/len(scores)
print('Average Score:',Average)
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
