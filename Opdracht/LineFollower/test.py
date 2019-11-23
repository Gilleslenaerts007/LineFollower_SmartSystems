import os
import pickle
import json
import warnings
from time import time, sleep

from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p

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

env = gym.make("LineFollower-v0")
env.reset()
for _ in range(100):
	for i in range(1000):
		#env.render()
		obsv, rew, done, info = env.step((1, 1))
		sleep(0.05)
		if done:
			break
	env.reset()
env.close()
