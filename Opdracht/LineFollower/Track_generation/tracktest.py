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

import matplotlib.pyplot as plt

''' Generate 1 track figure and show it
t = Track.generate(2.0, hw_ratio=0.7, seed=4125, spikeyness=0.2, nb_checkpoints=500)

img = t.render()
plt.imshow(img)
plt.show()
'''
#Generate multiple tracks
for i in range(9):
	t = Track.generate(2.0, hw_ratio=0.7, seed=None,
					   spikeyness=0.2, nb_checkpoints=500)
	img = t.render(ppm=1000)
	plt.subplot(3, 3, i+1)
	plt.imshow(img)
	plt.axis("off")
# plt.tight_layout()
plt.savefig("track_generator.png", dpi=300)
plt.show()