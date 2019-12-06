import os
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import gym

import gym_line_follower  # to register environment

# Number of past subsequent observations to take as input
window_length = 5
env = gym.make("LineFollower-v0")
  #  train(env, "ddpg_1", steps=100000, pretrained_path=None)
  #  test(env, "models/ddpg_1/last_weights.h5f")
