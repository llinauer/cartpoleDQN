"""

cartpole_dqn.py

Control the Cartpole environment with the DQN-method.

"""

import argparse
import collections
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_cartpole

BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEP = 1.5E-5
GAMMA = 0.99
REPLAY_SIZE = 1E5
LEARNING_RATE = 1E-3
SYNC_TARGET_FRAMES = 1000


class DQN(nn.Module):
    """ Deep-Q Network to estimate the Q-value of a state """

    def __init__(self):
        """ Constructor, define layers """
        super().__init__()

        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """ Forward pass through the network """

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc1(x)

        return x


def parse_args():
    """ Parse command-line arguments """
    
    parser = argparse.ArgumentParser(description='Train the cartpole system with DQN')
    parser.add_argument('--task', choices=['steady', 'upswing'], type=str,
                        help='Choose the type of task, either keeping the pole steady'
                             ' or doing an upswing', required=True)
    args = parser.parse_args()

    return args


def sample(replay_buffer, batch_size):
    """ Sample a batch of (state, action, next_state, reward) tuples from the replay buffer """

    # choose batch_size random indices from the replay buffer
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)

    states, actions, rewards, dones, next_states = zip(*[replay_buffer[idx] for idx in indices])

    # return the states, actions, next_states
    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
        np.array(next_states)


def main():
    """ main function """
    
    # parse cl-args
    args = parse_args()

    upswing = args.task == 'upswing'

    # create environment
    env = gym.make('CustomCartPole-v1', render_mode='rgb_array', upswing=upswing)

    # create two instances of DQN, the training and the target network
    train_net = DQN()
    target_net = DQN()

    # create a replay buffer
    replay_buffer = collections.deque(maxlen=REPLAY_SIZE)

    # training loop
    frame_idx = 0
    epsilon = EPSILON_START

    while True:
        # decrease epsilon every step until EPSILON_FINAL is reached
        epsilon -= EPSILON_DECAY_STEP
        epsilon = max(epsilon, EPSILON_FINAL)

        # do one step in the environment


if __name__ == '__main__':
    main()
