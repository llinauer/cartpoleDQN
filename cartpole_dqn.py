"""

cartpole_dqn.py

Control the Cartpole environment with the DQN-method and a MLP.

"""

import argparse

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPController(nn.Module):
    """ MLP controller with one hidden layer and tanh activation """

    def __init__(self):
        """ Constructor, define layers """
        super().__init__()

        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        """ Forward pass through the network """

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc1(x)

        return x


def parse_args():
    """ Parse command-line arguments """
    
    parser = argparse.ArgumentParser(description='Train the cartpole system with DQN')
    parser.add_argument('--task', choice=['steady', 'swing-up'], type=str,
                        help='Choose the type of task, either keeping the pole steady'
                             ' or doing a swing up')
    args = parser.parse_args()

    return args

def main():
    """ main function """
    
    # parse cl-args
    args = parse_args()

    # create environment
    env = gym.make('CartPole-v1', render_mode='rgb_array')
