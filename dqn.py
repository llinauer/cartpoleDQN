"""
dqn.py

Contains the definition of the Deep-Q network
"""

import torch
from torch import nn


class DQN(nn.Module):
    """ Deep-Q Network to estimate the Q-value of a state """

    def __init__(self):
        """ Constructor, define layers """
        super().__init__()

        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        """ Forward pass through the network """

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x
