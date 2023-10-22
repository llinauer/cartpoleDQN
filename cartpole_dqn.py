"""

cartpole_dqn.py

Control the Cartpole environment with the DQN-method and a MLP.

"""

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



def main():
    """ main function """
    pass
