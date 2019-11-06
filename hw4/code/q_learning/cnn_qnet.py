import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        h, w, c = env.observation_space.shape
        n = env.action_space.n
        his = config.state_history
        self.conv1 = nn.Conv2d(c * his, 16, (8, 8), 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, (4, 4), 2)
        self.relu2 = nn.ReLU()
        # dueling dqn head
        self.advantage1 = nn.Linear(2048, 128)
        self.relu3 = nn.ReLU()
        self.advantage2 = nn.Linear(128, n)
        self.value1 = nn.Linear(2048, 128)
        self.relu4 = nn.ReLU()
        self.value2 = nn.Linear(128, 1)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        x = state.transpose(1, 3)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.reshape(state.shape[0], -1)
        # compute advantage
        a = self.advantage1(x)
        a = self.relu3(a)
        a = self.advantage2(a)
        # compute value
        v = self.value1(x)
        v = self.relu4(v)
        v = self.value2(v)
        return v + (a - a.mean())
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
