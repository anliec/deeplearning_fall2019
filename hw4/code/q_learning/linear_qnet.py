import numpy as np

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, env, config):
        """
        A state-action (Q) network with a single fully connected
        layer, takes the state as input and gives action values
        for all actions.
        """
        super().__init__()

        #####################################################################
        # TODO: Define a fully connected layer for the forward pass. Some
        # useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H, W, C = env.observation_space.shape
        action_count = env.action_space.n
        self.linear = nn.Linear(in_features=H * W * C * config.state_history, out_features=action_count)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        """
        Returns Q values for all actions

        Args:
            state: tensor of shape (batch, H, W, C x config.state_history)

        Returns:
            q_values: tensor of shape (batch_size, num_actions)
        """
        #####################################################################
        # TODO: Implement the forward pass, 1-2 lines.
        #####################################################################
        print(state.shape)
        if len(state.shape) == 4:
            batch, _, _, _ = state.shape
        elif len(state.shape) == 3:
            batch = 1
        else:
            raise NotImplementedError
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).cuda()
        print(type(state))
        state = state.reshape((batch, -1))
        print(state.shape)
        return self.linear(state)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
