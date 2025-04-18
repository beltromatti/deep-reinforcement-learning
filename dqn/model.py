# dqn_rl/model.py
import torch
import torch.nn as nn

class NetworkBuilder:
    """Utility to build a PyTorch neural network from a list of layers."""
    def __init__(self, layers):
        """Initialize with a list of layers.

        Args:
            layers (list): List of PyTorch layers (e.g., [nn.Linear(128, 64), nn.ReLU()]).
        """
        self.layers = layers

    def build(self, state_size, action_size):
        """Build the network with input and output layers.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of actions.

        Returns:
            nn.Sequential: Constructed neural network.
        """
        return nn.Sequential(
            nn.Linear(state_size, self.layers[0].in_features),
            *self.layers,
            nn.Linear(self.layers[-1].out_features, action_size)
        )

class DQN(nn.Module):
    """Deep Q-Network (DQN) model."""
    def __init__(self, state_size, action_size, network_builder=None):
        """Initialize the DQN model.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of actions.
            network_builder (NetworkBuilder, optional): Custom network builder. Defaults to a standard architecture.
        """
        super(DQN, self).__init__()
        if network_builder is None:
            # Default architecture: two hidden layers with ReLU
            self.network = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
        else:
            self.network = network_builder.build(state_size, action_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor [batch_size, state_size].

        Returns:
            torch.Tensor: Q-values for each action [batch_size, action_size].
        """
        return self.network(x)