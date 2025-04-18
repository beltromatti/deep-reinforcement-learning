# dqn_rl/environment.py
import numpy as np
import torch

class EnvironmentWrapper:
    """Wrapper for generic environments compatible with DQN training.

    The environment must provide:
    - reset(): Returns initial state (numpy array or compatible).
    - step(action): Returns (next_state, reward, done, info).
    """
    def __init__(self, env):
        """Initialize the environment wrapper.

        Args:
            env: Environment object with reset() and step() methods.
        """
        self.env = env
        # Get state and action space dimensions (assuming discrete actions)
        state = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        self.state_size = len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0]
        try:
            self.action_size = self.env.action_space.n
        except AttributeError:
            raise ValueError("Environment must specify action_space.n for discrete actions.")

    def reset(self):
        """Reset the environment and return the initial state as a tensor.

        Returns:
            torch.Tensor: Initial state tensor [1, state_size].
        """
        state = self.env.reset()[0] if isinstance(self.env.reset(), tuple) else self.env.reset()
        return torch.FloatTensor(state).unsqueeze(0)

    def step(self, action):
        """Perform an action and return the result as tensors.

        Args:
            action (int): Action to perform.

        Returns:
            tuple: (next_state, reward, done, info) where next_state is a tensor.
        """
        result = self.env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done = result
            info = {}
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        return next_state, reward, done, info