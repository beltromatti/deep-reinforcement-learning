from abc import ABC, abstractmethod
import numpy as np
import torch
import gymnasium as gym

class Environment(ABC):
    """Abstract base class for reinforcement learning environments."""
    
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial state.
        
        Returns:
            torch.Tensor: Initial state tensor [1, state_size].
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """Perform an action and return the next state, reward, done, and info.
        
        Args:
            action: Action to perform (int for discrete spaces, array/tensor for continuous spaces).
        
        Returns:
            tuple: (next_state, reward, done, info) where next_state is a tensor,
                   done is True if the episode is terminated or truncated.
        """
        pass
    
    @property
    @abstractmethod
    def state_size(self):
        """Return the dimension of the flattened state space."""
        pass
    
    @property
    @abstractmethod
    def action_size(self):
        """Return the dimension of the action space (number of actions for discrete,
        flattened vector size for continuous)."""
        pass

class EnvironmentWrapper(Environment):
    """Wrapper for Gymnasium environments to conform to the Environment interface.
    
    This wrapper supports environments with discrete (Discrete) or continuous (Box) action spaces
    and Box observation spaces of any shape (e.g., 1D vectors or multidimensional arrays like images).
    Multidimensional states are flattened to 1D tensors of shape [1, state_size], where state_size
    is the total number of elements in the state (e.g., height * width * channels for images).
    For discrete action spaces, action_size is the number of actions (action_space.n).
    For continuous action spaces, action_size is the flattened vector size (prod(action_space.shape)).
    
    Args:
        gym_env: Gymnasium environment with a Discrete or Box action space.
        device (torch.device, optional): Device for tensor creation (default: CPU).
    
    Raises:
        ValueError: If the action space is not Discrete or Box, the observation space is not a Box,
                   or the observation/action space shape is invalid.
    """
    
    def __init__(self, gym_env, device=torch.device("cpu")):
        """Initialize the wrapper with a Gymnasium environment."""
        self.env = gym_env
        self.device = device
        
        # Validate and compute action size
        if isinstance(gym_env.action_space, gym.spaces.Discrete):
            self._action_size = gym_env.action_space.n
            self._action_space_shape = ()  # Scalar action for discrete
            self._is_discrete = True
        elif isinstance(gym_env.action_space, gym.spaces.Box):
            if not gym_env.action_space.shape:
                raise ValueError("Box action space shape cannot be empty")
            self._action_size = int(np.prod(gym_env.action_space.shape))
            self._action_space_shape = gym_env.action_space.shape
            self._is_discrete = False
        else:
            raise ValueError("Environment must have a Discrete or Box action space")
        
        # Validate and compute state size
        if not isinstance(gym_env.observation_space, gym.spaces.Box):
            raise ValueError("Environment must have a Box observation space")
        if not gym_env.observation_space.shape:
            raise ValueError("Observation space shape cannot be empty")
        self._state_size = int(np.prod(gym_env.observation_space.shape))
    
    def reset(self):
        """Reset the environment and return the initial state as a flattened tensor.
        
        Returns:
            torch.Tensor: Initial state tensor [1, state_size].
        
        Raises:
            ValueError: If the state is invalid (e.g., contains NaN or has incorrect shape).
        """
        result = self.env.reset()
        state = result[0] if isinstance(result, tuple) else result
        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.env.observation_space.shape:
            raise ValueError(f"Reset returned a state with shape {state.shape}, "
                             f"expected {self.env.observation_space.shape}")
        if np.any(np.isnan(state)):
            raise ValueError("Reset returned a state with NaN values")
        state_flat = state.flatten()
        return torch.tensor(state_flat, device=self.device, dtype=torch.float32).unsqueeze(0)
    
    def step(self, action):
        """Perform an action and return the result with the next state as a flattened tensor.
        
        Args:
            action: Action to perform (int for discrete spaces, array/tensor for continuous spaces).
        
        Returns:
            tuple: (next_state, reward, done, info) where next_state is a tensor,
                   done is True if the episode is terminated or truncated.
        
        Raises:
            ValueError: If the action or next state is invalid (e.g., wrong shape or contains NaN).
        """
        # Validate and format action for continuous spaces
        if not self._is_discrete:
            action = np.asarray(action, dtype=np.float32)
            if action.shape != self._action_space_shape:
                raise ValueError(f"Action shape {action.shape} does not match "
                                 f"action space shape {self._action_space_shape}")
            if np.any(np.isnan(action)):
                raise ValueError("Action contains NaN values")
        
        result = self.env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done = result
            info = {}
        
        next_state = np.asarray(next_state, dtype=np.float32)
        if next_state.shape != self.env.observation_space.shape:
            raise ValueError(f"Step returned a next_state with shape {next_state.shape}, "
                             f"expected {self.env.observation_space.shape}")
        if np.any(np.isnan(next_state)):
            raise ValueError("Step returned a next_state with NaN values")
        next_state_flat = next_state.flatten()
        next_state_tensor = torch.tensor(next_state_flat, device=self.device, dtype=torch.float32).unsqueeze(0)
        return next_state_tensor, reward, done, info
    
    @property
    def state_size(self):
        """Return the dimension of the flattened state space."""
        return self._state_size
    
    @property
    def action_size(self):
        """Return the dimension of the action space (number of actions for discrete,
        flattened vector size for continuous)."""
        return self._action_size