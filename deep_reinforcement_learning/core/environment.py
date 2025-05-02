from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Union, List, Tuple
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
    
    def __init__(self, gym_env, device: torch.device = torch.device("cpu"), 
                 truncate_episode: bool = True, 
                 state_max_values: Optional[Union[List, Tuple, np.ndarray]] = None, 
                 state_min_values: Optional[Union[List, Tuple, np.ndarray]] = None, 
                 norm_range: Tuple[float, float] = (0, 1)):
        """Initialize the wrapper with a Gymnasium environment.
        
        Args:
            gym_env: Gymnasium environment with a Discrete or Box action space.
            device (torch.device, optional): Device for tensor creation (default: CPU).
            truncate_episode (bool, optional): If True, episodes are considered done on truncation.
            state_max_values (list, tuple, or np.ndarray, optional): Maximum values for state normalization.
            state_min_values (list, tuple, or np.ndarray, optional): Minimum values for state normalization.
            norm_range (tuple, optional): Output range for normalization (default: (0, 1)).
        
        Raises:
            ValueError: If action/observation spaces are invalid, or normalization parameters are incorrect.
            TypeError: If state_max_values or state_min_values cannot be converted to np.ndarray.
        """
        self.env = gym_env
        self.device = device
        self.truncate_episode = truncate_episode
        
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

        # Validate and set state normalizer parameters
        if (state_max_values is None) != (state_min_values is None):
            raise ValueError("Both state_max_values and state_min_values must be provided for normalization, or neither")
        elif state_max_values is not None and state_min_values is not None:
            try:
                # Convert to np.ndarray with float32 dtype
                state_max_values = np.asarray(state_max_values, dtype=np.float32).flatten()
                state_min_values = np.asarray(state_min_values, dtype=np.float32).flatten()
            except (TypeError, ValueError) as e:
                raise TypeError("state_max_values and state_min_values must be convertible to np.ndarray") from e
            
            if state_max_values.size != self._state_size:
                raise ValueError(f"state_max_values must have size {self._state_size}, got {state_max_values.size}")
            if state_min_values.size != self._state_size:
                raise ValueError(f"state_min_values must have size {self._state_size}, got {state_min_values.size}")
            if np.any(state_max_values <= state_min_values):
                raise ValueError("state_max_values must be greater than state_min_values for all dimensions")
            
            self.state_max = state_max_values
            self.state_min = state_min_values
            self.norm_min, self.norm_max = norm_range
            self.normalize_state = True
        else:
            self.normalize_state = False

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
        if self.normalize_state:
            state_flat = self._normalize_state(state_flat)
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
            done = terminated or truncated if self.truncate_episode else terminated
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
        if self.normalize_state:
            next_state_flat = self._normalize_state(next_state_flat)
        next_state_tensor = torch.tensor(next_state_flat, device=self.device, dtype=torch.float32).unsqueeze(0)
        return next_state_tensor, reward, done, info
    
    def _normalize_state(self, state):
        denominator = self.state_max - self.state_min
        safe_denominator = np.where(denominator != 0, denominator, 1.0)
        state_norm = (state - self.state_min) / safe_denominator
        # Map in [norm_min, norm_max]
        state_norm = state_norm * (self.norm_max - self.norm_min) + self.norm_min
        state_norm = np.where(denominator != 0, state_norm, state)
        return state_norm
    
    @property
    def state_size(self):
        """Return the dimension of the flattened state space."""
        return self._state_size
    
    @property
    def action_size(self):
        """Return the dimension of the action space (number of actions for discrete,
        flattened vector size for continuous)."""
        return self._action_size