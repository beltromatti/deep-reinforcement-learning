# deep_reinforcement_learning/__init__.py
"""
Deep Reinforcement Learning: A modular Python package for training reinforcement learning agents.
Provides a high-level API for DQN and supports extension to other algorithms.
"""
from .core.environment import Environment, EnvironmentWrapper
from .core.model import Model, InputLayer, HiddenLayer, OutputLayer, ReLU
from .core.utils import setup_logger, save_model, load_model
from .algorithms.dqn.agent import DQNAgent
from .algorithms.dqn.config import DQNConfig

__version__ = "0.1.0"