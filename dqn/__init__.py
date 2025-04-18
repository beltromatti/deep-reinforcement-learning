# dqn/__init__.py
"""
DQN: A modular Python package for training Deep Q-Network (DQN) agents on generic environments.
"""
from .environment import EnvironmentWrapper
from .model import DQN, NetworkBuilder
from .memory import ReplayMemory, PrioritizedReplayMemory
from .policy import EpsilonGreedyPolicy
from .trainer import DQNTrainer
from .config import load_config

__version__ = "0.1.0"