# examples/cartpole_example.py
import gymnasium as gym
from dqn.environment import EnvironmentWrapper
from dqn.trainer import DQNTrainer
from dqn.model import NetworkBuilder
import torch.nn as nn

def main():
    # Create CartPole environment
    env = EnvironmentWrapper(gym.make("CartPole-v1"))

    # Define custom network architecture
    network_builder = NetworkBuilder([
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU()
    ])

    # Configuration
    config = {
        "episodes": 500,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.995,
        "memory_size": 20000,
        "learning_rate": 0.0005,
        "target_update": 10,
        "use_prioritized_replay": True,
        "use_double_dqn": True
    }

    # Initialize and train
    trainer = DQNTrainer(env, config, network_builder=network_builder)
    trainer.train()

if __name__ == "__main__":
    main()