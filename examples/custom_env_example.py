# examples/custom_env_example.py
from dqn.environment import EnvironmentWrapper
from dqn.trainer import DQNTrainer
import numpy as np

class CustomEnv:
    """Simple custom environment for testing."""
    def __init__(self):
        self.state = np.zeros(4)
        self.action_space = type("ActionSpace", (), {"n": 2})()
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        self.state = np.random.random(4)
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        next_state = self.state + np.random.random(4) * 0.1
        reward = 1.0 if action == 0 else 0.5
        done = self.step_count >= self.max_steps
        self.state = next_state
        return next_state, reward, done, {}

def main():
    env = EnvironmentWrapper(CustomEnv())
    config = {
        "episodes": 200,
        "batch_size": 32,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.99,
        "memory_size": 10000,
        "learning_rate": 0.001,
        "target_update": 5
    }
    trainer = DQNTrainer(env, config)
    trainer.train()

if __name__ == "__main__":
    main()