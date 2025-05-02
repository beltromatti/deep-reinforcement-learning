## deep-reinforcement-learning (Version: 1.1)

A modular, high-level Python package for Deep Reinforcement Learning, designed to simplify the implementation and study of DRL algorithms, offering an accessible and extensible framework for students, researchers, and developers.

## Overview

The **deep-reinforcement-learning** package is a modular, high-level Python library for implementing Deep Reinforcement Learning (DRL) algorithms. Developed by Mattia Beltrami, a Computer Science for Management student at the University of Bologna (UNIBO), this package aims to simplify the process of building, training, and experimenting with DRL agents. It is designed to be both educational and practical, offering a clear API for beginners while providing flexibility for advanced users to customize environments, models, and algorithms.

The package currently supports the Deep Q-Network (DQN) algorithm, implemented in the `deep_reinforcement_learning.algorithms.dqn` subpackage. Its modular architecture separates core components (environments, models, utilities) from algorithm-specific logic, making it easy to extend with new algorithms like Proximal Policy Optimization (PPO) or Dueling DQN in the future. The package abstracts low-level details, such as tensor management and neural network construction, while retaining full configurability for hyperparameters and network architectures.

### Technical Design

The package is built around a modular and extensible architecture, with a clear separation of concerns to ensure reusability and maintainability. Below is an overview of its key components and their roles:

- **Core Module (`deep_reinforcement_learning.core`)**:
  - **`environment.py`**: Defines the abstract `Environment` class and the concrete `EnvironmentWrapper` class. The `EnvironmentWrapper` adapts Gymnasium environments to a standardized interface, supporting both discrete and continuous action spaces. It flattens multidimensional states (e.g., images) into 1D tensors and ensures compatibility with PyTorch.
  - **`model.py`**: Implements a flexible neural network framework with `Model`, `Layer` (e.g., `InputLayer`, `HiddenLayer`, `OutputLayer`), and `ActivationFunction` (e.g., `ReLU`) classes. Users can define custom network architectures by specifying a list of layers, with a default architecture of two hidden layers (256 and 128 neurons) if none is provided.
  - **`utils.py`**: Provides utility functions for logging (`setup_logger`), saving models (`save_model`), and loading models (`load_model`). These utilities ensure consistent logging and model persistence across the package.

- **Algorithms Module (`deep_reinforcement_learning.algorithms`)**:
  - **`dqn/agent.py`**: Contains the `DQNAgent` class, which implements the DQN algorithm, and the `ReplayMemory`, `PrioritizedReplayMemory`, `SumTree` classes for the different types of experience replay (normal or PER). The `DQNAgent` handles action selection (epsilon-greedy), memory storage, model optimization, and training loops.
  - **`dqn/config.py`**: Defines the `DQNConfig` class, which manages hyperparameters (e.g., learning rate, gamma, epsilon decay) and supports loading configurations from YAML files.

- **Package Initialization (`__init__.py`)**:
  - Exposes key classes (`Environment`, `EnvironmentWrapper`, `Model`, `DQNAgent`, `DQNConfig`) and utilities for easy access.

The package leverages **PyTorch** for neural network computations, supporting both CPU and GPU devices. It uses **Gymnasium** for environment interactions, with the `EnvironmentWrapper` ensuring compatibility with any environment that provides `reset()` and `step()` methods. The design emphasizes extensibility: new algorithms can be added as subpackages under `algorithms/`, sharing the `core` module’s utilities and interfaces.

### Key Features

- **High-Level API**: Simplifies DRL agent training with intuitive classes (`DQNAgent`, `Model`, `EnvironmentWrapper`).
- **Flexible Environment Support**: Works with Gymnasium environments and custom environments via the `Environment` interface.
- **Customizable Neural Networks**: Allows users to define network architectures using the `Model` class and layer abstractions.
- **Configurable DQN**: Supports full customization of hyperparameters (e.g., gamma, epsilon, learning rate) via `DQNConfig`.
- **Model Persistence**: Supports saving and loading models with architecture and weights preserved.
- **Training Visualization**: Generates plots of episode rewards for performance analysis.
- **Educational Design**: Includes detailed docstrings, logging, and error handling to aid learning.
- **Extensible Structure**: Designed to support additional algorithms (e.g., PPO, Dueling DQN) as subpackages.

### DQN Algorithm Explanation

The package implements the **Deep Q-Network (DQN)** algorithm, a value-based DRL method that combines Q-learning with deep neural networks to approximate the optimal action-value function. Below is a detailed explanation of the DQN implementation in `deep_reinforcement_learning.algorithms.dqn`:

- **Core Components**:
  - **Q-Network (`Model`)**: A neural network that maps states to action values (Q-values). By default, it consists of an input layer, two hidden layers (256 and 128 neurons with ReLU activations), and an output layer matching the action space size.
  - **Target Network**: A separate copy of the Q-network, updated periodically to stabilize training by providing consistent Q-value targets.
  - **Replay Memory (`ReplayMemory`)**: A fixed-size buffer (default: 20,000 transitions) that stores experiences (state, action, reward, next_state, done). Random sampling from this buffer reduces temporal correlations, improving learning stability (Added support for Prioritized Experience Replay (PER) that gives more importance to more informative transitions that leads to a faster convergence).

- **Training Process**:
  - **Epsilon-Greedy Policy**: The agent selects actions using an epsilon-greedy strategy, balancing exploration (random actions) and exploitation (greedy actions based on Q-values). Epsilon decays over time (default: from 1.0 to 0.02) to favor exploitation as training progresses.
  - **Experience Collection**: At each step, the agent interacts with the environment, storing transitions in the replay memory.
  - **Optimization**: The agent samples a batch of transitions (default: 64) and optimizes the Q-network using the Adam optimizer and mean squared error (MSE) loss. The loss is computed as the difference between predicted Q-values and target Q-values, derived from the Bellman equation:  
    `Q(s, a) = r + γ * max(Q(s', a')) * (1 - done)`, where `γ` is the discount factor (default: 0.99).
  - **Target Network Update**: The target network’s weights are updated every `target_update` episodes (default: 10) to stabilize training.
  - **Gradient Clipping**: Gradients are clipped (default: max norm 1.0) to prevent exploding gradients.

- **Key Hyperparameters** (configurable via `DQNConfig`):
  - `episodes`: Number of training episodes (default: 1000).
  - `batch_size`: Number of transitions sampled per optimization step (default: 64).
  - `gamma`: Discount factor for future rewards (default: 0.99).
  - `epsilon_start`, `epsilon_end`, `epsilon_decay_mode`, `epsilon_exponential_decay`: Control exploration (default: 1.0, 0.02, 'exponential', 0.995).
  - `learning_rate`: Adam optimizer learning rate (default: 0.0005).
  - `memory_size`: Replay memory capacity (default: 20,000).
  - `target_update`: Frequency of target network updates (default: 10).
  - `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0).

- **Features**:
  - Supports discrete action spaces via `EnvironmentWrapper`.
  - Saves model checkpoints periodically (default: every 50 episodes).
  - Generates a reward plot (`rewards_plot.png`) to visualize training progress.
  - Logs episode statistics (reward, epsilon, loss) for monitoring.

The DQN implementation is robust and suitable for environments like CartPole-v1, where it can achieve high rewards (e.g., >400) after sufficient training. Future enhancements could include Double DQN, Dueling DQN and other features to improve performance.

## Installation

Install the package via pip:
```bash
pip install deep-reinforcement-learning
```

For development, clone the repository and install locally:
```bash
git clone https://github.com/beltromatti/deep-reinforcement-learning.git
cd deep-reinforcement-learning
pip install -e .
```

## Requirements

- Python >= 3.8
- torch >= 2.0.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- pyyaml >= 6.0
- gymnasium >= 0.29.1 (optional, for Gymnasium environments)

See `requirements.txt` for a complete list of dependencies.

## Usage

Below is an example of training a DQN agent on the CartPole-v1 environment using the package’s high-level API:

```python
import gymnasium as gym
import deep_reinforcement_learning as drl

def main():
    """
    Main function to set up, train, and evaluate a DQN agent on CartPole-v1.
    """
    # Create the CartPole-v1 environment with state normalization
    # State bounds are defined to normalize observations, ensuring stable training
    state_min = [-4.8, -3.5, -0.418, -3.5]  # Min bounds for [cart pos, cart vel, pole angle, pole vel]
    state_max = [4.8, 3.5, 0.418, 3.5]       # Max bounds
    env = drl.EnvironmentWrapper(
        gym.make("CartPole-v1"),
        truncate_episode=False,               # Continue episodes until environment termination
        state_max_values=state_max,          # Upper bounds for state normalization
        state_min_values=state_min           # Lower bounds for state normalization
    )
    
    # Define a neural network architecture for the DQN
    # The network maps states (4D input) to Q-values for each action (2 outputs)
    layers = [
        drl.InputLayer(),                    # Input layer (automatically sized to state_size)
        drl.HiddenLayer(256, drl.ReLU()),    # 256 neurons with ReLU activation
        drl.HiddenLayer(128, drl.ReLU()),    # 128 neurons with ReLU activation
        drl.OutputLayer()                    # Output layer (automatically sized to action_size)
    ]
    model = drl.Model(
        state_size=env.state_size,           # 4 (state dimensions)
        action_size=env.action_size,         # 2 (left or right)
        layers=layers
    )

    # Initialize weights using He initialization (suitable for ReLU activations)
    model.init_weights(mode='he_relu')
    
    # Configure DQN hyperparameters
    # These parameters control exploration, learning, and training stability
    config = drl.DQNConfig(
        episodes=800,                        # Train for 800 episodes
        batch_size=64,                       # Sample 64 experiences per training step
        gamma=0.99,                          # Discount factor for future rewards
        epsilon_start=1.0,                   # Start with full exploration
        epsilon_end=0.02,                    # End with minimal exploration
        epsilon_decay_mode='exponential',    # Decay epsilon exponentially
        epsilon_exponential_decay=0.995,     # Decay factor per episode
        memory_size=10000,                   # Store up to 10,000 experiences
        learning_rate=0.0005,                # Learning rate for Adam optimizer
        target_update=1000,                  # Update target network every 1000 steps
        max_grad_norm=1.0,                   # Clip gradients to avoid large updates
        save_checkpoint_every=50,            # Save model every 50 episodes
        checkpoint_path="dqn_cartpole.model",# Save model to this file
        plot_path="cartpole_rewards.png"     # Save reward plot to this file
    )

    # Create and train the DQN agent
    # The agent learns by interacting with the environment and updating the model
    agent = drl.DQNAgent(env, model, config)
    rewards = agent.train()                  # Returns list of episode rewards
    
    # Save the trained model for future use
    drl.save_model(model, config.checkpoint_path)
    
    # Evaluate the trained agent
    # Run the agent for up to 2000 steps with no exploration (greedy policy)
    agent.run(max_steps=2000)
    
    # Print a summary of training performance
    avg_reward = sum(rewards[-100:]) / 100
    print(f"Training completed. Average reward (last 100 episodes): {avg_reward:.2f}")

if __name__ == "__main__":
    main()
```

This script:
1. Wraps the CartPole-v1 environment with `EnvironmentWrapper` to ensure compatibility.
2. Creates a custom neural network with two hidden layers (256 and 128 neurons).
3. Configures the DQN agent with specified hyperparameters.
4. Trains the agent for 800 episodes, saving checkpoints and a reward plot.
5. Saves and run the trained model.

For additional examples, see the `examples/` directory:
- `cartpole_dqn.py`: Demonstrates DQN training on CartPole-v1.
- `lunarlander_dqn_per.py`: Shows how to use PER (Prioritized Experience Replay) on LunarLander-v3.

## Configuration

DQN hyperparameters can be customized via the `DQNConfig` class or a YAML file. Example `config.yaml`:

```yaml
episodes: 1000
batch_size: 64
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.02
epsilon_decay_mode: 'exponential'
epsilon_exponential_decay: 0.995
memory_size: 20000
learning_rate: 0.0005
target_update: 1000
max_grad_norm: 1.0
save_checkpoint_every: 50
checkpoint_path: "dqn_checkpoint.model"
plot_path: "rewards_plot.png"
```

Load the configuration:

```python
from deep_reinforcement_learning.algorithms.dqn.config import DQNConfig
config = DQNConfig()
config.load_yaml_config("config.yaml")
```

## Customization

The package supports extensive customization:
- **Environments**: Use any environment with `reset()` and `step()` methods. The `EnvironmentWrapper` handles state/action conversions.
- **Neural Networks**: Define custom architectures by passing a list of layers to `Model`. Example:
  ```python
  from deep_reinforcement_learning import Model, InputLayer, HiddenLayer, OutputLayer, ReLU
  model = Model(
      state_size=env_wrapper.state_size,
      action_size=env_wrapper.action_size,
      layers=[InputLayer(), HiddenLayer(128, ReLU()), HiddenLayer(64, ReLU()), OutputLayer()]
  )
  ```
- **Hyperparameters**: Adjust all DQN parameters via `DQNConfig`.

## Project Structure

```
deep-reinforcement-learning/
├── deep_reinforcement_learning/
│   ├── algorithms/
│   │   ├── dqn/
│   │   │   ├── agent.py        # DQNAgent and ReplayMemory
│   │   │   ├── config.py       # DQNConfig
│   ├── core/
│   │   ├── environment.py      # Environment and EnvironmentWrapper
│   │   ├── model.py            # Model and layer classes
│   │   ├── utils.py            # Logging and model persistence
│   ├── __init__.py             # Package initialization
├── examples/                   # Example scripts
├── tests/                      # Unit tests
├── models/                     # Trained and Ready-To-Use models for various environments
├── README.md                   # Documentation
├── LICENSE                     # GNU GPLv3 License
├── requirements.txt            # Pip requirements
├── MANIFEST.in                 # Adds non-python files to the package
├── .gitignore                  # Don't track some useless files to github
├── pyproject.toml              # Installation script for modern packages
├── setup.py                    # Installation script
```

## Future Directions

Planned enhancements include:
- Implementing advanced DQN variants (e.g., Double DQN, Dueling DQN...).
- Adding Proximal Policy Optimization (PPO) as a subpackage under `algorithms/ppo`.
- Integrating TensorBoard for real-time training visualization.
- Expanding example scripts and tutorials for educational use.

Contributions are welcome! See the **Contributing** section.

## Release Notes

### Version 1.1 (May 2025)

- **Added Support for Prioritized Experience Replay (PER) in DQN**: The `deep_reinforcement_learning.algorithms.dqn` subpackage now includes support for Prioritized Experience Replay (PER), enhancing the DQN algorithm's learning efficiency. The `PrioritizedReplayMemory` and `SumTree` classes have been added to prioritize transitions with higher temporal-difference (TD) errors, allowing the agent to focus on more informative experiences during training. This leads to faster convergence and improved performance in complex environments.
  
  **What is PER?**  
  Prioritized Experience Replay (PER) is an advanced replay memory strategy that assigns priorities to stored transitions based on their TD error, which measures the difference between predicted and actual Q-values. Transitions with higher TD errors are sampled more frequently, as they indicate areas where the agent can learn the most. The `SumTree` data structure efficiently manages these priorities, ensuring low computational overhead. Users can enable PER by configuring the `DQNAgent` with the `PrioritizedReplayMemory` option, as demonstrated in the `lunarlander_dqn_per.py` example.

## Use of Generative AI

This project used **Grok 3**, developed by xAI, to assist in generating documentation, code comments, and structural suggestions. All AI-generated content was thoroughly reviewed and validated by Mattia Beltrami to ensure accuracy and alignment with the package’s goals.

## Author

Developed by **Mattia Beltrami**, a Computer Science for Management student at the University of Bologna (UNIBO). Email: [mattia.beltrami@studio.unibo.it](mailto:mattia.beltrami@studio.unibo.it). GitHub: [beltromatti](https://github.com/beltromatti).

## Contributing

To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please include tests and follow the project’s code style.

## License

Licensed under the GNU General Public License v3 (GPLv3). See the `LICENSE` file.

## Contact

For questions or feedback, contact Mattia Beltrami at [mattia.beltrami@studio.unibo.it](mailto:mattia.beltrami@studio.unibo.it) or open an issue on the [GitHub repository](https://github.com/beltromatti/deep-reinforcement-learning).