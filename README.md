## deep-reinforcement-learning (Version: 1.0)

A modular Python package for Deep Reinforcement Learning, designed to provide a flexible and extensible framework for training reinforcement learning agents, with a focus on accessibility for students, researchers, and developers.

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
  - **`dqn/agent.py`**: Contains the `DQNAgent` class, which implements the DQN algorithm, and the `ReplayMemory` class for experience replay. The `DQNAgent` handles action selection (epsilon-greedy), memory storage, model optimization, and training loops.
  - **`dqn/config.py`**: Defines the `DQNConfig` class, which manages hyperparameters (e.g., learning rate, gamma, epsilon decay) and supports loading configurations from YAML files.

- **Package Initialization (`__init__.py`)**:
  - Exposes key classes (`Environment`, `EnvironmentWrapper`, `Model`, `DQNAgent`, `DQNConfig`) and utilities for easy access.

The package leverages **PyTorch** for neural network computations, supporting both CPU and GPU devices. It uses **Gymnasium** for environment interactions, with the `EnvironmentWrapper` ensuring compatibility with any environment that provides `reset()` and `step()` methods. The design emphasizes extensibility: new algorithms can be added as subpackages under `algorithms/`, sharing the `core` module’s utilities and interfaces.

### Key Features

- **High-Level API**: Simplifies DRL agent training with intuitive classes (`DQNAgent`, `Model`, `EnvironmentWrapper`).
- **Flexible Environment Support**: Works with Gymnasium environments and custom environments via the `Environment` interface.
- **Customizable Neural Networks**: Allows users to define network architectures using the `Model` class and layer abstractions.
- **Configurable DQN**: Supports full customization of hyperparameters (e.g., gamma, epsilon, learning rate) via `DQNConfig`.
- **Experience Replay**: Implements a fixed-size `ReplayMemory` for stable DQN training.
- **Model Persistence**: Supports saving and loading models with architecture and weights preserved.
- **Training Visualization**: Generates plots of episode rewards for performance analysis.
- **Educational Design**: Includes detailed docstrings, logging, and error handling to aid learning.
- **Extensible Structure**: Designed to support additional algorithms (e.g., PPO, Dueling DQN) as subpackages.

### DQN Algorithm Explanation

The package implements the **Deep Q-Network (DQN)** algorithm, a value-based DRL method that combines Q-learning with deep neural networks to approximate the optimal action-value function. Below is a detailed explanation of the DQN implementation in `deep_reinforcement_learning.algorithms.dqn`:

- **Core Components**:
  - **Q-Network (`Model`)**: A neural network that maps states to action values (Q-values). By default, it consists of an input layer, two hidden layers (256 and 128 neurons with ReLU activations), and an output layer matching the action space size.
  - **Target Network**: A separate copy of the Q-network, updated periodically to stabilize training by providing consistent Q-value targets.
  - **Replay Memory (`ReplayMemory`)**: A fixed-size buffer (default: 20,000 transitions) that stores experiences (state, action, reward, next_state, done). Random sampling from this buffer reduces temporal correlations, improving learning stability.

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
  - `epsilon_start`, `epsilon_end`, `epsilon_decay`: Control exploration (default: 1.0, 0.02, 0.995).
  - `learning_rate`: Adam optimizer learning rate (default: 0.0005).
  - `memory_size`: Replay memory capacity (default: 20,000).
  - `target_update`: Frequency of target network updates (default: 10).
  - `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0).

- **Features**:
  - Supports discrete action spaces via `EnvironmentWrapper`.
  - Saves model checkpoints periodically (default: every 50 episodes).
  - Generates a reward plot (`rewards_plot.png`) to visualize training progress.
  - Logs episode statistics (reward, epsilon, loss) for monitoring.

The DQN implementation is robust and suitable for environments like CartPole-v1, where it can achieve high rewards (e.g., >400) after sufficient training. Future enhancements could include prioritized experience replay, Double DQN, or Dueling DQN to improve performance.

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
    # Create the CartPole-v1 environment
    env = drl.EnvironmentWrapper(gym.make("CartPole-v1", render_mode="human"))
    
    # Define a custom neural network architecture
    layers = [
        drl.InputLayer(),
        drl.HiddenLayer(256, drl.ReLU()),
        drl.HiddenLayer(128, drl.ReLU()),
        drl.OutputLayer()
    ]
    model = drl.Model(state_size=env.state_size, action_size=env.action_size, layers=layers)
    
    # Configure training parameters
    config = drl.DQNConfig(
        episodes=500,              # Number of training episodes
        batch_size=64,             # Size of experience replay batch
        gamma=0.99,                # Discount factor
        epsilon_start=1.0,         # Initial exploration rate
        epsilon_end=0.02,          # Final exploration rate
        epsilon_decay=0.995,       # Exploration decay rate
        memory_size=20000,         # Replay memory capacity
        learning_rate=0.0005,      # Adam optimizer learning rate
        target_update=10,          # Frequency of target network updates
        max_grad_norm=1.0,         # Gradient clipping norm
        save_checkpoint_every=50,   # Save model every 50 episodes
        checkpoint_path="dqn_cartpole.model",  # Path to save model
        plot_path="cartpole_rewards.png"     # Path to save rewards plot
    )
    
    # Create and train the DQN agent
    agent = drl.DQNAgent(env, model, config)
    rewards = agent.train()
    
    # Save the trained model
    drl.save_model(model, config.checkpoint_path)
    
    # Evaluate the trained agent
    agent.run(max_steps=2000)
    
    # Print summary
    print(f"Training completed. Average reward (last 100 episodes): {sum(rewards[-100:])/100:.2f}")

if __name__ == "__main__":
    main()
```

This script:
1. Wraps the CartPole-v1 environment with `EnvironmentWrapper` to ensure compatibility.
2. Creates a custom neural network with two hidden layers (256 and 128 neurons).
3. Configures the DQN agent with specified hyperparameters.
4. Trains the agent for 500 episodes, saving checkpoints and a reward plot.
5. Saves and run the trained model.

For additional examples, see the `examples/` directory:
- `cartpole_dqn.py`: Demonstrates DQN training on CartPole-v1.
- `custom_env_dqn.py`: Shows how to use a custom environment.

## Configuration

DQN hyperparameters can be customized via the `DQNConfig` class or a YAML file. Example `config.yaml`:

```yaml
episodes: 500
batch_size: 64
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.02
epsilon_decay: 0.995
memory_size: 20000
learning_rate: 0.0005
target_update: 10
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
- Adding Proximal Policy Optimization (PPO) as a subpackage under `algorithms/ppo`.
- Implementing advanced DQN variants (e.g., Double DQN, Dueling DQN, Prioritized Experience Replay).
- Supporting continuous action spaces in `EnvironmentWrapper`.
- Integrating TensorBoard for real-time training visualization.
- Expanding example scripts and tutorials for educational use.

Contributions are welcome! See the **Contributing** section.

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