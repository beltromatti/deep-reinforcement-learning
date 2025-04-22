## deep-reinforcement-learning

A modular, high-level Python package for Deep Reinforcement Learning, designed to simplify the implementation and study of DRL algorithms for students, researchers, and enthusiasts.

## Overview

The **deep-reinforcement-learning** package provides a flexible and intuitive API for training and experimenting with Deep Reinforcement Learning **(DRL**) algorithms. Developed by Mattia Beltrami, a student of Computer Science for Management at the University of Bologna **(UNIBO**), this package aims to make DRL accessible to a wide audience, including students learning the foundations of reinforcement learning, researchers prototyping new algorithms, and developers building intelligent agents.

The package is structured to be highly modular, with a focus on reusability and customization. It currently supports the Deep Q-Network **(DQN**) algorithm, implemented as a subpackage under **deep_reinforcement_learning.algorithms.dqn**. The design allows for easy integration of additional algorithms, such as Proximal Policy Optimization **(PPO**), in future releases. By providing a high-level API, the package abstracts complex implementation details, enabling users to focus on understanding DRL concepts and experimenting with custom environments and network architectures.

## Key Features

* **High-Level API**: Simplifies the training of DRL agents with intuitive interfaces for environments, models, and trainers.  
* **Generic Environment Support**: Compatible with any environment providing **reset**()** and **step**()** methods, not limited to Gymnasium.  
* **Modular Design**: Organized into subpackages **(**dqn**, **core**)** for easy maintenance and extension to other algorithms **(e.g., PPO**).  
* **Configurable DQN**: Fully configurable parameters **(gamma, epsilon, learning rate, etc.**)** to adjust training precision.
* **Customizable Network Architecture**: An easy way to build a Deep Neural Network via PyTorch high level wrapper.
* **Advanced Features**: Supports prioritized experience replay, Double DQN, and checkpointing for robust training.  
* **Educational Focus**: Includes detailed documentation, examples, and logging to facilitate learning and experimentation.  
* **Extensibility**: Structured to accommodate new algorithms as subpackages under **algorithms/**.

## Project Logic and Motivation

Deep Reinforcement Learning combines reinforcement learning principles with deep neural networks to solve complex sequential decision-making problems, such as game playing, robotics, and autonomous systems. However, implementing DRL algorithms from scratch can be daunting due to the need to manage environments, neural networks, exploration strategies, and training loops. The **deep-reinforcement-learning** package addresses this challenge by providing a **unified, high-level framework** that abstracts low-level details while retaining flexibility.

The core logic of the package is built around modularity and reusability:
- **Environment Wrapper** **(**core.environment.GenericEnvironmentWrapper**)**: Ensures compatibility with any environment that provides a standard interface **(**reset**()** for initial states and **step**()** for state transitions, rewards, and termination flags**)**. This allows users to experiment with custom environments beyond standard Gymnasium ones.  
- **Algorithm Subpackages** **(**algorithms.dqn**, **algorithms.ppo**)**: Each algorithm is implemented as a self-contained subpackage, with dedicated modules for models, policies, memory, and trainers. This structure isolates algorithm-specific logic, making it easy to add new algorithms like PPO.  
- **Core Utilities** **(**core.config**, **core.utils**)**: Shared functionality, such as configuration management and logging, reduces code duplication and ensures consistency across algorithms.  
- **Customizable Models** **(**algorithms.dqn.model.NetworkBuilder**)**: A PyTorch-based wrapper allows users to define neural network architectures with a simple list of layers **(e.g., **[Linear**(**128, 64**)**, ReLU**()**]**)**, enabling rapid experimentation.  
- **Training Pipeline** **(**algorithms.dqn.trainer.DQNTrainer**)**: A high-level trainer class orchestrates the training process, handling exploration, optimization, and checkpointing, with configurable parameters for fine-tuning.

The package is designed with **education in mind**. By providing clear documentation, example scripts, and modular code, it serves as a learning tool for students exploring DRL concepts like Q-learning, experience replay, and epsilon-greedy policies. Simultaneously, its flexibility and advanced features **(e.g., prioritized replay, Double DQN**)** make it suitable for researchers prototyping new ideas or testing algorithms on custom environments.

## Installation

Install the package via pip:
*`*`*bash
pip install deep-reinforcement-learning
*`*`*

Or install from source for development:
*`*`*bash
git clone https://github.com/your-username/deep-reinforcement-learning.git
cd deep-reinforcement-learning
pip install -e .
*`*`*

## Requirements

* **Python >= 3.8**  
* **torch >= 2.0.0**  
* **numpy >= 1.21.0**  
* **matplotlib >= 3.5.0**  
* **pyyaml >= 6.0**  
* **gymnasium >= 0.29.1 **(optional, for Gymnasium environments**)**

See **requirements.txt** for a full list of dependencies.

## Usage

The package provides a high-level interface for training DRL agents. Below is an example of training a DQN agent on the CartPole-v1 environment from Gymnasium:

*`*`*python
import gymnasium as gym
from deep_reinforcement_learning.algorithms.dqn import EnvironmentWrapper, DQNTrainer, NetworkBuilder
import torch.nn as nn

# Create environment
env = EnvironmentWrapper**(**gym.make**(**"CartPole-v1"**)**)

# Define custom network architecture
network_builder = NetworkBuilder**(**[
    nn.Linear**(**128, 64**)**,
    nn.ReLU**(**)**,
    nn.Linear**(**64, 32**)**,
    nn.ReLU**(**)**
]**)

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
    "use_double_dqn": True,
    "save_checkpoint_every": 50,
    "plot_path": "rewards_plot.png"
}

# Train the agent
trainer = DQNTrainer**(**env, config, network_builder=network_builder**)
trainer.train**(**)**
*`*`*

This script trains a DQN agent with a custom neural network, saves checkpoints, and plots the training rewards. The resulting plot is saved as **rewards_plot.png**.

For more examples, including how to use custom environments, see the **examples/** directory:
- **cartpole_dqn.py**: Demonstrates DQN on CartPole-v1.
- **custom_env_dqn.py**: Shows how to use DQN with a user-defined environment.

## Configuration

All DQN parameters are customizable via a configuration dictionary or a YAML file. Example **config.yaml**:
*`*`*yaml
episodes: 500
batch_size: 64
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.02
epsilon_decay: 0.995
memory_size: 20000
learning_rate: 0.0005
target_update: 10
use_prioritized_replay: true
per_alpha: 0.6
per_beta: 0.4
use_double_dqn: true
max_grad_norm: 1.0
checkpoint_path: "dqn_checkpoint.pth"
plot_path: "rewards_plot.png"
*`*`*

Load the configuration:
*`*`*python
from deep_reinforcement_learning.core.config import load_config
config = load_config**(**"config.yaml"**)**
*`*`*

## Customization

The package is designed for maximum flexibility:
- **Environments**: Use any environment with **reset**()** and **step**()** methods. The **EnvironmentWrapper** handles state and action conversions automatically.  
- **Network Architecture**: Define custom DQN architectures using **NetworkBuilder**. For example:
  *`*`*python
  network_builder = NetworkBuilder**(**[nn.Linear**(**256, 128**)**, nn.ReLU**(**)**, nn.Linear**(**128, 64**)**, nn.ReLU**(**)**]**)**
  *`*`*
- **Parameters**: Adjust all DQN parameters **(e.g., **gamma**, **epsilon_decay**, **learning_rate**)** via the configuration.  
- **Advanced Features**: Enable prioritized experience replay or Double DQN by setting **use_prioritized_replay** or **use_double_dqn** to **true**.

## Project Structure

The package is organized for modularity and extensibility:
*`*`*
deep-reinforcement-learning/
├── deep_reinforcement_learning/
│   ├── algorithms/
│   │   ├── dqn/                # DQN implementation
│   │   ├── ppo/                # Placeholder for PPO **(future**)
│   ├── core/                   # Shared utilities **(config, logging, environment wrapper**)
├── examples/                   # Example scripts
├── tests/                      # Unit tests
├── README.md                   # Documentation
├── LICENSE                     # MIT License
├── pyproject.toml              # Build configuration
└── setup.py                    # Installation script
*`*`*

## Future Directions

The package is designed to grow with the field of DRL. Planned enhancements include:
- Implementation of Proximal Policy Optimization **(PPO**)** as a new subpackage under **algorithms/ppo**.  
- Support for Dueling DQN and Noisy Networks for improved exploration.  
- Integration with TensorBoard for real-time visualization of training metrics.  
- Multi-step learning and continuous action spaces for broader applicability.  
- Additional example environments and tutorials for educational purposes.

Contributions are welcome! See the **Contributing** section for details.

## Use of Generative AI

This project leveraged generative artificial intelligence, specifically Grok 3 developed by xAI, to assist in generating comments, documentation, and code structure suggestions. The AI was used to create clear, detailed, and educational content, enhancing the project's accessibility. However, every line of code, comment, and documentation was carefully reviewed and validated by Mattia Beltrami to ensure accuracy and correctness.

## Author

Developed by **Mattia Beltrami**, a student of Computer Science for Management at the University of Bologna **(UNIBO**). This project was created as part of an educational effort to explore and share knowledge about Deep Reinforcement Learning.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch **(**git checkout -b feature/your-feature**)**.
3. Commit your changes **(**git commit -m "Add your feature"**)**.
4. Push to the branch **(**git push origin feature/your-feature**)**.
5. Open a Pull Request.

Please include tests for new features and follow the code style used in the project.

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the **LICENSE** file for details.

## Contact

For questions or feedback, contact Mattia Beltrami at **[mattia.beltrami@studio.unibo.it**]**(**mailto:mattia.beltrami@studio.unibo.it**)** or open an issue on the **[GitHub repository**]**(**https://github.com/beltromatti/deep-reinforcement-learning**)**.