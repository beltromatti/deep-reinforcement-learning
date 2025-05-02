# deep_reinforcement_learning/examples/cartpole_dqn.py
"""
Educational example script to train a Deep Q-Network (DQN) agent on the CartPole-v1 environment using the deep-reinforcement-learning package.

**Purpose**:
This script serves as a hands-on tutorial to demonstrate the implementation of a DQN algorithm for reinforcement learning. It guides users through:
- Setting up a Gymnasium environment with a custom wrapper for state normalization.
- Designing a neural network architecture for the DQN model.
- Configuring DQN hyperparameters using the DQNConfig class.
- Training a DQN agent, saving the model, and evaluating its performance.
- Visualizing training progress with reward plots.

**What is DQN?**
Deep Q-Networks (DQN) is a reinforcement learning algorithm that combines Q-learning with deep neural networks to approximate the optimal action-value function. It uses experience replay and a target network to stabilize training, making it suitable for environments with discrete action spaces like CartPole-v1.

**CartPole-v1 Environment**:
In CartPole-v1 (from Gymnasium), the agent balances a pole on a cart by applying left or right forces. The state consists of four variables (cart position, cart velocity, pole angle, pole angular velocity), and the agent chooses between two actions (push left or right). The episode ends if the pole falls beyond a threshold, the cart moves too far, or the maximum steps are reached. The goal is to maximize the cumulative reward (1 per step, up to 500).

**Learning Objectives**:
- Understand how to set up and preprocess a reinforcement learning environment.
- Learn to define a neural network for Q-value approximation.
- Explore key DQN hyperparameters and their impact on training.
- Practice training, saving, and evaluating a DQN agent.
- Visualize and interpret training performance.

**Usage**:
Run the script from the command line:
    python -m deep_reinforcement_learning.examples.cartpole_dqn

**Prerequisites**:
- Install required packages: `gymnasium`, `deep-reinforcement-learning` (e.g., via `pip install gymnasium deep-reinforcement-learning`).
- Basic understanding of Python, reinforcement learning concepts, and neural networks.

**Output**:
- A trained DQN model saved to `dqn_cartpole.model`.
- A plot of training rewards saved to `cartpole_rewards.png`.
- Console output summarizing the average reward over the last 100 episodes.
"""
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