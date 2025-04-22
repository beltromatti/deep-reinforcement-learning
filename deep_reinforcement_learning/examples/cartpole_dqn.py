# deep_reinforcement_learning/examples/cartpole_dqn.py
"""
Example script to train a DQN agent on the CartPole-v1 environment using the deep-reinforcement-learning package.

This script demonstrates how to:
- Set up a Gymnasium environment with EnvironmentWrapper.
- Define a custom neural network architecture for the DQN model.
- Configure training parameters using the DQNConfig class.
- Train a DQN agent and save the model.
- Evaluate the trained agent and visualize the training rewards.

Usage:
    python -m deep_reinforcement_learning.examples.cartpole_dqn
"""
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