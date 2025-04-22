import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from ...core.utils import setup_logger, save_model
from ...core.model import Model
from ...core.environment import Environment
from .config import DQNConfig

# The `DQNAgent` class implements a Deep Q-Network (DQN) agent for reinforcement learning.
# It uses a neural network to approximate the Q-value function and learns optimal actions through
# an epsilon-greedy policy, experience replay, and periodic target network updates.
# The agent interacts with a provided environment, optimizes its model based on sampled experiences,
# and supports training and evaluation phases.
class DQNAgent:
    """Deep Q-Network (DQN) agent for reinforcement learning."""
    
    def __init__(self, env: Environment, model: Model, config=DQNConfig()):
        """Initialize the DQN agent.
        
        Args:
            env: Environment object (subclass of Environment).
                The environment provides the state space, action space, and step function for
                agent-environment interaction (e.g., CartPole-v1 from Gymnasium).
            model: Model object (subclass of nn.Module).
                The neural network used to approximate the Q-value function, mapping states to
                action values. It must be compatible with the environment's state and action sizes.
            config (DQNConfig): Configuration parameters object.
                Contains hyperparameters such as learning rate, gamma, epsilon values, memory size,
                batch size, and paths for saving models and plots.
        
        Explanation:
            - Initializes the agent by setting up the environment, model, and configuration.
            - Determines the device (GPU or CPU) for computation using PyTorch.
            - Moves the model to the selected device for efficient processing.
            - Creates a target model, a copy of the main model, used to stabilize training by
              providing consistent Q-value targets.
            - Initializes an Adam optimizer for updating the model's parameters.
            - Sets up a replay memory to store experiences (state, action, reward, next_state, done).
            - Initializes the exploration parameter (epsilon) for the epsilon-greedy policy.
            - Configures a logger for tracking training progress and debugging.
        """
        self.env = env
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize target model
        self.target_model = type(model)(env.state_size, env.action_size, model.layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Initialize optimizer and memory
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.memory = ReplayMemory(config.memory_size)
        
        # Initialize exploration parameters
        self.epsilon = config.epsilon_start
        self.logger = setup_logger(__name__)
    
    def choose_action(self, state):
        """Select an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state [1, state_size].
                A tensor representing the current observation from the environment, with shape
                [1, state_size] (batch dimension included for compatibility with the model).
        
        Returns:
            int: Selected action.
                An integer representing the action chosen from the environment's action space.
        
        Explanation:
            - Implements an epsilon-greedy policy to balance exploration and exploitation.
            - With probability (1 - epsilon), the agent selects the action with the highest
              Q-value predicted by the model (greedy action).
            - With probability epsilon, the agent selects a random action to explore the environment.
            - The state is moved to the appropriate device (GPU/CPU) before processing.
            - Uses `torch.no_grad()` to disable gradient computation during inference, improving efficiency.
            - Returns the index of the selected action.
        """
        state = state.to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.model(state)
                return q_values.argmax().item()
        return random.randrange(self.env.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory.
        
        Args:
            state (torch.Tensor): Current state [1, state_size].
                The state before taking the action.
            action (int): Action taken.
                The action chosen by the agent.
            reward (float): Reward received.
                The reward provided by the environment after taking the action.
            next_state (torch.Tensor): Next state [1, state_size].
                The state observed after taking the action.
            done (bool): Whether the episode is done.
                Indicates if the episode terminated (e.g., the agent failed or reached a goal).
        
        Explanation:
            - Stores a transition tuple (state, action, reward, next_state, done) in the replay memory.
            - The replay memory is used to sample experiences for training, enabling the agent to learn
              from past interactions and break temporal correlations in the data.
            - Transitions are stored in a `ReplayMemory` object, which has a fixed capacity and
              automatically discards old transitions when full.
        """
        self.memory.push((state, action, reward, next_state, done))
    
    def optimize_model(self):
        """Optimize the model using a batch of transitions.
        
        Returns:
            float: Loss value for the batch (or 0.0 if not enough transitions).
                The mean squared error loss computed for the batch, used for monitoring training.
        
        Explanation:
            - Performs a single optimization step by sampling a batch of transitions from the replay memory.
            - Returns 0.0 if there are insufficient transitions (less than batch_size) to form a batch.
            - Extracts components (states, actions, rewards, next_states, done flags) from the sampled
              transitions and converts them into tensors for batch processing.
            - Computes the current Q-values for the chosen actions using the main model.
            - Computes the target Q-values using the target model, applying the Bellman equation:
              target = reward + gamma * max(next_Q_values) * (1 - done).
              The `done` flag ensures that terminal states have no future value.
            - Calculates the mean squared error (MSE) loss between current and target Q-values.
            - Performs backpropagation to compute gradients, clips gradients to prevent exploding
              gradients, and updates the model's parameters using the Adam optimizer.
            - Returns the loss value for logging and monitoring.
        """
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        transitions = self.memory.sample(self.config.batch_size)
        
        # Extract components
        batch_state = torch.cat([t[0] for t in transitions]).to(self.device)
        batch_action = torch.LongTensor([t[1] for t in transitions]).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        batch_next_state = torch.cat([t[3] for t in transitions]).to(self.device)
        batch_done = torch.FloatTensor([t[4] for t in transitions]).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.model(batch_state).gather(1, batch_action).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(batch_next_state).max(1)[0]
            target_q_values = batch_reward + (self.config.gamma * next_q_values * (1 - batch_done))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, start_episode=0, episode_rewards=None):
        """Train the DQN agent.
        
        Args:
            start_episode (int): Starting episode for resuming training.
                Allows resuming training from a specific episode (default is 0 for fresh training).
            episode_rewards (list, optional): List of previous episode rewards.
                Enables appending new rewards to an existing list for resumed training.
        
        Returns:
            list: List of episode rewards.
                A list containing the total reward accumulated in each episode during training.
        
        Explanation:
            - Trains the agent for the number of episodes specified in `config.episodes`.
            - For each episode:
              - Resets the environment to obtain an initial state.
              - Interacts with the environment by selecting actions, observing rewards and next states,
                and storing transitions in the replay memory.
              - Optimizes the model after each step using a batch of sampled transitions.
              - Accumulates the total reward for the episode.
            - Periodically updates the target model (every `config.target_update` episodes) to stabilize
              training by copying the main model's weights.
            - Decays the exploration parameter (epsilon) to reduce randomness over time, using:
              epsilon = max(epsilon_end, epsilon * epsilon_decay).
            - Logs episode statistics (reward, epsilon, loss) for monitoring.
            - Saves the model periodically (every `config.save_checkpoint_every` episodes) to the
              specified checkpoint path.
            - Generates a plot of episode rewards at the end of training, saves it to `config.plot_path`,
              and closes the figure to free memory.
            - Returns the list of episode rewards for further analysis or visualization.
        """
        episodes = self.config.episodes
        episode_rewards = episode_rewards or []
        
        for episode in range(start_episode, episodes):
            state = self.env.reset().to(self.device)
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.to(self.device)
                total_reward += reward
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                loss = self.optimize_model()
            
            # Update target network
            if episode % self.config.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            
            # Log results
            episode_rewards.append(total_reward)
            self.logger.info(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f} | Loss: {loss:.4f}")
            
            # Save model periodically
            if (episode + 1) % self.config.save_checkpoint_every == 0:
                save_model(self.model, self.config.checkpoint_path)
        
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label='Reward')
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.config.plot_path)
        plt.close()
        print(f"Rewards plot saved to: {self.config.plot_path}")
        
        return episode_rewards
    
    def run(self, max_steps=10000):
        """Run the trained agent in the environment for one episode.
        
        Args:
            max_steps (int): Maximum number of steps to prevent infinite loops.
                Limits the episode length to avoid infinite execution in non-terminating environments.
        
        Returns:
            float: Total reward accumulated during the episode.
                The sum of rewards obtained by the agent during the evaluation episode.
        
        Explanation:
            - Evaluates the trained agent by running it for a single episode in the environment.
            - Uses the main model to select the greedy action (highest Q-value) at each step, with no
              exploration (epsilon is ignored).
            - Disables gradient computation with `torch.no_grad()` for efficiency during evaluation.
            - Accumulates rewards until the episode terminates or `max_steps` is reached.
            - Logs the total reward for the episode.
            - Returns the total reward as a performance metric.
        """
        state = self.env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            state = state.to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
                action = q_values.argmax().item()
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            step += 1
        
        self.logger.info(f"Episode reward: {total_reward:.2f}")
        return total_reward
    

# The `ReplayMemory` class implements a fixed-size buffer to store transitions for experience replay
# in DQN. It uses a deque to efficiently manage a rolling window of experiences, discarding the oldest
# transitions when the capacity is exceeded. This enables the agent to sample random batches of
# experiences for training, reducing temporal correlations and improving learning stability.
class ReplayMemory:
    """Replay memory for storing transitions in DQN."""
    
    def __init__(self, capacity):
        """Initialize the replay memory.
        
        Args:
            capacity (int): Maximum number of transitions to store.
                Determines the size of the memory buffer; older transitions are discarded when full.
        
        Explanation:
            - Initializes a `deque` (double-ended queue) with the specified `capacity`.
            - The `deque` automatically removes the oldest transition when a new one is added and the
              capacity is exceeded, ensuring constant memory usage.
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        """Add a transition to the memory.
        
        Args:
            transition (tuple): (state, action, reward, next_state, done).
                A tuple containing the experience to store, where:
                - state: Current state (torch.Tensor).
                - action: Action taken (int).
                - reward: Reward received (float).
                - next_state: Next state (torch.Tensor).
                - done: Whether the episode ended (bool).
        
        Explanation:
            - Appends the provided transition to the `deque`.
            - If the memory exceeds its capacity, the oldest transition is automatically removed.
        """
        self.memory.append(transition)
    
    def sample(self, batch_size):
        """Sample a random batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample.
                Specifies how many transitions to randomly select from the memory.
        
        Returns:
            list: List of sampled transitions.
                A list of `batch_size` transition tuples, randomly selected from the memory.
        
        Explanation:
            - Uses `random.sample` to select `batch_size` transitions randomly from the memory.
            - Random sampling reduces temporal correlations between experiences, improving the stability
              of the DQN training process.
            - Returns the sampled transitions as a list, which can be processed into tensors for training.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return the current size of the memory.
        
        Returns:
            int: Number of transitions currently stored.
        
        Explanation:
            - Returns the length of the `deque`, indicating how many transitions are currently stored.
            - Used to check if there are enough transitions to sample a batch (e.g., in `optimize_model`).
        """
        return len(self.memory)