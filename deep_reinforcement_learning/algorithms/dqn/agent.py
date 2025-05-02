import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

from ...core.utils import setup_logger, save_model
from ...core.model import Model
from ...core.environment import Environment
from .config import DQNConfig

logger = setup_logger(__name__)

# The `DQNAgent` class implements a Deep Q-Network (DQN) agent for reinforcement learning.
# It uses a neural network to approximate the Q-value function and learns optimal actions through
# an epsilon-greedy policy, experience replay, and periodic target network updates.
# The agent interacts with a provided environment, optimizes its model based on sampled experiences,
# and supports training and evaluation phases.
class DQNAgent:
    """Deep Q-Network (DQN) agent for reinforcement learning."""
    
    def __init__(self, env: Environment, model: Model, config=DQNConfig()):
        """
        Initialize the DQN agent.

        Args:
            env (Environment): Environment object (subclass of Environment).
                The environment provides the state space, action space, and step function for
                agent-environment interaction (e.g., CartPole-v1 from Gymnasium).
            model (Model): Model object (subclass of nn.Module).
                The neural network used to approximate the Q-value function, mapping states to
                action values. It must be compatible with the environment's state and action sizes.
            config (DQNConfig): Configuration parameters object.
                Contains hyperparameters such as learning rate, gamma, epsilon values, memory size,
                batch size, paths for saving models and plots, and settings for Prioritized Experience
                Replay (PER) like `use_per`, `per_alpha`, `per_beta_start`, and `per_beta_end`.

        Explanation:
            - Initializes the agent by setting up the environment, model, and configuration.
            - Determines the device (GPU or CPU) for computation using PyTorch and moves the model
            to the selected device for efficient processing.
            - Creates a target model, a copy of the main model, to stabilize training by providing
            consistent Q-value targets. The target model is set to evaluation mode.
            - Initializes an Adam optimizer for updating the model's parameters with the specified
            learning rate (`config.learning_rate`).
            - Sets up the replay memory to store experiences (state, action, reward, next_state, done):
                - If Prioritized Experience Replay (PER) is enabled (`config.use_per=True`), uses
                `PrioritizedReplayMemory` with parameters for prioritization (`per_alpha`, `per_beta_start`,
                `per_beta_end`, and annealing steps).
                - Otherwise, uses `ReplayMemory` for uniform sampling.
            - Initializes the exploration parameter (epsilon) for the epsilon-greedy policy with
            `config.epsilon_start`.
            - Configures a logger for tracking training progress and debugging.
        """
        self.env = env
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.total_step = 0
        
        # Initialize target model
        self.target_model = type(model)(env.state_size, env.action_size, model.layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        if config.use_scheduler: self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

        # Initialize memory
        if config.use_per:
            self.memory = PrioritizedReplayMemory(
                config.memory_size, 
                config.per_alpha, 
                config.per_beta_start, 
                config.per_beta_end,
                config.per_beta_annealing_steps
            )
        else:
            self.memory = ReplayMemory(config.memory_size)
        
        # Initialize exploration parameters
        self.epsilon = config.epsilon_start
    
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
        """
        Store a transition in the replay memory.

        Args:
            state (torch.Tensor): Current state [1, state_size].
                The state before taking the action, represented as a tensor with shape [1, state_size].
            action (int): Action taken.
                The action chosen by the agent, an integer from the environment's action space.
            reward (float): Reward received.
                The reward provided by the environment after taking the action.
            next_state (torch.Tensor): Next state [1, state_size].
                The state observed after taking the action, with shape [1, state_size].
            done (bool): Whether the episode is done.
                Indicates if the episode terminated (e.g., the agent failed or reached a goal).

        Explanation:
            - Stores a transition tuple (state, action, reward, next_state, done) in the replay memory.
            - The replay memory is used to sample experiences for training, enabling the agent to learn
            from past interactions and break temporal correlations in the data.
            - If Prioritized Experience Replay (PER) is enabled (`config.use_per=True`):
                - Computes the initial TD-error for the transition to assign a priority.
                - The TD-error is calculated as |Q(s, a) - (r + gamma * max(Q(s', a')) * (1 - done))|,
                where Q(s, a) is the predicted Q-value from the main model, and the target uses the
                target model for stability.
                - Stores the transition in `PrioritizedReplayMemory` with the computed priority.
            - If PER is disabled, stores the transition in `ReplayMemory` without computing priorities.
            - The replay memory has a fixed capacity (`config.memory_size`) and automatically discards
            the oldest transitions when full, using a deque for efficient memory management.
            - Moves state and next_state tensors to the appropriate device (CPU/GPU) for Q-value
            computations when PER is enabled.
        """
        if self.config.use_per:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            with torch.no_grad():
                current_q = self.model(state)[0, action]
                next_q = self.target_model(next_state).max(1)[0]
                target_q = reward + (self.config.gamma * next_q * (1 - done))
                error = (current_q - target_q).abs().item()
            self.memory.push((state, action, reward, next_state, done), error)
        else:
            self.memory.push((state, action, reward, next_state, done))
    
    def optimize_model(self):
        """
        Optimize the model using a batch of transitions sampled from the replay memory.

        Returns:
            float: Loss value for the batch (or 0.0 if not enough transitions).
                The mean squared error loss, optionally weighted by importance sampling weights
                when using Prioritized Experience Replay (PER), used for monitoring training.

        Explanation:
            - Performs a single optimization step by sampling a batch of transitions from the replay memory.
            - Returns 0.0 if there are fewer transitions than the batch size (`config.batch_size`).
            - If Prioritized Experience Replay (PER) is enabled (`config.use_per=True`):
                - Samples transitions proportional to their priorities, along with their indices and
                importance sampling weights to correct for sampling bias.
                - Updates the priorities of sampled transitions based on their TD-errors.
            - If PER is disabled, samples transitions uniformly and assigns equal weights (1.0).
            - Extracts components (states, actions, rewards, next_states, done flags) from the sampled
            transitions and converts them into tensors for batch processing.
            - Computes the current Q-values for the chosen actions using the main model.
            - Computes the target Q-values using the target model, applying the Bellman equation:
            target = reward + gamma * max(next_Q_values) * (1 - done).
            The `done` flag ensures terminal states have no future value.
            - Calculates the loss as the mean of weighted squared errors between current and target Q-values:
            - For PER: loss = mean(weights * (current_Q - target_Q)^2), where weights correct for sampling bias.
            - For uniform sampling: loss = mean((current_Q - target_Q)^2), equivalent to standard MSE.
            - Performs backpropagation to compute gradients, clips gradients to prevent exploding gradients
            (using `config.max_grad_norm`), and updates the model's parameters using the Adam optimizer.
            - Returns the loss value for logging and monitoring training progress.
        """
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        if self.config.use_per:
            transitions, indices, weights = self.memory.sample(self.config.batch_size)
        else:
            transitions = self.memory.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
        
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
        
        # Compute TD-errors for updating priorities
        errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
        
        # Compute weighted loss
        loss = (weights * nn.MSELoss(reduction='none')(current_q_values, target_q_values)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        #self.logger.info(f"Gradient norm: {grad_norm:.4f}") # DEBUG
        self.optimizer.step()

        # Update priorities if using PER
        if self.config.use_per:
            self.memory.update_priorities(indices, errors)
        
        return loss.item()
    
    def train(self, max_steps_per_episode = 5000, start_episode=0, episode_rewards=None):
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
            step = 0
            done = False
            
            while not done and step < max_steps_per_episode:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.to(self.device)
                total_reward += reward
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                loss = self.optimize_model()

                step += 1
                self.total_step += 1
                if self.total_step % self.config.target_update == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            
            # Decay epsilon
            if self.config.epsilon_decay_mode == 'linear':
                self.epsilon = max(self.config.epsilon_end, self.epsilon - self.config.epsilon_linear_decay)
            if self.config.epsilon_decay_mode == 'exponential':
                self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_exponential_decay)

            # Run scheduler
            if self.config.use_scheduler and loss != 0.0:
                self.scheduler.step()
            
            # Log results
            episode_rewards.append(total_reward)
            logger.info(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")
            
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
        
        logger.info(f"Episode reward: {total_reward:.2f}")
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
        state, action, reward, next_state, done = transition
        state = state.cpu()
        next_state = next_state.cpu()
        transition = (state, action, reward, next_state, done)
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



class SumTree:
    """
    SumTree data structure for efficient prioritized experience replay.
    Stores priorities in a binary tree where leaves represent transition priorities
    and internal nodes store the sum of their children's priorities.
    Enables O(log n) sampling and priority updates.
    """
    
    def __init__(self, capacity):
        """
        Initialize the SumTree.

        Args:
            capacity (int): Maximum number of transitions to store.
                Determines the size of the leaf nodes in the tree.

        Explanation:
            - The tree is represented as a 1D array of size 2*capacity - 1.
            - Leaves (priorities) are stored at indices [capacity-1, 2*capacity-2].
            - Internal nodes store the sum of their children's priorities.
            - self.size tracks the number of stored transitions (≤ capacity).
            - self.data_pointer tracks the next index for writing a new priority.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Array for tree nodes
        self.size = 0  # Current number of stored transitions
        self.data_pointer = 0  # Index for the next write operation

    def write(self, priority):
        """
        Write a new priority to the tree at the current data_pointer.

        Args:
            priority (float): Priority value for the transition (typically |TD-error|^alpha).

        Explanation:
            - Writes the priority to the leaf node at data_pointer.
            - Updates the tree by propagating the change to parent nodes.
            - Increments data_pointer and size (up to capacity).
            - When capacity is reached, overwrites oldest transitions.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        # Update size and data_pointer
        if self.size < self.capacity:
            self.size += 1
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, priority):
        """
        Update the priority at a given tree index and propagate changes.

        Args:
            tree_idx (int): Index in the tree array (leaf or internal node).
            priority (float): New priority value.

        Explanation:
            - Updates the priority at tree_idx and computes the change.
            - Propagates the change to all parent nodes up to the root.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        """
        Retrieve the leaf index and priority corresponding to a cumulative sum.

        Args:
            s (float): Cumulative sum value in [0, total_priority].

        Returns:
            tuple: (leaf_index, priority).
                - leaf_index (int): Index of the transition in [0, capacity-1].
                - priority (float): Priority value at the leaf.

        Explanation:
            - Traverses the tree from the root to a leaf based on the sum s.
            - At each node, chooses the left or right child based on their sums.
            - Returns the leaf's data index and its priority.
        """
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):  # Reached a leaf
                leaf_idx = parent_idx
                break
            if s <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                s -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx]

    def total(self):
        """
        Return the total priority sum.

        Returns:
            float: Sum of all priorities (stored at the root).
        """
        return self.tree[0]

    def min(self):
        """
        Return the minimum non-zero priority among stored transitions.

        Returns:
            float: Minimum priority, or 1.0 if no transitions are stored.

        Explanation:
            - Computes the minimum priority among the leaf nodes corresponding to stored transitions.
            - Uses self.size to limit the range to actual transitions.
            - Returns 1.0 if the tree is empty to avoid division by zero in weight calculations.
        """
        if self.size == 0:
            return 1.0
        return max(np.min(self.tree[self.capacity - 1:self.capacity - 1 + self.size]), 1e-6)


class PrioritizedReplayMemory:
    """
    Replay memory with prioritized experience replay for DQN.
    Stores transitions and samples them proportional to their priorities (based on TD-errors).
    Uses a SumTree for efficient priority management and importance sampling weights to correct bias.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_annealing_steps=100_000):
        """
        Initialize the prioritized replay memory.

        Args:
            capacity (int): Maximum number of transitions to store.
            alpha (float): Exponent for prioritizing transitions (0 = uniform, 1 = full prioritization).
            beta_start (float): Initial value of beta for importance sampling weights.
            beta_end (float): Final value of beta after annealing.
            beta_annealing_steps (int): Number of steps to anneal beta from beta_start to beta_end.

        Explanation:
            - Uses a deque to store transitions (state, action, reward, next_state, done).
            - Uses a SumTree to store and manage priorities.
            - Alpha controls the degree of prioritization.
            - Beta controls the importance sampling correction, annealed linearly to reduce bias over time.
            - max_priority ensures new transitions are sampled frequently.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.beta_increment = (beta_end - beta_start) / beta_annealing_steps
        self.memory = deque(maxlen=capacity)  # Stores transitions
        self.priorities = SumTree(capacity)  # Stores priorities
        self.max_priority = 1.0  # Default priority for new transitions
        self.step = 0  # Tracks sampling steps for beta annealing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, transition, error=None):
        """
        Add a transition to the memory with an associated priority.

        Args:
            transition (tuple): (state, action, reward, next_state, done).
                - state (torch.Tensor): Current state.
                - action (int): Action taken.
                - reward (float): Reward received.
                - next_state (torch.Tensor): Next state.
                - done (bool): Whether the episode ended.
            error (float, optional): TD-error for the transition.
                If None, uses max_priority to ensure new transitions are sampled frequently.

        Explanation:
            - Appends the transition to the deque.
            - Computes the priority as |error|^alpha or max_priority for new transitions.
            - Writes the priority to the SumTree at the current index.
        """
        state, action, reward, next_state, done = transition
        state = state.cpu()
        next_state = next_state.cpu()
        transition = (state, action, reward, next_state, done)
        priority = self.max_priority if error is None else abs(error) ** self.alpha
        self.memory.append(transition)
        self.priorities.write(priority)

    def sample(self, batch_size):
        """
        Sample a batch of transitions with prioritized sampling.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (transitions, indices, weights).
                - transitions (list): List of sampled transition tuples.
                - indices (list): Indices of sampled transitions in the memory.
                - weights (torch.Tensor): Importance sampling weights for loss correction.

        Explanation:
            - Samples transitions proportional to their priorities using the SumTree.
            - Computes importance sampling weights: w_i = (p_i / min(p_j) / N)^(-beta).
            - Normalizes weights to prevent large gradients.
            - Anneals beta linearly with each sampling step.
        """
        self.step += 1
        self.beta = min(self.beta_end, self.beta_start + self.step * self.beta_increment)
        
        indices = []
        transitions = []
        weights = []
        total_priority = self.priorities.total()
        min_priority = max(self.priorities.min(), 1e-6)  # Ensure min_priority is non-zero

        # Sample transitions proportional to priorities
        for _ in range(batch_size):
            r = random.uniform(0, total_priority)
            idx, priority = self.priorities.get(r)
            indices.append(idx)
            transitions.append(self.memory[idx])
            # Compute importance sampling weight
            weight = ((priority / min_priority) * (1.0 / self.capacity)) ** (-self.beta)  # (p_i / min(p_j) * 1/N)^(-beta)
            weights.append(weight)
        
        # Normalize weights to stabilize gradients
        weights = np.array(weights)
        weights = weights / np.max(weights + 1e-6)  # Normalize using max weight
        weights = torch.FloatTensor(weights).to(self.device)

        # Log per debugging
        #logger.info(f"Priority Weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        
        return transitions, indices, weights

    def update_priorities(self, indices, errors):
        """
        Update priorities for sampled transitions based on new TD-errors.

        Args:
            indices (list): Indices of sampled transitions.
            errors (list): TD-errors for the sampled transitions.

        Explanation:
            - Updates the priority for each sampled transition as |error|^alpha.
            - Updates max_priority to ensure new transitions have high sampling probability.
        """
        for idx, error in zip(indices, errors):
            priority = abs(error) ** self.alpha
            self.priorities.update(idx + self.capacity - 1, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Return the current size of the memory.

        Returns:
            int: Number of transitions stored.
        """
        return len(self.memory)