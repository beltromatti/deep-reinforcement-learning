import yaml

class DQNConfig:
    """
    Configuration class for the DQN Agent.
    This class holds all hyperparameters and settings required for training a Deep Q-Network (DQN) agent
    on an environment. Parameters can be set via the constructor or loaded from a YAML file.
    """

    def __init__(self, episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.02, epsilon_decay=0.995, memory_size=20000, 
                 learning_rate=0.0005, target_update=10, max_grad_norm=1.0, 
                 save_checkpoint_every=50,  checkpoint_path="dqn_checkpoint.model", 
                 plot_path="rewards_plot.png"):
        """
        Initialize the Config class with default hyperparameters for DQN training.

        Args:
            episodes (int): Number of episodes to train the DQN agent.
                An episode is a single run of the CartPole environment until the pole falls or a timeout is reached.
                Default is 1000, which is typically sufficient to train the agent to achieve high rewards (e.g., >400)
                in CartPole-v1. Increasing this value allows more training time but increases computational cost.
            
            batch_size (int): Number of transitions sampled from the replay memory for each optimization step.
                DQN uses experience replay to learn from past experiences, sampling a batch of transitions
                (state, action, reward, next_state, done). A batch size of 64 balances computational efficiency
                and stable learning by providing a diverse set of experiences.
            
            gamma (float): Discount factor for future rewards in the Bellman equation.
                Gamma determines how much the agent values future rewards compared to immediate rewards
                (0 < gamma <= 1). A value of 0.99 means the agent heavily considers future rewards, encouraging
                long-term planning in CartPole.
            
            epsilon_start (float): Initial value of epsilon for the epsilon-greedy exploration strategy.
                Epsilon controls the trade-off between exploration (random actions) and exploitation
                (actions based on Q-values). Starting at 1.0 means the agent initially takes random actions
                100% of the time, promoting exploration.
            
            epsilon_end (float): Final value of epsilon after decay.
                As training progresses, epsilon decreases to favor exploitation over exploration.
                A minimum of 0.02 ensures the agent retains a small amount of randomness (2% random actions)
                to avoid getting stuck in suboptimal policies.
            
            epsilon_decay (float): Decay rate for epsilon per episode.
                After each episode, epsilon is multiplied by this value
                (epsilon = max(epsilon_end, epsilon * epsilon_decay)). A decay rate of 0.995 reduces epsilon
                gradually, allowing sufficient exploration early in training while shifting to exploitation later.
            
            memory_size (int): Maximum number of transitions stored in the replay memory.
                The replay memory holds past experiences to break temporal correlations and improve learning
                stability. A size of 20,000 is large enough to store diverse experiences from many episodes
                in CartPole, ensuring robust training.
            
            learning_rate (float): Learning rate for the Adam optimizer used to update the DQN's weights.
                The learning rate controls the step size of weight updates during backpropagation.
                A small value like 0.0005 ensures stable convergence, preventing large updates that could
                destabilize learning.
            
            target_update (int): Frequency (in episodes) for updating the target network's weights.
                DQN uses a target network to stabilize Q-value estimates in the Bellman equation.
                Copying the main network's weights to the target network every 10 episodes balances
                stability and adaptation to new learning.
            
            max_grad_norm (float): Maximum norm for gradient clipping.
                Gradient clipping prevents exploding gradients by limiting the norm of gradients during
                backpropagation. A value of 1.0 ensures stable training by constraining large gradient updates,
                which is particularly useful in deep networks like DQN.
            
            save_checkpoint_every (int): Frequency (in episodes) for saving model checkpoints.
                Checkpoints include the model weights, optimizer state, and training progress
                (episode, epsilon, rewards). Saving every 50 episodes allows resuming training from recent
                states without excessive disk usage.
            
            checkpoint_path (str): File path for saving model checkpoints.
                Checkpoints are saved as PyTorch state dictionaries containing model weights and training states.
                The default path 'dqn_checkpoint.pth' is used to store the latest checkpoint, allowing training
                resumption or evaluation.
            
            plot_path (str): File path for saving the rewards plot.
                During training, a plot of episode rewards is generated to visualize learning progress.
                The default path 'rewards_plot.png' saves the plot as a PNG file, useful for analyzing training
                performance and convergence.
        """
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.max_grad_norm = max_grad_norm
        self.save_checkpoint_every = save_checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.plot_path = plot_path

    def load_yaml_config(self, config_path):
        """
        Load configuration from a YAML file and update the class attributes.

        This method reads a YAML file containing key-value pairs for configuration parameters and updates
        the corresponding attributes of the Config instance. Only valid attributes (those defined in the
        constructor) are updated, and invalid keys are ignored with a warning.

        Args:
            config_path (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            yaml.YAMLError: If the YAML file is invalid or cannot be parsed.

        Example:
            A YAML file (e.g., config.yaml) might look like:
            ```yaml
            episodes: 2000
            batch_size: 128
            gamma: 0.95
            ```
            Calling `config.load_yaml_config("config.yaml")` updates the corresponding attributes.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Update only the attributes that exist in the class
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: '{key}' is not a valid configuration parameter and will be ignored.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")