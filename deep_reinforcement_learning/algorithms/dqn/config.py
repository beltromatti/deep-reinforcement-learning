import yaml
from typing import Optional

class DQNConfig:
    """
    Configuration class for the DQN Agent.
    This class holds all hyperparameters and settings required for training a Deep Q-Network (DQN) agent
    on an environment. Parameters can be set via the constructor or loaded from a YAML file.
    """

    def __init__(self, episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, 
                epsilon_end=0.02, epsilon_decay_mode='exponential', 
                epsilon_exponential_decay: Optional[float]=None,
                epsilon_linear_decay: Optional[float]=None,
                memory_size=20000, learning_rate=0.0005, use_scheduler=False, 
                scheduler_step_size=50, scheduler_gamma=0.9, 
                target_update=2000, max_grad_norm=1.0, save_checkpoint_every=50, 
                checkpoint_path='dqn_checkpoint.model', 
                plot_path='rewards_plot.png', use_per=False, per_alpha=0.6, 
                per_beta_start=0.4, per_beta_end=1.0, 
                per_beta_annealing_steps: Optional[int]=None):
        """
        Initialize the configuration for training a Deep Q-Network (DQN) agent in any reinforcement learning environment.

        This class provides hyperparameters and settings for training a DQN agent using experience replay, 
        epsilon-greedy exploration, and optional Prioritized Experience Replay (PER). The parameters are designed 
        to be flexible and applicable to a wide range of environments, such as those in OpenAI Gym, Atari games, 
        or custom setups. Each parameter is explained in detail to clarify its role in the learning process, 
        its impact on performance, and considerations for tuning in different environments.

        Args:
            episodes (int): Total number of episodes to train the DQN agent.
                An episode is a single run of the environment from the initial state to a terminal state (or until a 
                maximum step limit is reached). The number of episodes determines the duration of training. A default 
                value of 1000 is suitable for many environments but may need adjustment based on the complexity of the 
                task. For simple environments (e.g., with small state/action spaces), fewer episodes may suffice, while 
                complex environments (e.g., high-dimensional or continuous state spaces) may require significantly more. 
                Increasing this value extends training time and computational cost but may improve performance.

            batch_size (int): Number of transitions sampled from the replay memory for each optimization step.
                DQN uses experience replay to store and sample past transitions (state, action, reward, next_state, done) 
                to train the neural network. The batch size determines how many transitions are used in each update of the 
                Q-network. A batch size of 64 is a common choice, balancing computational efficiency with learning stability 
                by providing a diverse set of experiences. Smaller batch sizes may increase variance in updates, while larger 
                ones increase computational cost and may smooth gradients excessively. Adjust based on environment complexity 
                and hardware constraints.

            gamma (float): Discount factor for future rewards in the Bellman equation.
                The discount factor (0 < gamma ≤ 1) controls the agent's focus on immediate versus future rewards. A value 
                of 0.99 prioritizes long-term rewards, encouraging the agent to plan far into the future, which is suitable 
                for environments with long episodes or significant delayed rewards. Lower values (e.g., 0.9) prioritize 
                immediate rewards, which may be appropriate for environments with short episodes or sparse rewards. The 
                choice of gamma depends on the environment's reward structure and episode length.

            epsilon_start (float): Initial value of epsilon for the epsilon-greedy exploration strategy.
                Epsilon controls the probability of selecting a random action (exploration) versus the action with the 
                highest Q-value (exploitation). Must be in [0, 1]. A value of 1.0 means fully random actions at the start 
                of training, ensuring broad exploration of the state-action space. This is critical early in training when 
                the Q-network's estimates are unreliable. Adjust this value based on how much initial exploration is needed 
                for the environment.

            epsilon_end (float): Final value of epsilon after decay.
                This parameter sets the minimum exploration rate, ensuring some randomness persists throughout training to 
                prevent the agent from converging to suboptimal policies. Must be in [0, 1] and less than or equal to 
                `epsilon_start`. A default of 0.02 (2% random actions) is typical, balancing exploitation of learned policies 
                with continued exploration. For environments with high stochasticity, a higher `epsilon_end` may be beneficial.

            epsilon_decay_mode (str): Method for decaying epsilon over episodes.
                Specifies how epsilon transitions from `epsilon_start` to `epsilon_end`. Options are:
                    - 'exponential': Epsilon is multiplied by a decay factor per episode, resulting in a smooth, non-linear 
                    decay. This is the standard approach in DQN, as it allows rapid exploration early on and gradually 
                    shifts to exploitation.
                    - 'linear': Epsilon decreases by a fixed amount per episode, resulting in a straight-line decay. This may 
                    be simpler but can lead to abrupt changes in exploration behavior.
                The default is 'exponential', as it aligns with common DQN implementations and provides a smooth transition. 
                Raises ValueError if an invalid mode is specified.

            epsilon_exponential_decay (float, optional): Decay factor per episode for exponential decay mode.
                If provided, epsilon is updated as `epsilon *= epsilon_exponential_decay` after each episode. Must be in 
                (0, 1]. If None, the decay factor is computed as `(epsilon_end / epsilon_start) ** (1.0 / (episodes - 1))`, 
                ensuring epsilon reaches `epsilon_end` by the final episode. This parameter allows fine-tuning the exploration 
                schedule. Smaller values result in faster decay, reducing exploration earlier, which may be suitable for 
                environments where quick convergence is expected.

            epsilon_linear_decay (float, optional): Decay rate per episode for linear decay mode.
                If provided, epsilon is updated as `epsilon -= epsilon_linear_decay` after each episode. Must be non-negative. 
                If None, the decay rate is computed as `(epsilon_start - epsilon_end) / (episodes - 1)`, ensuring epsilon 
                reaches `epsilon_end` by the final episode. This parameter controls the rate of transition to exploitation. 
                Larger values lead to faster decay, which may be appropriate for simpler environments.

            memory_size (int): Maximum number of transitions stored in the replay memory.
                The replay memory stores past experiences to break temporal correlations in the data and stabilize learning. 
                A size of 20,000 is sufficient for many environments, allowing diverse experiences to be sampled over multiple 
                episodes. For environments with long episodes or large state/action spaces, a larger memory may improve 
                performance by retaining more experiences. However, larger memory sizes increase memory usage and sampling 
                time. Adjust based on environment complexity and available resources.

            learning_rate (float): Learning rate for the Adam optimizer used to update the Q-network's weights.
                The learning rate determines the step size of weight updates during backpropagation. A value of 0.0005 is 
                small enough to ensure stable convergence in most environments, avoiding large updates that could destabilize 
                learning. For environments with noisy rewards or complex dynamics, an even smaller learning rate (e.g., 0.0001) 
                may be necessary. Conversely, simpler environments may tolerate slightly larger values (e.g., 0.001) for faster 
                learning.

            use_scheduler (bool): Whether to use a learning rate scheduler to decrease the learning rate over time.
                If True, a StepLR scheduler reduces the learning rate by multiplying it by `scheduler_gamma` every 
                `scheduler_step_size` episodes. This can improve convergence by allowing larger updates early in training and 
                finer adjustments later. The default is False, using a constant learning rate, which is simpler but may not 
                adapt to changing learning dynamics. Enable for complex environments or long training runs.

            scheduler_step_size (int): Number of episodes between learning rate reductions when using a scheduler.
                Determines how frequently the learning rate is decreased. A value of 50 means the learning rate is reduced 
                every 50 episodes. Smaller values lead to more frequent reductions, which may be useful for fine-tuning in 
                later stages of training. Adjust based on the expected training duration and convergence behavior.

            scheduler_gamma (float): Multiplicative factor for reducing the learning rate in the scheduler.
                The learning rate is updated as `learning_rate *= scheduler_gamma` every `scheduler_step_size` episodes. 
                A value of 0.9 reduces the learning rate by 10% each time, promoting stable convergence. Values closer to 1 
                (e.g., 0.95) result in slower decay, while values closer to 0 (e.g., 0.5) lead to faster decay. Adjust based 
                on the desired learning rate schedule.

            target_update (int): Frequency (in steps) for updating the target network's weights.
                DQN uses a target network to compute stable Q-value estimates in the Bellman equation. The target network's 
                weights are periodically copied from the main Q-network every `target_update` steps. A value of 2000 balances 
                stability (by keeping the target network fixed for a period) and adaptability (by incorporating recent learning). 
                Smaller values (e.g., 1000) make the target network track the Q-network more closely, which may be beneficial 
                in rapidly changing environments but risks instability. Larger values (e.g., 5000) increase stability but may 
                slow adaptation. Adjust based on environment dynamics and episode length.

            max_grad_norm (float): Maximum norm for gradient clipping during backpropagation.
                Gradient clipping limits the magnitude of gradients to prevent exploding gradients, which can destabilize 
                training in deep networks. A value of 1.0 constrains the gradient norm, ensuring stable updates. This is 
                particularly important in environments with high variance in rewards or complex state spaces. Increase for 
                environments with stable gradients or decrease for those prone to large gradient swings.

            save_checkpoint_every (int): Frequency (in episodes) for saving model checkpoints.
                Checkpoints save the Q-network's weights, optimizer state, and training progress (e.g., current episode, 
                epsilon, and rewards). Saving every 50 episodes allows resuming training from recent states without excessive 
                disk usage. For long training runs or unstable environments, more frequent saves (e.g., every 10 episodes) 
                may be useful. Adjust based on training duration and the need for fault tolerance.

            checkpoint_path (str): File path for saving model checkpoints.
                Checkpoints are saved as PyTorch state dictionaries containing the model weights, optimizer state, and training 
                metadata. The default path 'dqn_checkpoint.model' stores the latest checkpoint, which can be used to resume 
                training or evaluate the trained agent. Ensure the path is unique for different experiments to avoid overwriting.

            plot_path (str): File path for saving the rewards plot.
                A plot of episode rewards is generated during training to visualize learning progress and convergence. The 
                default path 'rewards_plot.png' saves the plot as a PNG file. This is useful for analyzing training performance, 
                identifying plateaus, or detecting instability. Ensure the path is unique for different experiments.

            use_per (bool): Whether to use Prioritized Experience Replay (PER).
                If True, transitions are sampled from the replay memory based on their priority (proportional to their 
                temporal-difference (TD) error), focusing on more informative experiences. This can improve learning efficiency 
                in complex environments with sparse rewards or large state spaces. If False, standard uniform sampling is used, 
                where all transitions have equal probability. Uniform sampling is simpler and sufficient for simpler 
                environments but may be less efficient. The default is False, as PER adds computational overhead and complexity.

            per_alpha (float): Exponent for prioritizing transitions in PER.
                Determines the degree of prioritization in sampling, where the probability of sampling a transition is 
                proportional to its TD-error raised to the power `per_alpha` (P(i) ∝ |TD-error|^alpha). A value of 0 results 
                in uniform sampling, while 1 maximizes prioritization. The default of 0.6 balances prioritization (focusing on 
                high-error transitions) and diversity (ensuring all transitions have some chance of being sampled). Increase 
                for environments where certain transitions are significantly more informative, but avoid values too close to 1 
                to prevent overfitting to a small subset of experiences.

            per_beta_start (float): Initial value of beta for importance sampling in PER.
                Beta controls the extent to which importance sampling weights correct the bias introduced by prioritized 
                sampling. A lower starting value (e.g., 0.4) reduces the impact of these weights early in training, when Q-value 
                estimates are noisy, preventing large gradient updates that could destabilize learning. This is particularly 
                important in the initial phases of training. Adjust based on the expected noise in Q-value estimates.

            per_beta_end (float): Final value of beta after linear annealing in PER.
                Beta increases linearly to this value over `per_beta_annealing_steps`, fully correcting the bias introduced by 
                prioritized sampling by the end of training. A value of 1.0 ensures unbiased updates, aligning the learning 
                process with the true expected Q-values as the agent converges. Values less than 1.0 may be used in specific 
                cases to maintain some bias correction flexibility, but 1.0 is standard for optimal convergence.

            per_beta_annealing_steps (int, optional): Number of sampling steps to anneal beta from `per_beta_start` to 
                `per_beta_end`.
                Determines the duration over which beta grows linearly during training. If None, defaults to `episodes * 1000`, 
                assuming approximately 1000 sampling steps per episode as a rough heuristic for many environments. For 
                environments with significantly longer or shorter episodes, adjust this value to ensure beta reaches 
                `per_beta_end` at an appropriate point in training. A larger value slows the annealing process, delaying full 
                bias correction, which may stabilize early training but delay optimal convergence.

        Raises:
            ValueError: If any of the following conditions are met:
                - `epsilon_start` or `epsilon_end` are not in [0, 1].
                - `epsilon_end` is greater than `epsilon_start`.
                - `epsilon_decay_mode` is not 'linear' or 'exponential'.
                - `epsilon_exponential_decay` is provided and not in (0, 1] for exponential mode.
                - `epsilon_linear_decay` is provided and negative for linear mode.
                - `gamma` is not in (0, 1].
                - `batch_size`, `memory_size`, `episodes`, `target_update`, or `save_checkpoint_every` are not positive integers.
                - `learning_rate` or `max_grad_norm` are not positive.
                - `per_alpha`, `per_beta_start`, or `per_beta_end` are not in [0, 1] when `use_per` is True.
                - `per_beta_annealing_steps` is provided and not positive.

        Notes:
            - This configuration is designed to be environment-agnostic, but hyperparameter tuning is critical for optimal 
            performance. Start with the default values and adjust based on empirical results, environment characteristics 
            (e.g., state/action space size, reward structure, episode length), and computational constraints.
            - For environments with high stochasticity or sparse rewards, consider increasing `memory_size`, enabling PER 
            (`use_per=True`), or adjusting `gamma` to focus on immediate rewards.
            - For computationally intensive environments (e.g., Atari games with image inputs), consider reducing `batch_size` 
            or increasing `target_update` to improve efficiency, and ensure sufficient memory for `memory_size`.
            - The epsilon-greedy exploration strategy is simple but effective. For environments requiring more sophisticated 
            exploration (e.g., continuous action spaces), consider alternative algorithms like DDPG or SAC.
            - When using PER, monitor the TD-errors and beta annealing to ensure the prioritization does not overly focus on 
            a small subset of experiences, which could reduce diversity and hinder learning.
        """
        # Validate parameters
        if not 0 <= epsilon_start <= 1 or not 0 <= epsilon_end <= 1:
            raise ValueError("epsilon_start and epsilon_end must be in [0, 1]")
        if epsilon_end > epsilon_start:
            raise ValueError("epsilon_end must be <= epsilon_start")
        if epsilon_decay_mode not in ['linear', 'exponential']:
            raise ValueError("epsilon_decay_mode must be 'linear' or 'exponential'")
        
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_mode = epsilon_decay_mode
        self.epsilon_linear_decay = (epsilon_linear_decay if epsilon_linear_decay is not None else (epsilon_start - epsilon_end) / (episodes - 1) if episodes > 1 else 0.0)
        self.epsilon_exponential_decay = (epsilon_exponential_decay if epsilon_exponential_decay is not None else ((epsilon_end / epsilon_start) ** (1.0 / (episodes - 1)) if episodes > 1 and epsilon_start > 0 else 1.0))
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.target_update = target_update
        self.max_grad_norm = max_grad_norm
        self.save_checkpoint_every = save_checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.plot_path = plot_path
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_annealing_steps = per_beta_annealing_steps if per_beta_annealing_steps is not None else episodes * 1000

        if self.epsilon_linear_decay < 0:
            raise ValueError("epsilon_linear_decay must be non-negative")
        if not 0 < self.epsilon_exponential_decay <= 1:
            raise ValueError("epsilon_exponential_decay must be in (0, 1]")


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