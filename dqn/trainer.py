# dqn_rl/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from .model import DQN
from .memory import ReplayMemory, PrioritizedReplayMemory
from .policy import EpsilonGreedyPolicy

class DQNTrainer:
    """Trainer for Deep Q-Network (DQN) agents."""
    def __init__(self, env, config, model=None, network_builder=None):
        """Initialize the trainer.

        Args:
            env: Environment object (wrapped or raw).
            config (dict): Configuration parameters (gamma, batch_size, etc.).
            model (DQN, optional): Pre-initialized DQN model.
            network_builder (NetworkBuilder, optional): Custom network builder.
        """
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = env.state_size
        self.action_size = env.action_size

        # Initialize models
        self.model = model if model else DQN(self.state_size, self.action_size, network_builder).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size, network_builder).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Initialize optimizer and memory
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        if config.get("use_prioritized_replay", False):
            self.memory = PrioritizedReplayMemory(config["memory_size"], config.get("per_alpha", 0.6), config.get("per_beta", 0.4))
        else:
            self.memory = ReplayMemory(config["memory_size"])

        # Initialize policy
        self.policy = EpsilonGreedyPolicy(config["epsilon_start"], config["epsilon_end"], config["epsilon_decay"])

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def optimize(self):
        """Optimize the DQN model using a batch of transitions."""
        if len(self.memory) < self.config["batch_size"]:
            return 0.0

        if isinstance(self.memory, PrioritizedReplayMemory):
            transitions, indices, weights = self.memory.sample(self.config["batch_size"])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.memory.sample(self.config["batch_size"])
            weights = None

        batch_state = torch.cat([t[0] for t in transitions]).to(self.device)
        batch_action = torch.LongTensor([t[1] for t in transitions]).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        batch_next_state = torch.cat([t[3] for t in transitions]).to(self.device)
        batch_done = torch.FloatTensor([t[4] for t in transitions]).to(self.device)

        current_q_values = self.model(batch_state).gather(1, batch_action).squeeze(1)

        with torch.no_grad():
            if self.config.get("use_double_dqn", False):
                next_actions = self.model(batch_next_state).argmax(1, keepdim=True)
                next_q_values = self.target_model(batch_next_state).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_model(batch_next_state).max(1)[0]
            target_q_values = batch_reward + (self.config["gamma"] * next_q_values * (1 - batch_done))

        loss = nn.MSELoss()(current_q_values, target_q_values)
        if weights is not None:
            loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))
        self.optimizer.step()

        if isinstance(self.memory, PrioritizedReplayMemory):
            td_errors = (current_q_values - target_q_values).abs().cpu().numpy()
            self.memory.update_priorities(indices, td_errors + 1e-5)

        return loss.item()

    def train(self, start_episode=0, episode_rewards=None):
        """Train the DQN agent.

        Args:
            start_episode (int): Starting episode for resuming training.
            episode_rewards (list, optional): List of previous episode rewards.

        Returns:
            DQN: Trained model.
        """
        episode_rewards = episode_rewards or []
        for episode in range(start_episode, self.config["episodes"]):
            state = self.env.reset().to(self.device)
            total_reward = 0
            done = False

            while not done:
                action = self.policy.select_action(state, self.model, self.action_size)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.to(self.device)
                total_reward += reward

                transition = (state, action, reward, next_state, done)
                self.memory.push(transition)
                state = next_state
                loss = self.optimize()

            if episode % self.config["target_update"] == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.policy.decay_epsilon()
            episode_rewards.append(total_reward)
            self.logger.info(f"Episode {episode+1}/{self.config['episodes']} | Reward: {total_reward:.2f} | Epsilon: {self.policy.epsilon:.3f} | Loss: {loss:.4f}")

            if (episode + 1) % self.config.get("save_checkpoint_every", 50) == 0:
                self.save_checkpoint(episode, episode_rewards)

        plt.plot(episode_rewards)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(self.config.get("plot_path", "rewards_plot.png"))
        plt.close()

        return self.model

    def save_checkpoint(self, episode, rewards, filepath=None):
        """Save model and training state.

        Args:
            episode (int): Current episode.
            rewards (list): List of episode rewards.
            filepath (str, optional): Path to save checkpoint.
        """
        filepath = filepath or self.config.get("checkpoint_path", "dqn_checkpoint.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": self.policy.epsilon,
            "rewards": rewards
        }, filepath)
        self.logger.info(f"Checkpoint saved to: {filepath}")

    def load_checkpoint(self, filepath=None):
        """Load model and training state.

        Args:
            filepath (str, optional): Path to load checkpoint.

        Returns:
            tuple: (episode, rewards) from the checkpoint.
        """
        filepath = filepath or self.config.get("checkpoint_path", "dqn_checkpoint.pth")
        if not torch.cuda.is_available():
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.epsilon = checkpoint["epsilon"]
        self.logger.info(f"Checkpoint loaded from: {filepath}")
        return checkpoint["episode"], checkpoint["rewards"]