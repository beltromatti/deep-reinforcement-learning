# dqn_rl/policy.py
import random
import torch

class EpsilonGreedyPolicy:
    """Epsilon-greedy policy for action selection."""
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        """Initialize the policy.

        Args:
            epsilon_start (float): Initial epsilon value.
            epsilon_end (float): Final epsilon value.
            epsilon_decay (float): Decay rate per episode.
        """
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, model, action_size):
        """Select an action using epsilon-greedy strategy.

        Args:
            state (torch.Tensor): Current state [1, state_size].
            model (nn.Module): DQN model.
            action_size (int): Number of possible actions.

        Returns:
            int: Selected action.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = model(state)
                return q_values.argmax().item()
        return random.randrange(action_size)

    def decay_epsilon(self):
        """Decay epsilon for the next episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)