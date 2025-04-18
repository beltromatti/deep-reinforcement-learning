# dqn_rl/memory.py
from collections import deque
import random
import numpy as np

class ReplayMemory:
    """Standard replay memory for DQN."""
    def __init__(self, capacity):
        """Initialize the replay memory.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """Add a transition to the memory.

        Args:
            transition (tuple): (state, action, reward, next_state, done).
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list: List of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)

class PrioritizedReplayMemory:
    """Prioritized replay memory with importance sampling."""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        """Initialize the prioritized replay memory.

        Args:
            capacity (int): Maximum number of transitions.
            alpha (float): Prioritization exponent (0 = uniform sampling).
            beta (float): Importance sampling weight.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, transition):
        """Add a transition with maximum priority.

        Args:
            transition (tuple): (state, action, reward, next_state, done).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample transitions based on priorities.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (transitions, indices, weights) for importance sampling.
        """
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions.

        Args:
            indices (list): Indices of sampled transitions.
            priorities (list): New priority values.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)