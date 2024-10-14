import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4):
        super().__init__(buffer_size, batch_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        self.priorities.append(max(self.priorities, default=1))

    def sample(self):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon