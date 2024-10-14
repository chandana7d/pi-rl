# base_agent.py
import torch
from torch import nn, optim
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize optimizer and networks
        self.optimizer = None
        self.model = None

    @abstractmethod
    def build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_network(self, output_dim: int, output_activation: nn.Module):
        """Abstract method to define network architecture."""
        pass

    @abstractmethod
    def _build_optimizer(self):
        """Abstract method to define the optimizer."""
        pass

    def save_model(self, filepath: str):
        """Save model to disk."""
        torch.save(self.model, filepath)

    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = torch.load(filepath, map_location=self.device)
        self.model.to(self.device)

    def train(self):
        """Training logic goes here, to be implemented by subclasses."""
        raise NotImplementedError("Train method must be implemented in subclasses.")
