# a2c_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from pi.agents.base_agent import BaseAgent

class A2CAgent(BaseAgent):
    def __init__(self, state_size: int, action_size: int, config: dict):
        super().__init__(state_size, action_size, config)

        # Initialize policy and value networks
        self.policy_net = self._build_network(self.action_size, nn.Softmax(dim=-1))
        self.value_net = self._build_network(1, nn.Identity())

        # Store networks in a model dictionary
        self.model = self.build_model()
        self.optimizer = self._build_optimizer()

    def build_model(self) -> dict:
        """Build the policy and value networks."""
        return {
            "policy_net": self.policy_net,
            "value_net": self.value_net
        }

    def _build_network(self, output_dim: int, output_activation: nn.Module) -> nn.Module:
        """Builds a simple neural network."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            output_activation
        ).to(self.device)

    def _build_optimizer(self) -> optim.Optimizer:
        """Build an Adam optimizer for both networks."""
        return optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.config.get("learning_rate", 0.001)
        )

    def train(self, states, actions, rewards):
        """Train the agent using A2C logic."""
        self.optimizer.zero_grad()

        # Calculate policy loss and value loss
        policy_output = self.policy_net(states)
        value_output = self.value_net(states).squeeze()

        log_probs = torch.log(policy_output.gather(1, actions.unsqueeze(1)).squeeze())
        advantages = rewards - value_output.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(value_output, rewards)

        # Combine losses
        loss = policy_loss + self.config.get("value_loss_weight", 0.5) * value_loss
        loss.backward()

        self.optimizer.step()

        return loss.item()

