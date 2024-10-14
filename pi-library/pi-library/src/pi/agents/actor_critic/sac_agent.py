import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Pipe
from pi.agents.base_agent import BaseAgent


class SACAgent(BaseAgent):
    def __init__(self, state_size: int, action_size: int, config: dict):
        super().__init__(state_size, action_size, config)

        # Initialize policy and Q-value networks
        self.policy_net = self._build_policy_network()
        self.q1_net = self._build_q_network()
        self.q2_net = self._build_q_network()
        self.value_net = self._build_value_network()

        # Target value network
        self.target_value_net = self._build_value_network()
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Store optimizers
        self.policy_optimizer = self._build_optimizer(self.policy_net)
        self.q1_optimizer = self._build_optimizer(self.q1_net)
        self.q2_optimizer = self._build_optimizer(self.q2_net)
        self.value_optimizer = self._build_optimizer(self.value_net)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.alpha = config.get("alpha", 0.2)  # Entropy coefficient

    def _build_policy_network(self) -> nn.Module:
        return self._build_network(self.action_size, nn.Tanh())

    def _build_q_network(self) -> nn.Module:
        return self._build_network(1, nn.Identity())

    def _build_value_network(self) -> nn.Module:
        return self._build_network(1, nn.Identity())

    def _build_network(self, output_dim: int, output_activation: nn.Module) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            output_activation
        ).to(self.device)

    def _build_optimizer(self, network: nn.Module) -> optim.Optimizer:
        return optim.Adam(network.parameters(), lr=self.config.get("learning_rate", 0.001))

    def train(self, states, actions, rewards, next_states, dones):
        """Train the agent using SAC logic."""
        # Convert inputs to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Update Q-values
        with torch.no_grad():
            next_action_prob = self.policy_net(next_states)
            next_q1_value = self.q1_net(next_states)
            next_q2_value = self.q2_net(next_states)
            next_value = self.value_net(next_states)

            target_q_value = rewards + (1 - dones) * self.gamma * (torch.min(next_q1_value, next_q2_value) - self.alpha * next_action_prob)

        q1_value = self.q1_net(states)
        q2_value = self.q2_net(states)

        q1_loss = nn.MSELoss()(q1_value, target_q_value)
        q2_loss = nn.MSELoss()(q2_value, target_q_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update value network
        value_loss = nn.MSELoss()(self.value_net(states), torch.min(q1_value, q2_value) - self.alpha * next_action_prob)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        policy_loss = (self.alpha * next_action_prob - self.q1_net(states)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target value network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return q1_loss.item(), q2_loss.item(), value_loss.item(), policy_loss.item()

    def worker(self, worker_id, conn, env, config):
        """A worker process for parallel training."""
        while True:
            message = conn.recv()
            if message == "train":
                # Run one episode and train on the result
                state = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                total_reward = 0

                while not done:
                    action = self.policy_net(state.unsqueeze(0)).squeeze().cpu().detach().numpy()
                    next_state, reward, done, _ = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)

                    total_reward += reward
                    state = next_state

                conn.send(total_reward)

            elif message == "stop":
                break
