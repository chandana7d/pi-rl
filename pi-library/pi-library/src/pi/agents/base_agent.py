from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Tuple, List, Dict
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import os
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseAgent(ABC):
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        """Initialize the base agent with given state and action sizes and configuration."""
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory: Deque[Tuple[Any, Any, float, Any, bool]] = deque(maxlen=config.get("memory_size", 100000))
        self.model: torch.nn.Module = self._build_model()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()
        self.distributed: bool = config.get("distributed", False)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.num_workers: int = config.get("num_workers", 1)

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Build the neural network model."""
        pass

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer for training."""
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Select an action based on the current state."""
        pass

    @abstractmethod
    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> Dict[str, float]:
        """Update the model based on the provided experiences."""
        pass

    def remember(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """Store the experience in memory."""
        self.memory.append(experience)

    def save(self, path: str) -> None:
        """Save the agent's state to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logging.info(f"Agent state saved to {path}")

    def load(self, path: str) -> None:
        """Load the agent's state from a checkpoint file."""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config.update(checkpoint['config'])
            logging.info(f"Agent state loaded from {path}")
        except FileNotFoundError:
            logging.error(f"Checkpoint file not found: {path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")

    def train_distributed(self, num_iterations: int) -> None:
        """Train the agent in a distributed manner."""
        if not self.distributed:
            raise ValueError("Agent is not configured for distributed training")

        def train_worker(rank, world_size):
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            for i in range(num_iterations):
                metrics = self.learn(self._sample_experiences())
                if rank == 0:
                    logging.info(f"Iteration {i + 1}/{num_iterations}, Metrics: {metrics}")
            dist.destroy_process_group()

        mp.spawn(train_worker, args=(self.num_workers,), nprocs=self.num_workers, join=True)

    def _sample_experiences(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Sample experiences from memory for training."""
        if len(self.memory) < self.config.get("batch_size", 32):
            logging.warning("Not enough experiences in memory to sample.")
        return [self.memory[i] for i in
                np.random.choice(len(self.memory), self.config.get("batch_size", 32), replace=False)]

    def sync_parameters(self) -> None:
        """Synchronize model parameters in a distributed setting."""
        if self.distributed:
            dist.barrier()
            for param in self.model.parameters():
                dist.broadcast(param.data, 0)

    def tune_hyperparameters(self, param_ranges: Dict[str, List[Any]], num_trials: int) -> Dict[str, Any]:
        """Tune hyperparameters using a random search strategy."""

        def objective(config):
            self.config.update(config)
            self.model = self._build_model()
            self.optimizer = self._build_optimizer()
            total_reward = 0
            for _ in range(self.config.get("eval_episodes", 10)):
                episode_reward = self._evaluate_episode()
                total_reward += episode_reward
            return total_reward / self.config.get("eval_episodes", 10)

        best_config = None
        best_reward = float('-inf')

        for _ in range(num_trials):
            config = {k: np.random.choice(v) for k, v in param_ranges.items()}
            reward = objective(config)
            if reward > best_reward:
                best_reward = reward
                best_config = config

        logging.info(f"Best hyperparameters found: {best_config}")
        return best_config

    def _evaluate_episode(self) -> float:
        """Evaluate the agent's performance in a single episode."""
        # Implement episode evaluation logic here
        pass

    def parallel_learning(self, num_processes: int, num_iterations: int) -> None:
        """Train the agent in parallel across multiple processes."""

        def worker(worker_id):
            local_agent = type(self)(self.state_size, self.action_size, self.config)
            for i in range(num_iterations):
                metrics = local_agent.learn(local_agent._sample_experiences())
                logging.info(f"Worker {worker_id}, Iteration {i + 1}/{num_iterations}, Metrics: {metrics}")
            return local_agent.model.state_dict()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(worker, range(num_processes)))

        # Aggregate results (e.g., average model parameters)
        avg_state_dict = {k: sum(d[k] for d in results) / num_processes for k in results[0].keys()}
        self.model.load_state_dict(avg_state_dict)

    def save_config(self, path: str) -> None:
        """Save the agent's configuration to a file."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logging.info(f"Configuration saved to {path}")

    def load_config(self, path: str) -> None:
        """Load the agent's configuration from a file."""
        try:
            with open(path, 'r') as f:
                self.config.update(json.load(f))
            logging.info(f"Configuration loaded from {path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {path}")
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from the configuration file: {path}")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")

    def checkpoint(self, path: str, iteration: int) -> None:
        """Save a checkpoint of the agent's state and configuration."""
        checkpoint_path = os.path.join(path, f"checkpoint_{iteration}.pt")
        self.save(checkpoint_path)
        config_path = os.path.join(path, f"config_{iteration}.json")
        self.save_config(config_path)

    def resume_from_checkpoint(self, path: str, iteration: int) -> None:
        """Resume the agent's state and configuration from a checkpoint."""
        checkpoint_path = os.path.join(path, f"checkpoint_{iteration}.pt")
        self.load(checkpoint_path)
        config_path = os.path.join(path, f"config_{iteration}.json")
        self.load_config(config_path)
