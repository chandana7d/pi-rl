import torch
import gym
from pi.agents.actor_critic.a2c_agent import A2CAgent
from pi.agents.actor_critic.a3c_agent import A3CAgent
from pi.agents.actor_critic.sac_agent import SACAgent
from torch.multiprocessing import Process, Pipe

def test_a2c_agent():
    # Initialize parameters
    config = {
        "memory_size": 10000,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "value_loss_weight": 0.5,
        "num_iterations": 100,
        "num_processes": 2,
        "log_dir": "./logs"
    }

    state_size = 4  # Example state size
    action_size = 2  # Example action size

    # Initialize the agent
    agent = A2CAgent(state_size=state_size, action_size=action_size, config=config)
    print("A2C Agent initialized successfully!")

    # Example of running training (dummy input for demonstration purposes)
    states = torch.randn(5, state_size)  # 5 dummy states
    actions = torch.randint(0, action_size, (5,))  # 5 dummy actions
    rewards = torch.randn(5)  # 5 dummy rewards

    # Train the agent and print the loss
    loss = agent.train(states, actions, rewards)
    print(f"Training loss: {loss}")

def test_a3c_agent():
    # Initialize parameters
    config = {
        "memory_size": 10000,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "value_loss_weight": 0.5,
        "num_workers": 4,
        "num_iterations": 100
    }

    state_size = 4  # Example state size for CartPole
    action_size = 2  # Example action size for CartPole

    # Initialize the agent
    agent = A3CAgent(state_size=state_size, action_size=action_size, config=config)
    print("A3C Agent initialized successfully!")

    # Setup multiprocessing with Gym environments
    envs = [gym.make("CartPole-v1") for _ in range(config["num_workers"])]
    workers = []
    parent_conns, child_conns = zip(*[Pipe() for _ in range(config["num_workers"])])

    for worker_id, (parent_conn, child_conn) in enumerate(zip(parent_conns, child_conns)):
        worker = Process(target=agent.worker, args=(worker_id, child_conn, envs[worker_id], config))
        worker.start()
        workers.append(worker)

    # Train the agent
    for iteration in range(config["num_iterations"]):
        for parent_conn in parent_conns:
            parent_conn.send("train")

        rewards = [parent_conn.recv() for parent_conn in parent_conns]
        print(f"Iteration {iteration}, Rewards: {rewards}")

    # Stop workers
    for parent_conn in parent_conns:
        parent_conn.send("stop")

    for worker in workers:
        worker.join()

def test_sac_agent():
    def main():
        config = {
            "memory_size": 10000,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "num_workers": 4,
            "num_iterations": 100
        }

        state_size = 4  # Example state size for CartPole
        action_size = 2  # Example action size for CartPole

        # Initialize the agent
        agent = SACAgent(state_size=state_size, action_size=action_size, config=config)
        print("SAC Agent initialized successfully!")

        # Setup multiprocessing with Gym environments
        envs = [gym.make("CartPole-v1") for _ in range(config["num_workers"])]
        workers = []
        parent_conns, child_conns = zip(*[Pipe() for _ in range(config["num_workers"])])

        for worker_id, (parent_conn, child_conn) in enumerate(zip(parent_conns, child_conns)):
            worker = Process(target=agent.worker, args=(worker_id, child_conn, envs[worker_id], config))
            worker.start()
            workers.append(worker)

        # Train the agent
        for iteration in range(config["num_iterations"]):
            for parent_conn in parent_conns:
                parent_conn.send("train")

            rewards = [parent_conn.recv() for parent_conn in parent_conns]
            print(f"Iteration {iteration}, Rewards: {rewards}")

        # Stop workers
        for parent_conn in parent_conns:
            parent_conn.send("stop")

        for worker in workers:
            worker.join()


if __name__ == "__main__":
    print("Testing A2C Agent...")
    test_a2c_agent()
    print("\nTesting A3C Agent...")
    test_a3c_agent()
    print("\nTesting A3C Agent...")
    test_sac_agent()
