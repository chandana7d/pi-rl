# main.py
import numpy as np
import torch
#from a2c_agent import A2CAgent
from pi.agents.actor_critc.a2c_agent import BaseAgent

def main():
    # Configuration for the A2C agent
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

    # Initialize the A2C agent
    state_size = 4  # Example state size
    action_size = 2  # Example action space size
    agent = A2CAgent(state_size=state_size, action_size=action_size, config=config)

    # Train the agent for a few episodes
    num_episodes = 5
    for episode in range(num_episodes):
        state = np.random.rand(state_size)  # Dummy initial state
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state = np.random.rand(state_size)  # Simulated next state
            reward = np.random.rand()  # Random reward
            done = np.random.choice([True, False], p=[0.1, 0.9])  # Random episode termination

            agent.remember((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train the agent if enough data is collected
            agent.learn()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Save the trained model
    agent.save("a2c_agent_checkpoint.pth")

    # Load and test the model
    agent.load("a2c_agent_checkpoint.pth")
    test_state = np.random.rand(state_size)
    test_action = agent.act(test_state)
    print(f"Test State Action: {test_action}")

    # Optional parallel training
    agent.train_in_parallel(num_processes=2, num_iterations=10)

if __name__ == "__main__":
    main()
