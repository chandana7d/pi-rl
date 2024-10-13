from pi.agents.dqn_agent import DQNAgent
from pi.env.base_env import BaseEnv
from pi.train.trainer import Trainer

def main():
    env = BaseEnv()
    agent = DQNAgent()
    trainer = Trainer(agent, env)
    trainer.train(episodes=1000)

if __name__ == "__main__":
    main()
