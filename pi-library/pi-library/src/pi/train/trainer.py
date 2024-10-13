class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                # Handle training logic
                state = next_state
