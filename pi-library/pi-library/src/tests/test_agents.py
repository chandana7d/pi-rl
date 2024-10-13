import unittest
from pi.agents.dqn_agent import DQNAgent

class TestDQNAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = DQNAgent()
        self.assertIsNotNone(agent)

if __name__ == "__main__":
    unittest.main()
