import unittest
from pi.policies.dqn_policy import DQNPolicy

class TestDQNPolicy(unittest.TestCase):
    def test_policy_initialization(self):
        policy = DQNPolicy()
        self.assertIsNotNone(policy)

if __name__ == "__main__":
    unittest.main()
