import unittest
from pi.models.neural_networks import DQNModel
import torch

class TestDQNModel(unittest.TestCase):
    def test_forward_pass(self):
        model = DQNModel()
        input_tensor = torch.randn(1, 4)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 2))

if __name__ == "__main__":
    unittest.main()
