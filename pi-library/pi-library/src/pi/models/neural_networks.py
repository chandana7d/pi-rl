import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)
