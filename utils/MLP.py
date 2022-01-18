import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, data_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(data_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        a = self.fc1(x)
        a = F.relu(a)

        b = self.fc2(a)
        b = F.relu(b)

        c = self.fc3(b)

        return c