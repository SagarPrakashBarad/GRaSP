import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNConv, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.fc1(torch.mm(adj, x)))
        x = self.fc2(torch.mm(adj, x))
        return x
