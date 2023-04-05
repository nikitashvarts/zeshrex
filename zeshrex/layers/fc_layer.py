from torch import nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 256,
            dropout_rate: float = 0.0,
            use_activation: bool = True
    ):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.relu = F.relu

    def forward(self, x):
        out = self.fc1(x)
        if self.use_activation:
            out = self.tanh(out)
            # out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
