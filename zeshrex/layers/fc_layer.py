from typing import Optional

from torch import nn


class FCLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 0,
        dropout_rate: float = 0.2,
        use_activation: bool = True,
    ):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()

        fc1_output = hidden_dim if hidden_dim != 0 else output_dim
        self.fc1: nn.Linear = nn.Linear(input_dim, fc1_output)
        self.fc2: Optional[nn.Linear] = nn.Linear(hidden_dim, output_dim) if hidden_dim != 0 else None

    def forward(self, x):
        out = x
        if self.fc2 is not None:
            out = self.fc1(out)

        if self.use_activation:
            out = self.tanh(out)
        out = self.dropout(out)

        if self.fc2 is not None:
            out = self.fc2(out)
        else:
            out = self.fc1(out)
        return out
