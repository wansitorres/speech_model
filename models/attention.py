import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Attention-based pooling over time.
    Input: (batch, features, time)
    Output: (batch, features)
    """

    def __init__(self, in_features, hidden=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: (batch, features, time)
        x = x.transpose(1, 2)   # (batch, time, features)

        weights = self.attention(x)   # (batch, time, 1)
        weights = torch.softmax(weights, dim=1)  # normalize over time

        pooled = torch.sum(x * weights, dim=1)   # (batch, features)
        return pooled
