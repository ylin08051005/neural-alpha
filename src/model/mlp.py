import torch
from torch import nn


class MAMinusMA(nn.Module):
    def __init__(self, max_history: int = 20, n_alpha: int = 10) -> None:
        super().__init__()
        self.n_alpha = n_alpha
        self.alpha_linear = nn.Linear(max_history, n_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_weight = self.alpha_linear.weight.softmax(dim=-1)
        alpha = nn.functional.linear(x, alpha_weight)
        alpha = (alpha[:, : self.n_alpha] - alpha[:, self.n_alpha :]).tanh()

        return alpha


class NeuralAlpha(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

    def _reset_parameters(self):
        for layer in self.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
