from typing import Literal

import torch
from torch import nn


def soft_rank(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute soft ranks using a differentiable approximation.

    Args:
        x: Input tensor
        temperature: Temperature parameter to control smoothness (lower = closer to hard ranking)
    """
    ranks = 1 / (
        1 + torch.exp(
            -1.83 * ((x - x.mean()) / 2 * x.std())
        )
    )

    return ranks


class ICLoss(nn.Module):
    def __init__(self, ic_type: Literal["cross_entropy", "spearman"]) -> None:
        """
        Initialize the loss function.
        :param correlation_type: Type of correlation ('pearson' or 'spearman'). Default is 'pearson'.
        """
        super().__init__()
        assert ic_type in [
            "spearman",
            "cross_entropy",
        ], "correlation_type must be 'cross_entropy' or 'spearman'."
        self.ic_type = ic_type

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss to maximize absolute Information Coefficient (IC).
        :param predictions: Tensor of model predictions (batch_size,).
        :param targets: Tensor of true values (batch_size,).
        :return: Loss value (to minimize).
        """
        correlation = torch.tensor(0.0)

        if self.ic_type == "cross_entropy":
            pass

        elif self.ic_type == "spearman":
            pred_rank = soft_rank(predictions)
            target_rank = soft_rank(targets)
            pred_n = pred_rank - pred_rank.mean()
            target_n = target_rank - target_rank.mean()
            pred_n = pred_n / (pred_n.norm() + 1e-8)
            target_n = target_n / (target_n.norm() + 1e-8)
            correlation = (pred_n * target_n).sum()

        loss = -torch.abs(correlation)

        return loss
