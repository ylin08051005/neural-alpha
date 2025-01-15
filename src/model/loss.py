from typing import Literal, List

import torch
import torchsort
from torch import nn


class ICLoss(nn.Module):
    def __init__(self, correlation_type: Literal["pearson", "spearman"]):
        """
        Initialize the loss function.
        :param correlation_type: Type of correlation ('pearson' or 'spearman'). Default is 'pearson'.
        """
        super().__init__()
        assert correlation_type in [
            "pearson",
            "spearman",
        ], "correlation_type must be 'pearson' or 'spearman'."
        self.correlation_type = correlation_type

    def forward(self, predictions, targets):
        """
        Compute the loss to maximize absolute Information Coefficient (IC).
        :param predictions: Tensor of model predictions (batch_size,).
        :param targets: Tensor of true values (batch_size,).
        :return: Loss value (to minimize).
        """
        correlation = torch.tensor(0.0)

        if self.correlation_type == "pearson":
            pred_mean = torch.mean(predictions)
            target_mean = torch.mean(targets)
            cov = torch.sum((predictions - pred_mean) * (targets - target_mean))
            pred_std = torch.std(predictions)
            target_std = torch.std(targets)
            correlation = cov / (pred_std * target_std + 1e-6)

        elif self.correlation_type == "spearman":
            pred_rank = torchsort.soft_rank(predictions)
            target_rank = torchsort.soft_rank(targets)

            if pred_rank is not None and target_rank is not None:
                pred_n = pred_rank - pred_rank.mean()
                target_n = target_rank - target_rank.mean()
                pred_n = pred_n / pred_n.norm()
                target_n = target_n / target_n.norm()
                correlation = (pred_n * target_n).sum()

        loss = -torch.abs(correlation)
        return loss
