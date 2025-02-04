from typing import Optional
import torch
import numpy as np


def dcg_at_k(relevance: torch.Tensor, k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain
    Args:
        relevance: Array of relevance scores
        k: Number of elements to consider
    """
    if k is not None:
        relevance = relevance[:k]

    gains = 2**relevance - 1
    discounts = torch.log2(torch.arange(2, len(relevance) + 2, device=relevance.device))

    return (gains / discounts).sum().item()


def ndcg(
    batch_preds: torch.Tensor,
    batch_targets: torch.Tensor,
    k: Optional[int],
) -> Optional[float]:
    """
    preds (torch.Tensor): predicted alphas (1, n_assets, n_alphas)
    target (torch.Tensor): true alphas (1, n_assets, 1)
    k (int): number of alphas
    """
    top_ndcgs = []

    if k is not None:
        for batch_pred, batch_target in zip(batch_preds, batch_targets):
            batch_pred = batch_pred.permute(1, 0)

            for pred_alpha in batch_pred:
                _, top_indices = torch.topk(
                    pred_alpha, k, dim=-1, largest=True, sorted=True
                )
                if batch_target.dim() == 2:
                    top_indices = top_indices.unsqueeze(-1)

                true_top_sorted = torch.gather(batch_target, dim=0, index=top_indices)
                dcg = dcg_at_k(true_top_sorted, k)
                ideal_top_sorted, _ = torch.topk(
                    batch_target, k, dim=0, largest=True, sorted=True
                )
                idcg = dcg_at_k(ideal_top_sorted, k)

                if idcg > 0:
                    top_ndcgs.append(dcg / idcg)
                else:
                    top_ndcgs.append(torch.tensor(0.0, device=batch_target.device))

                top_ndcgs.append((dcg / idcg))

        top_ndcgs = np.array(top_ndcgs)

        return float(np.mean(top_ndcgs))
