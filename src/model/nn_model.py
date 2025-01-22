from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F


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
            nn.Linear(hidden_dim, hidden_dim),
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


class WeightValue(nn.Module):
    def __init__(
        self,
        weight_type: Literal["vanilla", "two_layer", "time_mixer"],
        input_dim: int,
        hidden_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.weight_type = weight_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if weight_type == "vanilla":
            self.model = nn.Linear(input_dim, 1)
        if hidden_dim:
            if weight_type == "two_layer":
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                )
            elif weight_type == "time_mixer":
                raise NotImplementedError("TimeMixer is not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AlphaSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        value_weight_type: Literal["vanilla", "two_layer", "time_mixer"] = "vanilla",
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.w_q = nn.Linear(input_dim, embed_dim)
        self.w_k = nn.Linear(input_dim, embed_dim)
        self.w_v = WeightValue(value_weight_type, input_dim, vdim)
        self.attn_drop = nn.Dropout(dropout)

    def _reset_parameters(self):
        for layer in self.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, input_dim = x.shape
        query = (
            self.w_q(x)
            .reshape(batch_size, num_patch, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        key = (
            self.w_k(x)
            .reshape(batch_size, num_patch, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.w_v(x)
            .reshape(batch_size, num_patch, self.num_heads, 1)
            .permute(0, 2, 1, 3)
        )

        attn_score = self.attn_drop(
            F.softmax(
                torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5, dim=-1
            )
        )
        alpha_rank = (
            torch.matmul(attn_score, value)
            .transpose(1, 2)
            .reshape(batch_size, num_patch, 1)
        )

        return alpha_rank
