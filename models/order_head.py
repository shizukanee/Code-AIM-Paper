from __future__ import annotations

import torch
import torch.nn as nn


class OrderHead(nn.Module):
    """Predict whether dependent is left of head from hidden states."""

    def __init__(self, hidden_size: int, proj: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = hidden_size * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj, 1),
        )

    @staticmethod
    def pair_features(h_dep: torch.Tensor, h_head: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(h_dep - h_head)
        hadamard = h_dep * h_head
        return torch.cat([h_dep, h_head, abs_diff, hadamard], dim=-1)

    def forward(self, h_dep: torch.Tensor, h_head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pair_features(h_dep, h_head)
        # Keep head numerics consistent under mixed precision training.
        weight_dtype = self.mlp[0].weight.dtype
        if features.dtype != weight_dtype:
            features = features.to(weight_dtype)
        logits = self.mlp(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, logits
