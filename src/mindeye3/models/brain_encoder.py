from __future__ import annotations

import torch
from torch import nn


class BrainEncoder(nn.Module):
    def __init__(
        self,
        fmri_dim: int,
        hidden_dim: int,
        scene_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, scene_dim),
        )

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        return self.network(fmri)

