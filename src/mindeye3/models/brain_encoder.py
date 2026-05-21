from __future__ import annotations

import torch
from torch import nn


class BrainEncoder(nn.Module):
    def __init__(
        self,
        fmri_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        scene_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        input_dim = fmri_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, scene_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        return self.network(fmri)
