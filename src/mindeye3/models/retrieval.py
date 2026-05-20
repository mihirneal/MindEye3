from __future__ import annotations

import torch
from torch import nn

from mindeye3.models.brain_encoder import BrainEncoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projection(x), dim=-1)


class RetrievalModel(nn.Module):
    def __init__(
        self,
        fmri_dim: int,
        hidden_dim: int,
        scene_dim: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = BrainEncoder(
            fmri_dim=fmri_dim,
            hidden_dim=hidden_dim,
            scene_dim=scene_dim,
            dropout=dropout,
        )
        self.image_head = ProjectionHead(scene_dim, embedding_dim)

    def encode_brain(self, fmri: torch.Tensor) -> torch.Tensor:
        return self.encoder(fmri)

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        scene = self.encode_brain(fmri)
        return self.image_head(scene)

