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
        num_subjects: int = 0,
        subject_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if subject_embedding_dim < 0:
            raise ValueError("subject_embedding_dim must be non-negative")

        self.subject_embedding: nn.Embedding | None = None
        if subject_embedding_dim > 0:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be positive when subject conditioning is enabled")
            self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)

        layers: list[nn.Module] = []
        input_dim = fmri_dim + subject_embedding_dim
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

    def forward(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.subject_embedding is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when subject conditioning is enabled")
            subject = self.subject_embedding(subject_id.long())
            fmri = torch.cat([fmri, subject], dim=-1)
        return self.network(fmri)
