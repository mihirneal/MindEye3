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
        hidden_layers: int,
        scene_dim: int,
        embedding_dim: int,
        dropout: float = 0.0,
        num_subjects: int = 0,
        subject_embedding_dim: int = 0,
        subject_input_adapter: bool = False,
        clip_token_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = BrainEncoder(
            fmri_dim=fmri_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            scene_dim=scene_dim,
            dropout=dropout,
            num_subjects=num_subjects,
            subject_embedding_dim=subject_embedding_dim,
            subject_input_adapter=subject_input_adapter,
        )
        self.image_head = ProjectionHead(scene_dim, embedding_dim)
        self.clip_token_shape = clip_token_shape
        if clip_token_shape is None:
            self.clip_token_queries: nn.Parameter | None = None
            self.clip_token_norm: nn.LayerNorm | None = None
            self.clip_token_head: nn.Linear | None = None
        else:
            num_tokens, token_dim = clip_token_shape
            self.clip_token_queries = nn.Parameter(torch.randn(num_tokens, scene_dim) * 0.02)
            self.clip_token_norm = nn.LayerNorm(scene_dim)
            self.clip_token_head = nn.Linear(scene_dim, token_dim)

    def encode_brain(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(fmri, subject_id=subject_id)

    def forward(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        scene = self.encode_brain(fmri, subject_id=subject_id)
        return self.image_head(scene)

    def forward_with_reconstruction(
        self,
        fmri: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene = self.encode_brain(fmri, subject_id=subject_id)
        outputs = {"image": self.image_head(scene), "scene": scene}
        if (
            self.clip_token_head is not None
            and self.clip_token_norm is not None
            and self.clip_token_queries is not None
        ):
            token_state = scene[:, None, :] + self.clip_token_queries[None, :, :]
            outputs["clip_tokens"] = self.clip_token_head(self.clip_token_norm(token_state))
        return outputs
