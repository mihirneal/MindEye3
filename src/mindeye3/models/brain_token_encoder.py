from __future__ import annotations

import math

import torch
from torch import nn


class BrainTokenEncoder(nn.Module):
    def __init__(
        self,
        fmri_dim: int,
        num_tokens: int,
        token_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        scene_dim: int,
        dropout: float = 0.0,
        num_subjects: int = 0,
        subject_embedding_dim: int = 0,
        subject_input_adapter: bool = False,
        feature_group_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1")
        if transformer_heads < 1:
            raise ValueError("transformer_heads must be positive")
        if token_dim % transformer_heads != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if subject_embedding_dim < 0:
            raise ValueError("subject_embedding_dim must be non-negative")

        self.fmri_dim = fmri_dim
        self.token_dim = token_dim
        if feature_group_ids is None:
            self.num_tokens = num_tokens
            self.chunk_size = math.ceil(fmri_dim / num_tokens)
            self.padded_dim = self.num_tokens * self.chunk_size
            self.register_buffer("group_feature_index", None, persistent=False)
            self.register_buffer("group_feature_mask", None, persistent=False)
        else:
            feature_group_ids = torch.as_tensor(feature_group_ids, dtype=torch.long)
            if feature_group_ids.ndim != 1 or feature_group_ids.numel() != fmri_dim:
                raise ValueError("feature_group_ids must be a rank-1 tensor with length fmri_dim")
            if int(feature_group_ids.min()) < 0:
                raise ValueError("feature_group_ids must be non-negative")
            group_index, group_mask = _build_group_feature_index(feature_group_ids)
            self.num_tokens = int(group_index.shape[0])
            self.chunk_size = int(group_index.shape[1])
            self.padded_dim = fmri_dim
            self.register_buffer("group_feature_index", group_index, persistent=False)
            self.register_buffer("group_feature_mask", group_mask, persistent=False)

        self.subject_gain: nn.Embedding | None = None
        self.subject_bias: nn.Embedding | None = None
        if subject_input_adapter:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be positive when subject input adapters are enabled")
            self.subject_gain = nn.Embedding(num_subjects, fmri_dim)
            self.subject_bias = nn.Embedding(num_subjects, fmri_dim)
            nn.init.zeros_(self.subject_gain.weight)
            nn.init.zeros_(self.subject_bias.weight)

        self.local_weight = nn.Parameter(torch.empty(self.num_tokens, self.chunk_size, token_dim))
        self.local_bias = nn.Parameter(torch.zeros(self.num_tokens, token_dim))
        nn.init.normal_(self.local_weight, std=1.0 / math.sqrt(self.chunk_size))
        self.position = nn.Parameter(torch.randn(self.num_tokens, token_dim) * 0.02)

        self.subject_embedding: nn.Embedding | None = None
        self.subject_projection: nn.Linear | None = None
        if subject_embedding_dim > 0:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be positive when subject conditioning is enabled")
            self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
            self.subject_projection = nn.Linear(subject_embedding_dim, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.scene = nn.Linear(token_dim, scene_dim)

    def forward_tokens(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.subject_gain is not None and self.subject_bias is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when subject input adapters are enabled")
            fmri = fmri * (1.0 + self.subject_gain(subject_id.long())) + self.subject_bias(subject_id.long())

        if self.group_feature_index is not None and self.group_feature_mask is not None:
            flat_index = self.group_feature_index.reshape(-1)
            chunks = fmri[:, flat_index].reshape(fmri.shape[0], self.num_tokens, self.chunk_size)
            chunks = chunks * self.group_feature_mask[None, :, :]
        elif self.padded_dim != self.fmri_dim:
            fmri = torch.nn.functional.pad(fmri, (0, self.padded_dim - self.fmri_dim))
            chunks = fmri.reshape(fmri.shape[0], self.num_tokens, self.chunk_size)
        else:
            chunks = fmri.reshape(fmri.shape[0], self.num_tokens, self.chunk_size)
        tokens = torch.einsum("btc,tcd->btd", chunks, self.local_weight) + self.local_bias
        tokens = tokens + self.position[None, :, :]

        if self.subject_embedding is not None and self.subject_projection is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when subject conditioning is enabled")
            subject = self.subject_projection(self.subject_embedding(subject_id.long()))
            tokens = tokens + subject[:, None, :]

        tokens = self.transformer(tokens)
        return self.norm(tokens)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        return self.scene(pooled)

    def forward(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        return self.pool_tokens(self.forward_tokens(fmri, subject_id=subject_id))


def _build_group_feature_index(feature_group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups = []
    for group_id in torch.unique(feature_group_ids, sorted=True):
        groups.append(torch.nonzero(feature_group_ids == group_id, as_tuple=False).flatten())
    max_features = max(int(group.numel()) for group in groups)
    index = torch.zeros((len(groups), max_features), dtype=torch.long)
    mask = torch.zeros((len(groups), max_features), dtype=torch.float32)
    for row, group in enumerate(groups):
        index[row, : group.numel()] = group
        mask[row, : group.numel()] = 1.0
    return index, mask
