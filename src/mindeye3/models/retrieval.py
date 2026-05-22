from __future__ import annotations

import torch
from torch import nn

from mindeye3.models.brain_encoder import BrainEncoder
from mindeye3.models.brain_token_encoder import BrainTokenEncoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projection(x), dim=-1)


class SceneQueryReconstructionHead(nn.Module):
    def __init__(self, scene_dim: int, output_tokens: int, output_dim: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_tokens, scene_dim) * 0.02)
        self.norm = nn.LayerNorm(scene_dim)
        self.head = nn.Linear(scene_dim, output_dim)

    def forward(
        self,
        scene: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_state = scene[:, None, :] + self.queries[None, :, :]
        return self.head(self.norm(token_state))


class CrossAttentionReconstructionHead(nn.Module):
    """Brain-IT-style decoder: image query tokens attend to brain-cluster tokens."""

    def __init__(
        self,
        brain_token_dim: int,
        output_tokens: int,
        output_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("clip_token_decoder_layers must be at least 1")
        if heads < 1:
            raise ValueError("clip_token_decoder_heads must be positive")
        if brain_token_dim % heads != 0:
            raise ValueError("brain token dimension must be divisible by clip_token_decoder_heads")

        self.queries = nn.Parameter(torch.randn(output_tokens, brain_token_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=brain_token_dim,
            nhead=heads,
            dim_feedforward=brain_token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(brain_token_dim)
        self.head = nn.Linear(brain_token_dim, output_dim)

    def forward(
        self,
        scene: torch.Tensor,
        brain_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if brain_tokens is None:
            raise ValueError("bit_cross_attention reconstruction requires brain token encoder features")
        queries = self.queries[None, :, :].expand(brain_tokens.shape[0], -1, -1)
        token_state = self.decoder(tgt=queries, memory=brain_tokens)
        return self.head(self.norm(token_state))


class RetrievalModel(nn.Module):
    def __init__(
        self,
        fmri_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        scene_dim: int,
        embedding_dim: int,
        dropout: float = 0.0,
        encoder_type: str = "mlp",
        num_subjects: int = 0,
        subject_embedding_dim: int = 0,
        subject_input_adapter: bool = False,
        brain_tokens: int = 128,
        brain_token_dim: int = 256,
        brain_transformer_layers: int = 2,
        brain_transformer_heads: int = 8,
        clip_token_shape: tuple[int, int] | None = None,
        clip_token_decoder: str = "scene_query",
        clip_token_decoder_layers: int = 2,
        clip_token_decoder_heads: int = 8,
        feature_group_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type == "mlp":
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
        elif encoder_type == "brain_tokens":
            self.encoder = BrainTokenEncoder(
                fmri_dim=fmri_dim,
                num_tokens=brain_tokens,
                token_dim=brain_token_dim,
                transformer_layers=brain_transformer_layers,
                transformer_heads=brain_transformer_heads,
                scene_dim=scene_dim,
                dropout=dropout,
                num_subjects=num_subjects,
                subject_embedding_dim=subject_embedding_dim,
                subject_input_adapter=subject_input_adapter,
                feature_group_ids=feature_group_ids,
            )
        else:
            raise ValueError("encoder_type must be one of: mlp, brain_tokens")
        self.image_head = ProjectionHead(scene_dim, embedding_dim)
        self.clip_token_shape = clip_token_shape
        if clip_token_shape is None:
            self.clip_token_decoder: nn.Module | None = None
        else:
            num_tokens, token_dim = clip_token_shape
            clip_token_decoder = clip_token_decoder.lower()
            if clip_token_decoder == "scene_query":
                self.clip_token_decoder = SceneQueryReconstructionHead(scene_dim, num_tokens, token_dim)
            elif clip_token_decoder == "bit_cross_attention":
                if not isinstance(self.encoder, BrainTokenEncoder):
                    raise ValueError("bit_cross_attention clip token decoder requires encoder_type='brain_tokens'")
                self.clip_token_decoder = CrossAttentionReconstructionHead(
                    brain_token_dim=self.encoder.token_dim,
                    output_tokens=num_tokens,
                    output_dim=token_dim,
                    layers=clip_token_decoder_layers,
                    heads=clip_token_decoder_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError("clip_token_decoder must be one of: scene_query, bit_cross_attention")

    def encode_brain(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(fmri, subject_id=subject_id)

    def encode_brain_features(
        self,
        fmri: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(self.encoder, BrainTokenEncoder):
            brain_tokens = self.encoder.forward_tokens(fmri, subject_id=subject_id)
            scene = self.encoder.pool_tokens(brain_tokens)
            return scene, brain_tokens
        return self.encoder(fmri, subject_id=subject_id), None

    def forward(self, fmri: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        scene, _ = self.encode_brain_features(fmri, subject_id=subject_id)
        return self.image_head(scene)

    def forward_with_reconstruction(
        self,
        fmri: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene, brain_tokens = self.encode_brain_features(fmri, subject_id=subject_id)
        outputs = {"image": self.image_head(scene), "scene": scene}
        if self.clip_token_decoder is not None:
            outputs["clip_tokens"] = self.clip_token_decoder(scene, brain_tokens)
        return outputs
