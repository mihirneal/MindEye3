from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class BrainSample:
    subject_id: int
    fmri: torch.Tensor
    stimulus_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StimulusEmbedding:
    image: torch.Tensor | None = None
    clip_tokens: torch.Tensor | None = None
    text: torch.Tensor | None = None
    video: torch.Tensor | None = None
    audio: torch.Tensor | None = None

    def require(self, key: str) -> torch.Tensor:
        value = getattr(self, key)
        if value is None:
            raise KeyError(f"Missing stimulus embedding: {key}")
        return value
