from __future__ import annotations

from dataclasses import dataclass

import torch

from mindeye3.data.types import BrainSample, StimulusEmbedding


@dataclass(frozen=True)
class PairedBatch:
    subject_id: torch.Tensor
    fmri: torch.Tensor
    stimulus_id: torch.Tensor
    image: torch.Tensor
    clip_tokens: torch.Tensor | None = None


def collate_paired_batch(
    items: list[tuple[BrainSample, StimulusEmbedding]],
) -> PairedBatch:
    return PairedBatch(
        subject_id=torch.tensor([sample.subject_id for sample, _ in items], dtype=torch.long),
        fmri=torch.stack([sample.fmri for sample, _ in items]),
        stimulus_id=torch.tensor(
            [sample.stimulus_id if sample.stimulus_id is not None else -1 for sample, _ in items],
            dtype=torch.long,
        ),
        image=torch.stack([embedding.require("image") for _, embedding in items]),
        clip_tokens=_stack_optional(items, "clip_tokens"),
    )


def _stack_optional(
    items: list[tuple[BrainSample, StimulusEmbedding]],
    key: str,
) -> torch.Tensor | None:
    values = [getattr(embedding, key) for _, embedding in items]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise KeyError(f"Some items are missing stimulus embedding: {key}")
    return torch.stack([value for value in values if value is not None])
