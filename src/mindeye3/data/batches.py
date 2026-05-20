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
    )
