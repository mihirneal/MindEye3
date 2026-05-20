from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from mindeye3.config import DataConfig
from mindeye3.data.types import BrainSample, StimulusEmbedding


@dataclass(frozen=True)
class PairedBatch:
    subject_id: torch.Tensor
    fmri: torch.Tensor
    stimulus_id: torch.Tensor
    image: torch.Tensor


class SyntheticBrainDataset(Dataset[tuple[BrainSample, StimulusEmbedding]]):
    """Synthetic paired fMRI and image embeddings with a learnable signal."""

    def __init__(self, config: DataConfig, seed: int = 0) -> None:
        self.config = config
        generator = torch.Generator().manual_seed(seed)

        latent_dim = min(config.fmri_dim, config.embedding_dim)
        latent = torch.randn(config.num_samples, latent_dim, generator=generator)
        fmri_projection = torch.randn(latent_dim, config.fmri_dim, generator=generator)
        image_projection = torch.randn(latent_dim, config.embedding_dim, generator=generator)

        fmri = latent @ fmri_projection
        image = latent @ image_projection
        fmri = fmri + config.noise_std * torch.randn(fmri.shape, generator=generator)
        image = image + config.noise_std * torch.randn(image.shape, generator=generator)

        self.fmri = torch.nn.functional.normalize(fmri.float(), dim=-1)
        self.image = torch.nn.functional.normalize(image.float(), dim=-1)
        self.subject_ids = torch.arange(config.num_samples) % config.num_subjects
        self.stimulus_ids = torch.arange(config.num_samples)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> tuple[BrainSample, StimulusEmbedding]:
        return (
            BrainSample(
                subject_id=int(self.subject_ids[index]),
                fmri=self.fmri[index],
                stimulus_id=int(self.stimulus_ids[index]),
            ),
            StimulusEmbedding(image=self.image[index]),
        )


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


def create_dataloaders(
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader[PairedBatch], DataLoader[PairedBatch]]:
    if config.name != "synthetic":
        raise ValueError(f"Unsupported dataset for V0 scaffold: {config.name}")
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    dataset = SyntheticBrainDataset(config, seed=seed)
    train_size = int(len(dataset) * config.train_fraction)
    eval_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=split_generator,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_paired_batch,
        ),
        DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_paired_batch,
        ),
    )

