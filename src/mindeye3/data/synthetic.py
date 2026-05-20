from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from mindeye3.config import DataConfig
from mindeye3.data.batches import PairedBatch, collate_paired_batch
from mindeye3.data.types import BrainSample, StimulusEmbedding


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


def create_synthetic_dataloaders(
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader[PairedBatch], DataLoader[PairedBatch]]:
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


create_dataloaders = create_synthetic_dataloaders
