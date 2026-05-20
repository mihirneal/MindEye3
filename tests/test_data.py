import torch

from mindeye3.config import DataConfig
from mindeye3.data.synthetic import SyntheticBrainDataset, create_dataloaders


def test_synthetic_dataset_shapes_and_determinism() -> None:
    config = DataConfig(num_samples=8, fmri_dim=16, embedding_dim=12)
    dataset_a = SyntheticBrainDataset(config, seed=3)
    dataset_b = SyntheticBrainDataset(config, seed=3)

    sample, embedding = dataset_a[0]
    sample_b, embedding_b = dataset_b[0]

    assert len(dataset_a) == 8
    assert sample.fmri.shape == (16,)
    assert embedding.require("image").shape == (12,)
    assert torch.allclose(sample.fmri, sample_b.fmri)
    assert torch.allclose(embedding.require("image"), embedding_b.require("image"))


def test_create_dataloaders_returns_paired_batches() -> None:
    config = DataConfig(num_samples=20, fmri_dim=16, embedding_dim=12, batch_size=5)
    train_loader, eval_loader = create_dataloaders(config, seed=5)

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))

    assert train_batch.fmri.shape == (5, 16)
    assert train_batch.image.shape == (5, 12)
    assert eval_batch.fmri.shape[-1] == 16

