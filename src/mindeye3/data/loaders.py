from __future__ import annotations

from torch.utils.data import DataLoader

from mindeye3.config import DataConfig
from mindeye3.data.batches import PairedBatch
from mindeye3.data.nsd import create_nsd_dataloaders
from mindeye3.data.synthetic import create_synthetic_dataloaders


def create_dataloaders(
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader[PairedBatch], DataLoader[PairedBatch]]:
    if config.name == "synthetic":
        return create_synthetic_dataloaders(config, seed)
    if config.name == "nsd":
        return create_nsd_dataloaders(config, seed)
    raise ValueError(f"Unsupported dataset: {config.name}")
