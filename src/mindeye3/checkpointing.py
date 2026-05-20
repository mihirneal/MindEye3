from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mindeye3.config import MindEyeConfig, config_to_dict


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: MindEyeConfig,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_to_dict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)

