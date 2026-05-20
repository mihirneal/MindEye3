from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str = "synthetic"
    num_samples: int = 256
    num_subjects: int = 4
    fmri_dim: int = 128
    embedding_dim: int = 64
    noise_std: float = 0.05
    train_fraction: float = 0.8
    batch_size: int = 32


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    scene_dim: int = 64
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    device: str = "cpu"
    log_every: int = 10


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: list[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass(frozen=True)
class MindEyeConfig:
    seed: int = 7
    output_dir: str = "outputs/v0_synthetic"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")
    return value


def load_config(path: str | Path) -> MindEyeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _as_dict(raw)
    return MindEyeConfig(
        seed=int(root.get("seed", MindEyeConfig.seed)),
        output_dir=str(root.get("output_dir", MindEyeConfig.output_dir)),
        data=DataConfig(**_as_dict(root.get("data"))),
        model=ModelConfig(**_as_dict(root.get("model"))),
        training=TrainingConfig(**_as_dict(root.get("training"))),
        evaluation=EvaluationConfig(**_as_dict(root.get("evaluation"))),
    )


def config_to_dict(config: MindEyeConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "output_dir": config.output_dir,
        "data": vars(config.data),
        "model": vars(config.model),
        "training": vars(config.training),
        "evaluation": vars(config.evaluation),
    }

