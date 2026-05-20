from pathlib import Path

import yaml

from mindeye3.checkpointing import load_checkpoint
from mindeye3.cli.build_nsd_embeddings import build_parser as build_embeddings_parser
from mindeye3.cli.eval import build_parser as build_eval_parser
from mindeye3.cli.train import build_parser as build_train_parser
from mindeye3.config import load_config
from mindeye3.training import train


def test_cli_parsers_accept_required_args() -> None:
    train_args = build_train_parser().parse_args(["--config", "configs/v0_synthetic.yaml"])
    eval_args = build_eval_parser().parse_args(
        ["--config", "configs/v0_synthetic.yaml", "--checkpoint", "checkpoint.pt"]
    )
    embedding_args = build_embeddings_parser().parse_args(
        ["--stimuli", "nsd_stimuli.hdf5", "--output", "cache.pt", "--limit", "10"]
    )

    assert train_args.config == "configs/v0_synthetic.yaml"
    assert eval_args.checkpoint == "checkpoint.pt"
    assert embedding_args.limit == 10


def test_training_smoke_writes_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config = {
        "seed": 11,
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "name": "synthetic",
            "num_samples": 32,
            "num_subjects": 2,
            "fmri_dim": 16,
            "embedding_dim": 12,
            "noise_std": 0.05,
            "train_fraction": 0.75,
            "batch_size": 8,
        },
        "model": {
            "hidden_dim": 24,
            "scene_dim": 10,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "temperature": 0.1,
            "device": "cpu",
            "log_every": 0,
        },
        "evaluation": {
            "top_k": [1, 3],
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    checkpoint_path = train(load_config(config_path))
    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert checkpoint["epoch"] == 1
    assert "top1" in checkpoint["metrics"]
