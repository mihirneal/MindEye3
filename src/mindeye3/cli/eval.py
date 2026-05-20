from __future__ import annotations

import argparse
import torch

from mindeye3.checkpointing import load_checkpoint
from mindeye3.config import load_config
from mindeye3.data.synthetic import create_dataloaders
from mindeye3.evaluation import evaluate_model
from mindeye3.training import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MindEye3 retrieval checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    device = torch.device(config.training.device)
    _, eval_loader = create_dataloaders(config.data, seed=config.seed)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate_model(model, eval_loader, config.evaluation.top_k, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

