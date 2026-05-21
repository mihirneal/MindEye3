from __future__ import annotations

import argparse
import torch

from mindeye3.checkpointing import load_checkpoint
from mindeye3.config import load_config
from mindeye3.data import create_dataloaders
from mindeye3.evaluation import evaluate_model
from mindeye3.training import build_model, infer_batch_dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MindEye3 retrieval checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    parser.add_argument(
        "--split",
        choices=["eval", "train", "both"],
        default="eval",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Optional candidate-pool size for MindEye2-style top-1 retrieval.",
    )
    parser.add_argument(
        "--candidate-repeats",
        type=int,
        default=30,
        help="Number of random candidate pools to average when candidate-pool evaluation is enabled.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=0,
        help="Random seed for candidate-pool evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    device = torch.device(config.training.device)
    train_loader, eval_loader = create_dataloaders(config.data, seed=config.seed)
    reference_loader = train_loader if args.split == "train" else eval_loader
    fmri_dim, embedding_dim, clip_token_shape = infer_batch_dims(reference_loader)
    model = build_model(
        config,
        fmri_dim=fmri_dim,
        embedding_dim=embedding_dim,
        clip_token_shape=clip_token_shape,
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    loaders = {"train": train_loader, "eval": eval_loader}
    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for split in splits:
        metrics = evaluate_model(
            model,
            loaders[split],
            config.evaluation.top_k,
            device,
            candidate_pool_size=args.candidate_pool_size,
            candidate_repeats=args.candidate_repeats,
            candidate_seed=args.candidate_seed,
        )
        for key, value in metrics.items():
            print(f"{split}_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
