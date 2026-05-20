from __future__ import annotations

import argparse

from mindeye3.config import load_config
from mindeye3.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MindEye3 retrieval model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

