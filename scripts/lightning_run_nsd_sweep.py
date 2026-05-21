from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the subj01 NSD retrieval sweep on Lightning.")
    parser.add_argument("--name", default="subj01_hparam_grid", help="Sweep name.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Shard index to run.")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of shards.")
    parser.add_argument("--summarize", action="store_true", help="Only summarize completed jobs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweep_root = Path("outputs") / "sweeps" / args.name
    config_root = sweep_root / "configs"
    log_root = sweep_root / "logs"
    jobs = build_jobs(args.name)

    if args.summarize:
        summarize(sweep_root, jobs)
        return

    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    assigned = [job for index, job in enumerate(jobs) if index % args.num_gpus == args.gpu_index]
    print(f"running {len(assigned)} of {len(jobs)} jobs for shard {args.gpu_index}/{args.num_gpus}")
    for job in assigned:
        config_path = config_root / f"{job['name']}.yaml"
        log_path = log_root / f"{job['name']}.log"
        write_config(config_path, job["config"])
        print(f"START {job['name']} -> {log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                ["uv", "run", "mindeye3-train", "--config", str(config_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            print(f"FAILED {job['name']} returncode={result.returncode}", file=sys.stderr)
            continue
        print(f"DONE {job['name']}")
    summarize(sweep_root, jobs)


def build_jobs(sweep_name: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dropout in [0.4, 0.5, 0.6]:
        for hidden_layers in [1, 2, 3]:
            for batch_size in [64, 128]:
                for lr_schedule in ["none", "cosine"]:
                    job_name = (
                        f"d{int(dropout * 10):02d}"
                        f"_l{hidden_layers}"
                        f"_b{batch_size}"
                        f"_{lr_schedule}"
                    )
                    jobs.append(
                        {
                            "name": job_name,
                            "config": make_config(
                                sweep_name=sweep_name,
                                job_name=job_name,
                                dropout=dropout,
                                hidden_layers=hidden_layers,
                                batch_size=batch_size,
                                lr_schedule=lr_schedule,
                            ),
                        }
                    )
    return jobs


def make_config(
    sweep_name: str,
    job_name: str,
    dropout: float,
    hidden_layers: int,
    batch_size: int,
    lr_schedule: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": 7,
        "output_dir": f"outputs/sweeps/{sweep_name}/runs/{job_name}",
        "data": {
            "name": "nsd",
            "root": "/teamspace/studios/this_studio/nsd",
            "subjects": [1],
            "beta_space": "fsaverage",
            "beta_version": "betas_fithrf_GLMdenoise_RR",
            "embedding_cache": "/teamspace/studios/this_studio/nsd/cache/embeddings/openclip_vith14.pt",
            "max_samples": None,
            "max_cached_sessions": 64,
            "normalize_fmri": True,
            "include_missing_data": False,
            "average_repeats": True,
            "ncsnr_topk": 20000,
            "train_fraction": 0.8,
            "batch_size": batch_size,
        },
        "model": {
            "hidden_dim": 2048,
            "hidden_layers": hidden_layers,
            "scene_dim": 1024,
            "dropout": dropout,
        },
        "training": {
            "epochs": 20,
            "learning_rate": 0.0001,
            "min_learning_rate": 0.000001,
            "lr_schedule": lr_schedule,
            "weight_decay": 0.01,
            "temperature": 0.07,
            "device": "cuda",
            "log_every": 25,
            "best_metric": "image_top10",
        },
        "evaluation": {
            "top_k": [1, 5, 10],
        },
    }
    return deepcopy(config)


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def summarize(sweep_root: Path, jobs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        checkpoint_path = sweep_root / "runs" / job["name"] / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except RuntimeError as exc:
            print(f"skipping unreadable checkpoint {checkpoint_path}: {exc}", file=sys.stderr)
            continue
        row = {
            "name": job["name"],
            "epoch": checkpoint["epoch"],
            **job["config"]["model"],
            "batch_size": job["config"]["data"]["batch_size"],
            "lr_schedule": job["config"]["training"]["lr_schedule"],
            **checkpoint["metrics"],
        }
        rows.append(row)

    rows.sort(key=lambda row: row.get("image_top10", -1.0), reverse=True)
    summary_path = sweep_root / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")
    print(f"wrote {summary_path}")
    for row in rows[:10]:
        print(
            f"{row['name']} epoch={row['epoch']} "
            f"top1={row['image_top1']:.3f} top5={row['image_top5']:.3f} top10={row['image_top10']:.3f}"
        )


if __name__ == "__main__":
    main()
