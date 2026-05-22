from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from mindeye3.checkpointing import load_checkpoint
from mindeye3.config import load_config
from mindeye3.data import create_dataloaders
from mindeye3.data.batches import PairedBatch
from mindeye3.training import build_model, infer_batch_dims, infer_feature_group_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export NSD retrieval reconstruction preview grids.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    parser.add_argument("--stimuli", required=True, help="Path to nsd_stimuli.hdf5.")
    parser.add_argument("--output-dir", required=True, help="Directory for preview PNGs and metadata.")
    parser.add_argument("--split", choices=["eval", "train"], default="eval", help="Dataset split to export.")
    parser.add_argument("--limit", type=int, default=24, help="Number of query samples to visualize.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of retrieved images per query.")
    parser.add_argument("--device", default=None, help="Override config training.device.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    export_retrieval_reconstructions(
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        stimuli_path=Path(args.stimuli),
        output_dir=Path(args.output_dir),
        split=args.split,
        limit=args.limit,
        top_k=args.top_k,
        device_override=args.device,
    )


@torch.no_grad()
def export_retrieval_reconstructions(
    config_path: Path,
    checkpoint_path: Path,
    stimuli_path: Path,
    output_dir: Path,
    split: str,
    limit: int,
    top_k: int,
    device_override: str | None = None,
) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    try:
        import h5py
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Exporting NSD reconstructions requires the vision extra.") from exc

    config = load_config(config_path)
    device = torch.device(device_override or config.training.device)
    train_loader, eval_loader = create_dataloaders(config.data, seed=config.seed)
    loader = train_loader if split == "train" else eval_loader
    fmri_dim, embedding_dim, clip_token_shape = infer_batch_dims(loader)
    feature_group_ids = infer_feature_group_ids(loader)
    model = build_model(
        config,
        fmri_dim=fmri_dim,
        embedding_dim=embedding_dim,
        clip_token_shape=clip_token_shape,
        feature_group_ids=feature_group_ids,
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions, targets, stimulus_ids, subject_ids = _collect(loader, model, device)
    similarities = torch.nn.functional.normalize(predictions, dim=-1) @ torch.nn.functional.normalize(targets, dim=-1).T
    top_scores, top_indices = similarities.topk(k=min(top_k, similarities.shape[1]), dim=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = min(limit, predictions.shape[0])
    with h5py.File(stimuli_path, "r") as handle:
        images = handle["imgBrick"]
        tiles: list[Image.Image] = []
        metadata_rows: list[dict[str, str | int | float]] = []
        for row_index in range(rows):
            query_id = int(stimulus_ids[row_index])
            query_image = _load_stimulus_image(images, query_id, Image)
            row_tiles = [_label_image(query_image, f"target s{int(subject_ids[row_index])}", ImageDraw)]
            for rank, target_index in enumerate(top_indices[row_index].tolist(), start=1):
                retrieved_id = int(stimulus_ids[target_index])
                score = float(top_scores[row_index, rank - 1])
                retrieved_image = _load_stimulus_image(images, retrieved_id, Image)
                row_tiles.append(_label_image(retrieved_image, f"top{rank} {score:.2f}", ImageDraw))
                metadata_rows.append(
                    {
                        "row": row_index,
                        "subject_id": int(subject_ids[row_index]),
                        "query_stimulus_id": query_id,
                        "rank": rank,
                        "retrieved_stimulus_id": retrieved_id,
                        "score": score,
                    }
                )
            tiles.append(_hstack(row_tiles, Image))

    grid = _vstack(tiles, Image)
    grid_path = output_dir / "retrieval_reconstruction_grid.png"
    metadata_path = output_dir / "retrieval_reconstruction_metadata.tsv"
    grid.save(grid_path)
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["row", "subject_id", "query_stimulus_id", "rank", "retrieved_stimulus_id", "score"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"wrote preview grid: {grid_path}")
    print(f"wrote metadata: {metadata_path}")


def _collect(
    loader,
    model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    stimulus_ids: list[torch.Tensor] = []
    subject_ids: list[torch.Tensor] = []
    for batch in loader:
        assert isinstance(batch, PairedBatch)
        fmri = batch.fmri.to(device)
        subject_id = batch.subject_id.to(device)
        predictions.append(model(fmri, subject_id=subject_id).cpu())
        targets.append(batch.image.cpu())
        stimulus_ids.append(batch.stimulus_id.cpu())
        subject_ids.append(batch.subject_id.cpu())
    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(stimulus_ids, dim=0),
        torch.cat(subject_ids, dim=0),
    )


def _load_stimulus_image(images, stimulus_id: int, image_module) -> object:
    return image_module.fromarray(images[stimulus_id - 1]).convert("RGB")


def _label_image(image, label: str, image_draw_module):
    labeled = image.copy()
    draw = image_draw_module.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return labeled


def _hstack(images: list[object], image_module):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = image_module.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    return output


def _vstack(images: list[object], image_module):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    output = image_module.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for image in images:
        output.paste(image, (0, y))
        y += image.height
    return output


if __name__ == "__main__":
    main()
