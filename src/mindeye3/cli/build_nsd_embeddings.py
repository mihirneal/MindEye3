from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an NSD image embedding cache.")
    parser.add_argument("--stimuli", required=True, help="Path to nsd_stimuli.hdf5.")
    parser.add_argument("--output", required=True, help="Path to write the torch cache.")
    parser.add_argument("--model", default="ViT-H-14", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", help="OpenCLIP pretrained tag.")
    parser.add_argument("--batch-size", type=int, default=64, help="Image batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for smoke tests.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_embedding_cache(
        stimuli_path=Path(args.stimuli),
        output_path=Path(args.output),
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        limit=args.limit,
    )


@torch.no_grad()
def build_embedding_cache(
    stimuli_path: Path,
    output_path: Path,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> Path:
    try:
        import h5py
        import open_clip
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Building NSD embeddings requires the vision extra: "
            "`uv sync --extra vision` or `uv run --extra vision ...`."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings: list[torch.Tensor] = []
    stimulus_ids: list[int] = []

    with h5py.File(stimuli_path, "r") as handle:
        images = handle["imgBrick"]
        total = images.shape[0] if limit is None else min(limit, images.shape[0])
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = [
                preprocess(Image.fromarray(images[index])).unsqueeze(0)
                for index in range(start, end)
            ]
            image_tensor = torch.cat(batch, dim=0).to(device)
            encoded = torch.nn.functional.normalize(model.encode_image(image_tensor), dim=-1)
            embeddings.append(encoded.cpu())
            stimulus_ids.extend(range(start + 1, end + 1))
            print(f"encoded {end}/{total}")

    payload = {
        "stimulus_ids": torch.tensor(stimulus_ids, dtype=torch.long),
        "embeddings": torch.cat(embeddings, dim=0).float(),
        "model": model_name,
        "pretrained": pretrained,
        "source": str(stimuli_path),
    }
    torch.save(payload, output_path)
    print(f"wrote embedding cache: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
