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
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Optional 0-based visual transformer block index to encode instead of the final image embedding.",
    )
    parser.add_argument(
        "--intermediate-pool",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for intermediate visual transformer features.",
    )
    parser.add_argument(
        "--project-intermediate",
        action="store_true",
        help="Apply the visual projection matrix to intermediate features when available.",
    )
    parser.add_argument(
        "--include-clip-tokens",
        action="store_true",
        help="Also store spatial OpenCLIP visual tokens for reconstruction conditioning.",
    )
    parser.add_argument(
        "--token-layer-index",
        type=int,
        default=-1,
        help="Visual transformer layer index to use for spatial token features when --include-clip-tokens is set.",
    )
    parser.add_argument(
        "--token-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for optional CLIP tokens.",
    )
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
        layer_index=args.layer_index,
        intermediate_pool=args.intermediate_pool,
        project_intermediate=args.project_intermediate,
        include_clip_tokens=args.include_clip_tokens,
        token_layer_index=args.token_layer_index,
        token_dtype=args.token_dtype,
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
    layer_index: int | None,
    intermediate_pool: str,
    project_intermediate: bool,
    include_clip_tokens: bool,
    token_layer_index: int,
    token_dtype: str,
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
    clip_tokens: list[torch.Tensor] = []
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
            encoded = encode_images(
                model=model,
                image_tensor=image_tensor,
                layer_index=layer_index,
                intermediate_pool=intermediate_pool,
                project_intermediate=project_intermediate,
            )
            embeddings.append(encoded.cpu())
            if include_clip_tokens:
                clip_tokens.append(encode_clip_tokens(model, image_tensor, token_layer_index).cpu())
            stimulus_ids.extend(range(start + 1, end + 1))
            print(f"encoded {end}/{total}")

    payload = {
        "stimulus_ids": torch.tensor(stimulus_ids, dtype=torch.long),
        "embeddings": torch.cat(embeddings, dim=0).float(),
        "model": model_name,
        "pretrained": pretrained,
        "layer_index": layer_index,
        "intermediate_pool": intermediate_pool if layer_index is not None else None,
        "project_intermediate": project_intermediate if layer_index is not None else None,
        "source": str(stimuli_path),
    }
    if clip_tokens:
        token_tensor = torch.cat(clip_tokens, dim=0).float()
        if token_dtype == "float16":
            token_tensor = token_tensor.half()
        elif token_dtype != "float32":
            raise ValueError("token_dtype must be float16 or float32")
        payload["clip_tokens"] = token_tensor
        payload["token_layer_index"] = token_layer_index
        payload["token_dtype"] = token_dtype
    torch.save(payload, output_path)
    print(f"wrote embedding cache: {output_path}")
    return output_path


def encode_images(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    layer_index: int | None,
    intermediate_pool: str,
    project_intermediate: bool,
) -> torch.Tensor:
    if layer_index is None:
        return torch.nn.functional.normalize(model.encode_image(image_tensor), dim=-1)

    visual = getattr(model, "visual", None)
    if visual is None or not hasattr(visual, "forward_intermediates"):
        raise ValueError("Intermediate image embeddings require an OpenCLIP visual module with forward_intermediates")

    outputs = visual.forward_intermediates(
        image_tensor,
        indices=[layer_index],
        stop_early=True,
        normalize_intermediates=True,
        intermediates_only=True,
        output_fmt="NLC",
        output_extra_tokens=True,
    )
    intermediates = outputs["image_intermediates"]
    if not intermediates:
        raise ValueError(f"No intermediate activations returned for layer_index={layer_index}")

    if intermediate_pool == "cls":
        prefix_tokens = outputs.get("image_intermediates_prefix")
        if not prefix_tokens:
            raise ValueError("CLS pooling requested but OpenCLIP did not return prefix tokens")
        encoded = prefix_tokens[0][:, 0]
    elif intermediate_pool == "mean":
        encoded = intermediates[0].mean(dim=1)
    else:
        raise ValueError(f"Unknown intermediate_pool {intermediate_pool!r}")

    projection = getattr(visual, "proj", None)
    if project_intermediate and projection is not None:
        encoded = encoded @ projection
    return torch.nn.functional.normalize(encoded, dim=-1)


def encode_clip_tokens(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    token_layer_index: int,
) -> torch.Tensor:
    visual = getattr(model, "visual", None)
    if visual is None or not hasattr(visual, "forward_intermediates"):
        raise ValueError("CLIP token export requires an OpenCLIP visual module with forward_intermediates")

    outputs = visual.forward_intermediates(
        image_tensor,
        indices=[token_layer_index],
        stop_early=False,
        normalize_intermediates=True,
        intermediates_only=True,
        output_fmt="NLC",
        output_extra_tokens=False,
    )
    intermediates = outputs["image_intermediates"]
    if not intermediates:
        raise ValueError(f"No token activations returned for token_layer_index={token_layer_index}")
    return intermediates[0].float()


if __name__ == "__main__":
    main()
