# MindEye3

MindEye3 is a retrieval-first fMRI decoding scaffold for image, video, and
multimodal reconstruction research.

The current V0 implementation provides a CPU-runnable synthetic retrieval
pipeline, package structure, CLI entry points, and tests. It intentionally does
not download real fMRI datasets or generator weights.

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run mindeye3-train --config configs/v0_synthetic.yaml
uv run mindeye3-eval --config configs/v0_synthetic.yaml --checkpoint outputs/v0_synthetic/checkpoint.pt
```

