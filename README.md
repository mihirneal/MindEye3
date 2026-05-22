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

## NSD Retrieval V1

The first real-data path targets NSD retrieval with fsaverage
`betas_fithrf_GLMdenoise_RR` betas and a cached image-embedding table.

Build an image embedding cache on the machine that has
`nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`:

```bash
uv run --extra vision mindeye3-build-nsd-embeddings \
  --stimuli /teamspace/studios/this_studio/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  --output /teamspace/studios/this_studio/nsd/cache/embeddings/openclip_vith14.pt
```

Then train retrieval:

```bash
uv run mindeye3-train --config configs/v1_nsd_retrieval.yaml
```

## Reconstruction V2

The reconstruction path starts with a Brain-IT-inspired joint objective: pooled
CLIP retrieval plus spatial OpenCLIP-token prediction for diffusion
conditioning. The joint configs use brain-token encoders and a BIT-style
cross-attention decoder, where learned image-feature queries attend directly to
functional brain-token features while the pooled CLIP retrieval head stays active.
They also checkpoint on 300-way MindEye2-style brain-to-image retrieval. Build a
token cache on Lightning before running the joint config:

```bash
uv run --extra vision mindeye3-build-nsd-embeddings \
  --stimuli /teamspace/studios/this_studio/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  --output /teamspace/studios/this_studio/nsd/cache/embeddings/openclip_bigG14_laion2b_s39b_b160k_tokens.pt \
  --model ViT-bigG-14 \
  --pretrained laion2b_s39b_b160k \
  --include-clip-tokens \
  --batch-size 32
```

Then train the first joint model:

```bash
uv run mindeye3-train --config configs/v2_nsd_brainit_joint.yaml
```
