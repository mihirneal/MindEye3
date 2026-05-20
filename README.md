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
