# MindEye3 V1 Plan

## Summary

MindEye3 V1 will be a unified fMRI decoder for image reconstruction,
video reconstruction, and multimodal semantic retrieval. MindEye2 is a
reference point for goals and benchmarks, not an architectural template.

The first version should optimize for fast, credible progress on SOTA-facing
benchmarks by using official preprocessed fMRI derivatives rather than raw
scanner data. The core model should learn a generator-agnostic scene
representation from brain activity, then condition modern image and video
generation systems.

## Core Decisions

- Use preprocessed fMRI data only for V1.
- Treat retrieval as a first-class objective and benchmark, equal in importance
  to generation quality.
- Build one shared brain encoder that can support image, video, audio-language,
  and semantic decoding tasks.
- Use FLUX.2 as the first image generation base.
- Add LTX-2 as the staged video and audio-video generation base.
- Keep generator adapters modular so newer image or video models can be swapped
  in without rewriting the brain encoder.

## Generator Strategy

### Image Generation

FLUX.2 should be the primary image reconstruction base for V1. It is the best
initial fit for high-quality still-image reconstruction, NSD-style benchmarks,
LAION-fMRI image decoding, low-level fidelity, object detail, and layout
reconstruction.

Stable Diffusion 3.5 Large should remain a reproducible fallback because of its
mature tooling and broad ecosystem. Qwen-Image and newer Qwen image models
should be tracked as experimental branches, especially for strong text,
editing, and semantic-image alignment capabilities.

### Video Generation

LTX-2 should be the primary staged video base. It is especially aligned with the
MindEye3 direction because it supports open audio-video generation and maps well
to movie datasets with audio, transcripts, and naturalistic temporal structure.

Wan2.2 and HunyuanVideo should be treated as comparative video-only backends.
Closed systems such as Veo can be used as qualitative oracle comparisons, but
not as the core scientific substrate because they do not provide enough control
over latent representations, fine-tuning, or reproducible conditioning.

### Staging

V0 should focus on FLUX.2 image reconstruction and retrieval on image datasets.
V0.5 should add temporal brain modeling and the LTX-2 video head. V1 should
unify still images and videos as scene decoding problems, where still images are
treated as one-frame scenes.

## Brain Data Strategy

MindEye3 V1 should use official preprocessed derivatives, not raw or lightly
processed fMRI. This choice prioritizes model progress, reproducible
benchmarks, and faster iteration.

Use GLMsingle or single-trial betas for image datasets where available,
especially NSD and LAION-fMRI. Use official parcel or time-series data for
CNeuroMod and Algonauts-style naturalistic movie datasets.

Raw or lightly processed fMRI may become a V2 research track, but only after V1
proves that the model architecture, generator-control interface, retrieval
objectives, and video extension are strong.

## Dataset Scope

Primary image datasets:

- Natural Scenes Dataset (NSD)
- LAION-fMRI
- CNeuroMod-THINGS
- BOLD5000
- Generic Object Decoding (GOD)

Primary video and multimodal datasets:

- CNeuroMod movie and Friends data
- Algonauts 2025 CNeuroMod competition data
- CineBrain
- Gallant Lab natural movie datasets, including vim-style datasets where usable

Exploratory extensions:

- NSD-synthetic for out-of-distribution testing
- NSD-Imagery for mental imagery transfer
- CNeuroMod Harry Potter and other language-rich stimuli
- Game or embodied naturalistic datasets where public derivatives are usable

## Model Architecture Direction

The main architecture should be a brain-to-generator-control system rather than
a MindEye2-style brain-to-CLIP-only pipeline.

The shared brain encoder should map fMRI into a structured scene representation
with separate but coordinated components:

- semantic content
- spatial and layout information
- low-level appearance, color, texture, and contrast
- temporal and motion structure
- audio-language context for movie stimuli
- uncertainty estimates for sampling multiple plausible reconstructions

Generator-specific adapters should translate this shared representation into
conditioning signals for FLUX.2, LTX-2, and later generator backends.

## Retrieval Strategy

Retrieval should be built into training and evaluation from the beginning.

Train contrastive heads from fMRI to image, video, text, and audio embeddings.
Support image retrieval, video clip retrieval, caption retrieval, transcript
retrieval, and embedding-space retrieval.

Generation should use retrieval as both a benchmark and a control signal.
MindEye3 should generate multiple candidates per brain sample, then rerank them
using brain-to-embedding agreement, low-level similarity predictors, semantic
consistency, and temporal consistency for video.

## Evaluation Plan

Image evaluation:

- top-k retrieval
- pairwise identification
- CLIP, SigLIP, and DINO-style semantic similarity
- low-level perceptual similarity
- qualitative reconstruction grids
- comparison against MindEye2-style benchmark results

Video evaluation:

- video clip retrieval
- frame and scene semantic alignment
- temporal consistency
- motion similarity
- transcript and audio alignment where available
- qualitative side-by-side video reconstructions

Generalization evaluation:

- cross-subject transfer
- few-shot new-subject adaptation
- cross-dataset transfer
- out-of-distribution image testing
- held-out movie and held-out stimulus evaluation

Ablations:

- semantic-only decoding
- low-level-only decoding
- layout-only decoding
- motion-only decoding
- retrieval-only training
- generation-only training
- retrieval-reranked generation

## Assumptions

- V1 assumes access to a multi-GPU lab setup with enough storage for large
  preprocessed fMRI datasets and cached stimulus embeddings.
- The first implementation should optimize scientific speed and reproducible
  benchmarks rather than custom raw-fMRI preprocessing.
- Image generation starts with FLUX.2.
- Video generation is added through LTX-2 after the image and retrieval spine is
  working.
- Audio and transcript information are in scope for movie datasets.
- Dataset access, licensing, redistribution limits, and citation requirements
  must be preserved for every dataset adapter.
