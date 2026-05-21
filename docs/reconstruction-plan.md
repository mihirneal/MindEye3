# Reconstruction Plan

This track follows Brain-IT as the reconstruction target architecture, while keeping
the current MindEye-style retrieval loop alive as a stabilizer and comparison point.

## Baseline To Beat

- Compare against MindEye2 on the NSD standard subjects: 1, 2, 5, and 7.
- Report forward retrieval (brain to image) and backward retrieval (image to brain).
- Report reconstruction metrics used by Brain-IT and MindEye-style work: PixCorr,
  SSIM, AlexNet/Inception/CLIP two-way identification, EfficientNet distance, SwAV
  distance, LPIPS, and FID when the sample count is large enough.

## Architecture Path

1. Train a multi-subject, subject-conditioned retrieval backbone on fsaverage betas.
2. Add Brain-IT-style semantic reconstruction supervision by predicting spatial
   OpenCLIP tokens, not only a pooled CLIP vector.
3. Add a low-level branch that predicts VGG feature maps for DIP coarse layouts.
4. Condition an open diffusion backend from the predicted semantic tokens, then
   initialize/refine from the low-level branch.
5. Jointly fine-tune the brain model and diffusion conditioning path after feature
   alignment is stable.

## Diffusion Backend Choice

The first practical backend should be SDXL-unCLIP-style conditioning, because
Brain-IT and MindEye2 both rely on SDXL/OpenCLIP-token conditioning and the
conditioning target is explicit. FLUX/SD3.5 are stronger text-to-image generators,
but they do not immediately solve fMRI reconstruction because our conditioning is
visual-token based, not prompt-only text conditioning.

## Current Implementation State

- `configs/v2_nsd_multisubject_retrieval.yaml` trains a stronger subject-conditioned
  retrieval backbone across subjects 1, 2, 5, and 7.
- `configs/v2_nsd_brainit_joint.yaml` enables joint pooled-CLIP retrieval plus
  spatial-CLIP-token prediction when the cache contains `clip_tokens`.
- `mindeye3-build-nsd-embeddings --include-clip-tokens` can build the first semantic
  token cache for the joint run.
