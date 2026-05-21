from __future__ import annotations

import torch
from torch import nn


class SymmetricContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        positive_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted = torch.nn.functional.normalize(predicted, dim=-1)
        target = torch.nn.functional.normalize(target, dim=-1)
        logits = predicted @ target.T / self.temperature
        if positive_ids is not None:
            return _symmetric_multi_positive_loss(logits, positive_ids)
        labels = torch.arange(logits.shape[0], device=logits.device)
        brain_to_stimulus = torch.nn.functional.cross_entropy(logits, labels)
        stimulus_to_brain = torch.nn.functional.cross_entropy(logits.T, labels)
        return 0.5 * (brain_to_stimulus + stimulus_to_brain)


def _symmetric_multi_positive_loss(
    logits: torch.Tensor,
    positive_ids: torch.Tensor,
) -> torch.Tensor:
    if positive_ids.ndim != 1:
        raise ValueError("positive_ids must be rank-1")
    if positive_ids.shape[0] != logits.shape[0]:
        raise ValueError("positive_ids length must match batch size")

    positive_ids = positive_ids.to(logits.device)
    positive_mask = positive_ids[:, None] == positive_ids[None, :]
    brain_to_stimulus = _multi_positive_cross_entropy(logits, positive_mask)
    stimulus_to_brain = _multi_positive_cross_entropy(logits.T, positive_mask.T)
    return 0.5 * (brain_to_stimulus + stimulus_to_brain)


def _multi_positive_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if not positive_mask.any(dim=1).all():
        raise ValueError("Each row must have at least one positive")
    all_logsumexp = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    positive_logsumexp = torch.logsumexp(positive_logits, dim=1)
    return (all_logsumexp - positive_logsumexp).mean()
