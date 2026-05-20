from __future__ import annotations

import torch
from torch import nn


class SymmetricContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = torch.nn.functional.normalize(predicted, dim=-1)
        target = torch.nn.functional.normalize(target, dim=-1)
        logits = predicted @ target.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        brain_to_stimulus = torch.nn.functional.cross_entropy(logits, labels)
        stimulus_to_brain = torch.nn.functional.cross_entropy(logits.T, labels)
        return 0.5 * (brain_to_stimulus + stimulus_to_brain)

