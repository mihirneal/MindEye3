from __future__ import annotations

import torch


def topk_retrieval_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    top_k: list[int] | tuple[int, ...],
) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target embeddings must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("retrieval embeddings must be rank-2 tensors")

    predicted = torch.nn.functional.normalize(predicted, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    scores = predicted @ target.T
    labels = torch.arange(scores.shape[0], device=scores.device)
    max_k = min(max(top_k), scores.shape[1])
    indices = scores.topk(k=max_k, dim=1).indices

    metrics: dict[str, float] = {}
    for k in top_k:
        effective_k = min(k, scores.shape[1])
        correct = indices[:, :effective_k].eq(labels[:, None]).any(dim=1)
        metrics[f"top{effective_k}"] = correct.float().mean().item()
    return metrics

