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


def candidate_retrieval_accuracy(
    query: torch.Tensor,
    target: torch.Tensor,
    pool_size: int,
    top_k: list[int] | tuple[int, ...] = (1,),
    num_repeats: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    if query.shape != target.shape:
        raise ValueError("query and target embeddings must have the same shape")
    if query.ndim != 2:
        raise ValueError("retrieval embeddings must be rank-2 tensors")
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")

    num_items = query.shape[0]
    if pool_size > num_items:
        raise ValueError("pool_size cannot exceed the number of retrieval items")

    query = torch.nn.functional.normalize(query, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    labels = torch.arange(num_items)
    generator = torch.Generator().manual_seed(seed)
    correct_counts = {k: 0.0 for k in top_k}

    for _ in range(num_repeats):
        candidate_indices = torch.empty(num_items, pool_size, dtype=torch.long)
        candidate_indices[:, 0] = labels
        for index in range(num_items):
            negatives = _sample_negative_indices(num_items, index, pool_size - 1, generator)
            candidate_indices[index, 1:] = negatives

        candidates = target[candidate_indices]
        scores = torch.einsum("bd,bkd->bk", query, candidates)
        max_k = min(max(top_k), pool_size)
        ranked = scores.topk(k=max_k, dim=1).indices
        for k in top_k:
            effective_k = min(k, pool_size)
            correct_counts[k] += ranked[:, :effective_k].eq(0).any(dim=1).float().mean().item()

    return {f"top{min(k, pool_size)}": value / num_repeats for k, value in correct_counts.items()}


def _sample_negative_indices(
    num_items: int,
    positive_index: int,
    num_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.randperm(num_items - 1, generator=generator)[:num_negatives]
    return permutation + (permutation >= positive_index).long()
