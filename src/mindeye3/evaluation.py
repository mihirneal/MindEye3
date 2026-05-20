from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mindeye3.data.batches import PairedBatch
from mindeye3.metrics import topk_retrieval_accuracy
from mindeye3.models import RetrievalModel


@torch.no_grad()
def collect_embeddings(
    model: RetrievalModel,
    loader: DataLoader[PairedBatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        fmri = batch.fmri.to(device)
        image = batch.image.to(device)
        predictions.append(model(fmri).cpu())
        targets.append(image.cpu())
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def evaluate_model(
    model: RetrievalModel,
    loader: DataLoader[PairedBatch],
    top_k: list[int],
    device: torch.device,
) -> dict[str, float]:
    predictions, targets = collect_embeddings(model, loader, device)
    return topk_retrieval_accuracy(predictions, targets, top_k)
