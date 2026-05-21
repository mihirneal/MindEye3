from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from mindeye3.data.batches import PairedBatch
from mindeye3.metrics import candidate_retrieval_accuracy, topk_retrieval_accuracy
from mindeye3.models import RetrievalModel


@torch.no_grad()
def collect_embeddings(
    model: RetrievalModel,
    loader: DataLoader[PairedBatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    stimulus_ids: list[torch.Tensor] = []
    for batch in loader:
        fmri = batch.fmri.to(device)
        image = batch.image.to(device)
        predictions.append(model(fmri).cpu())
        targets.append(image.cpu())
        stimulus_ids.append(batch.stimulus_id.cpu())
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0), torch.cat(stimulus_ids, dim=0)


def evaluate_model(
    model: RetrievalModel,
    loader: DataLoader[PairedBatch],
    top_k: list[int],
    device: torch.device,
    candidate_pool_size: int | None = None,
    candidate_repeats: int = 30,
    candidate_seed: int = 0,
) -> dict[str, float]:
    predictions, targets, stimulus_ids = collect_embeddings(model, loader, device)
    metrics = {f"trial_{key}": value for key, value in topk_retrieval_accuracy(predictions, targets, top_k).items()}
    image_predictions, image_targets = average_by_stimulus(predictions, targets, stimulus_ids)
    metrics.update(
        {f"image_{key}": value for key, value in topk_retrieval_accuracy(image_predictions, image_targets, top_k).items()}
    )
    if candidate_pool_size is not None:
        metrics.update(
            {
                f"mindeye2_brain_to_image_{key}": value
                for key, value in candidate_retrieval_accuracy(
                    image_predictions,
                    image_targets,
                    pool_size=candidate_pool_size,
                    top_k=[1],
                    num_repeats=candidate_repeats,
                    seed=candidate_seed,
                ).items()
            }
        )
        metrics.update(
            {
                f"mindeye2_image_to_brain_{key}": value
                for key, value in candidate_retrieval_accuracy(
                    image_targets,
                    image_predictions,
                    pool_size=candidate_pool_size,
                    top_k=[1],
                    num_repeats=candidate_repeats,
                    seed=candidate_seed,
                ).items()
            }
        )
    return metrics


def average_by_stimulus(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stimulus_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"predictions": [], "targets": []})
    for stimulus_id, prediction, target in zip(stimulus_ids.tolist(), predictions, targets):
        grouped[int(stimulus_id)]["predictions"].append(prediction)
        grouped[int(stimulus_id)]["targets"].append(target)

    averaged_predictions: list[torch.Tensor] = []
    averaged_targets: list[torch.Tensor] = []
    for stimulus_id in sorted(grouped):
        values = grouped[stimulus_id]
        averaged_predictions.append(torch.stack(values["predictions"]).mean(dim=0))
        averaged_targets.append(torch.stack(values["targets"]).mean(dim=0))
    return torch.stack(averaged_predictions), torch.stack(averaged_targets)
