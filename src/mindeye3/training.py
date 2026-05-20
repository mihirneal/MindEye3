from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mindeye3.checkpointing import save_checkpoint
from mindeye3.config import MindEyeConfig
from mindeye3.data import create_dataloaders
from mindeye3.data.batches import PairedBatch
from mindeye3.evaluation import evaluate_model
from mindeye3.losses import SymmetricContrastiveLoss
from mindeye3.models import RetrievalModel


def build_model(
    config: MindEyeConfig,
    fmri_dim: int | None = None,
    embedding_dim: int | None = None,
) -> RetrievalModel:
    return RetrievalModel(
        fmri_dim=fmri_dim if fmri_dim is not None else config.data.fmri_dim,
        hidden_dim=config.model.hidden_dim,
        scene_dim=config.model.scene_dim,
        embedding_dim=embedding_dim if embedding_dim is not None else config.data.embedding_dim,
        dropout=config.model.dropout,
    )


def infer_batch_dims(loader: DataLoader[PairedBatch]) -> tuple[int, int]:
    batch = next(iter(loader))
    return int(batch.fmri.shape[-1]), int(batch.image.shape[-1])


def train(config: MindEyeConfig) -> Path:
    torch.manual_seed(config.seed)
    device = torch.device(config.training.device)
    train_loader, eval_loader = create_dataloaders(config.data, seed=config.seed)
    fmri_dim, embedding_dim = infer_batch_dims(train_loader)
    model = build_model(config, fmri_dim=fmri_dim, embedding_dim=embedding_dim).to(device)
    criterion = SymmetricContrastiveLoss(config.training.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    global_step = 0
    latest_metrics: dict[str, float] = {}
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            fmri = batch.fmri.to(device)
            image = batch.image.to(device)
            prediction = model(fmri)
            loss = criterion(prediction, image)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            if config.training.log_every > 0 and global_step % config.training.log_every == 0:
                average_loss = running_loss / max(1, config.training.log_every)
                print(f"epoch={epoch} step={global_step} loss={average_loss:.4f}")
                running_loss = 0.0

        latest_metrics = evaluate_model(model, eval_loader, config.evaluation.top_k, device)
        metric_text = " ".join(f"{key}={value:.3f}" for key, value in latest_metrics.items())
        print(f"epoch={epoch} eval {metric_text}")

    checkpoint_path = Path(config.output_dir) / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=config.training.epochs,
        metrics=latest_metrics,
    )
    print(f"wrote checkpoint: {checkpoint_path}")
    return checkpoint_path
