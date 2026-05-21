import torch

from mindeye3.losses import SymmetricContrastiveLoss
from mindeye3.metrics import topk_retrieval_accuracy
from mindeye3.models import BrainEncoder, RetrievalModel


def test_brain_encoder_forward_shape() -> None:
    encoder = BrainEncoder(fmri_dim=16, hidden_dim=32, hidden_layers=2, scene_dim=8)
    output = encoder(torch.randn(4, 16))

    assert output.shape == (4, 8)


def test_retrieval_model_forward_shape() -> None:
    model = RetrievalModel(fmri_dim=16, hidden_dim=32, hidden_layers=2, scene_dim=8, embedding_dim=12)
    output = model(torch.randn(4, 16))

    assert output.shape == (4, 12)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_symmetric_contrastive_loss_is_lower_for_matching_pairs() -> None:
    loss_fn = SymmetricContrastiveLoss(temperature=0.1)
    target = torch.eye(4)
    matching = target.clone()
    mismatched = target.roll(shifts=1, dims=0)

    assert loss_fn(matching, target) < loss_fn(mismatched, target)


def test_topk_retrieval_accuracy() -> None:
    embeddings = torch.eye(4)
    metrics = topk_retrieval_accuracy(embeddings, embeddings, top_k=[1, 2])

    assert metrics["top1"] == 1.0
    assert metrics["top2"] == 1.0
