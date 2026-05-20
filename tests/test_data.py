from pathlib import Path

import torch

from mindeye3.config import DataConfig
from mindeye3.data import create_dataloaders as create_project_dataloaders
from mindeye3.data.nsd import NSDDataset, StimulusEmbeddingCache, split_by_stimulus
from mindeye3.data.synthetic import SyntheticBrainDataset, create_dataloaders


def test_synthetic_dataset_shapes_and_determinism() -> None:
    config = DataConfig(num_samples=8, fmri_dim=16, embedding_dim=12)
    dataset_a = SyntheticBrainDataset(config, seed=3)
    dataset_b = SyntheticBrainDataset(config, seed=3)

    sample, embedding = dataset_a[0]
    sample_b, embedding_b = dataset_b[0]

    assert len(dataset_a) == 8
    assert sample.fmri.shape == (16,)
    assert embedding.require("image").shape == (12,)
    assert torch.allclose(sample.fmri, sample_b.fmri)
    assert torch.allclose(embedding.require("image"), embedding_b.require("image"))


def test_create_dataloaders_returns_paired_batches() -> None:
    config = DataConfig(num_samples=20, fmri_dim=16, embedding_dim=12, batch_size=5)
    train_loader, eval_loader = create_dataloaders(config, seed=5)

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))

    assert train_batch.fmri.shape == (5, 16)
    assert train_batch.image.shape == (5, 12)
    assert eval_batch.fmri.shape[-1] == 16


def test_nsd_dataset_pairs_betas_with_embedding_cache(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path)
    cache_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "stimulus_ids": torch.tensor([101, 102, 103]),
            "embeddings": torch.eye(3),
        },
        cache_path,
    )

    config = DataConfig(
        name="nsd",
        root=str(root),
        embedding_cache=str(cache_path),
        subjects=[1],
        train_fraction=0.5,
        batch_size=2,
    )
    dataset = NSDDataset(config)
    sample, embedding = dataset[0]

    assert len(dataset) == 3
    assert sample.subject_id == 1
    assert sample.stimulus_id == 101
    assert sample.fmri.shape == (5,)
    assert torch.allclose(embedding.require("image"), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.isclose(sample.fmri.mean(), torch.tensor(0.0), atol=1e-6)


def test_nsd_dataloaders_split_by_stimulus(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path)
    cache_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "stimulus_ids": torch.tensor([101, 102, 103]),
            "embeddings": torch.eye(3),
        },
        cache_path,
    )

    config = DataConfig(
        name="nsd",
        root=str(root),
        embedding_cache=str(cache_path),
        subjects=[1],
        train_fraction=0.67,
        batch_size=4,
    )
    dataset = NSDDataset(config)
    train_dataset, eval_dataset = split_by_stimulus(dataset, train_fraction=0.67, seed=1)
    train_stimuli = {dataset.trials[index].stimulus_id for index in train_dataset.indices}
    eval_stimuli = {dataset.trials[index].stimulus_id for index in eval_dataset.indices}

    assert train_stimuli
    assert eval_stimuli
    assert train_stimuli.isdisjoint(eval_stimuli)

    train_loader, eval_loader = create_project_dataloaders(config, seed=1)
    assert next(iter(train_loader)).fmri.shape[-1] == 5
    assert next(iter(eval_loader)).image.shape[-1] == 3


def test_stimulus_embedding_cache_validates_shape(tmp_path: Path) -> None:
    cache_path = tmp_path / "bad.pt"
    torch.save({"stimulus_ids": torch.tensor([1, 2]), "embeddings": torch.eye(3)}, cache_path)

    try:
        StimulusEmbeddingCache(cache_path)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected invalid cache shape to fail")


def _write_nsd_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "nsd"
    behav = root / "nsddata" / "ppdata" / "subj01" / "behav"
    beta = (
        root
        / "nsddata_betas"
        / "ppdata"
        / "subj01"
        / "fsaverage"
        / "betas_fithrf_GLMdenoise_RR"
    )
    behav.mkdir(parents=True)
    beta.mkdir(parents=True)
    (behav / "responses.tsv").write_text(
        "\t".join(
            [
                "SUBJECT",
                "SESSION",
                "RUN",
                "TRIAL",
                "73KID",
                "10KID",
                "TIME",
                "ISOLD",
                "ISCORRECT",
                "RT",
                "CHANGEMIND",
                "MEMORYRECENT",
                "MEMORYFIRST",
                "ISOLDCURRENT",
                "ISCORRECTCURRENT",
                "TOTAL1",
                "TOTAL2",
                "BUTTON",
                "MISSINGDATA",
            ]
        )
        + "\n"
        + "1\t1\t1\t1\t101\t1\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n"
        + "1\t1\t1\t2\t102\t2\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n"
        + "1\t1\t1\t3\t103\t3\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n"
        + "1\t1\t1\t4\t999\t4\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n",
        encoding="utf-8",
    )
    torch.save(torch.arange(6, dtype=torch.float32).reshape(3, 2), beta / "lh.betas_session01.mgh.pt")
    torch.save(torch.arange(9, dtype=torch.float32).reshape(3, 3), beta / "rh.betas_session01.mgh.pt")
    return root
