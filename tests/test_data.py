import shutil
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


def test_nsd_dataset_can_average_repeats_and_select_ncsnr_topk(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path, include_repeat=True)
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
        average_repeats=True,
        ncsnr_topk=2,
        normalize_fmri=False,
    )
    dataset = NSDDataset(config)
    sample, embedding = dataset[0]

    assert len(dataset) == 3
    assert sample.stimulus_id == 101
    assert sample.metadata["num_repeats"] == 2
    assert sample.fmri.shape == (2,)
    assert torch.allclose(sample.fmri, torch.tensor([2.0, 3.5]))
    assert torch.allclose(embedding.require("image"), torch.tensor([1.0, 0.0, 0.0]))


def test_nsd_dataset_selects_multi_subject_ncsnr_topk(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path, include_repeat=True)
    shutil.copytree(
        root / "nsddata" / "ppdata" / "subj01",
        root / "nsddata" / "ppdata" / "subj02",
    )
    shutil.copytree(
        root / "nsddata_betas" / "ppdata" / "subj01",
        root / "nsddata_betas" / "ppdata" / "subj02",
    )
    beta2 = root / "nsddata_betas" / "ppdata" / "subj02" / "fsaverage" / "betas_fithrf_GLMdenoise_RR"
    torch.save(torch.tensor([[9.0], [2.0]]), beta2 / "lh.ncsnr.mgh.pt")
    torch.save(torch.tensor([[1.0], [8.0], [7.0]]), beta2 / "rh.ncsnr.mgh.pt")
    cache_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "stimulus_ids": torch.tensor([101, 102, 103]),
            "embeddings": torch.eye(3),
            "clip_tokens": torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4),
        },
        cache_path,
    )

    config = DataConfig(
        name="nsd",
        root=str(root),
        embedding_cache=str(cache_path),
        subjects=[1, 2],
        average_repeats=True,
        ncsnr_topk=2,
        normalize_fmri=False,
    )
    dataset = NSDDataset(config)
    sample, embedding = dataset[0]

    assert len(dataset) == 6
    assert sample.fmri.shape == (2,)
    assert embedding.require("clip_tokens").shape == (2, 4)


def test_nsd_dataset_builds_atlas_feature_groups_after_topk(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path, include_repeat=True)
    label = root / "nsddata" / "freesurfer" / "fsaverage" / "label"
    label.mkdir(parents=True)
    torch.save(torch.tensor([2, 2]), label / "lh.streams.mgz.pt")
    torch.save(torch.tensor([3, 3, 3]), label / "rh.streams.mgz.pt")
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
        average_repeats=True,
        ncsnr_topk=4,
        normalize_fmri=False,
        feature_grouping="streams",
        feature_group_max_features=2,
    )
    dataset = NSDDataset(config)

    assert dataset.feature_group_ids is not None
    assert dataset.feature_group_ids.shape == (4,)
    assert dataset.feature_group_ids.max().item() == 2


def test_nsd_dataloaders_can_group_batches_by_stimulus(tmp_path: Path) -> None:
    root = _write_nsd_fixture(tmp_path, include_repeat=True)
    shutil.copytree(
        root / "nsddata" / "ppdata" / "subj01",
        root / "nsddata" / "ppdata" / "subj02",
    )
    shutil.copytree(
        root / "nsddata_betas" / "ppdata" / "subj01",
        root / "nsddata_betas" / "ppdata" / "subj02",
    )
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
        subjects=[1, 2],
        average_repeats=True,
        normalize_fmri=False,
        train_fraction=0.67,
        batch_size=4,
        group_batches_by_stimulus=True,
    )
    train_loader, _ = create_project_dataloaders(config, seed=1)
    batch = next(iter(train_loader))
    _, counts = torch.unique(batch.stimulus_id, return_counts=True)

    assert int(counts.max()) == 2


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
    train_stimuli = {dataset.samples[index].stimulus_id for index in train_dataset.indices}
    eval_stimuli = {dataset.samples[index].stimulus_id for index in eval_dataset.indices}

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


def _write_nsd_fixture(tmp_path: Path, include_repeat: bool = False) -> Path:
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
    rows = [
        "1\t1\t1\t1\t101\t1\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n",
        "1\t1\t1\t2\t102\t2\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n",
        "1\t1\t1\t3\t103\t3\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n",
        "1\t1\t1\t4\t999\t4\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n",
    ]
    if include_repeat:
        rows.insert(1, "1\t1\t1\t2\t101\t1\t0\t0\t1\t1\t0\tNaN\tNaN\t0\t1\t1\t0\t1\t0\n")

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
        + "".join(rows),
        encoding="utf-8",
    )
    num_trials = 5 if include_repeat else 4
    lh = torch.arange(num_trials * 2, dtype=torch.float32).reshape(num_trials, 2)
    rh = torch.arange(num_trials * 3, dtype=torch.float32).reshape(num_trials, 3)
    torch.save(lh, beta / "lh.betas_session01.mgh.pt")
    torch.save(rh, beta / "rh.betas_session01.mgh.pt")
    torch.save(torch.tensor([[1.0], [4.0]]), beta / "lh.ncsnr.mgh.pt")
    torch.save(torch.tensor([[3.0], [2.0], [5.0]]), beta / "rh.ncsnr.mgh.pt")
    return root
