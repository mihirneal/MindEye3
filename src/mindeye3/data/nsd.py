from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from mindeye3.config import DataConfig
from mindeye3.data.batches import PairedBatch, collate_paired_batch
from mindeye3.data.types import BrainSample, StimulusEmbedding


@dataclass(frozen=True)
class NSDTrialRef:
    subject_id: int
    session: int
    session_trial_index: int
    stimulus_id: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NSDAveragedRef:
    subject_id: int
    stimulus_id: int
    trials: tuple[NSDTrialRef, ...]
    metadata: dict[str, Any]


NSDSampleRef = NSDTrialRef | NSDAveragedRef


class StimulusEmbeddingCache:
    """Torch-backed stimulus feature cache keyed by NSD 73K stimulus id."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("embedding cache must be a dict saved with torch.save")

        ids = payload.get("stimulus_ids", payload.get("image_ids"))
        embeddings = payload.get("embeddings", payload.get("image"))
        if ids is None or embeddings is None:
            raise KeyError("embedding cache requires stimulus_ids and embeddings tensors")

        ids_tensor = torch.as_tensor(ids, dtype=torch.long)
        embeddings_tensor = torch.as_tensor(embeddings, dtype=torch.float32)
        if ids_tensor.ndim != 1:
            raise ValueError("stimulus_ids must be a rank-1 tensor")
        if embeddings_tensor.ndim != 2:
            raise ValueError("embeddings must be a rank-2 tensor")
        if ids_tensor.shape[0] != embeddings_tensor.shape[0]:
            raise ValueError("stimulus_ids and embeddings must have the same length")

        self.embedding_dim = int(embeddings_tensor.shape[1])
        clip_tokens = payload.get("clip_tokens")
        clip_tokens_tensor = None if clip_tokens is None else torch.as_tensor(clip_tokens, dtype=torch.float32)
        if clip_tokens_tensor is not None and clip_tokens_tensor.shape[0] != ids_tensor.shape[0]:
            raise ValueError("stimulus_ids and clip_tokens must have the same length")

        self.clip_token_shape: tuple[int, int] | None = None
        if clip_tokens_tensor is not None:
            if clip_tokens_tensor.ndim != 3:
                raise ValueError("clip_tokens must be a rank-3 tensor of shape [images, tokens, dim]")
            self.clip_token_shape = (int(clip_tokens_tensor.shape[1]), int(clip_tokens_tensor.shape[2]))

        self._by_stimulus_id: dict[int, StimulusEmbedding] = {}
        for index, (stimulus_id, embedding) in enumerate(zip(ids_tensor.tolist(), embeddings_tensor)):
            clip_token_value = None if clip_tokens_tensor is None else clip_tokens_tensor[index].float()
            self._by_stimulus_id[int(stimulus_id)] = StimulusEmbedding(
                image=torch.nn.functional.normalize(embedding.float(), dim=0),
                clip_tokens=clip_token_value,
            )

    def __contains__(self, stimulus_id: int) -> bool:
        return stimulus_id in self._by_stimulus_id

    def __getitem__(self, stimulus_id: int) -> StimulusEmbedding:
        return self._by_stimulus_id[stimulus_id]


class NSDBetaLoader:
    def __init__(
        self,
        root: str | Path,
        beta_space: str,
        beta_version: str,
        max_cached_sessions: int = 2,
    ) -> None:
        self.root = Path(root)
        self.beta_space = beta_space
        self.beta_version = beta_version
        self.max_cached_sessions = max_cached_sessions
        self.feature_indices: torch.Tensor | None = None
        self._cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def session_exists(self, subject_id: int, session: int) -> bool:
        return all(path.exists() or path.with_suffix(path.suffix + ".pt").exists() for path in self._hemisphere_paths(subject_id, session))

    def load_trial(self, trial: NSDTrialRef) -> torch.Tensor:
        key = (trial.subject_id, trial.session)
        if key not in self._cache:
            self._cache[key] = self._load_session(*key)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cached_sessions:
                self._cache.popitem(last=False)
        session = self._cache[key]
        return session[trial.session_trial_index].clone()

    def load_ncsnr(self, subject_id: int) -> torch.Tensor:
        hemispheres = [self._load_beta_file(path) for path in self._ncsnr_paths(subject_id)]
        return torch.cat([hemi.flatten() for hemi in hemispheres], dim=0).float()

    def _hemisphere_paths(self, subject_id: int, session: int) -> tuple[Path, Path]:
        base = (
            self.root
            / "nsddata_betas"
            / "ppdata"
            / f"subj{subject_id:02d}"
            / self.beta_space
            / self.beta_version
        )
        return (
            base / f"lh.betas_session{session:02d}.mgh",
            base / f"rh.betas_session{session:02d}.mgh",
        )

    def _ncsnr_paths(self, subject_id: int) -> tuple[Path, Path]:
        base = (
            self.root
            / "nsddata_betas"
            / "ppdata"
            / f"subj{subject_id:02d}"
            / self.beta_space
            / self.beta_version
        )
        return (base / "lh.ncsnr.mgh", base / "rh.ncsnr.mgh")

    def _load_session(self, subject_id: int, session: int) -> torch.Tensor:
        hemispheres = [self._load_beta_file(path) for path in self._hemisphere_paths(subject_id, session)]
        if hemispheres[0].shape[0] != hemispheres[1].shape[0]:
            raise ValueError(f"hemisphere trial counts differ for subj{subject_id:02d} session {session}")
        session_betas = torch.cat(hemispheres, dim=1)
        if self.feature_indices is not None:
            session_betas = session_betas[:, self.feature_indices]
        return session_betas

    def _load_beta_file(self, path: Path) -> torch.Tensor:
        if path.exists():
            return _load_mgh(path)

        fixture_path = path.with_suffix(path.suffix + ".pt")
        if fixture_path.exists():
            return _load_tensor_beta_fixture(fixture_path)

        raise FileNotFoundError(path)


class NSDDataset(Dataset[tuple[BrainSample, StimulusEmbedding]]):
    def __init__(self, config: DataConfig) -> None:
        if config.root is None:
            raise ValueError("NSD data.root must point to the downloaded NSD directory")
        if config.embedding_cache is None:
            raise ValueError("NSD data.embedding_cache must point to a torch embedding cache")

        self.config = config
        self.root = Path(config.root)
        self.embedding_cache = StimulusEmbeddingCache(config.embedding_cache)
        self.beta_loader = NSDBetaLoader(
            self.root,
            beta_space=config.beta_space,
            beta_version=config.beta_version,
            max_cached_sessions=config.max_cached_sessions,
        )
        self.trials = self._build_trials()
        self.samples: list[NSDSampleRef] = self._build_samples()
        self.feature_indices = self._build_feature_indices()
        self.beta_loader.feature_indices = self.feature_indices
        if not self.samples:
            raise ValueError("No NSD trials matched the requested subjects, betas, and embedding cache")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[BrainSample, StimulusEmbedding]:
        sample_ref = self.samples[index]
        fmri = self._load_sample_fmri(sample_ref)
        if self.config.normalize_fmri:
            fmri = _zscore(fmri)
        return (
            BrainSample(
                subject_id=sample_ref.subject_id,
                fmri=fmri,
                stimulus_id=sample_ref.stimulus_id,
                metadata=sample_ref.metadata,
            ),
            self.embedding_cache[sample_ref.stimulus_id],
        )

    @property
    def fmri_dim(self) -> int:
        sample, _ = self[0]
        return int(sample.fmri.numel())

    @property
    def embedding_dim(self) -> int:
        return self.embedding_cache.embedding_dim

    def _load_sample_fmri(self, sample_ref: NSDSampleRef) -> torch.Tensor:
        if isinstance(sample_ref, NSDTrialRef):
            return self.beta_loader.load_trial(sample_ref).float()
        return torch.stack([self.beta_loader.load_trial(trial).float() for trial in sample_ref.trials]).mean(dim=0)

    def _build_trials(self) -> list[NSDTrialRef]:
        trials: list[NSDTrialRef] = []
        for subject_id in self.config.subjects:
            response_path = self.root / "nsddata" / "ppdata" / f"subj{subject_id:02d}" / "behav" / "responses.tsv"
            trials.extend(self._read_subject_trials(subject_id, response_path))
            if self.config.max_samples is not None and len(trials) >= self.config.max_samples:
                return trials[: self.config.max_samples]
        return trials

    def _build_samples(self) -> list[NSDSampleRef]:
        if not self.config.average_repeats:
            return list(self.trials)

        grouped: dict[tuple[int, int], list[NSDTrialRef]] = defaultdict(list)
        for trial in self.trials:
            grouped[(trial.subject_id, trial.stimulus_id)].append(trial)

        samples: list[NSDSampleRef] = []
        for (subject_id, stimulus_id), trials in grouped.items():
            samples.append(
                NSDAveragedRef(
                    subject_id=subject_id,
                    stimulus_id=stimulus_id,
                    trials=tuple(trials),
                    metadata={
                        "73kid": stimulus_id,
                        "num_repeats": len(trials),
                        "sessions": sorted({trial.session for trial in trials}),
                    },
                )
            )
        return samples

    def _read_subject_trials(self, subject_id: int, response_path: Path) -> list[NSDTrialRef]:
        if not response_path.exists():
            raise FileNotFoundError(response_path)

        session_counts: dict[int, int] = defaultdict(int)
        trials: list[NSDTrialRef] = []
        with response_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                session = int(row["SESSION"])
                session_trial_index = session_counts[session]
                session_counts[session] += 1

                if not self.config.include_missing_data and int(row.get("MISSINGDATA", "0")) != 0:
                    continue
                stimulus_id = int(row["73KID"])
                if stimulus_id not in self.embedding_cache:
                    continue
                if not self.beta_loader.session_exists(subject_id, session):
                    continue

                trials.append(
                    NSDTrialRef(
                        subject_id=subject_id,
                        session=session,
                        session_trial_index=session_trial_index,
                        stimulus_id=stimulus_id,
                        metadata={
                            "run": int(row["RUN"]),
                            "trial": int(row["TRIAL"]),
                            "73kid": stimulus_id,
                            "10kid": int(row["10KID"]),
                        },
                    )
                )
        return trials

    def _build_feature_indices(self) -> torch.Tensor | None:
        topk = self.config.ncsnr_topk
        if topk is None:
            return None
        if topk <= 0:
            raise ValueError("ncsnr_topk must be positive")

        ncsnr_values = [self.beta_loader.load_ncsnr(subject_id) for subject_id in self.config.subjects]
        lengths = {int(value.numel()) for value in ncsnr_values}
        if len(lengths) != 1:
            raise ValueError("ncsnr vectors must have the same length for multi-subject selection")

        stacked = torch.stack(ncsnr_values)
        aggregation = self.config.ncsnr_aggregation.lower()
        if aggregation == "mean":
            ncsnr = stacked.mean(dim=0)
        elif aggregation == "min":
            ncsnr = stacked.min(dim=0).values
        elif aggregation == "max":
            ncsnr = stacked.max(dim=0).values
        else:
            raise ValueError("ncsnr_aggregation must be one of: mean, min, max")
        effective_topk = min(topk, ncsnr.numel())
        return torch.topk(ncsnr, k=effective_topk).indices.sort().values


def create_nsd_dataloaders(
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader[PairedBatch], DataLoader[PairedBatch]]:
    dataset = NSDDataset(config)
    train_dataset, eval_dataset = split_by_stimulus(dataset, config.train_fraction, seed)
    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_paired_batch,
        ),
        DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_paired_batch,
        ),
    )


def split_by_stimulus(
    dataset: NSDDataset,
    train_fraction: float,
    seed: int,
) -> tuple[Subset[tuple[BrainSample, StimulusEmbedding]], Subset[tuple[BrainSample, StimulusEmbedding]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    stimulus_ids = sorted({sample.stimulus_id for sample in dataset.samples})
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(stimulus_ids), generator=generator).tolist()
    train_count = max(1, min(len(stimulus_ids) - 1, int(len(stimulus_ids) * train_fraction)))
    train_stimuli = {stimulus_ids[index] for index in permutation[:train_count]}

    train_indices = [index for index, sample in enumerate(dataset.samples) if sample.stimulus_id in train_stimuli]
    eval_indices = [index for index, sample in enumerate(dataset.samples) if sample.stimulus_id not in train_stimuli]
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def _load_mgh(path: Path) -> torch.Tensor:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise ImportError("Reading NSD .mgh beta files requires nibabel. Run `uv sync`.") from exc

    data = torch.as_tensor(nib.load(str(path)).get_fdata(dtype=np.float32)).squeeze()
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Expected rank-2 beta data after squeezing {path}, got shape {tuple(data.shape)}")
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data.contiguous().float()


def _load_tensor_beta_fixture(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False).float()
    if data.ndim != 2:
        raise ValueError("tensor beta fixtures must be rank-2")
    return data


def _zscore(value: torch.Tensor) -> torch.Tensor:
    std = value.std(unbiased=False)
    if std <= 0:
        return value - value.mean()
    return (value - value.mean()) / std
