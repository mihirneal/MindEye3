from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GeneratorAdapter(ABC):
    @abstractmethod
    def generate(self, conditioning: torch.Tensor) -> object:
        raise NotImplementedError


class ImageGeneratorAdapter(GeneratorAdapter):
    pass


class VideoGeneratorAdapter(GeneratorAdapter):
    pass


class FluxAdapter(ImageGeneratorAdapter):
    def generate(self, conditioning: torch.Tensor) -> object:
        raise NotImplementedError("FLUX generation is not implemented in the V0 scaffold")


class LTXAdapter(VideoGeneratorAdapter):
    def generate(self, conditioning: torch.Tensor) -> object:
        raise NotImplementedError("LTX video generation is not implemented in the V0 scaffold")

