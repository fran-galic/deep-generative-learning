from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Collection, TypeVar

import torch
import torch.nn.functional as F

from torch_utils import TorchData

DistType = TypeVar("DistType", bound="Foo")

@dataclass
class Dist(ABC):
    dim: int

    def __default_pdf(self, sample: TorchData = None) -> TorchData:
        return torch.exp(-self.score(sample))

    def __default_score(self, sample: TorchData = None) -> TorchData:
        return -torch.log(self.pdf(sample).clamp_min(1e-37))

    def __post_init__(self):
        is_overriden = lambda attr: getattr(Dist, attr) != getattr(type(self), attr)

        if not (is_overriden("pdf") or is_overriden("score")):
            raise TypeError("At least one of `pdf` or `score` must be implemented.")

        if not is_overriden("pdf"):
            setattr(self, "pdf", self.__default_pdf)

        if not is_overriden("score"):
            setattr(self, "score", self.__default_score)

    @property
    @abstractmethod
    def params(self) -> Collection[str]:
        pass

    @property
    @abstractmethod
    def param_shapes(self) -> Collection[int]:
        pass

    def update_params(self, *args) -> DistType:
        for param, value in zip(self.params, args):
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Parameter `{param}` of `{type(self)}` not found.")

        return self

    @abstractmethod
    def sample(self, eps: TorchData = None) -> TorchData:
        pass

    def score(self, sample: TorchData) -> TorchData:
        return NotImplemented

    def pdf(self, sample: TorchData) -> TorchData:
        pass

@dataclass
class Normal(Dist):
    mean: TorchData = field(default=None, init=False)
    logvar: TorchData = field(default=None, init=False)

    @property
    def params(self) -> Collection[str]:
        return "mean", "logvar"

    @property
    def param_shapes(self) -> Collection[int]:
        return self.dim, self.dim

    def sample(self, eps: TorchData = None) -> TorchData:
        if eps is None:
            eps = torch.randn_like(self.mean)

        stddev = torch.exp(self.logvar / 2)

        return eps * stddev + self.mean

    def score(self, sample: TorchData) -> TorchData:
        var = torch.exp(self.logvar)
        score = 0.5 * ((sample - self.mean) ** 2 / var + self.logvar + math.log(2 * math.pi))
        score = torch.sum(score, dim=-1)

        return score

@dataclass
class Binary(Dist):
    logpi: TorchData = field(default=None, init=False)

    @property
    def params(self) -> Collection[str]:
        return "logpi",

    @property
    def param_shapes(self) -> Collection[int]:
        return 1,

    def sample(self, eps: TorchData = None) -> TorchData:
        if eps is None:
            eps = torch.rand_like(self.logpi)

        pi = torch.sigmoid(self.logpi)

        return torch.where(eps < pi, torch.ones_like(pi), torch.zeros_like(pi))

    def score(self, sample: TorchData) -> TorchData:
        signed_logit = torch.where(sample == 1, -self.logpi, self.logpi)

        return F.softplus(signed_logit)
