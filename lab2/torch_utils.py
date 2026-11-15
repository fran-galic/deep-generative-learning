from typing import Callable, Union

import torch

TorchData = Union[torch.Tensor, torch.nn.Parameter, float]
TorchBuiltin = Union[Callable, str]
