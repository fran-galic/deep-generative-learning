import math
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Union

import torch
from torch import nn

from dists import Dist
from torch_utils import TorchBuiltin, TorchData


class RegularizedSequential(nn.Sequential):
    def __init__(self, *args, l1_decay=0.0, l2_decay=0.0):
        super().__init__(*args)
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay

    def regularization_loss(self):
        parameter = next(self.parameters())
        loss = torch.zeros((), dtype=parameter.dtype, device=parameter.device)

        for layer in self:
            if isinstance(layer, nn.Linear):
                loss = loss + self.l1_decay * layer.weight.abs().sum()
                loss = loss + self.l2_decay * layer.weight.square().sum()

        return loss


def _activation_layer(activation: TorchBuiltin) -> Optional[nn.Module]:
    if activation is None:
        return None
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()
    if isinstance(activation, str):
        activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "softplus": nn.Softplus,
        }
        return activations[activation.lower()]()

    raise ValueError(f"Nepodržana aktivacijska funkcija: {activation}")

@dataclass
class NetworkConfiguration:
    input_dim: Union[tuple[int, ...], int]
    hidden_dims: Sequence[int]
    output_dim: Union[tuple[int, ...], int]
    activation_fn: TorchBuiltin = None
    output_fn: TorchBuiltin = None
    l1_decay: float = 0.0
    l2_decay: float = 0.0
    dropout_rate: float = None
    kernel_initializer: Optional[TorchBuiltin] = None
    output_bias_initializer: Optional[TorchBuiltin] = None

    def build(self, name: Optional[str] = None) -> nn.Module:
        del name
        input_dim = math.prod(self.input_dim) if isinstance(self.input_dim, tuple) else self.input_dim
        output_dim = math.prod(self.output_dim) if isinstance(self.output_dim, tuple) else self.output_dim
        dimensions = [input_dim, *self.hidden_dims, output_dim]
        layers = []

        if isinstance(self.input_dim, tuple) and len(self.input_dim) > 1:
            layers.append(nn.Flatten())

        for index, (in_features, out_features) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            linear = nn.Linear(in_features, out_features)
            if self.kernel_initializer is not None:
                self.kernel_initializer(linear.weight)
            layers.append(linear)

            is_output = index == len(dimensions) - 2
            activation = self.output_fn if is_output else self.activation_fn
            activation_layer = _activation_layer(activation)
            if activation_layer is not None:
                layers.append(activation_layer)

            if not is_output and self.dropout_rate is not None and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        output_layer = next(layer for layer in reversed(layers) if isinstance(layer, nn.Linear))
        if self.output_bias_initializer is not None:
            self.output_bias_initializer(output_layer.bias)

        return RegularizedSequential(
            *layers,
            l1_decay=self.l1_decay,
            l2_decay=self.l2_decay,
        )

@dataclass
class Coder(Callable):
    network: nn.Module
    distribution: Dist

    def __call__(self, sample: TorchData, **kwargs) -> Dist:
        outputs = self.network(sample, **kwargs)
        output_shapes = self.distribution.param_shapes
        params = torch.split(outputs, output_shapes, dim=-1)

        self.distribution.update_params(*params)

        return self.distribution
