from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@torch.compile
class CFLayerNorm(nn.Module):
    """Channels-first layernorm."""

    def __init__(self, num_channels: int, **kwargs):
        """Initialize."""
        super().__init__()
        self._ln = nn.LayerNorm(normalized_shape=num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
        """
        assert (
            len(x.shape)
        ) == 4, f"Channels-first layernorm is only defined for rank 4 spatial tensors but got rank {len(x.shape)}."
        # (B, C, H, W) -> (B, H, W, C) -> (B, C, H, W)
        # This is slow without compilation because the tensor will become noncontiguous in GPU
        # memory before the layernorm computation, which slows down the kernel. But torch.compile
        # successcully fuses all operations in this case.
        return self._ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable = F.relu,
        dropout: float = 0.0,
        hidden_layernorm: bool = False,
    ):
        """Initialize the MLP.
        Args:
            input_dim: Number of input channels.
            hidden_dim: Number of hidden channels.
            output_dim: Number of output channels.
            num_layers: Number of layers.
            activation: Activation function.
            dropout: Dropout rate.
            hidden_layernorm: Apply layernorm before hidden layers. This is common in subnormalized
                              transformers.
        """
        super().__init__()
        self._dropout = nn.Dropout(p=dropout)
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self._linears = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )
        self._activation = activation
        if hidden_layernorm:
            layernorm_dim = list(hidden_dims)
            self._layer_norms = nn.ModuleList(nn.LayerNorm(h) for h in layernorm_dim)
        else:
            self._layer_norms = [lambda x: x for _ in hidden_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        for i, linear in enumerate(self._linears):
            x = linear(x)
            if i < len(self._linears) - 1:
                x = self._activation(x)
                x = self._layer_norms[i](x)
            x = self._dropout(x)
        return x
