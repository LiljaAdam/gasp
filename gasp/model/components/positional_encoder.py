import math

import torch


def _frequencies_orig(embed_dim: int, pos_dim: int, max_freq: float) -> torch.Tensor:
    log = math.log(max_freq)
    axis_dim = math.ceil(embed_dim / pos_dim)
    fqs = torch.arange(axis_dim // 2, dtype=torch.float)
    fqs = torch.exp(-2 * (fqs / axis_dim) * log)
    return fqs


def _frequencies_perceiver(embed_dim: int, max_freq: float) -> torch.Tensor:
    fqs = torch.pi * torch.linspace(0, max_freq, embed_dim // 2, dtype=torch.float).flip(dims=[-1])
    return fqs


def _frequencies(embed_dim: int, pos_dim: int, max_freq: float, perceiver: bool) -> torch.Tensor:
    if perceiver:
        return _frequencies_perceiver(embed_dim, max_freq)
    return _frequencies_orig(embed_dim, pos_dim, max_freq)
