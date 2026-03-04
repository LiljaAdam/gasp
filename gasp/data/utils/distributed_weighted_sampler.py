import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
        seed: int = 0,
    ):
        """
        Args:
            weights (sequence): a sequence of weights, not normalized.
            num_samples (int, optional): number of samples per process. Defaults to ceil(len(weights) / num_replicas).
            replacement (bool): whether to sample with replacement.
            rank (int, optional): rank of the current process.
            num_replicas (int, optional): number of processes participating in DDP.
            seed (int): random seed.
        """
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, dtype=torch.double)
        self.weights = weights
        self.replacement = replacement
        self.seed = seed

        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.rank = rank
        self.num_replicas = num_replicas
        self.total_size = len(self.weights)
        self.epoch = 0

        if num_samples is None:
            self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        else:
            self.num_samples = num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample indices (same across all processes)
        indices = torch.multinomial(
            self.weights, self.num_samples * self.num_replicas, self.replacement, generator=g
        ).tolist()

        # Subsample for this rank
        indices_rank = indices[self.rank :: self.num_replicas]
        return iter(indices_rank)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Sets the epoch for deterministic behavior across epochs."""
        self.epoch = epoch
