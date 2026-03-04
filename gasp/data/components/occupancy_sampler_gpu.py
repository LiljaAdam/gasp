from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch import Tensor

import gasp.config.base_config as cfg
from gasp.common.types import PointCloudRange
from gasp.data.utils.ray_intersection import calculate_max_ray_distances_pt
from gasp.model.ops.functions.interpolate import interpolate_trajectories


def _transform(points: Tensor, poses: Tensor) -> Tensor:
    return (poses[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + poses[..., :3, 3]


def _stack(input: List[Tensor]) -> Optional[Tensor]:
    if input is None or len(input) == 0:
        return None
    return torch.stack(input)


@dataclass
class LidarGPUOccupancySamplerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: LidarGPUOccupancySampler)

    num_lidar_rays: int = 100_000
    """Number of lidar points to sample (* batchsize)."""
    num_missing_rays: int = 20_000
    """Number of missing rays to sample (* batchsize)."""
    num_query_points: int = 900_000
    """Number of positive samples to generate."""

    buffer_size: float = 0.1
    """Buffer size behind the positive samples."""

    negative_bias_power: Union[int, float] = 1
    """Whether to bias the negatives to be close to positives by taking (jitter)^power. 1 is no bias."""
    negative_from_points_ratio: float = -1.0  # [0.0, 1.0] for static ratio, -1.0 for dynamic ratio
    """Ratio of negative samples to generate from points in bev range (vs negative directions)."""
    negative_samples_based_on_ray_length: bool = True
    """Whether to sample negatives based on the length of the ray. This effectively samples more queries from longer rays."""

    positive_samples_based_on: Literal[
        "ray_length",
        "uniform",
    ] = "uniform"
    """Whether to sample positives based on the length of the ray. This effectively samples more queries from longer rays."""
    positive_samples_based_on_ray_length_power: float = 1.0
    """Power to raise the ray length to. 1.0 is no power."""


class LidarGPUOccupancySampler(torch.nn.Module):
    def __init__(
        self,
        config: LidarGPUOccupancySamplerConfig,
        point_cloud_range: PointCloudRange,
    ):
        super().__init__()
        self.config = config
        self.point_cloud_range = point_cloud_range

    def _uniform_subsample(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            idx = torch.randint(0, pc.shape[0], (num_per_sample,), dtype=torch.long, device=pc.device)
            subsampled_points.append(pc[idx])

            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idx])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_ray_length(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            lengths = pc[..., :3].norm(dim=-1, keepdim=False)
            probabilities = lengths**self.config.positive_samples_based_on_ray_length_power
            probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)
            idxs = torch.multinomial(probabilities, num_per_sample, replacement=True)
            subsampled_points.append(pc[idxs])

            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idxs])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.training:
            if self.config.positive_samples_based_on == "ray_length":
                return self._subsample_based_on_ray_length(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "uniform":
                return self._uniform_subsample(points, num_per_sample, labels)
            else:
                raise ValueError(f"Invalid positive samples based on: {self.config.positive_samples_based_on}")
        else:
            return self._uniform_subsample(points, num_per_sample, labels)

    def _compensate(
        self, points: Tensor, ego_poses: Tensor, ego_times: Tensor, extrinsics: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute compensated points and point origins (to t=0)."""
        poses = interpolate_trajectories(ego_poses, ego_times, points[..., -1].contiguous())  # B, N, 3, 4
        lidar2target = (poses @ extrinsics.unsqueeze(1)).to(points.dtype)  # B, N, 4, 4
        points[..., :3] = _transform(points[..., :3], lidar2target)
        origins = lidar2target[..., :3, 3]
        return points, origins

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        positive_points: Tensor,  # variable per sample
        ego_poses: Tensor,
        ego_times: Tensor,
        extrinsics: Tensor,
        negative_rays: Tensor,  # variable per sample
        future_points_ratio_inside: Tensor,
        positive_point_labels: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample occupancy queries and labels."""
        bs, device = positive_points.shape[0], positive_points.device

        num_positive_points = positive_points.shape[1]
        if num_positive_points == 0:
            return (
                torch.empty((bs, 0, 4), device=device),
                torch.empty((bs, 0, 1), device=device),
                {k: torch.empty_like(v) for k, v in positive_point_labels.items()},
            )

        points, labels = self._subsample(positive_points, self.config.num_lidar_rays, positive_point_labels)

        # compensate to ego frame
        points, origins = self._compensate(points, ego_poses, ego_times, extrinsics)

        # compute lengths and directions
        unnorm_directions = points[..., :3] - origins
        lengths = unnorm_directions.norm(dim=-1, keepdim=True)
        directions = unnorm_directions / lengths

        # sample num_query_points positive queries from points in range
        pos_idxs = torch.randint(0, points.shape[1], (bs, self.config.num_query_points), device=device)
        batch_idx = torch.arange(bs, device=device).unsqueeze(-1)
        positive_queries = points[batch_idx, pos_idxs]

        # Create positive queries
        jitter = torch.rand((bs, self.config.num_query_points, 1), device=device)
        positive_queries[..., :3] += directions[batch_idx, pos_idxs] * jitter * self.config.buffer_size

        # needs to be computed on the fly to handle case where negative rays are not available
        negative_from_points_ratio = self._compute_negative_from_points_ratio(future_points_ratio_inside)
        num_neg_from_points = self._num_neg_from_points(negative_rays, negative_from_points_ratio)
        num_neg_from_rays = self.config.num_query_points - num_neg_from_points

        # sample negative queries from positive points
        if self.config.negative_samples_based_on_ray_length:
            probabilities = lengths.squeeze(-1)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            neg_idxs = torch.multinomial(probabilities, num_neg_from_points, replacement=True)
        else:
            neg_idxs = torch.randint(0, points.shape[1], (bs, num_neg_from_points), device=device)

        negative_queries = points[batch_idx, neg_idxs]
        jitter = torch.rand((bs, num_neg_from_points, 1), device=device)
        if self.config.negative_bias_power != 1:
            jitter **= self.config.negative_bias_power
        negative_queries[..., :3] -= directions[batch_idx, neg_idxs] * jitter * lengths[batch_idx, neg_idxs]

        # sample negative queries from missing rays
        if num_neg_from_rays:
            neg_ray_queries = self._sample_from_missing_rays(
                negative_rays, num_neg_from_rays, ego_poses, ego_times, extrinsics
            )
            negative_queries = torch.cat([negative_queries, neg_ray_queries], dim=1)

        # concatenate positive and negative queries
        all_queries = torch.cat([positive_queries, negative_queries], dim=1)
        all_occupancy = torch.cat(
            [
                torch.ones((bs, self.config.num_query_points, 1), device=device, dtype=torch.float32),
                torch.zeros((bs, self.config.num_query_points, 1), device=device, dtype=torch.float32),
            ],
            dim=1,
        )

        all_labels = {}
        for key, label in labels.items():
            positive_labels = label[batch_idx, pos_idxs].long()
            if "valid" in key:
                negative_labels = positive_labels.clone()
            elif "vfm" in key:
                negative_labels = torch.zeros_like(positive_labels)
            else:
                negative_labels = (
                    torch.ones((bs, self.config.num_query_points, 1), device=device, dtype=torch.long) * -1
                )
            all_labels[key] = torch.cat([positive_labels, negative_labels], dim=1)

        return all_queries, all_occupancy, all_labels

    def _num_neg_from_points(self, negative_rays: Tensor, negative_from_points_ratio: float) -> int:
        """Determine ratio of negative queries from positive points vs missing rays."""
        if len(negative_rays) and negative_from_points_ratio < 1.0:
            if not all((neg_rays.shape[0] for neg_rays in negative_rays)):
                # special case: if no negative rays are available (for any sample in the batch), sample all negatives from points
                num_neg_from_points = self.config.num_query_points
            else:
                num_neg_from_points = int(self.config.num_query_points * negative_from_points_ratio)
        else:
            num_neg_from_points = self.config.num_query_points
        return num_neg_from_points

    def _compute_negative_from_points_ratio(self, future_points_ratio_inside: Tensor) -> float:
        """Compute the ratio of negative queries from positive points vs missing rays."""
        if self.config.negative_from_points_ratio < 0.0:
            # We need to have a ratio that is the same for all samples in the batch, so we take the mean
            ratio = future_points_ratio_inside.sum() / future_points_ratio_inside.shape[0]
        else:
            ratio = self.config.negative_from_points_ratio
        return ratio

    def _sample_from_missing_rays(
        self, negative_rays: Tensor, num_queries: int, ego_poses: Tensor, ego_times: Tensor, extrinsics: Tensor
    ) -> Tensor:
        """Sample negative queries from missing rays."""
        neg_rays = self._uniform_subsample(negative_rays, min(self.config.num_missing_rays, num_queries))[
            0
        ]  # TODO: Verify that it's better to do uniform subsampling here rather than a potentially dynamic one...
        device, bs = neg_rays.device, neg_rays.shape[0]
        batch_idx = torch.arange(bs, device=device).unsqueeze(-1)

        neg_rays, neg_origins = self._compensate(neg_rays, ego_poses, ego_times, extrinsics)
        neg_rays[..., :3] = neg_rays[..., :3] - neg_origins
        neg_rays[..., :3] /= neg_rays[..., :3].norm(dim=-1, keepdim=True)

        neg_lengths = calculate_max_ray_distances_pt(
            origins=neg_origins,
            normalized_directions=neg_rays[..., :3],
            bounds_min=torch.tensor(self.point_cloud_range[:3], device=device),
            bounds_max=torch.tensor(self.point_cloud_range[3:], device=device),
        ).unsqueeze(-1)

        neg_idxs = torch.randint(0, neg_rays.shape[1], (bs, num_queries), device=device)
        jitter = torch.rand((bs, num_queries, 1), device=device)
        neg_ray_queries = neg_rays[batch_idx, neg_idxs]
        neg_ray_queries[..., :3] *= neg_lengths[batch_idx, neg_idxs] * jitter
        neg_ray_queries[..., :3] += neg_origins[batch_idx, neg_idxs]

        return neg_ray_queries


if __name__ == "__main__":
    from gasp.model.base import DEFAULT_PCD_RANGE_FULL

    sampler = LidarGPUOccupancySamplerConfig().setup(point_cloud_range=DEFAULT_PCD_RANGE_FULL)
    sampler.sample(
        positive_points=torch.rand((2, 1000, 4)),
        ego_poses=torch.rand((2, 100, 4, 4)),
        ego_times=(torch.arange(0, 100, 1) / 100).unsqueeze(0).repeat(2, 1),
        extrinsics=torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        negative_rays=None,
    )
