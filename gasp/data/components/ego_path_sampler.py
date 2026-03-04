from dataclasses import dataclass, field
from typing import Tuple, Type

import numpy as np
import scipy
from shapely.geometry import LineString, box
from zod.data_classes.ego_motion import EgoMotion

import gasp.config.base_config as cfg
from gasp.common.types import PointCloudRange
from gasp.data.components.pose_utils import fast_ego_motion_poses, inverse_np

ZOD_MAX_INTENSITY = 255
ZOD_FRAMES_MAX_N_FUTURE = 10
CENTER_FRAME = 10
EPS = 1e-3

QUERY_DTYPE = np.float32
LABEL_DTYPE = np.int32


class NoEgoPathError(ValueError):
    pass


def min_distance_between_ego_and_point(ego_path_points, point):
    return np.min(np.linalg.norm(ego_path_points - point, axis=-1))


# this one one is very slow, ye be warned
def _interpolate_linestring(line: LineString, distances: np.ndarray) -> np.ndarray:
    return np.array([line.interpolate(d, normalized=True).coords[0] for d in distances])


def _interpolate_numpy(line: LineString, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a linestring at given distances.

    Assumes that distances are in the range [0,1].
    """
    x, y = line.xy
    ego_path_points = np.array([x, y]).T
    diff = np.diff(ego_path_points, axis=0)
    cumsum = np.cumsum(np.linalg.norm(diff, axis=1))
    cumsum = cumsum / cumsum[-1]
    cumsum = np.concatenate([[0], cumsum])
    x_ = np.interp(distances, cumsum, x)
    y_ = np.interp(distances, cumsum, y)
    dir = np.apply_along_axis(lambda col: np.interp(distances, cumsum, col), axis=0, arr=np.vstack((diff, diff[-1:])))
    dir /= np.linalg.norm(dir, axis=-1, keepdims=True)
    return np.array([x_, y_]).T, dir


@dataclass
class EgoPathSamplerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: EgoPathSampler)

    num_query_points: int = 10_000
    """Number of positive samples to generate."""

    buffer_size: float = 1.0
    """Buffer size around the positive samples."""

    oversample_factor: float = 5.0
    """Oversample factor for negative sample close to the ego."""

    query_dim: int = 2
    """Dimension of the query points."""

    sprinkle_points: bool = True
    """Whether to sprinkle points randomly within the point cloud range."""
    sprinkle_points_ratio: float = 0.15
    """Ratio of points to sprinkle randomly within the point cloud range."""
    sprinkle_points_not_on_ego_path: bool = True
    """Whether to sprinkle points not on the ego path."""


class EgoPathSampler:
    def __init__(
        self,
        config: EgoPathSamplerConfig,
        point_cloud_range: PointCloudRange,
    ):
        self.config = config
        self.point_cloud_range = point_cloud_range

    def _get_ego_path_in_range(
        self, oxts: EgoMotion, current_timestamp: float, bev_oxts_pose: np.ndarray
    ) -> LineString:
        last_oxts_timestamp = oxts.timestamps[-1]

        poses_timestamps = np.linspace(current_timestamp, last_oxts_timestamp, 1000)

        ego_path = fast_ego_motion_poses(oxts, poses_timestamps)

        # Transform to current frame
        ego_path = inverse_np(bev_oxts_pose) @ ego_path

        # Get the ego path in the local patch
        ego_path = LineString(ego_path[:, : self.config.query_dim, 3])
        local_patch = box(
            self.point_cloud_range[0],
            self.point_cloud_range[1],
            self.point_cloud_range[3],
            self.point_cloud_range[4],
        )

        ego_path = ego_path.intersection(local_patch)

        if ego_path.is_empty:
            raise NoEgoPathError("Ego path is empty")

        if ego_path.geom_type == "MultiLineString":
            ego_path = ego_path.geoms[0]  # type: ignore

        return ego_path  # type: ignore

    def _sample_positive_queries(self, positive_queries: np.ndarray) -> np.ndarray:
        # Add random buffer around the positive samples
        ranges = np.random.uniform(0, self.config.buffer_size, (self.config.num_query_points, self.config.query_dim))
        angles_x_y = np.random.uniform(0, 2 * np.pi, (self.config.num_query_points, 1))

        if self.config.query_dim == 2:
            positive_queries += ranges * np.hstack([np.cos(angles_x_y), np.sin(angles_x_y)])
        elif self.config.query_dim == 3:
            angles_z = np.random.uniform(0, 2 * np.pi, (self.config.num_query_points, 1))
            positive_queries += ranges * np.hstack([np.cos(angles_x_y), np.sin(angles_x_y), np.cos(angles_z)])
        else:
            raise ValueError("Query dimension must be 2 or 3")

        return positive_queries

    def _sample_negative_queries(self, queries, dirs) -> np.ndarray:
        # random offset along path
        queries += dirs * np.random.uniform(-0.5, 0.5, (self.config.num_query_points, 1))
        if self.config.query_dim == 3:
            # random height
            queries[:, 2] += np.random.normal(
                -self.config.buffer_size, self.config.buffer_size, self.config.num_query_points
            )
        # random perpendicular offset using a truncated normal distribution
        perpendicular_offset = scipy.stats.truncnorm.rvs(
            0, np.inf, self.config.buffer_size + EPS, self.config.oversample_factor, self.config.num_query_points
        )  # TODO: improve this to make sure we get enough coverage far away
        perpendicular_offset *= np.random.choice([-1, 1], self.config.num_query_points)
        queries[:, :2] += perpendicular_offset[:, None] * np.array([-dirs[:, 1], dirs[:, 0]]).T
        return queries

    def sample(self, oxts: EgoMotion, current_timestamp: float, bev_oxts_pose: np.ndarray) -> dict:
        try:
            ego_path = self._get_ego_path_in_range(oxts, current_timestamp, bev_oxts_pose)
        except NoEgoPathError:
            # note that we create these dummy samples to avoid breaking the pipeline
            samples = {
                "ego_path_queries": np.zeros((2, self.config.query_dim), dtype=QUERY_DTYPE),
                "ego_path_label": np.ones((2, 1), dtype=LABEL_DTYPE),
                "ego_path_valid": np.ones((2, 1), dtype=LABEL_DTYPE),
            }
            return samples

        # Draw self.config.num_query_points samples from the ego path
        s = np.random.uniform(0, 1, self.config.num_query_points)
        queries, dirs = _interpolate_numpy(ego_path, s)

        positive_queries = self._sample_positive_queries(queries.copy())
        negative_queries = self._sample_negative_queries(queries, dirs)

        # Generate labels
        positive_labels = np.ones((self.config.num_query_points, 1), dtype=LABEL_DTYPE)
        negative_labels = np.zeros((self.config.num_query_points, 1), dtype=LABEL_DTYPE)

        labels = np.vstack([positive_labels, negative_labels])
        queries = np.vstack([positive_queries, negative_queries])
        if self.config.query_dim == 2:
            queries[:, :2] = np.clip(queries[:, :2], self.point_cloud_range[:2], self.point_cloud_range[3:5])
        else:
            queries[:, :3] = np.clip(queries[:, :3], self.point_cloud_range[:3], self.point_cloud_range[3:])

        if self.config.sprinkle_points:
            n_sprinkle_points = int(self.config.sprinkle_points_ratio * self.config.num_query_points)
            sprinkle_points = np.random.uniform(
                self.point_cloud_range[: self.config.query_dim],
                self.point_cloud_range[3 : 3 + self.config.query_dim],
                (n_sprinkle_points, self.config.query_dim),
            )

            if self.config.sprinkle_points_not_on_ego_path:
                # see if sprinkle points are close to ego_path
                deltas = np.array(self.point_cloud_range[3 : 3 + self.config.query_dim]) - np.array(
                    self.point_cloud_range[: self.config.query_dim]
                )
                distances = deltas**2
                max_range = np.sqrt(np.sum(distances))
                min_ego_path_samples = int(max_range / self.config.buffer_size)
                ego_path_points = np.array(
                    [
                        ego_path.interpolate(p, normalized=True).coords
                        for p in np.linspace(0, 1, min_ego_path_samples, dtype=float)
                    ]
                ).squeeze(1)
                # check the distance between ego_path_points and sprinkle_points
                d_sprinkles_ep = (sprinkle_points[:, None] - ego_path_points[None, :]) ** 2
                d_sprinkles_ep = np.sum(d_sprinkles_ep, axis=-1)
                d_sprinkles_ep = np.min(d_sprinkles_ep, axis=-1)
                # only use the points that are far away from ego path
                m_dist = d_sprinkles_ep > self.config.buffer_size**2
                sprinkle_points = sprinkle_points[m_dist]

            n_sprinkle_points = len(sprinkle_points)

            # Replace the last n_sprinkle_points negative samples with the sprinkle points
            queries[-n_sprinkle_points:] = sprinkle_points
            labels[-n_sprinkle_points:] = 0

        samples = {
            "ego_path_queries": queries.astype(QUERY_DTYPE),
            "ego_path_label": labels.astype(LABEL_DTYPE),
            "ego_path_valid": np.ones_like(labels),
        }

        return samples


if __name__ == "__main__":
    import time

    from zod import ZodFrame, ZodFrames

    from gasp.model.base import DEFAULT_PCD_RANGE_FULL

    zod = ZodFrames("/datasets/zod", "mini")
    frame: ZodFrame = next(iter(zod))
    sampler = EgoPathSamplerConfig(num_query_points=50_000).setup(point_cloud_range=DEFAULT_PCD_RANGE_FULL)

    # warmup
    samples = sampler.sample(frame, frame.info.keyframe_time)

    n_iters = 10
    print(f"running benchmark with {n_iters} iterations")
    start = time.time()
    for _ in range(n_iters):
        samples = sampler.sample(frame, frame.info.keyframe_time)
    print("Time:", time.time() - start)
    import matplotlib.pyplot as plt

    plt.scatter(
        samples["ego_path_queries"][:, 0], samples["ego_path_queries"][:, 1], c=samples["ego_path_label"], s=0.1
    )
    plt.savefig("ego_path_sampler.png")
