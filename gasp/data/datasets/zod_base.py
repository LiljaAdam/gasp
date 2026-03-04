import datetime
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Type, Union

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from zod import ZodDrives, ZodFrames, ZodSequences
from zod.constants import Lidar
from zod.data_classes.calibration import Calibration
from zod.data_classes.ego_motion import EgoMotion
from zod.data_classes.sensor import CameraFrame, LidarData

from gasp.common.types import PointCloudRange, Zample
from gasp.data.components.cached_vfm_reader import CachedVFMFeaturesReader, CachedVFMFeaturesReaderConfig
from gasp.data.components.ego_path_sampler import EgoPathSampler, EgoPathSamplerConfig
from gasp.data.components.pose_utils import inverse_np

ZOD_MAX_INTENSITY = 255
FRAMES_LIDAR_FREQUENCY = 10
FRAMES_IMAGES_FREQUENCY = 10
warnings.filterwarnings(action="ignore", category=UserWarning)


@staticmethod
def base_collate_fn(batch: List[dict]):
    res = {}
    keys_to_pop = ("poses", "poses_timestamps")
    collate_lidar = "lidars" in batch[0] and all(batch[0]["lidars"].shape[1] == b["lidars"].shape[1] for b in batch)
    if not collate_lidar:
        keys_to_pop += ("lidars", "lidar_timestamps")

    collate_offset_pointclouds = "offset_pointclouds" in batch[0] and all(
        batch[0]["offset_pointclouds"].shape[1] == b["offset_pointclouds"].shape[1] for b in batch
    )
    if not collate_offset_pointclouds:
        keys_to_pop += ("offset_pointclouds", "offset_pointclouds_indices")

    # pop the keys that can be collated manually as they are variable length
    for key in keys_to_pop:
        if key in batch[0]:
            res[key] = [b.pop(key) for b in batch]

    # run the default collate function for the rest of the keys
    res.update(default_collate(batch))  # type: ignore

    return res


@staticmethod
def new_base_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    res = {}
    keys_to_pop = ("poses", "poses_timestamps")

    # Determine if we can safely collate lidars
    collate_lidar = "lidars" in batch[0] and all(batch[0]["lidars"].shape[1] == b["lidars"].shape[1] for b in batch)
    if not collate_lidar:
        keys_to_pop += ("lidars", "lidar_timestamps")

    # Determine if we can safely collate offset pointclouds
    collate_offset_pointclouds = "offset_pointclouds" in batch[0] and all(
        batch[0]["offset_pointclouds"].shape[1] == b["offset_pointclouds"].shape[1] for b in batch
    )
    if not collate_offset_pointclouds:
        keys_to_pop += ("offset_pointclouds", "offset_pointclouds_indices")

    # Copy the batch to avoid mutating original dicts
    batch_copy = [b.copy() for b in batch]

    # Extract and store the variable-length fields manually
    for key in keys_to_pop:
        if key in batch_copy[0]:
            res[key] = [b.pop(key) for b in batch_copy]

    # Use PyTorch's default_collate for the rest
    res.update(default_collate(batch_copy))  # type: ignore

    return res


@dataclass
class ZODBaseDatasetConfig:
    data: Path = Path("/datasets/zod/")
    """The path to the dataset."""
    version: Literal["full", "mini"] = "full"
    """What version of the dataset to use."""
    ids_to_load: List[int] = field(default_factory=lambda: [])
    """The sequence ids to load. If empty, all ids are loaded."""
    frames_to_load: List[str] = field(default_factory=lambda: [])
    """The frames within the sequences to load. If empty, all frames are loaded."""

    load_gasp_infos_if_exists: bool = True
    """Whether to load the gasp infos if they exist."""
    save_gasp_infos: bool = True
    """Whether to save the gasp infos."""

    make_batchable: bool = False
    """Whether to make the dataset batchable. If False, the dataset will only be able to run in single sample mode."""
    only_tokens: bool = True
    """Whether to only load the tokens as inputs. (Only applicable for nuscenes)"""
    only_tokens_for_offsets: bool = True
    """Whether to only load the tokens for the offsets as supervision. (Only applicable for nuscenes)"""
    upsample_key_frames: bool = True
    """Whether to upsample the key frames to the same frequency as the other frames."""
    key_frames_weight: float = 5.0
    """The weight of the key frames."""

    # lidar
    load_lidar: bool = False
    """Whether to load the lidar data."""
    n_prev_lidar: int = 3
    """How many lidar scans to load as input. (In addition to the current scan)."""
    prev_lidar_max_time: float = 3.0
    """The maximum time difference between the current lidar scan and the previous lidar scans. (seconds)"""
    n_future_lidar: int = -1
    """How many future lidar scans to load AS INPUT to the model. (In addition to the current scan)."""
    future_lidar_max_time: float = 3.0
    """The maximum time difference between the current lidar scan and the future lidar scans. (seconds)"""
    aggregate_lidar: bool = True
    """Whether to aggregate the lidar scans."""
    n_max_lidar_points: int = -1
    """The maximum number of points to load. -1 for no limit."""
    point_cloud_range: PointCloudRange = (0, 0, 0, 0, 0, 0)  # will be set by model
    """The range of the point cloud (x_min, y_min, z_min, x_max, y_max, z_max)."""

    # Offset frames settings for both occupancy and dino features
    num_future_frames_to_sample: int = 3
    """Number of future frames to sample. If 0 (or negative), no future frames will be sampled."""
    min_future_frame_s: float = 0.05
    """Minimum time in the future to sample num_future_frames_to_sample from."""
    max_future_frame_s: float = 3.0
    """Maximum time in the future to sample num_future_frames_to_sample from."""
    num_past_frames_to_sample: int = 0
    """Number of past frames to sample. If 0 (or negative), no past frames will be sampled."""
    min_past_frame_s: float = 0.05
    """Minimum time in the past to sample num_past_frames_to_sample from."""
    max_past_frame_s: float = 3.0
    """Maximum time in the past to sample num_past_frames_to_sample from."""
    num_diodes: int = 128
    """Number of diodes to use."""
    always_add_current_frame_to_offsets: bool = True
    """Whether to always add the current frame to the offsets. """

    # lidar occupancy query sampler
    load_offset_pointclouds: bool = False
    """Whether to load the offset lidar scan to use for occupancy data."""
    n_max_offset_points_to_load_per_scan: int = -1
    """The maximum number of future points to load, for dataloader efficiency (-1 = no limit)."""
    n_max_missing_rays_to_load_per_scan: int = -1
    """The maximum number of missing rays to send to gpu (-1 = no limit)."""
    n_max_poses_to_load_per_scan: int = 650
    """The maximum number of poses to send to gpu (-1 = no limit)."""

    add_missing_points: bool = False
    """Whether to complete the future point clouds with missing points data."""
    missing_points_num_azimuths: int = 180
    """Number of azimuth bins in the lidar. Lower values will result in fewer missing points."""
    missing_points_min_elevation: float = 0.0  # [deg]
    """The minimum elevation of the created missing points."""
    missing_points_range: float = 500  # [m]
    """The range of the created missing points."""
    missing_points_downsample_factor: float = -1.0
    """The downsample factor for the created missing points. -1.0 for no downsampling."""

    # VFM features projection with cached features
    load_cached_vfm_features: bool = False
    """Whether to load the VFM features projection data."""
    load_cached_vfm_compression_info: bool = False
    """Whether to load the VFM features compressed information."""
    cached_vfm_cfg: CachedVFMFeaturesReaderConfig = field(default_factory=lambda: CachedVFMFeaturesReaderConfig())
    """The VFM features projection sampler configuration."""
    load_vfm_stuff: bool = False
    """Whether to load the VFM stuff."""

    # ego path query sampler
    load_ego_path_samples: bool = False
    """Whether to load the ego path data."""
    eps_cfg: EgoPathSamplerConfig = field(default_factory=lambda: EgoPathSamplerConfig())
    """The ego path sampler configuration."""

    # Configuration for all downstream tasks
    tasks_region_of_interest: PointCloudRange = (0, 0, 0, 0, 0, 0)  # will be set by model
    tasks_canvas_size: Tuple[int, int] = (0, 0)  # will be set by model

    # Augmentations
    rotation_augmentation: bool = True
    """Whether to apply rotation augmentations."""
    rotation_range: Tuple[float, float] = (-20, 20)
    """The range of the rotation augmentation. (degrees)"""
    translation_augmentation: bool = False
    """Whether to apply rotation and translation augmentations."""
    translation_range: Tuple[float, float] = (-1.0, 1.0)
    """The range of the translation augmentation. (meters)"""

    @property
    def n_max_offset_points(self) -> int:
        if self.n_max_offset_points_to_load_per_scan < 0:
            return self.n_max_offset_points_to_load_per_scan

        num_frames = 0
        if self.num_future_frames_to_sample > 0:
            num_frames += self.num_future_frames_to_sample
        if self.num_past_frames_to_sample > 0:
            num_frames += self.num_past_frames_to_sample

        return self.n_max_offset_points_to_load_per_scan * num_frames

    @property
    def n_max_missing_rays(self) -> int:
        if self.n_max_missing_rays_to_load_per_scan < 0:
            return self.n_max_missing_rays_to_load_per_scan

        num_frames = 0
        if self.num_future_frames_to_sample > 0:
            num_frames += self.num_future_frames_to_sample
        if self.num_past_frames_to_sample > 0:
            num_frames += self.num_past_frames_to_sample
        return self.n_max_missing_rays_to_load_per_scan * num_frames

    @property
    def n_max_poses(self) -> int:
        return self.n_max_poses_to_load_per_scan * (self.num_future_frames_to_sample + self.num_past_frames_to_sample)


class ZODBaseDataset(Dataset):
    config: ZODBaseDatasetConfig
    _ZOD_DEVKIT_CLS: Union[Type[ZodFrames], Type[ZodSequences], Type[ZodDrives]] = ZodSequences
    _BLACKLIST: List[str] = []

    def __init__(
        self,
        config: ZODBaseDatasetConfig,
        train_or_val: Literal["train", "val"],
    ):
        super().__init__()
        assert config.point_cloud_range != (0, 0, 0, 0, 0, 0), "Model did not set point_cloud_range."

        self.config = config
        self.train_or_val = train_or_val
        assert config.load_image or config.load_lidar, "Must load either image or lidar data."

        if config.load_cached_vfm_features or config.load_cached_vfm_compression_info:
            self.cached_vfm_reader: CachedVFMFeaturesReader = config.cached_vfm_cfg.setup()
        if config.load_ego_path_samples:
            self.ego_path_sampler: EgoPathSampler = config.eps_cfg.setup(point_cloud_range=config.point_cloud_range)

        zod = self._ZOD_DEVKIT_CLS(str(self.config.data), version=config.version, mp=False)
        ids_ = sorted(list(zod.get_split(self.train_or_val)))
        self.ids_ = [frame_id for frame_id in ids_ if frame_id not in self._BLACKLIST]

        if config.ids_to_load:
            ids_to_load = [str(seq).zfill(6) for seq in config.ids_to_load]
            self.ids_ = [frame_id for frame_id in self.ids_ if frame_id in ids_to_load]

        del zod

    @property
    def is_train(self) -> bool:
        return self.train_or_val == "train"

    @property
    def is_val(self) -> bool:
        return self.train_or_val == "val"

    @property
    def any_augmentation(self) -> bool:
        return self.config.rotation_augmentation or self.config.translation_augmentation

    @property
    def image_frequency(self) -> float:
        return FRAMES_IMAGES_FREQUENCY

    @property
    def lidar_frequency(self) -> float:
        return FRAMES_LIDAR_FREQUENCY

    @property
    def max_future_frames(self) -> int:
        return max(1, self.max_future_offset)

    @property
    def max_prev_images_frames(self) -> int:
        return int(np.ceil(self.config.prev_images_max_time * self.image_frequency))

    @property
    def max_prev_lidar_frames(self) -> int:
        return int(np.ceil(self.config.prev_lidar_max_time * self.lidar_frequency))

    @property
    def max_future_images_frames(self) -> int:
        return int(np.ceil(self.config.future_images_max_time * self.image_frequency))

    @property
    def max_future_lidar_frames(self) -> int:
        return int(np.ceil(self.config.future_lidar_max_time * self.lidar_frequency))

    @property
    def load_future_offset_frames(self) -> bool:
        return self.config.num_future_frames_to_sample > 0

    @property
    def load_past_offset_frames(self) -> bool:
        return self.config.num_past_frames_to_sample > 0

    @property
    def min_future_offset(self) -> float:
        return int(self.config.min_future_frame_s * self.lidar_frequency)

    @property
    def max_future_offset(self) -> float:
        return int(self.config.max_future_frame_s * self.lidar_frequency)

    @property
    def min_past_offset(self) -> float:
        return int(self.config.min_past_frame_s * self.lidar_frequency)

    @property
    def max_past_offset(self) -> float:
        return int(self.config.max_past_frame_s * self.lidar_frequency)

    def get_start_and_end_frames_buffer(self) -> Tuple[int, int]:
        start_frames_buffer_alternatives = []
        end_frames_buffer_alternatives = []
        if self.config.load_image:
            start_frames_buffer_alternatives.append(self.max_prev_images_frames)
            end_frames_buffer_alternatives.append(self.max_future_images_frames)
        if self.config.load_lidar:
            start_frames_buffer_alternatives.append(self.max_prev_lidar_frames)
            end_frames_buffer_alternatives.append(self.max_future_lidar_frames)
        if len(start_frames_buffer_alternatives) == 0:
            start_frames_buffer_alternatives.append(self.max_prev_lidar_frames)
            end_frames_buffer_alternatives.append(self.max_future_lidar_frames)

        start_frames_buffer = max([2] + start_frames_buffer_alternatives)
        end_frames_buffer = max([1] + end_frames_buffer_alternatives)
        return start_frames_buffer, end_frames_buffer

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self._getitem(index)
        except Exception as e:
            print(f"Error at index {index}")
            print(f"Which is this bad boy: {self._get_zample(index)[0].info.id}")
            print(f"With the error: {e}")
            # raise e
            return self.__getitem__(index + 1)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _get_zample(self, index: int) -> Tuple[Zample, int]:
        pass

    @abstractmethod
    def _get_current_time_from_zample(self, zample: Zample, frame_idx: int) -> datetime.datetime:
        pass

    @abstractmethod
    def _get_camera_frames(self, zample: Zample) -> List[CameraFrame]:
        pass

    @abstractmethod
    def _aggregate_lidar_frames(self, zample: Zample, current_idx: int, bev_oxts_pose: np.ndarray) -> LidarData:
        pass


def points_in_target_bev(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: Calibration,
    timestamp: float,
    point_cloud_range: PointCloudRange,
    bev_oxts_pose: np.ndarray,
) -> np.ndarray:
    old_points = lidar_data.points
    # motion compensate scanwise to the target timestamp
    source_pose = ego_motion.get_poses(lidar_data.core_timestamp)
    # target_pose = ego_motion.get_poses(timestamp)
    target_pose = bev_oxts_pose
    lidar_data.transform(calibration.lidars[Lidar.VELODYNE].extrinsics)
    lidar_data.transform(inverse_np(target_pose) @ source_pose)
    # check bev range
    min_range = np.array(point_cloud_range[:3])
    max_range = np.array(point_cloud_range[3:])
    in_range = np.all((min_range <= lidar_data.points) & (lidar_data.points <= max_range), axis=-1)
    # restore lidar_data (we don't want to overwrite)
    lidar_data.points = old_points
    return in_range
