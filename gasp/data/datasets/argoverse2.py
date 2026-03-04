import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import av2
import av2.geometry
import av2.geometry.geometry
import av2.utils
import av2.utils.io
import numpy as np
import torch
from av2.datasets.sensor.constants import RingCameras as Cam
from av2.structures.sweep import Sweep
from torch.utils.data import Dataset
from zod.data_classes.ego_motion import EgoMotion

import gasp.config.base_config as cfg
from gasp.common.types import PointCloudRange
from gasp.data.components.cached_vfm_reader import AV2CachedVFMFeaturesReader, AV2CachedVFMFeaturesReaderConfig
from gasp.data.components.ego_path_sampler import EgoPathSampler
from gasp.data.components.pose_utils import fast_ego_motion_poses, inverse_np
from gasp.data.components.vfm_reader import VFMReader, VFMReaderConfig
from gasp.data.datasets.zod_base import ZODBaseDatasetConfig

AV2_MAX_REFLECTANCE: float = 255.0
AV2_LIDAR_FREQUENCY: int = 10
AV2_IMAGE_FREQUENCY: int = 10
# (top, right, bottom, left)
AV2_CAM_CROP = {
    Cam.RING_FRONT_CENTER.value: (0, 0, 250, 0),
    Cam.RING_FRONT_LEFT.value: (0, 0, 0, 0),
    Cam.RING_FRONT_RIGHT.value: (0, 0, 0, 0),
    Cam.RING_REAR_LEFT.value: (0, 0, 250, 0),
    Cam.RING_REAR_RIGHT.value: (0, 0, 250, 0),
    Cam.RING_SIDE_LEFT.value: (0, 0, 0, 0),
    Cam.RING_SIDE_RIGHT.value: (0, 0, 0, 0),
}

AV2_SPLITS_DIR = "gasp/data/splits/argoverse2/"

warnings.filterwarnings(action="ignore", category=UserWarning)


@dataclass
class AV2DatasetConfig(ZODBaseDatasetConfig, cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: AV2Dataset)
    """The target class for this configuration."""

    data: Path = Path("/datasets/argoverse/")
    """The path to the dataset."""
    subset: Literal["sensor", "lidar"] = "sensor"
    """The subset of the dataset to use."""

    av2_cached_missing_points_dir: Path = Path("/datasets/av2_missing_points")  # Path("/outputs/cached_missing_points")

    cached_vfm_cfg: AV2CachedVFMFeaturesReaderConfig = field(default_factory=lambda: AV2CachedVFMFeaturesReaderConfig())
    """The VFM features projection sampler configuration."""

    vfm_reader_cfg: VFMReaderConfig = field(default_factory=lambda: VFMReaderConfig())
    """The VFM reader configuration."""

    lidar_sweep_subsample_factor: int = 1
    """The subsample factor for the lidar sweeps. (1 out of every n sweeps)"""


class AV2Dataset(Dataset):
    _BLACKLIST: List[str] = [
        "75e8adad-50a6-3245-8726-5e612db3d165",
        "54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca",
        "af170aac-8465-3d7b-82c5-64147e94af7d",
        "6e106cf8-f6dd-38f6-89c8-9be7a71e7275",
    ]

    def __init__(
        self,
        config: AV2DatasetConfig,
        train_or_val: Literal["train", "val"],
        split: Literal[
            "original",
            "geo_near",
            "geo_near_050",
            "geo_near_100",
            "geo_near_500",
            "geo_far_A",
            "geo_far_B",
            "geo_far_C",
        ],
    ):
        super().__init__()
        assert config.point_cloud_range != (0, 0, 0, 0, 0, 0), "Model did not set point_cloud_range."

        if config.subset == "lidar":
            assert not config.load_image, "Cannot load images in lidar subset."
            assert not config.load_cached_vfm_features, "Cannot load VFM features in lidar subset."

        self.config = config
        self.train_or_val = train_or_val
        self.split = split

        # get all sequences for current split
        all_train = sorted([d for d in (config.data / config.subset / "train").iterdir() if d.is_dir()])
        all_val = sorted([d for d in (config.data / config.subset / "val").iterdir() if d.is_dir()])

        # read splits
        if split == "sensor_geo_near" and train_or_val == "val" and config.load_4d_forecasting:
            split_name = "val_no_og_test"
        else:
            split_name = train_or_val

        with open(AV2_SPLITS_DIR + f"{self.split}/{split_name}.txt", "r") as f:
            sequences_in_split = [line.strip() for line in f]

        # Get the sequences from the train or val folder
        sequences = [seq for seq in all_train if seq.stem in sequences_in_split]
        sequences += [seq for seq in all_val if seq.stem in sequences_in_split]
        # remove blacklisted sequences
        sequences = [seq for seq in sequences if seq.stem not in self._BLACKLIST]
        # only use specified ids if provided
        if len(config.ids_to_load) > 0:
            sequences = [seq for seq in sequences if seq.stem in config.ids_to_load]
            print(f"Using {len(sequences)} sequences from {split_name}")
        else:
            assert len(sequences) > 0, f"No sequences found in {config.data / config.subset / split_name}"

        if self.config.version == "mini":
            sequences = sequences[:10]

        # we want to roll out the sequences to individual samples
        self.dataset_idx_to_seq_and_frame: List[Tuple[str, str]] = []
        # ensure that we have enough frames to predict all the way forward during validation
        stop_frames_before = max(1, int(self.config.max_future_frame_s * AV2_LIDAR_FREQUENCY))

        # check how many previous frames we need to be able to load
        n_prevs = []

        if config.load_lidar:
            n_prevs.append(config.prev_lidar_max_time * AV2_LIDAR_FREQUENCY)
        if len(n_prevs) == 0:
            n_prevs.append(config.prev_lidar_max_time * AV2_LIDAR_FREQUENCY)

        n_prev = int(max([2] + n_prevs))

        for seq in sequences:
            lidar_files = sorted((seq / "sensors" / "lidar").iterdir())
            samples_to_select = len(lidar_files) - n_prev - stop_frames_before
            assert samples_to_select > 0, (
                f"No lidar files found in {config.data / config.subset / train_or_val / seq / 'sensors' / 'lidar'}"
            )
            self.dataset_idx_to_seq_and_frame += [
                (str(seq), str(lidar_files[i + n_prev]))
                for i in range(0, samples_to_select, config.lidar_sweep_subsample_factor)
            ]

        if len(config.frames_to_load) > 0:
            self.dataset_idx_to_seq_and_frame = [
                (seq, frame)
                for seq, frame in self.dataset_idx_to_seq_and_frame
                if Path(frame).stem in config.frames_to_load
            ]

        if config.load_ego_path_samples:
            self.ego_path_sampler: EgoPathSampler = config.eps_cfg.setup(point_cloud_range=config.point_cloud_range)

        if config.load_cached_vfm_features or config.load_cached_vfm_compression_info:
            self.cached_vfm_reader: AV2CachedVFMFeaturesReader = config.cached_vfm_cfg.setup()

        if config.load_vfm_stuff:
            self.vfm_reader: VFMReader = config.vfm_reader_cfg.setup(dataset_name="av2", data_root=config.data, point_cloud_range=config.point_cloud_range)

    @property
    def is_train(self) -> bool:
        return "train" in self.train_or_val

    @property
    def is_val(self) -> bool:
        return "val" in self.train_or_val

    @property
    def any_augmentation(self) -> bool:
        return self.config.rotation_augmentation or self.config.translation_augmentation

    @property
    def lidar_frequency(self) -> float:
        return AV2_LIDAR_FREQUENCY

    @property
    def image_frequency(self) -> float:
        return AV2_IMAGE_FREQUENCY

    @property
    def max_future_frames(self) -> int:
        return max(1, self.max_future_offset)

    @property
    def max_prev_images_frames(self) -> int:
        return int(np.ceil(self.config.prev_images_max_time * self.image_frequency))

    @property
    def max_future_images_frames(self) -> int:
        return int(np.ceil(self.config.future_images_max_time * self.image_frequency))

    @property
    def max_prev_lidar_frames(self) -> int:
        return int(np.ceil(self.config.prev_lidar_max_time * self.lidar_frequency))

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
        min_ = 1 if self.config.min_future_frame_s > 0.0 else 0
        return max(int(self.config.min_future_frame_s * self.lidar_frequency), min_)

    @property
    def max_future_offset(self) -> float:
        return int(self.config.max_future_frame_s * self.lidar_frequency)

    @property
    def min_past_offset(self) -> float:
        min_ = 1 if self.config.min_past_frame_s > 0.0 else 0
        return max(int(self.config.min_past_frame_s * self.lidar_frequency), min_)

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

    def __len__(self):
        return len(self.dataset_idx_to_seq_and_frame)

    def _get_ego_motion(self, seq_root: Path) -> EgoMotion:
        df = av2.utils.io.read_feather(
            seq_root / "city_SE3_egovehicle.feather",
        )
        quats = df[["qw", "qx", "qy", "qz"]].to_numpy()
        trans = df[["tx_m", "ty_m", "tz_m"]].to_numpy()
        rot = av2.geometry.geometry.quat_to_mat(quats)
        poses = np.zeros((len(quats), 4, 4))
        poses[:, :3, :3] = rot
        poses[:, :3, 3] = trans
        poses[:, 3, 3] = 1

        # make all poses relative first pose
        origin = poses[0].copy()
        poses = inverse_np(origin) @ poses
        poses[0] = np.eye(4)

        # interpolate speeds from poses
        d_s = trans[1:] - trans[:-1]
        timestamps = df["timestamp_ns"].to_numpy().astype(np.float64) / 1e9
        d_t = timestamps[1:] - timestamps[:-1]
        d_t[d_t < 1e-9] = 1e-9
        velocities = d_s / d_t[:, np.newaxis].repeat(3, axis=-1)
        velocities = np.vstack(
            [velocities, np.zeros((1, 3))]
        )  # assume constant velocity between final pose and proceeding

        return EgoMotion(
            poses=poses,
            timestamps=timestamps,
            velocities=velocities,  # we dont have velocities, so we interpolate from poses
            accelerations=np.zeros((len(quats), 3)),  # we dont have accelerations
            angular_rates=np.zeros((len(quats), 3)),  # we dont have angular rates
            origin_lat_lon=origin,  # This is used for extracting map data
        )

    def _rotation_augmentation(
        self, ego_motion: EgoMotion, current_timestamp: float
    ) -> Tuple[EgoMotion, np.ndarray, float, np.ndarray, np.ndarray]:
        if self.is_train and self.any_augmentation:
            rotation_range = self.config.rotation_range
            translation_range = self.config.translation_range

            if self.config.rotation_augmentation:
                angle = np.random.uniform(rotation_range[0], rotation_range[1])
                rotation_matrix = np.array(
                    [
                        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
                    ]
                )
            else:
                angle = 0
                rotation_matrix = np.eye(2)

            if self.config.translation_augmentation:
                translation = np.random.uniform(translation_range[0], translation_range[1], size=(2,))
            else:
                translation = np.zeros(2)

            transform = np.eye(4)
            transform[:2, :2] = rotation_matrix[:2, :2]
            transform[:2, 3] = translation

            # we want to transform the oxts to the current time first and then apply the transformation
            current_oxts_pose = ego_motion.get_poses(current_timestamp)
            bev_oxts_pose = current_oxts_pose.copy()
            oxts_poses = ego_motion.poses.copy()
            oxts_poses = current_oxts_pose @ transform @ inverse_np(current_oxts_pose) @ oxts_poses
            ego_motion.poses = oxts_poses
        else:
            bev_oxts_pose = ego_motion.get_poses(current_timestamp)
            transform = np.eye(4)
            angle = 0
            translation = np.zeros(2)

        return ego_motion, bev_oxts_pose, angle, translation, transform

    def _crop_pointclouds(self, point_clouds: List[np.ndarray]):
        min_range = np.array(self.config.point_cloud_range[:3])
        max_range = np.array(self.config.point_cloud_range[3:])
        for i in range(len(point_clouds)):
            in_range = np.all((min_range <= point_clouds[i][:, :3]) & (point_clouds[i][:, :3] <= max_range), axis=-1)
            point_clouds[i] = point_clouds[i][in_range]
        return point_clouds

    def _aggregate_lidar_frames(
        self, root_folder: Path, current_nstimestamp_str: str, ego_motion: EgoMotion, target_pose: np.ndarray
    ) -> Dict[str, Any]:
        lidar_nstimestamps = sorted(f.stem for f in (root_folder / "sensors" / "lidar").iterdir() if f.is_file())
        # find the index of the current lidar frame
        current_idx = lidar_nstimestamps.index(current_nstimestamp_str)
        current_nstimestamp = int(current_nstimestamp_str)
        # get the previous lidar frames
        # used indices

        n_prev_max = self.max_prev_lidar_frames
        n_prev_lidar = self.config.n_prev_lidar

        if n_prev_max <= 0 or n_prev_lidar <= 0:
            used_indices: List[int] = []
        else:
            # Generate uniformly spaced indices from max previous time to current
            indices = np.linspace(current_idx - n_prev_max, current_idx, n_prev_lidar + 1)[:-1]
            used_indices = np.floor(indices).astype(np.int32).tolist()

        n_future_max = self.max_future_lidar_frames
        n_future_lidar = self.config.n_future_lidar

        if n_future_max <= 0 or n_future_lidar <= 0:
            future_indices: List[int] = []
        else:
            future_indices = np.linspace(current_idx, current_idx + n_future_max, n_future_lidar + 1)[1:]
            future_indices = np.floor(future_indices).astype(np.int32).tolist()

        used_indices = [current_idx] + used_indices + future_indices
        used_indices = list(set(used_indices))
        used_indices.sort()

        lidar_nstimestamps = [lidar_nstimestamps[i] for i in used_indices]
        sweeps = [
            Sweep.from_feather(root_folder / "sensors" / "lidar" / f"{frame}.feather") for frame in lidar_nstimestamps
        ]
        aggregated = np.empty((0, 5), dtype=np.float32)
        for sweep in sweeps:
            comp_sweep = _ego_motion_compensate_scanwise(sweep, ego_motion, target_pose=target_pose)
            times = (comp_sweep.offset_ns + comp_sweep.timestamp_ns - current_nstimestamp) / 1e9  # seconds
            intensity = comp_sweep.intensity / AV2_MAX_REFLECTANCE
            points = np.concatenate([comp_sweep.xyz, intensity[:, None], times[:, None]], axis=1)
            aggregated = np.concatenate([aggregated, points], axis=0)

        aggregated = [aggregated.astype(np.float32)]
        # # remove points outside the range if specified,
        if self.config.point_cloud_range is not None:
            aggregated = self._crop_pointclouds(aggregated)

        return {
            "lidars": aggregated,
            "lidar_timestamps": [int(current_nstimestamp / 1e3)],  # ms
        }

    def _select_offset_frames(self, all_lidar_frames: List, frame_idx: int) -> List:
        offset_frames = []
        n_frames = len(all_lidar_frames)

        if self.load_future_offset_frames:
            start_idx = min(frame_idx + self.min_future_offset, n_frames)
            end_idx = min(frame_idx + self.max_future_offset, n_frames)
            if start_idx <= end_idx:
                lidar_frames = all_lidar_frames[start_idx : end_idx + 1]
                num_frames_to_select = min(self.config.num_future_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False
                        ).tolist()
                    else:
                        # Uniformly select frames
                        select_frame_idx = (
                            np.linspace(0, len(lidar_frames) - 1, num_frames_to_select).astype(np.int32).tolist()
                        )

                    offset_frames += [lidar_frames[i] for i in select_frame_idx]

        if self.load_past_offset_frames:
            start_idx = max(frame_idx - self.max_past_offset, 0)
            end_idx = max(frame_idx - self.min_past_offset, 0)
            if start_idx <= end_idx:
                lidar_frames = all_lidar_frames[start_idx : end_idx + 1]
                num_frames_to_select = min(self.config.num_past_frames_to_sample, len(lidar_frames))
                if num_frames_to_select > 0:
                    if self.is_train:
                        select_frame_idx: List[int] = np.random.choice(
                            len(lidar_frames), num_frames_to_select, replace=False
                        ).tolist()
                    else:
                        # Uniformly select frames
                        select_frame_idx = (
                            np.linspace(0, len(lidar_frames) - 1, num_frames_to_select).astype(np.int32).tolist()
                        )

                    offset_frames += [lidar_frames[i] for i in select_frame_idx]

        if self.config.always_add_current_frame_to_offsets:
            offset_frames.append(all_lidar_frames[frame_idx])

        return offset_frames

    def _get_offset_lidar_frames(self, root_folder: Path, current_nstimestamp_str: str) -> List[Path]:
        lidar_nstimestamps = sorted(f.stem for f in (root_folder / "sensors" / "lidar").iterdir() if f.is_file())
        # find the index of the current lidar frame
        current_idx = lidar_nstimestamps.index(current_nstimestamp_str)
        # get the offset lidar frames
        selected = self._select_offset_frames(lidar_nstimestamps, current_idx)
        selected = [root_folder / "sensors" / "lidar" / f"{name}.feather" for name in selected]
        selected = sorted(selected, key=lambda x: int(x.stem))
        return selected

    def _get_offset_pointclouds(
        self, root_folder: Path, current_nstimestamp_str: str, bev_oxts_pose: np.ndarray, ego_motion: EgoMotion
    ) -> Tuple[Dict[str, np.ndarray], List[float]]:
        # load the future lidar scans
        lidar_filepaths = self._get_offset_lidar_frames(root_folder, current_nstimestamp_str)
        lidar_sweeps = [Sweep.from_feather(lidar_file) for lidar_file in lidar_filepaths]

        # lidar_datas = self._get_lidar_datas_in_relevant_diodes(lidar_datas)
        lidar_timestamps = [int(f.stem) / 1e9 for f in lidar_filepaths]

        # slice relevant part of oxts (ego motion) and store for use on gpu
        current_nstimestamp = int(current_nstimestamp_str)  # ns to s
        current_timestamp = current_nstimestamp / 1e9  # ns to s
        sweep_eps = 0.2  # 0.2 seconds, sweep is roughly 0.1 seconds so this has margin
        max_timestamp = max(lidar_timestamps + [current_timestamp]) + sweep_eps
        min_timestamp = min(lidar_timestamps + [current_timestamp]) - sweep_eps
        lower_mask = min_timestamp < ego_motion.timestamps
        upper_mask = ego_motion.timestamps < max_timestamp
        oxts_range_mask = lower_mask & upper_mask
        sliced_oxts = EgoMotion(  # dummy for use with compensation code
            ego_motion.poses[oxts_range_mask],
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            ego_motion.timestamps[oxts_range_mask],
            None,  # type: ignore
        )

        # downsample the pointclouds if needed
        if (max_per_scan := self.config.n_max_offset_points_to_load_per_scan) >= 0:
            for i, lidar_sweep in enumerate(lidar_sweeps):
                if len(lidar_sweeps) > max_per_scan:
                    idx = np.random.permutation(len(lidar_sweeps))[:max_per_scan]
                    lidar_sweeps[i] = Sweep(
                        xyz=lidar_sweep.xyz[idx],
                        intensity=lidar_sweep.intensity[idx],
                        laser_number=lidar_sweep.laser_number[idx],
                        offset_ns=lidar_sweep.offset_ns[idx],
                        timestamp_ns=lidar_sweep.timestamp_ns,
                        ego_SE3_up_lidar=lidar_sweep.ego_SE3_up_lidar,
                        ego_SE3_down_lidar=lidar_sweep.ego_SE3_down_lidar,
                    )

        # compute in-range mask
        in_range_mask = [
            _points_in_target_bev(
                lidar_sweep,
                sliced_oxts,
                self.config.point_cloud_range,
                bev_oxts_pose,
            )
            for lidar_sweep in lidar_sweeps
        ]

        # uncompensate clouds if needed
        lidar_sweeps = [fast_uncompensate_pointwise(s, sliced_oxts) for s in lidar_sweeps]

        for sweep in lidar_sweeps:
            sweep.xyz = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz)
        lidar_extrinsics = sweep.ego_SE3_up_lidar.transform_matrix.astype(np.float64)

        # fetch these before we mess up the timestamps
        if self.config.add_missing_points:
            missing_points = self._load_missing_points(lidar_filepaths, lidar_sweeps, sliced_oxts, current_nstimestamp)

        # set everything in reference to the current frame
        for sweep in lidar_sweeps:
            t_ = (sweep.offset_ns.astype(np.int64) + sweep.timestamp_ns) - current_nstimestamp
            sweep.timestamp_ns = 0
            sweep.offset_ns = t_

        offset_pointclouds = [np.hstack((sweep.xyz, sweep.offset_ns[:, None] / 1e9)) for sweep in lidar_sweeps]
        future_pcs_in_range = np.vstack([pc[mask] for pc, mask in zip(offset_pointclouds, in_range_mask)])
        offset_pointclouds_indices = np.concatenate(
            [np.full((in_range_mask[i].sum(),), i, dtype=np.uint8) for i in range(len(in_range_mask))]
        )

        # points outside the range become missing rays
        missing_rays = [pc[~mask] for pc, mask in zip(offset_pointclouds, in_range_mask)]

        if self.config.add_missing_points:
            missing_rays += missing_points
        missing_rays = np.vstack(missing_rays).astype(np.float32)

        # get the ratio of points inside the range to be used to samples queries in occupancy sampler
        ratio_inside = len(future_pcs_in_range) / (len(future_pcs_in_range) + len(missing_rays))

        if self.config.n_max_offset_points >= 0:
            idx = np.random.permutation(len(future_pcs_in_range))[: self.config.n_max_offset_points]
            future_pcs_in_range = future_pcs_in_range[idx]
            offset_pointclouds_indices = offset_pointclouds_indices[idx]

        if self.config.n_max_missing_rays >= 0:
            idx = np.random.permutation(len(missing_rays))[: self.config.n_max_missing_rays]
            missing_rays = missing_rays[idx]

        sliced_oxts.timestamps -= current_timestamp
        sliced_oxts.poses = inverse_np(bev_oxts_pose) @ sliced_oxts.poses

        if self.config.make_batchable:
            (
                future_pcs_in_range,
                offset_pointclouds_indices,
                missing_rays,
                sliced_oxts,
            ) = self._make_batchable(
                future_pcs_in_range,
                offset_pointclouds_indices,
                missing_rays,
                sliced_oxts,
            )

        data = {
            "ego_poses": sliced_oxts.poses.astype(np.float64),
            "ego_pose_times": sliced_oxts.timestamps.astype(np.float32),
            "lidar_extrinsics": lidar_extrinsics,
            "offset_pointclouds": future_pcs_in_range.astype(np.float32),
            "offset_pointclouds_indices": offset_pointclouds_indices,
            "missing_rays": missing_rays,
            "future_points_ratio_inside": ratio_inside,
        }

        return data, lidar_timestamps

    def _make_batchable(
        self,
        future_pcs_in_range: np.ndarray,
        offset_pointclouds_indices: np.ndarray,
        missing_rays: np.ndarray,
        sliced_oxts: EgoMotion,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EgoMotion, Optional[Dict[str, np.ndarray]]]:
        # Repeat points to make it batchable
        if future_pcs_in_range.shape[0] < self.config.n_max_offset_points:
            num_additional_points = self.config.n_max_offset_points - future_pcs_in_range.shape[0]
            selected_additional_points = np.random.choice(
                range(future_pcs_in_range.shape[0]), size=num_additional_points, replace=True
            )
            future_pcs_in_range = np.concatenate(
                [future_pcs_in_range, future_pcs_in_range[selected_additional_points]], axis=0
            )
            offset_pointclouds_indices = np.concatenate(
                [offset_pointclouds_indices, offset_pointclouds_indices[selected_additional_points]], axis=0
            )

        if missing_rays.shape[0] < self.config.n_max_missing_rays:
            # Repeat missing rays to make it batchable
            num_additional_missing_rays = self.config.n_max_missing_rays - missing_rays.shape[0]
            selected_additional_missing_rays = np.random.choice(
                range(missing_rays.shape[0]), size=num_additional_missing_rays, replace=True
            )
            missing_rays = np.concatenate([missing_rays, missing_rays[selected_additional_missing_rays]], axis=0)

        if sliced_oxts.timestamps.shape[0] < self.config.n_max_poses:
            # Repeat the last pose to make it batchable
            num_additional_poses = self.config.n_max_poses - sliced_oxts.timestamps.shape[0]
            sliced_oxts.poses = np.concatenate(
                [sliced_oxts.poses, sliced_oxts.poses[-1:].repeat(num_additional_poses, axis=0)], axis=0
            )
            sliced_oxts.timestamps = np.concatenate(
                [sliced_oxts.timestamps, sliced_oxts.timestamps[-1:].repeat(num_additional_poses, axis=0)], axis=0
            )

        return (
            future_pcs_in_range,
            offset_pointclouds_indices,
            missing_rays,
            sliced_oxts,
        )

    def _load_missing_points(
        self, lidar_filepaths: List[Path], lidar_sweeps: List[Sweep], sliced_oxts: EgoMotion, current_nstimestamp: int
    ) -> List[np.ndarray]:
        split_names = []
        for f in lidar_filepaths:
            if "train" in f.parts:
                split_names.append("train")
            elif "val" in f.parts:
                split_names.append("val")
            elif "test" in f.parts:
                split_names.append("test")
        mps_dir = [str(f).split("/sensors")[0].split("/")[-1] for f in lidar_filepaths]
        mps_fnames = [f"{s}/{d}/{f.stem}_missing.npy" for s, d, f in zip(split_names, mps_dir, lidar_filepaths)]
        mps_fpaths = [self.config.av2_cached_missing_points_dir / f for f in mps_fnames]

        existing = [f.exists() for f in mps_fpaths]

        xyzts = [np.load(f) for f, e in zip(mps_fpaths, existing) if e]
        xyzts = [np.concatenate((pc[:, :3], pc[:, 4:5]), axis=1) for pc in xyzts]
        swps = [s for s, e in zip(lidar_sweeps, existing) if e]

        # Create sweeps of missing points
        missing_sweeps = []
        for xyzt, swp in zip(xyzts, swps):
            points = xyzt[:, :3]
            offsets = xyzt[:, 3]
            z_mask = points[:, 2] > 0.0
            points = points[z_mask]
            offsets_ns = (offsets[z_mask] * 1e9).astype(np.int64)
            mps = Sweep(
                xyz=points,
                intensity=np.zeros((len(points),), dtype=swp.intensity.dtype),
                laser_number=np.zeros((len(points),), dtype=np.uint8),
                offset_ns=offsets_ns,
                timestamp_ns=swp.timestamp_ns,
                ego_SE3_up_lidar=swp.ego_SE3_up_lidar,
                ego_SE3_down_lidar=swp.ego_SE3_down_lidar,
            )
            missing_sweeps.append(mps)

        # uncompensate clouds if needed
        missing_sweeps = [fast_uncompensate_pointwise(s, sliced_oxts) for s in missing_sweeps]

        for sweep in missing_sweeps:
            sweep.xyz = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz)

        # set everything in reference to the current frame
        for sweep in missing_sweeps:
            t_ = (sweep.offset_ns.astype(np.int64) + sweep.timestamp_ns) - current_nstimestamp
            sweep.timestamp_ns = 0
            sweep.offset_ns = t_

        missing_points_cloud = [np.hstack((sweep.xyz, sweep.offset_ns[:, None] / 1e9)) for sweep in missing_sweeps]

        # Subsample points randomly
        if self.config.missing_points_downsample_factor > 0.0:
            subsampled_missing_points_cloud = []
            for pc in missing_points_cloud:
                selected_indeces = np.random.choice(
                    range(pc.shape[0]),
                    size=int(pc.shape[0] * self.config.missing_points_downsample_factor),
                    replace=False,
                )
                subsampled_missing_points_cloud.append(pc[selected_indeces])
            missing_points_cloud = subsampled_missing_points_cloud

        return missing_points_cloud  # , filtered_missing_points_cloud

    def __getitem__(self, idx: int):
        root_folder, lidar_file = self.dataset_idx_to_seq_and_frame[idx]
        current_nstimestamp_str = lidar_file.split("/")[-1].split(".")[0]
        current_timestamp = float(current_nstimestamp_str) / 1e9  # ns to s
        sequence_id = root_folder.split("/")[-1]

        data: Dict[str, Any] = {
            "id": sequence_id,
            "frame_idx": current_nstimestamp_str,
        }

        root_folder = Path(root_folder)
        ego_motion = self._get_ego_motion(root_folder)

        ego_motion, bev_oxts_pose, aug_angle, aug_translation, _ = self._rotation_augmentation(
            ego_motion, current_timestamp
        )

        data["augmentation_angle"] = aug_angle
        data["augmentation_translation"] = aug_translation

        if self.config.load_lidar:
            # get the lidar frames
            if self.config.aggregate_lidar:
                data.update(
                    self._aggregate_lidar_frames(
                        root_folder, current_nstimestamp_str, ego_motion, target_pose=bev_oxts_pose
                    )
                )
            else:
                raise NotImplementedError

        future_lidar_timestamps = None
        if self.config.load_offset_pointclouds:
            future_lidar_data, future_lidar_timestamps = self._get_offset_pointclouds(
                root_folder, current_nstimestamp_str, bev_oxts_pose, ego_motion
            )
            data.update(future_lidar_data)

        if self.config.load_cached_vfm_features or self.config.load_cached_vfm_compression_info:
            cached_vfm_data = self.cached_vfm_reader.read(
                root_folder,
                current_nstimestamp_str,
                lidar_timestamps=future_lidar_timestamps,
            )
            data.update(cached_vfm_data)
            data.update(self.cached_vfm_reader.get_compression_info())

        if self.config.load_ego_path_samples:
            ego_path_samples = self.ego_path_sampler.sample(ego_motion, current_timestamp, bev_oxts_pose)
            data.update(ego_path_samples)

        if self.config.load_vfm_stuff:
            vfm_data = self.vfm_reader.read(
                root_folder=root_folder,
                current_timestamp=current_timestamp,
                lidar_timestamps=future_lidar_timestamps,
                ego_motion=ego_motion,
                bev_oxts_pose=bev_oxts_pose,
            )
            data.update(vfm_data)

        def _to_tensor(data: Union[np.ndarray, list, str]) -> Union[str, torch.Tensor]:
            if isinstance(data, (str, int, float, dict)):
                return data
            data = np.stack(data, axis=0) if isinstance(data, list) else data
            return torch.from_numpy(data)

        return {k: _to_tensor(v) for k, v in data.items()}


def _ego_motion_compensate_scanwise(
    sweep: Sweep,
    ego_motion: EgoMotion,
    target_timestamp: Optional[float] = None,
    target_pose: Optional[np.ndarray] = None,
) -> Sweep:
    assert target_timestamp is not None or target_pose is not None, (
        "Either target_timestamp or target_pose must be provided."
    )
    xyz = sweep.xyz.copy()
    sweep_pose = ego_motion.get_poses(sweep.timestamp_ns / 1e9)

    if target_pose is None:
        assert target_timestamp is not None, "Target timestamp must be provided if target_pose is not."
        target_pose = ego_motion.get_poses(target_timestamp)

    odometry = inverse_np(target_pose) @ sweep_pose

    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    sweep.xyz = xyz
    return sweep


def _points_in_target_bev(
    lidar_sweep: Sweep,
    ego_motion: EgoMotion,
    point_cloud_range: PointCloudRange,
    bev_oxts_pose: np.ndarray,
) -> np.ndarray:
    xyz = lidar_sweep.xyz.copy()
    # motion compensate scanwise to the target timestamp
    source_pose = ego_motion.get_poses(lidar_sweep.timestamp_ns / 1e9)
    target_pose = bev_oxts_pose
    odometry = inverse_np(target_pose) @ source_pose
    rotations = odometry[..., :3, :3]
    translations = odometry[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)

    # check bev range
    min_range = np.array(point_cloud_range[:3])
    max_range = np.array(point_cloud_range[3:])
    in_range = np.all((min_range <= xyz) & (xyz <= max_range), axis=-1)
    # restore lidar_data (we don't want to overwrite)

    # remove points that are within the ego-vehicle
    mask_x = np.logical_and(xyz[..., 0] > -1.75, xyz[..., 0] < 3.75)
    mask_y = np.logical_and(xyz[..., 1] > -1.25, xyz[..., 1] < 1.25)
    not_in_vehicle = ~np.logical_and(mask_x, mask_y)

    in_bev = np.logical_and(in_range, not_in_vehicle)

    return in_bev


def fast_uncompensate_pointwise(sweep: Sweep, ego_motion: EgoMotion) -> Sweep:
    """Uncompensate an already scan-compensated point cloud. (inverse of fast_compensate_pointwise)

    Given a point cloud that has been compensated to the scan timestamp, un-compensate each point to its original
    pose given the delta-time encoded in each point (x,y,z,r,t)
    """
    # get source (scan) and target (pointwise) poses and compute relative transform (odometry)
    point_pose = ego_motion.get_poses(sweep.timestamp_ns / 1e9)
    target_times = (sweep.timestamp_ns + sweep.offset_ns) / 1e9  # should this be + or -?
    target_poses = fast_ego_motion_poses(ego_motion, target_times)
    odometry = inverse_np(target_poses) @ point_pose
    # Project ego -> new-ego
    sweep = _transform_sweep(sweep, odometry)
    return sweep


def _transform_sweep(sweep: Sweep, transform: np.ndarray) -> Sweep:
    xyz = sweep.xyz.copy()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    xyz = (xyz[..., None, :] @ rotations.swapaxes(-2, -1) + translations[..., None, :]).squeeze(-2)
    sweep.xyz = xyz
    return sweep


if __name__ == "__main__":
    AV2_SPLITS_DIR = "/gasp/gasp/data/splits/argoverse2/"
    config = AV2DatasetConfig()
    config.load_dufo = True
    config.dufo_cfg.save_visualizations = False
    config.dufo_cfg.force_cache = True
    config.data = Path("/datasets/argoverse")
    config.version = "full"
    config.point_cloud_range = (-100, -100, -100, 100, 100, 100)
    dataset = config.setup(train_or_val="train", split="sensor_original")
    dataset = config.setup(train_or_val="val", split="sensor_original")

    print(len(dataset))
    print(dataset[0])
