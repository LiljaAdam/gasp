from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import cv2
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from torch import Tensor, clamp as torch_clamp, load as torch_load
from zod.constants import Camera
from zod.data_classes.frame import ZodFrame
from zod.data_classes.sequence import ZodSequence

import gasp.config.base_config as cfg
from gasp.common.types import Zample
from gasp.data.components.pose_utils import inverse_np

QUERY_DTYPE = np.float32
FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.int32
CENTER_FRAME = 10
EPS = 1e-3

RESOLUTION = {
    "full": "1036x3836",
    "half": "518x1918",
}

AV2_CENTER_CROP_DINO: Tuple[int, int, int, int] = (286, 5, 212, 5)  # (top, right, bottom, left)
AV2_OTHERS_CROP_DINO: Tuple[int, int, int, int] = (2, 5, 2, 5)  # (top, right, bottom, left)

AV2_CENTER_CROP_CLIP: Tuple[int, int, int, int] = (287, 6, 213, 6)  # (top, right, bottom, left)
AV2_OTHERS_CROP_CLIP: Tuple[int, int, int, int] = (3, 4, 3, 4)  # (top, right, bottom, left)

AV2_SCALE = 2.0

NUSCENES_CROP: Tuple[int, int, int, int] = (4, 2, 0, 2)  # (top, right, bottom, left)
NUSCENES_SCALE = 2.0

MAXIMAL_CAMERA_LIDAR_TIMEDIFF = 0.1  # seconds


@dataclass
class CachedVFMFeaturesReaderConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: CachedVFMFeaturesReader)

    extracted_features_crop: Tuple[int, int, int, int] = (382, 6, 750, 6)  # (top, right, bottom, left)
    """The crop that was used when extracting the features."""  # TODO: We probably should do this in a better way. Will be buggered if we change the crop in the future.

    extracted_features_dim: int = 16
    """Feature dimension to read (depends on preprocessed dino files)."""
    _load_feature_dim: int = 16
    """Feature dimension that should be loaded to gpu. !! Note: This will be overwritten by the model config. !! """

    extracted_features_path: Path = Path("/datasets/zod_dino_features")
    """Path to cached dino features."""

    extracted_features_version: str = "distilled_base"
    """Name of the features to extract."""

    extracted_features_resolution: str = "half"  # full: 1036x3836 or half: 518x1918
    """Which image resolution to use."""

    extracted_features_stride: int = 7
    """Which ViT stride to use."""

    num_sampled_feature_maps: int = 2
    """How many feature maps to load."""

    sample_center_camera_factor: float = 2.0
    """How much more to sample the center camera features."""

    align_with_future_lidar_times: bool = True
    """If we want to use future lidar times as base for loaded future dino features."""

    rectify: bool = False
    """Whether to rectify the dino featuremaps. """

    def get_downscale_factor(self):
        scale = self.extracted_features_stride
        if self.extracted_features_resolution == "half":
            scale *= 2
        return scale


class CachedVFMFeaturesReader:
    def __init__(
        self,
        config: CachedVFMFeaturesReaderConfig,
    ):
        self.config = config

    def _get_dino_filepaths(self, zample: Zample):
        if self.config.extracted_features_dim > 0:
            featname = f"_compressed_{self.config.extracted_features_dim}.npy"
        else:
            featname = ".npy"

        feature_filenames = [
            cf.filepath.split("/")[-1].replace(".jpg", featname) for cf in zample.info.get_camera_frames()
        ]

        if isinstance(zample, ZodFrame):
            dataset_name = "single_frames"
        elif isinstance(zample, ZodSequence):
            if "drives" in zample.info.calibration_path:
                dataset_name = "drives"
            else:
                dataset_name = "sequences"
        else:
            raise ValueError("Unknown dataset type")

        features_dir = (
            self.config.extracted_features_path
            / Path(dataset_name)
            / Path(self.config.extracted_features_version)
            / Path(
                RESOLUTION[self.config.extracted_features_resolution]
                + "_s"
                + str(self.config.extracted_features_stride)
            )
        )
        feature_filepaths = [features_dir / ff for ff in feature_filenames]

        return feature_filepaths

    def _get_dino_features(self, camera_indices: List[int], zample: Zample) -> np.ndarray:
        feature_filepaths = self._get_dino_filepaths(zample)
        features = []
        for ci in camera_indices:
            filepath = str(feature_filepaths[ci])
            f = np.load(filepath)[..., : self.config._load_feature_dim]
            features.append(f)
        return np.stack(features, axis=0)

    def _rectify_features(
        self, features: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify a batch of features (B,H,W,C) using the given intrinsics and distortion."""
        distortion = np.array([distortion[0], distortion[1], distortion[2], distortion[3]])
        intrinsics = intrinsics[:3, :3]
        shape = (features.shape[2], features.shape[1])  # first dim is batch size
        new_intrinsics = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsics, distortion, shape, np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsics, distortion, np.eye(3), new_intrinsics, shape, cv2.CV_32FC1
        )
        features = np.stack([cv2.remap(feats, map1, map2, interpolation=cv2.INTER_LINEAR) for feats in features])
        return features, new_intrinsics

    def read(
        self,
        zample: Zample,
        current_timestamp: float,
        lidar_timestamps: Optional[List[float]],
    ) -> Dict[str, np.ndarray]:
        camera_frames = zample.info.get_camera_frames(camera=Camera.FRONT)
        camera_timestamps = np.array([frame.time.timestamp() for frame in camera_frames])
        camera_indices = np.arange(len(camera_frames))
        if self.config.align_with_future_lidar_times:
            assert lidar_timestamps is not None, "Need to provide lidar timestamps if aligning with future lidar times"
            lid_cam_timediff = np.abs(np.array(lidar_timestamps)[:, None] - camera_timestamps[None, :])
            nearby_camera = np.any(lid_cam_timediff < MAXIMAL_CAMERA_LIDAR_TIMEDIFF, axis=0)
            camera_indices_ = [ind for ind, valid in zip(camera_indices, nearby_camera) if valid]
            if len(camera_indices_) > 0:
                camera_indices = camera_indices_
        else:
            min_useful_timestamp = min(lidar_timestamps) - MAXIMAL_CAMERA_LIDAR_TIMEDIFF
            max_useful_timestamp = max(lidar_timestamps) + MAXIMAL_CAMERA_LIDAR_TIMEDIFF
            useful_mask = (min_useful_timestamp <= camera_timestamps) & (camera_timestamps <= max_useful_timestamp)
            camera_indices = camera_indices[useful_mask]
            camera_timestamps = camera_timestamps[useful_mask]

        if len(camera_indices) == 0:
            raise ValueError("No valid camera frames found")
        elif len(camera_indices) < self.config.num_sampled_feature_maps:
            cam_indices = (
                camera_indices
                + np.random.choice(
                    camera_indices, self.config.num_sampled_feature_maps - len(camera_indices), replace=True
                ).tolist()
            )
        else:
            cam_indices = np.random.choice(camera_indices, self.config.num_sampled_feature_maps, replace=False).tolist()

        dino_features = self._get_dino_features(cam_indices, zample)  # N, H, W, C

        extrinsics = zample.calibration.cameras[Camera.FRONT].extrinsics.transform
        intrinsics = zample.calibration.cameras[Camera.FRONT].intrinsics.copy()
        # Adjust intrinsics based on crop
        intrinsics[0, 2] -= self.config.extracted_features_crop[3]  # subtract left crop
        intrinsics[1, 2] -= self.config.extracted_features_crop[0]  # subtract top crop
        intrinsics[:2, :3] /= self.config.get_downscale_factor()

        distortion = zample.calibration.cameras[Camera.FRONT].distortion
        if self.config.rectify:
            dino_features, intrinsics = self._rectify_features(dino_features, intrinsics, distortion)
            distortion = np.zeros_like(distortion)

        dino_timestamps = np.array([camera_frames[i].time.timestamp() - current_timestamp for i in cam_indices])

        return {
            "vfm_timestamps": dino_timestamps.astype(np.float32),  # N_dino_feats
            "vfm_features": dino_features.astype(np.float32),  # N_dino_feats x h x w x c
            "vfm_intrinsics": intrinsics.astype(np.float32),  # 3 x 3
            "vfm_distortion": distortion.astype(np.float32),  # 4
            "vfm_extrinsics": extrinsics.astype(np.float64),  # 4 x 4
            "vfm_extrinsics_inverse": inverse_np(extrinsics).astype(np.float64),  # 4 x 4
        }

    def get_compression_info(self):
        compression_min = np.array(
            [
                -12.6621,
                -21.2744,
                -13.3906,
                -20.5065,
                -19.4904,
                -13.6711,
                -18.1193,
                -17.6182,
                -18.5727,
                -15.6117,
                -13.1612,
                -11.0931,
                -11.0776,
                -12.1945,
                -10.0991,
                -12.1445,
                -11.4880,
                -8.3002,
                -16.4026,
                -14.0660,
                -13.3529,
                -10.7962,
                -9.7955,
                -10.7200,
                -9.0564,
                -9.0463,
                -9.3013,
                -9.6953,
                -16.4178,
                -10.5235,
                -11.8855,
                -9.9040,
            ]
        )
        compression_max = np.array(
            [
                28.8876,
                16.0384,
                21.9716,
                13.9501,
                10.6534,
                18.5596,
                13.2123,
                16.1770,
                15.7204,
                9.7507,
                12.1231,
                14.5465,
                13.9269,
                10.2081,
                13.8290,
                11.8317,
                10.5687,
                14.0697,
                9.7659,
                11.0826,
                10.7266,
                11.2101,
                11.0520,
                8.9164,
                9.5093,
                13.6300,
                11.1215,
                8.9330,
                16.2625,
                13.4111,
                10.3097,
                10.3123,
            ]
        )

        return {
            "dino_compression_min": compression_min[: self.config._load_feature_dim],
            "dino_compression_max": compression_max[: self.config._load_feature_dim],
        }


@dataclass
class AV2CachedVFMFeaturesReaderConfig(CachedVFMFeaturesReaderConfig):
    _target: Type = field(default_factory=lambda: AV2CachedVFMFeaturesReader)

    extracted_features_path: Path = Path("/datasets/av2_dino_features")
    """Path to cached dino features."""
    extracted_features_version: str = "base"
    """Name of the features to extract."""

    resolution_name_dino: str = "770x1022"
    resolution_name_clip: str = "768x1024"

    cameras: Tuple[str, ...] = (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    )
    """From what cameras to load features"""

    @classmethod
    def from_dino_feature_reader_config(cls, config: CachedVFMFeaturesReaderConfig):
        return cls(
            extracted_features_crop=config.extracted_features_crop,
            extracted_features_dim=config.extracted_features_dim,
            extracted_features_path=config.extracted_features_path,
            # extracted_features_version=config.extracted_features_version,
            extracted_features_resolution=config.extracted_features_resolution,
            extracted_features_stride=config.extracted_features_stride,
            num_sampled_feature_maps=config.num_sampled_feature_maps,
            align_with_future_lidar_times=config.align_with_future_lidar_times,
            rectify=config.rectify,
            _load_feature_dim=config._load_feature_dim,
        )

    @property
    def is_reading_dino_features(self):
        return "dino" in self.extracted_features_path.name

    @property
    def is_reading_clip_features(self):
        return "clip" in self.extracted_features_path.name

    @property
    def resolution_name(self):
        if self.is_reading_dino_features:
            return self.resolution_name_dino
        elif self.is_reading_clip_features:
            return self.resolution_name_clip
        else:
            raise ValueError(f"Unknown resolution name: {self.extracted_features_path}")


class AV2CachedVFMFeaturesReader:
    def __init__(
        self,
        config: AV2CachedVFMFeaturesReaderConfig,
    ):
        self.config = config

        compression_info = get_compression_info(config)

        if compression_info is None:
            print("Warning: Compression info not found. Hope you won't need it.")
        else:
            self.compression_min = compression_info["compression_min"][: config._load_feature_dim]
            self.compression_max = compression_info["compression_max"][: config._load_feature_dim]

            self.compression_to_rgb_matrix = compression_info[f"compression_matrix_{config._load_feature_dim}_to_rgb"]
            self.compression_to_rgb_min = compression_info[f"compression_min_{config._load_feature_dim}_to_rgb"]
            self.compression_to_rgb_max = compression_info[f"compression_max_{config._load_feature_dim}_to_rgb"]

        if not (self.is_reading_dino_features or self.is_reading_clip_features):
            raise ValueError(f"Unknown feature version: {self.config.extracted_features_version}")

        sample_probability_per_cam = {cam: 1.0 / len(self.config.cameras) for cam in self.config.cameras}
        if "ring_front_center" in self.config.cameras:
            sample_probability_per_cam["ring_front_center"] *= self.config.sample_center_camera_factor
        if "CAM_FRONT" in self.config.cameras:
            sample_probability_per_cam["CAM_FRONT"] *= self.config.sample_center_camera_factor
        if "CAM_BACK" in self.config.cameras:
            sample_probability_per_cam["CAM_BACK"] *= self.config.sample_center_camera_factor
        # normalize the probabilities
        sample_probability_per_cam = np.array(list(sample_probability_per_cam.values()))
        sample_probability_per_cam = sample_probability_per_cam / np.sum(sample_probability_per_cam)
        self.sample_probability_per_cam = sample_probability_per_cam

    @property
    def is_reading_dino_features(self):
        return self.config.is_reading_dino_features

    @property
    def is_reading_clip_features(self):
        return self.config.is_reading_clip_features

    @property
    def resolution_name(self):
        return self.config.resolution_name

    def _crop_size_for_cam(self, cam_name: str) -> Tuple[int, int, int, int]:
        if self.is_reading_dino_features:
            return AV2_CENTER_CROP_DINO if cam_name == "ring_front_center" else AV2_OTHERS_CROP_DINO
        elif self.is_reading_clip_features:
            return AV2_CENTER_CROP_CLIP if cam_name == "ring_front_center" else AV2_OTHERS_CROP_CLIP
        else:
            raise ValueError(f"Unknown feature version: {self.config.extracted_features_version}")

    def normalize(self, features_in: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        if isinstance(features_in, np.ndarray):
            features = features_in.copy()
            features = (features - self.compression_min) / (self.compression_max - self.compression_min)
        else:
            features = features_in.clone()
            features = (features - self.compression_min.to(features.device)) / (
                self.compression_max.to(features.device) - self.compression_min.to(features.device)
            )
        return features

    def denormalize(self, features_in: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        if isinstance(features_in, np.ndarray):
            features = features_in.copy()
            features = features * (self.compression_max - self.compression_min) + self.compression_min
        else:
            features = features_in.clone()
            features = features * (
                self.compression_max.to(features.device) - self.compression_min.to(features.device)
            ) + self.compression_min.to(features.device)
        return features

    def compress_to_rgb(self, features_in: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        if isinstance(features_in, np.ndarray):
            features = features_in.copy()
            rgb_mat = self.compression_to_rgb_matrix
            rgb_min = self.compression_to_rgb_min
            rgb_max = self.compression_to_rgb_max
            clamp = np.clip
        else:
            features = features_in.clone()
            rgb_mat = self.compression_to_rgb_matrix.to(features.device)
            rgb_min = self.compression_to_rgb_min.to(features.device)
            rgb_max = self.compression_to_rgb_max.to(features.device)
            clamp = torch_clamp

        features = features @ rgb_mat
        features = (features - rgb_min) / (rgb_max - rgb_min)

        features = clamp(features, 0, 1)

        return features

    def read(
        self,
        root_folder: Path,
        current_nstimestamp_str: str,
        lidar_timestamps: Optional[List[float]],
    ) -> Dict[str, np.ndarray]:
        # Note that we are only sampling one camera per sample. This simplifies much as this is
        # sample one of the cameras to use.
        cam_name = np.random.choice(self.config.cameras, p=self.sample_probability_per_cam)
        cam_files: List[Path] = sorted((root_folder / "sensors" / "cameras" / cam_name).glob("*.jpg"))
        cam_times = [int(f.stem) for f in cam_files]
        current_nstimestamp = int(current_nstimestamp_str)

        min_useful_timestamp = min(lidar_timestamps) - MAXIMAL_CAMERA_LIDAR_TIMEDIFF
        max_useful_timestamp = max(lidar_timestamps) + MAXIMAL_CAMERA_LIDAR_TIMEDIFF

        # use only the relevant frames
        filtered = [
            (f, t) for f, t in zip(cam_files, cam_times) if min_useful_timestamp <= t / 1e9 <= max_useful_timestamp
        ]
        if filtered:
            cam_files, cam_times = zip(*filtered)
            cam_files = list(cam_files)
            cam_times = list(cam_times)
        else:
            cam_files, cam_times = [], []
            print("We really should not be here")

        if self.config.align_with_future_lidar_times:
            assert lidar_timestamps is not None, "Need to provide lidar timestamps if aligning with future lidar times"
            lt = np.array(lidar_timestamps)
            cam_files, cam_times = zip(
                *[
                    (f, t)
                    for f, t in zip(cam_files, cam_times)
                    if np.any(np.abs(lt - t / 1e9) < MAXIMAL_CAMERA_LIDAR_TIMEDIFF)
                ]
            )

        if len(cam_files) == 0:
            raise ValueError("No valid camera frames found")
        elif len(cam_files) < self.config.num_sampled_feature_maps:
            cam_files = (
                cam_files
                + np.random.choice(
                    np.array(cam_files), self.config.num_sampled_feature_maps - len(cam_files), replace=True
                ).tolist()
            )
        else:
            cam_files = np.random.choice(
                np.array(cam_files), self.config.num_sampled_feature_maps, replace=False
            ).tolist()

        dino_features = []
        dino_timestamps = []
        for cam_file in cam_files:
            fn = (
                self.config.extracted_features_path
                / self.config.extracted_features_version
                / self.resolution_name
                / "compressed_features"
                / f"{root_folder.name}_{cam_file.stem}_{cam_name}_compressed_{self.config.extracted_features_dim}.npy"
            )
            if not fn.exists():
                print(f"Warning: {fn} does not exist")
                continue
            dino_features.append(np.load(fn)[..., : self.config._load_feature_dim])
            dino_timestamps.append(int(cam_file.stem) - current_nstimestamp)

        if len(dino_features) == 0:
            raise ValueError("No valid VLM features found")

        dino_features = np.stack(dino_features, axis=0)
        dino_timestamps = np.array(dino_timestamps).astype(np.float32) / 1e9
        distortion = np.zeros(4)

        pinhole_cam = PinholeCamera.from_feather(root_folder, cam_name)

        intrinsics = pinhole_cam.intrinsics.K.copy()
        crop_size = self._crop_size_for_cam(cam_name)
        # Adjust intrinsics based on crop
        intrinsics[0, 2] -= crop_size[3]  # subtract left crop
        intrinsics[1, 2] -= crop_size[0]  # subtract top crop
        intrinsics[:2, :3] /= self.config.extracted_features_stride * AV2_SCALE

        extrinsics = inverse_np(pinhole_cam.extrinsics.copy())

        return {
            "vfm_timestamps": dino_timestamps,  # N_dino_feats
            "vfm_features": dino_features.astype(np.float32),  # N_dino_feats x h x w x c
            "vfm_intrinsics": intrinsics.astype(np.float32),  # 3 x 3
            "vfm_distortion": distortion.astype(np.float32),  # 4
            "vfm_extrinsics": extrinsics.astype(np.float64),  # 4 x 4
            "vfm_extrinsics_inverse": inverse_np(extrinsics).astype(np.float64),  # 4 x 4
        }

    def get_compression_info(self):
        return {
            "vfm_compression_min": self.compression_min.numpy(),
            "vfm_compression_max": self.compression_max.numpy(),
        }


def get_compression_info(config: CachedVFMFeaturesReaderConfig):
    path_ = (
        config.extracted_features_path
        / config.extracted_features_version
        / config.resolution_name
        / "compression_info.pt"
    )

    if not path_.exists():
        return None

    compression_info = torch_load(
        path_,
        weights_only=True,
    )

    return compression_info

