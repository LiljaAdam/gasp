from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import torch
from av2.geometry.camera.pinhole_camera import PinholeCamera
from PIL import Image
from torch import load as torch_load
from zod.constants import Camera
from zod.data_classes.ego_motion import EgoMotion

import gasp.config.base_config as cfg
from gasp.common.types import PointCloudRange, Zample
from gasp.data.components.pose_utils import inverse_np
from gasp.data.datasets.utils import (
    crop_images,
    crop_intrinsics,
    resize_images_with_scale,
    resize_intrinsics_with_scale,
)

DELETE_STUFF = True

QUERY_DTYPE = np.float32
FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.int32
CENTER_FRAME = 10
EPS = 1e-3

RESOLUTION = {
    "full": "1036x3836",
    "half": "518x1918",
}

RGB_MEAN = (123.675, 116.28, 103.53)  # (0.485, 0.456, 0.406)
RGB_STD = (58.395, 57.12, 57.375)  # (0.229, 0.224, 0.225)

ZOD_CROP: Tuple[int, int, int, int] = (382, 6, 750, 6)  # (top, right, bottom, left)
ZOD_SCALE = 2.0

AV2_CENTER_CROP: Tuple[int, int, int, int] = (314, 0, 212, 0)  # (top, right, bottom, left)
AV2_OTHERS_CROP: Tuple[int, int, int, int] = (12, 0, 2, 0)  # (top, right, bottom, left)
AV2_SCALE = 2.0


MAXIMAL_CAMERA_LIDAR_TIMEDIFF = 0.15  # seconds

DATASET_TO_CAMERAS = {
    "zod": (Camera.FRONT,),
    "av2": (
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
    ),
}


@dataclass
class VFMReaderConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: VFMReader)

    feature_extractor_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    """The model to use for feature extraction."""

    downsample_factor: float = 1.0
    """Downsample factor to use before cropping."""

    image_final_size: Tuple[int, int] = (-224, -224)
    """Final size of the images. Set to negative to not crop."""

    extracted_features_downsample_factor: int = 16
    """Which downsample factor to use."""

    sample_center_camera_factor: float = 2.0
    """How much more to sample the center camera features."""

    feature_reduction_root: Path = Path("/outputs/feature_reductions")
    """Path to cached feature reductions."""

    read_rgb: bool = True
    """Whether to read the rgb."""

    _feature_layers: Tuple[int, ...] = (-1,)
    """The layers to use for feature extraction. Expected to be set by the model config."""

    _feature_dim: int = -1
    """The dimension of the features. Expected to be set by the model config."""

    @property
    def model_name(self) -> str:
        return self.feature_extractor_model.split("/")[-1].replace("-", "_")

    @property
    def reduction_info_path(self) -> Path:
        return self.feature_reduction_root / f"{self.model_name}_features" / "reduction_info.pt"


def get_sample_probability_per_camera(cameras: Tuple[str, ...], sample_center_camera_factor: float):
    sample_probability_per_cam = {cam: 1.0 / len(cameras) for cam in cameras}
    if "ring_front_center" in cameras:
        sample_probability_per_cam["ring_front_center"] *= sample_center_camera_factor
    if "CAM_FRONT" in cameras:
        sample_probability_per_cam["CAM_FRONT"] *= sample_center_camera_factor
    if "CAM_BACK" in cameras:
        sample_probability_per_cam["CAM_BACK"] *= sample_center_camera_factor
    # normalize the probabilities
    sample_probability_per_cam = np.array(list(sample_probability_per_cam.values()))
    sample_probability_per_cam = sample_probability_per_cam / np.sum(sample_probability_per_cam)
    return sample_probability_per_cam


def get_crop_per_camera(dataset_name: str) -> Dict[str, Tuple[int, int, int, int]]:
    if dataset_name == "zod":
        return {Camera.FRONT: ZOD_CROP}
    elif dataset_name == "av2":
        return {
            "ring_front_center": AV2_CENTER_CROP,
            "ring_front_left": AV2_OTHERS_CROP,
            "ring_front_right": AV2_OTHERS_CROP,
            "ring_rear_left": AV2_OTHERS_CROP,
            "ring_rear_right": AV2_OTHERS_CROP,
            "ring_side_left": AV2_OTHERS_CROP,
            "ring_side_right": AV2_OTHERS_CROP,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def crop_images_and_intrinsics(
    images: np.ndarray, calibrations: np.ndarray, crop_per_camera: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    images = crop_images(images, crop_per_camera)
    intrinsics = crop_intrinsics(calibrations["intrinsics"], crop_per_camera)
    calibrations["intrinsics"] = intrinsics
    return images, calibrations


def downsample_images_and_intrinsics(
    images: np.ndarray, calibrations: np.ndarray, downsample_factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    images = resize_images_with_scale(images, 1 / float(downsample_factor))
    intrinsics = resize_intrinsics_with_scale(calibrations["intrinsics"], 1 / float(downsample_factor))
    calibrations["intrinsics"] = intrinsics
    return images, calibrations


def random_crop_images_and_intrinsics(
    images: np.ndarray, calibrations: np.ndarray, image_final_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if image_final_size[0] < 0 or image_final_size[1] < 0:
        return images, calibrations
    h, w = images.shape[-2:]
    h_crop, w_crop = image_final_size
    h_start = np.random.randint(0, h - h_crop)
    w_start = np.random.randint(0, w - w_crop)
    h_end = h - (h_start + h_crop)
    w_end = w - (w_start + w_crop)
    crop = (h_start, w_end, h_end, w_start)
    images, calibrations = crop_images_and_intrinsics(images, calibrations, crop)
    return images, calibrations


def get_relative_timestamps(timestamps: np.ndarray, current_timestamp: float) -> np.ndarray:
    """Get the relative timestamps to the current timestamp."""
    return timestamps - float(current_timestamp)


def get_intrinsics_for_feature_extraction(intrinsics: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Get the intrinsics for feature extraction."""
    intrinsics = resize_intrinsics_with_scale(intrinsics, 1 / float(downsample_factor))
    return intrinsics


def get_downsample_factor(dataset_name: str, downsample_factor: float) -> float:
    if dataset_name == "av2":
        return float(downsample_factor * AV2_SCALE)
    elif dataset_name == "zod":
        return float(downsample_factor * ZOD_SCALE)
    else:
        return float(downsample_factor)


class VFMReader:
    def __init__(
        self,
        config: VFMReaderConfig,
        dataset_name: str,
        data_root: Path,
        point_cloud_range: PointCloudRange,
    ):
        self.config = config
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.cameras = DATASET_TO_CAMERAS[dataset_name]
        self.sample_probability_per_cam = get_sample_probability_per_camera(
            self.cameras, self.config.sample_center_camera_factor
        )
        self.crop_per_camera = get_crop_per_camera(dataset_name)

        if self.config._feature_layers[0] > 0:
            reduction_info = torch_load(self.config.reduction_info_path, weights_only=False)
            reduction_means = np.stack(
                [
                    reduction_info[f"layer_{layer_idx}"][f"dim_{self.config._feature_dim}"]["pca_mean"]
                    for layer_idx in self.config._feature_layers
                ]
            )
            reduction_stds = np.stack(
                [
                    reduction_info[f"layer_{layer_idx}"][f"dim_{self.config._feature_dim}"]["pca_std"]
                    for layer_idx in self.config._feature_layers
                ]
            )

            reduction_means = reduction_means.transpose(1, 0)
            reduction_stds = reduction_stds.transpose(1, 0)

            reduction_means = reduction_means.reshape(-1)
            reduction_stds = reduction_stds.reshape(-1)

            self.vfm_means = reduction_means.astype(np.float32)
            self.vfm_stds = reduction_stds.astype(np.float32)
        else:
            self.vfm_means = None
            self.vfm_stds = None

        self.point_cloud_range = point_cloud_range

    def _get_camera_info(
        self,
        camera_name: str,
        zample: Optional[Zample] = None,
        root_folder: Optional[Path] = None,
        cameras: Optional[Dict[str, List[str]]] = None,
        cameras_timestamps: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the camera info for the given camera name.
        Args:
            camera_name: The name of the camera to get the info for.
            zample: The zample to get the info from.
            root_folder: The root folder to get the info from.
            cameras: The cameras to get the info from.
            camera_timestamps: The camera timestamps to get the info from.
        Returns:
            files: The files to get the info from.
            timestamps: The timestamps to get the info from.
        """
        if self.dataset_name == "zod":
            assert zample is not None, "Need to provide zample if dataset is zod"
            frames = zample.info.get_camera_frames(camera=camera_name)
            files = [Path(frame.filepath) for frame in frames]
            timestamps = np.array([frame.time.timestamp() for frame in frames])  # Convert to seconds
        elif self.dataset_name == "av2":
            assert root_folder is not None, "Need to provide root_folder if dataset is av2"
            files: List[Path] = sorted((root_folder / "sensors/cameras" / camera_name).glob("*.jpg"))
            timestamps = np.array([int(f.stem) for f in files], dtype=np.float64) / 1e9  # Convert to seconds
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return files, np.array(timestamps)

    def _filter_closest(self, files: List[Path], timestamps: np.ndarray, lidar_timestamps: np.ndarray):
        min_useful_timestamp = min(lidar_timestamps) - MAXIMAL_CAMERA_LIDAR_TIMEDIFF
        max_useful_timestamp = max(lidar_timestamps) + MAXIMAL_CAMERA_LIDAR_TIMEDIFF

        filtered = [(f, t) for f, t in zip(files, timestamps) if min_useful_timestamp <= t <= max_useful_timestamp]
        if filtered:
            files, timestamps = zip(*filtered)
            files = list(files)
            timestamps = np.array(timestamps)
        else:
            print("We really should not be here")
            return [], []

        # Find closest file to each lidar timestamp
        closest_file = []
        closest_timestamp = []
        for lidar_timestamp in lidar_timestamps:
            time_diffs = np.abs(timestamps - lidar_timestamp)
            closest_idx = np.argmin(time_diffs)
            closest_file.append(files[closest_idx])
            closest_timestamp.append(timestamps[closest_idx])

        return closest_file, np.array(closest_timestamp)

    def _read_images(self, files: List[Path], read_img_fn: Optional[Callable[[str], np.ndarray]]) -> List[np.ndarray]:
        if read_img_fn is not None:
            images = [np.array(read_img_fn(str(file)), dtype=np.float32) for file in files]
        else:
            images = [np.array(Image.open(self.data_root / file).convert("RGB"), dtype=np.float32) for file in files]
        images = np.stack(images, axis=0)
        images = np.transpose(images, (0, 3, 1, 2))  # num_images, 3, h, w
        return images

    def _get_camera_calibrations(
        self,
        camera_name: str,
        zample: Optional[Zample] = None,
        root_folder: Optional[Path] = None,
        cameras_calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if self.dataset_name == "zod":
            assert zample is not None, "Need to provide zample if dataset is zod"
            return zample.calibration.cameras[camera_name]
        elif self.dataset_name == "av2":
            assert root_folder is not None, "Need to provide root_folder if dataset is av2"
            pinhole_cam = PinholeCamera.from_feather(root_folder, camera_name)
            intrinsics = pinhole_cam.intrinsics.K.copy()
            extrinsics = inverse_np(pinhole_cam.extrinsics.copy())
            distortion = np.zeros(4)
            return {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "extrinsics_inverse": inverse_np(extrinsics),
                "distortion": distortion,
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _downsample_vfm_data(self, vfm_data: np.ndarray, scale: float) -> np.ndarray:
        scalex = scale
        scaley = scale
        vfm_data = vfm_data.transpose(0, 2, 3, 1)
        vfm_data = np.array(
            [
                cv2.resize(data, (0, 0), fx=scalex, fy=scaley or scalex, interpolation=cv2.INTER_NEAREST_EXACT)
                for data in vfm_data
            ]
        )
        return np.expand_dims(vfm_data, axis=1)

    def read(
        self,
        zample: Optional[Zample] = None,
        root_folder: Optional[Path] = None,
        cameras: Optional[Dict[str, List[str]]] = None,
        cameras_timestamps: Optional[Dict[str, List[int]]] = None,
        lidar_timestamps: Optional[List[float]] = None,
        current_timestamp: Optional[float] = None,
        cameras_calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        read_img_fn: Optional[Callable[[str], np.ndarray]] = None,
        ego_motion: Optional[EgoMotion] = None,
        bev_oxts_pose: Optional[np.ndarray] = None,
        is_train: bool = False,
    ) -> Dict[str, np.ndarray]:
        # Select only one camera per sample for efficiency
        camera_name = np.random.choice(self.cameras, p=self.sample_probability_per_cam)

        # Get the camera files
        files, timestamps = self._get_camera_info(camera_name, zample, root_folder, cameras, cameras_timestamps)

        # Filter the closest files and timestamps
        closest_files, closest_timestamps = self._filter_closest(
            files, timestamps, np.array(lidar_timestamps, dtype=np.float64)
        )

        # Read files
        images = self._read_images(closest_files, read_img_fn)

        # Get the camera calibrations
        calibrations = self._get_camera_calibrations(camera_name, zample, root_folder, cameras_calibrations)

        # Crop the images to remove useless pixels
        crop = self.crop_per_camera[camera_name]
        calibrations["intrinsics"] = crop_intrinsics(calibrations["intrinsics"], crop)

        images = crop_images(images, crop)

        # Downsample
        scale = 1 / get_downsample_factor(self.dataset_name, self.config.downsample_factor)
        calibrations["intrinsics"] = resize_intrinsics_with_scale(calibrations["intrinsics"], scale)
        if self.config.read_rgb:
            images = resize_images_with_scale(images, scale)

        # Reconstruct images with properly downsampled masks and depths
        contents = []
        stack = []
        if self.config.read_rgb:
            contents.append(["rgb"] * 3)
            stack.append(images)
            del images

        if len(stack) > 0:
            stack = np.concatenate(stack, axis=1)
            contents = np.concatenate(contents, axis=0)

        # Randomly crop the images if needed
        stack, calibrations = random_crop_images_and_intrinsics(stack, calibrations, self.config.image_final_size)

        # Get the relative timestamps
        closest_timestamps_relative = get_relative_timestamps(closest_timestamps, current_timestamp)

        # Change the intrinsics to match the feature extraction
        intrinsics = get_intrinsics_for_feature_extraction(
            calibrations["intrinsics"], self.config.extracted_features_downsample_factor
        )

        # Final extraction before output:
        if self.config.read_rgb:
            images = stack[:, contents == "rgb", :, :]

        p_then = ego_motion.get_poses(closest_timestamps_relative + current_timestamp)
        T_cam_then = inverse_np(calibrations["extrinsics"].copy())
        T_then_nows = inverse_np(p_then) @ bev_oxts_pose  # t x 4 x 4
        T_cam_ego = np.expand_dims(T_cam_then, axis=0) @ T_then_nows  # t x 4 x 4

        out = {
            "vfm_timestamps": closest_timestamps_relative,
            "vfm_extrinsics": calibrations["extrinsics"],
            "vfm_extrinsics_inverse": calibrations["extrinsics_inverse"],
            "vfm_distortion": calibrations["distortion"],
            "vfm_intrinsics": intrinsics,
            "rgb_intrinsics": calibrations["intrinsics"],
            "rgb_mean": np.array(RGB_MEAN, dtype=np.float32),
            "rgb_std": np.array(RGB_STD, dtype=np.float32),
            "vfm_T_cam_ego": T_cam_ego,
        }

        if self.config._feature_layers[0] > 0:
            out["vfm_mean"] = self.vfm_means
            out["vfm_std"] = self.vfm_stds

        if self.config.read_rgb:
            out["vfm_images"] = images.astype(np.float32)

        return out

    def get_sliced_ego_motion(self, bev_oxts_pose, ego_motion, timestamps, current_timestamp) -> EgoMotion:
        sweep_eps = 0.2  # 0.2 seconds, sweep is roughly 0.1 seconds so this has margin
        max_timestamp = max(list(timestamps) + [current_timestamp]) + sweep_eps
        min_timestamp = min(list(timestamps) + [current_timestamp]) - sweep_eps
        lower_mask = min_timestamp < ego_motion.timestamps
        upper_mask = ego_motion.timestamps < max_timestamp
        oxts_range_mask = lower_mask & upper_mask
        sliced_ego_motion = EgoMotion(  # dummy for use with compensation code
            ego_motion.poses[oxts_range_mask],
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            ego_motion.timestamps[oxts_range_mask],
            None,  # type: ignore
        )
        sliced_ego_motion.timestamps -= current_timestamp
        sliced_ego_motion.poses = inverse_np(bev_oxts_pose) @ sliced_ego_motion.poses
        return sliced_ego_motion


def generate_image_coords(height: int, width: int) -> np.ndarray:
    """Generate the image coordinates."""
    heights = torch.linspace(0, int(height - 1), int(height), device="cpu", dtype=torch.float64)
    widths = torch.linspace(0, int(width - 1), int(width), device="cpu", dtype=torch.float64)
    grid_heights, grid_widths = torch.meshgrid(heights, widths, indexing="ij")
    image_coords = torch.stack((grid_heights, grid_widths, torch.ones_like(grid_heights)), dim=-1)
    image_coords = image_coords.round()
    return image_coords
