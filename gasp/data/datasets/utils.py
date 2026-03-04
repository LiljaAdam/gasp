import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from zod.data_classes.calibration import LidarCalibration
from zod.data_classes.ego_motion import EgoMotion
from zod.data_classes.sensor import LidarData

from gasp.data.components.pose_utils import inverse_np


def custom_motion_compensate_scanwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: float,
    target_pose: Optional[np.ndarray] = None,
) -> LidarData:
    """Motion compensate a (pointwise compensated) lidar point cloud."""
    lidar_data = lidar_data.copy()
    source_pose = ego_motion.get_poses(lidar_data.core_timestamp)

    if target_pose is None:
        target_pose = ego_motion.get_poses(target_timestamp)

    # Compute relative transformation between target pose and source pose
    odometry = inverse_np(target_pose) @ source_pose

    # Project to ego vehicle frame using calib
    lidar_data.transform(calibration.extrinsics)

    # Project to center frame using odometry
    lidar_data.transform(odometry)

    # Project back to lidar frame using calib
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data


def crop_images(images, crop_size: Tuple[int, int, int, int]):
    """
    Args:
        images: The images to crop. (num_images, channels, height, width)
        crop_size: Tuple[int, int, int, int] = (top, right, bottom, left)
    """
    return images[..., crop_size[0] : -crop_size[2] or None, crop_size[3] : -crop_size[1] or None]


def crop_intrinsics(intrinsics: np.ndarray, crop_size: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the intrinsics.

    Args:
        intrinsics: The intrinsics to crop.
        crop_size: The crop size.

    Returns:
        The cropped intrinsics.
    """
    out_intrinsics = intrinsics.copy()
    out_intrinsics[0, 2] -= crop_size[3]
    out_intrinsics[1, 2] -= crop_size[0]
    return out_intrinsics


def resize_images(images, new_size):
    """Resize an image and its intrinsics.

    Args:
        images: The images to resize.
        intrinsics: The intrinsics to resize.
        new_size: The new size of the image.

    Returns:
        The resized image and intrinsics.
    """
    new_height, new_width = new_size
    images = images.transpose(0, 2, 3, 1)
    images = np.array([cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR) for image in images])
    images = images.transpose(0, 3, 1, 2)
    return images


def resize_images_with_scale(images, scalex, scaley=None):
    images = images.transpose(0, 2, 3, 1)
    images = np.array(
        [cv2.resize(image, (0, 0), fx=scalex, fy=scaley or scalex, interpolation=cv2.INTER_LINEAR) for image in images]
    )
    images = images.transpose(0, 3, 1, 2)
    return images


def resize_intrinsics(intrinsics, new_size, old_size):
    """Resize the intrinsics.

    Args:
        intrinsics: The intrinsics to resize.
        new_size: The new size of the image.

    Returns:
        The resized intrinsics.
    """
    old_width, old_height = old_size
    new_height, new_width = new_size

    scale_height = new_height / old_height
    scale_width = new_width / old_width

    out_intrinsics = intrinsics.copy()
    out_intrinsics[0, 0] *= scale_width
    out_intrinsics[1, 1] *= scale_height
    out_intrinsics[0, 2] *= scale_width
    out_intrinsics[1, 2] *= scale_height

    return out_intrinsics


def resize_intrinsics_with_scale(intrinsics, scale_width, scale_height=None):
    if isinstance(intrinsics, torch.Tensor):
        out_intrinsics = intrinsics.clone()
    else:
        out_intrinsics = intrinsics.copy()
    out_intrinsics[0, 0] *= scale_width
    out_intrinsics[1, 1] *= scale_height or scale_width
    out_intrinsics[0, 2] *= scale_width
    out_intrinsics[1, 2] *= scale_height or scale_width

    return out_intrinsics


class TorchList:
    """Python list stored as serialized objects in a torch tensor.

    Using this in data loaders will avoid memory copies between the dataloader workers.
    """

    def __init__(self, python_list: List):
        serialized_list = [pickle.dumps(d, protocol=-1) for d in python_list]
        addresses = [len(s) for s in serialized_list]

        # Use np.frombuffer instead of torch.frombuffer directly to avoid warnings about nonwritable
        # buffers or arrays.
        serialized_list = [np.frombuffer(s, dtype=np.uint8) for s in serialized_list]
        serialized_list = np.concatenate(serialized_list)
        self._serialized_list = torch.from_numpy(serialized_list)

        self._addresses = torch.tensor(addresses, dtype=torch.int64)
        self._addresses = torch.cumsum(self._addresses, dim=-1)

    def __len__(self):
        return len(self._addresses)

    def __getitem__(self, index):
        if index == 0:
            start_address = 0
        else:
            start_address = int(self._addresses[index - 1])
        end_address = int(self._addresses[index])
        list_entry_bytes = memoryview(self._serialized_list[start_address:end_address].numpy())
        return pickle.loads(list_entry_bytes)


VELODYNE_128_ELEVATION_MAPPING = {
    0: -11.742,
    1: -1.99,
    2: 3.4,
    3: -5.29,
    4: -0.78,
    5: 4.61,
    6: -4.08,
    7: 1.31,
    8: -6.5,
    9: -1.11,
    10: 4.28,
    11: -4.41,
    12: 0.1,
    13: 6.48,
    14: -3.2,
    15: 2.19,
    16: -3.86,
    17: 1.53,
    18: -9.244,
    19: -1.77,
    20: 2.74,
    21: -5.95,
    22: -0.56,
    23: 4.83,
    24: -2.98,
    25: 2.41,
    26: -6.28,
    27: -0.89,
    28: 3.62,
    29: -5.07,
    30: 0.32,
    31: 7.58,
    32: -0.34,
    33: 5.18,
    34: -3.64,
    35: 1.75,
    36: -25.0,
    37: -2.43,
    38: 2.96,
    39: -5.73,
    40: 0.54,
    41: 9.7,
    42: -2.76,
    43: 2.63,
    44: -7.65,
    45: -1.55,
    46: 3.84,
    47: -4.85,
    48: 3.18,
    49: -5.51,
    50: -0.12,
    51: 5.73,
    52: -4.3,
    53: 1.09,
    54: -16.042,
    55: -2.21,
    56: 4.06,
    57: -4.63,
    58: 0.76,
    59: 15.0,
    60: -3.42,
    61: 1.97,
    62: -6.85,
    63: -1.33,
    64: -5.62,
    65: -0.23,
    66: 5.43,
    67: -3.53,
    68: 0.98,
    69: -19.582,
    70: -2.32,
    71: 3.07,
    72: -4.74,
    73: 0.65,
    74: 11.75,
    75: -2.65,
    76: 1.86,
    77: -7.15,
    78: -1.44,
    79: 3.95,
    80: -2.1,
    81: 3.29,
    82: -5.4,
    83: -0.01,
    84: 4.5,
    85: -4.19,
    86: 1.2,
    87: -13.565,
    88: -1.22,
    89: 4.17,
    90: -4.52,
    91: 0.87,
    92: 6.08,  # 8.95
    93: -3.31,
    94: 2.08,
    95: -6.65,
    96: 1.42,
    97: -10.346,
    98: -1.88,
    99: 3.51,
    100: -6.06,
    101: -0.67,
    102: 4.72,
    103: -3.97,
    104: 2.3,
    105: -6.39,
    106: -1.0,
    107: 4.39,
    108: -5.18,
    109: 0.21,
    110: 6.98,
    111: -3.09,
    112: 4.98,
    113: -3.75,
    114: 1.64,
    115: -8.352,
    116: -2.54,
    117: 2.85,
    118: -5.84,
    119: -0.45,
    120: 8.43,
    121: -2.87,
    122: 2.52,
    123: -6.17,
    124: -1.66,
    125: 3.73,
    126: -4.96,
    127: 0.43,
}  # degrees
