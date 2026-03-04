import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import av2.utils.io as io_utils
import numpy as np
import torch
import tyro
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader, convert_pose_dataframe_to_SE3
from av2.structures.sweep import Sweep
from av2.utils.io import read_ego_SE3_sensor
from torch import Tensor
from tqdm import tqdm

from gasp.config.method_configs import convert_markup_to_ansi
from gasp.utils.pose_utils import interpolate_trajectories, pose_inverse, to4x4, transform_points

DUMMY_DISTANCE_VALUE = 500.0
MAX_REFLECTANCE_VALUE = 255.0

ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING = {
    31: -25.0,
    13: -1.0,
    1: -1.667,
    17: -15.639,
    29: -11.31,
    9: 0.0,
    3: -0.667,
    21: -8.843,
    27: -7.254,
    12: 0.333,
    5: -0.333,
    16: -6.148,
    25: -5.333,
    8: 1.333,
    7: 0.667,
    20: -4.0,
    22: -4.667,
    2: 1.667,
    10: 1.0,
    28: -3.667,
    23: -3.333,
    6: 3.333,
    11: 2.333,
    24: -2.667,
    18: -3.0,
    0: 7.0,
    14: 4.667,
    30: -2.333,
    19: -2.0,
    4: 15.0,
    15: 10.333,
    26: -1.333,
}

AV2_VELO_VLP32C_ELEVATION_MAPPING = ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING
AV2_ELEVATION_MAPPING = {
    "lidar_down": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar_up": AV2_VELO_VLP32C_ELEVATION_MAPPING,
    "lidar": AV2_VELO_VLP32C_ELEVATION_MAPPING,
}

AV2_AZIMUTH_RESOLUTION = {
    "lidar_down": 0.2,
    "lidar_up": 0.2,
    "lidar": 0.2,
}  # degrees
AV2_SKIP_ELEVATION_CHANNELS = {
    "lidar": (0, 1, 2),
    "lidar_down": (4, 15, 0),  # corresponding to 15,10.333,7.0 degrees as lidar_down is upside down
    "lidar_up": (0, 3, 4),  # corresponding to -25,-15.639,-11.31 degrees
}
AV2_IGNORE_REGIONS = {
    "lidar_down": [[140, 200, -30.0, 20.0], [-200.0, -170.0, -30.0, 20.0]],
    "lidar_up": [],
    "lidar": [],
}


@dataclass
class ExtractMissingPointsConfig:
    """Configuration for extracting missing LiDAR points in Argoverse 2."""

    output_root: Path = Path("/cached_missing_points")
    """Directory where missing-point .npy files will be written."""

    data_dir: Path = Path("/av2")
    """Root directory of the Argoverse 2 sensor dataset."""

    split: str = "train"
    """Dataset split to process (e.g. 'train', 'val', 'test')."""


@dataclass(init=False)
class Lidars:
    """Dataparser outputs for the lidar dataset and the ray generator.

    If a single value is provided, it is broadcasted to all lidars.

    Args:
        lidar_to_worlds: Lidar to world matrices. Tensor of per-image l2w matrices, in [R | t] format
        lidar_type: Type of lidar sensor. This will be an int corresponding to the LidarType enum.
        assume_ego_compensated: Whether or not to assume that points are ego-compensated.
        times: Timestamps for each lidar
        n_points: Number of points in each lidar
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the lidars
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    lidar_to_worlds: Tensor
    lidar_type: Tensor
    times: Optional[Tensor]
    metadata: Optional[Dict]
    azimuths: Optional[Tensor]
    elevations: Optional[Tensor]

    def __init__(
        self,
        lidar_to_worlds: Tensor,
        assume_ego_compensated: bool = True,
        times: Optional[Tensor] = None,
        metadata: Optional[Dict] = None,
        azimuths: Optional[Tensor] = None,
        elevations: Optional[Tensor] = None,
    ) -> None:
        """Initializes the Lidars object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"lidar_to_worlds": 2, "azimuths": 2, "elevations": 2}

        self.lidar_to_worlds = lidar_to_worlds

        # @dataclass's post_init will take care of broadcasting
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.azimuths = self._init_get_azimuths(azimuths)
        self.elevations = self._init_get_elevations(elevations)

        # self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

        self.assume_ego_compensated = assume_ego_compensated

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    def _init_get_azimuths(self, azimuths: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if azimuths is None:
            azimuths = None
        elif isinstance(azimuths, torch.Tensor):
            if azimuths.ndim == 0 or azimuths.shape[-1] != 1:
                azimuths = azimuths.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"azimuths must be None or a tensor, got {type(azimuths)}")

        return azimuths

    def _init_get_elevations(self, elevations: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if elevations is None:
            elevations = None
        elif isinstance(elevations, torch.Tensor):
            if elevations.ndim == 0 or elevations.shape[-1] != 1:
                elevations = elevations.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"elevations must be None or a tensor, got {type(elevations)}")

        return elevations

    @property
    def device(self) -> Union[torch.device, str]:
        """Returns the device that the camera is on."""
        return self.lidar_to_worlds.device

    @property
    def width(self) -> int:
        """Returns the width of the lidar."""
        return self.azimuths.shape[1] if self.azimuths is not None else 0

    @property
    def height(self) -> int:
        """Returns the height of the lidar."""
        return self.elevations.shape[1] if self.elevations is not None else 0


def _remove_ego_motion_compensation(
    point_cloud: torch.Tensor, l2ws: torch.Tensor, times: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes ego motion compensation from point cloud.

    Args:
        point_cloud: Point cloud to remove ego motion compensation from (in world frame). Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
        l2ws: Poses of the lidar. Shape: [num_poses, 4, 4]
        times: Timestamps of the lidar poses. Shape: [num_poses]

    Returns:
        Point cloud without ego motion compensation in sensor frame. Shape: [num_points, 5+N] x,y,z,intensity,timestamp,(channel_id)
        Lidar pose for each point in the point cloud. Shape: [num_points, 4, 4]
    """

    interpolated_l2ws, _, _ = interpolate_trajectories(
        l2ws.unsqueeze(1), times - times.min(), point_cloud[:, 4] - times.min(), clamp_frac=False
    )
    interpolated_l2ws = interpolated_l2ws[:, :3, :4]
    interpolated_w2ls = pose_inverse(interpolated_l2ws)
    homogen_points = torch.cat([point_cloud[:, :3], torch.ones_like(point_cloud[:, -1:])], dim=-1)
    points = torch.matmul(interpolated_w2ls, homogen_points.unsqueeze(-1))[:, :, 0]
    return torch.cat([points, point_cloud[:, 3:]], dim=-1), interpolated_l2ws


def _get_lidars(av2: AV2SensorDataLoader, sequence: str) -> Tuple[Lidars, List[Path]]:
    """Returns lidar info and loaded point clouds."""

    (
        lidar_filenames,
        times,
        poses_up,
        poses_down,
        idxs,
    ) = ([], [], [], [], [])

    lidar_times = av2.get_ordered_log_lidar_timestamps(sequence)

    for t in lidar_times:
        # get the path to the lidar point cloud
        pc_path = av2.get_lidar_fpath(sequence, t)
        lidar_filenames.append(pc_path)

        log_dir = pc_path.parent.parent.parent
        sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
        ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
        ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]

        # get the ego-vehicle pose at the lidar timestamp
        ego_pose = av2.get_city_SE3_ego(sequence, t)
        # move to the actual lidar position
        ego_pose_up = ego_pose.compose(ego_SE3_up_lidar)
        ego_pose_down = ego_pose.compose(ego_SE3_down_lidar)
        # add the pose
        poses_up.append(ego_pose_up.transform_matrix)
        poses_down.append(ego_pose_down.transform_matrix)
        idxs.append(0)  # add the lidar index (there is only one so lets use 0)
        times.append(t / 1e9)  # convert to seconds

    poses_up = torch.tensor(np.array(poses_up), dtype=torch.float32)
    poses_down = torch.tensor(np.array(poses_down), dtype=torch.float32)
    down2up = pose_inverse(poses_up[0, ...]) @ poses_down[0, ...]
    times = torch.tensor(np.array(times, dtype=np.float64), dtype=torch.float64)  # need higher precision
    idxs = torch.tensor(np.array(idxs)).int().unsqueeze(-1)
    # we will store the down2up transformation in the metadata, as we will need it later
    down2up = down2up.repeat(len(poses_up), 1, 1).view(len(poses_up), -1)

    lidars = Lidars(
        lidar_to_worlds=torch.cat([poses_up, poses_down], dim=0)[:, :3, :4],
        times=torch.cat([times, times], dim=0),
        metadata={
            "sensor_idxs": torch.cat([idxs, idxs + (idxs.max() + 1)], dim=0),
            "down2up": torch.cat([down2up, down2up], dim=0),  # will be removed later, here for compatibility
            "poses_down": torch.cat(
                [poses_down.view(len(poses_up), -1), poses_down.view(len(poses_up), -1)], dim=0
            ),  # will be removed later, here for compatibility
        },
    )

    return lidars, lidar_filenames


def _read_lidars(
    lidars: Lidars, filepaths: List[Path], data_dir: Path, sequence: str
) -> Tuple[List[np.ndarray], List[Path]]:
    point_clouds_up, point_clouds_down = [], []

    sorted_filepaths = sorted(filepaths)
    for pc_path in sorted_filepaths:
        sweep = Sweep.from_feather(pc_path)
        # get laser_number, 0-31 is up_lidar, 32-63 is down_lidar according to
        laser_number = sweep.laser_number
        # transform the points back to the sensor position
        xyz_up = sweep.ego_SE3_up_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number < 32])
        xyz_down = sweep.ego_SE3_down_lidar.inverse().transform_point_cloud(sweep.xyz[laser_number >= 32])
        # normalize the intensity values to be in [0, 1]
        intensity = sweep.intensity / MAX_REFLECTANCE_VALUE
        # Add relative time
        time = sweep.offset_ns / 1e9
        # add the intensity as the last channel
        xyz_intensity_up = np.concatenate(
            [
                xyz_up,
                intensity[laser_number < 32, None],
                time[laser_number < 32, None],
                laser_number[laser_number < 32, None],
            ],
            axis=1,
        )
        xyz_intensity_down = np.concatenate(
            [
                xyz_down,
                intensity[laser_number >= 32, None],
                time[laser_number >= 32, None],
                laser_number[laser_number >= 32, None],
            ],
            axis=1,
        )
        # add the point cloud
        point_clouds_up.append(torch.from_numpy(xyz_intensity_up))
        point_clouds_down.append(torch.from_numpy(xyz_intensity_down))

    missing_points_up = []
    missing_points_down = []

    assert lidars.times is not None  # typehints
    assert lidars.metadata is not None  # typehints

    poses_down = lidars.metadata["poses_down"].view(len(lidars.times), 4, 4)
    poses_up = lidars.lidar_to_worlds
    times = lidars.times
    log_pose_df = io_utils.read_feather(data_dir / sequence / "city_SE3_egovehicle.feather")
    all_ego2w = [convert_pose_dataframe_to_SE3(pose_df) for _, pose_df in log_pose_df.iterrows()]
    # calibration is constant, just use the last one
    assert sweep is not None
    uplidar2ego = sweep.ego_SE3_up_lidar
    all_lup2w = torch.tensor(
        np.array([e2w.compose(uplidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
    )
    downlidar2ego = sweep.ego_SE3_down_lidar
    all_ldown2w = torch.tensor(
        np.array([e2w.compose(downlidar2ego).transform_matrix for e2w in all_ego2w]), dtype=torch.float32
    )
    all_times = torch.from_numpy(log_pose_df["timestamp_ns"].to_numpy() / 1e9)

    for i, (pc_up, pc_down, lup2w, ldown2w, time) in enumerate(
        zip(point_clouds_up, point_clouds_down, poses_up, poses_down, times)
    ):
        for j, (pc_, l2w) in enumerate(zip((pc_up, pc_down), (lup2w, ldown2w))):
            pc = pc_.clone().to(torch.float64)
            # relative -> absolute time
            pc[:, -2] += time
            # project to world frame
            pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
            # remove ego motion compensation and move to sensor frame
            if j == 0:  # up_lidar
                l2ws = all_lup2w
                lidar_name = "lidar_up"
            else:  # down_lidar
                l2ws = all_ldown2w
                lidar_name = "lidar_down"
            pc, interpolated_poses = _remove_ego_motion_compensation(pc, l2ws, all_times)
            # reset time to relative
            pc[:, 4] = pc_[:, 4].clone()
            # transform to common lidar frame again, interpolated_poses is l2w for each point at its timestamp
            interpolated_poses = torch.matmul(pose_inverse(l2w.unsqueeze(0)).float(), to4x4(interpolated_poses).float())
            # move channel from index 5 to 3
            pc = pc[..., [0, 1, 2, 5, 3, 4]]
            if j == 1:  # down_lidar
                pc[..., 3] -= 32
            # add missing points
            miss_points = _get_missing_points(
                pc,
                interpolated_poses,
                lidar_name,
                dist_cutoff=0.0,
                outlier_thresh=0.2,
                ignore_regions=AV2_IGNORE_REGIONS[lidar_name],
            )
            if j == 1:  # down_lidar
                miss_points[..., 3] += 32

            # move channel back from index 3 to 5
            miss_points = miss_points[..., [0, 1, 2, 4, 5, 3]]
            if j == 0:  # up_lidar
                missing_points_up.append(miss_points.float())
            else:  # down_lidar
                missing_points_down.append(miss_points.float())

    missing_points_up_in_ego = [
        torch.cat([uplidar2ego.transform_point_cloud(pc[:, :3]), pc[:, 3:]], dim=-1) for pc in missing_points_up
    ]
    missing_points_down_in_ego = [
        torch.cat([downlidar2ego.transform_point_cloud(pc[:, :3]), pc[:, 3:]], dim=-1) for pc in missing_points_down
    ]
    missing_points_in_ego = [
        torch.cat([pc_up, pc_down], dim=0).cpu().numpy()
        for pc_up, pc_down in zip(missing_points_up_in_ego, missing_points_down_in_ego)
    ]
    return missing_points_in_ego, sorted_filepaths


def _get_missing_points(
    point_cloud: Tensor,
    l2ws: Tensor,
    lidar_name: str,
    dist_cutoff: float = 1.0,
    ignore_regions: list = [],
    outlier_thresh: float = 0.2,
) -> Tensor:
    """Finds missing points in the point cloud according to sensor spec (self.config.lidar_elevation_mapping)

    Args:
        point_cloud: Point cloud to find missing points in (in sensor frame). Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
        l2ws: Poses of the lidar. Shape: [num_points, 4, 4]
        lidar_name: Name of the lidar
        dist_cutoff: Distance cutoff for points to consider. Points closer than this will be ignored.
        ignore_regions: List of regions to ignore. Each region is a list of [min_azimuth, max_azimuth, min_elevation, max_elevation]
        outlier_thresh: Threshold for outlier elevation values. If the median elevation of the missing points is more than this value away from the median elevation of the points in the channel, we ignore the missing points.

    Returns:
        Missing points in the point cloud, in world_frame. Shape: [num_points, 4+x] x,y,z,channel_id(timestamp, intensity, etc.)
    """
    dist = torch.norm(point_cloud[:, :3], dim=-1)
    dist_mask = dist > dist_cutoff
    dist = dist[dist_mask]
    point_cloud = point_cloud[dist_mask]
    l2ws = l2ws[dist_mask]
    elevation = torch.arcsin(point_cloud[:, 2] / dist)
    elevation = torch.rad2deg(elevation)
    azimuth = torch.atan2(point_cloud[:, 1], point_cloud[:, 0])
    azimuth = torch.rad2deg(azimuth)

    # find missing points
    missing_points = []
    missing_points_sensor_frame = []
    for channel_id, expected_elevation in AV2_ELEVATION_MAPPING[lidar_name].items():
        if channel_id in AV2_SKIP_ELEVATION_CHANNELS[lidar_name]:
            continue
        channel_mask = (point_cloud[:, 3] - channel_id).abs() < 0.1  # handle floats
        curr_azimuth = azimuth[channel_mask]
        curr_l2ws = l2ws[channel_mask]
        curr_elev = elevation[channel_mask]
        if not channel_mask.any():
            continue
        curr_azimuth, sort_idx = curr_azimuth.sort()
        curr_l2ws = curr_l2ws[sort_idx]
        curr_elev = curr_elev[sort_idx]

        # find missing azimuths, we should have 360 / lidar_azimuth_resolution azimuths
        num_expected_azimuths = int(360 / AV2_AZIMUTH_RESOLUTION[lidar_name]) + 1
        expected_idx = torch.arange(num_expected_azimuths, device=curr_azimuth.device)
        # find offset
        offset = curr_azimuth[0] % AV2_AZIMUTH_RESOLUTION[lidar_name]
        current_idx = ((curr_azimuth - offset + 180) / AV2_AZIMUTH_RESOLUTION[lidar_name]).round().int()
        missing_idx = expected_idx[torch.isin(expected_idx, current_idx, invert=True)]
        # interpolate missing azimuths
        missing_azimuth = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            (-180 + offset - AV2_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                            curr_azimuth,
                            (180 + offset + AV2_AZIMUTH_RESOLUTION[lidar_name]).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        missing_elevation = (
            torch.from_numpy(
                np.interp(
                    missing_idx,
                    torch.cat([torch.tensor(-1).view(1), current_idx, torch.tensor(num_expected_azimuths).view(1)]),
                    torch.cat(
                        [
                            torch.tensor(expected_elevation).view(1),
                            curr_elev.view(-1),
                            torch.tensor(expected_elevation).view(1),
                        ]
                    ),
                )
            )
            .float()
            .view(-1, 1)
        )
        elevation_outlier_mask = (missing_elevation - curr_elev.median()).abs() > outlier_thresh
        missing_elevation[elevation_outlier_mask] = curr_elev.median()

        ignore_mask = torch.zeros_like(missing_azimuth.squeeze(-1)).bool()
        for ignore_region in ignore_regions:
            ignore_mask = (
                (missing_azimuth > ignore_region[0])
                & (missing_azimuth < ignore_region[1])
                & (missing_elevation > ignore_region[2])
                & (missing_elevation < ignore_region[3])
            ).squeeze(-1) | ignore_mask
        missing_azimuth = missing_azimuth[~ignore_mask]
        missing_elevation = missing_elevation[~ignore_mask]

        # missing_elevation = torch.ones_like(missing_azimuth) * current_elevation
        missing_distance = torch.ones_like(missing_azimuth) * DUMMY_DISTANCE_VALUE
        x = torch.cos(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        y = torch.sin(torch.deg2rad(missing_azimuth)) * torch.cos(torch.deg2rad(missing_elevation)) * missing_distance
        z = torch.sin(torch.deg2rad(missing_elevation)) * missing_distance
        points = torch.cat([x, y, z], dim=-1)
        missing_points_sensor_frame.append(points)
        # transform points from sensor space to world space
        points = torch.cat([points, torch.ones_like(points[:, -1:])], dim=-1)
        # find closest pose idx
        closest_pose_idx = torch.searchsorted(curr_azimuth, missing_azimuth.squeeze())
        closest_pose_idx = closest_pose_idx.clamp(max=len(curr_azimuth) - 1)
        if closest_pose_idx.shape == ():
            closest_pose_idx = closest_pose_idx.unsqueeze(0)
        closest_l2w = curr_l2ws[closest_pose_idx]
        points = torch.matmul(closest_l2w.float(), points.unsqueeze(-1))[:, :, 0]
        closest_pc_values = point_cloud[channel_mask][sort_idx][closest_pose_idx, 3:]
        points = torch.cat([points, closest_pc_values], dim=-1)
        missing_points.append(points)

    missing_points = torch.cat(missing_points, dim=0)
    missing_points_sensor_frame = torch.cat(missing_points_sensor_frame, dim=0)
    return missing_points


def extract_missing_points(output_root: Path, data_root: Path, split: str):
    datapath = data_root / "sensor" / split
    print("Setting up data loader...")
    av2 = AV2SensorDataLoader(datapath, datapath)

    out_dir = output_root / split
    sequences = av2.get_log_ids()[350:]
    print(f"Processing {len(sequences)} sequences in {split}-split")
    for sequence in tqdm(sequences):
        extract_sequence(av2, out_dir, sequence)


def extract_sequence(av2: AV2SensorDataLoader, out_dir: Path, sequence: str):
    lidars, pc_filenames = _get_lidars(av2, sequence)
    missing_points_in_ego, sorted_filepaths = _read_lidars(lidars, pc_filenames, av2._data_dir, sequence)

    assert len(sorted_filepaths) == len(missing_points_in_ego)

    out_dir = out_dir / sequence
    os.makedirs(out_dir, exist_ok=True)
    for missing_points, pc_path in zip(missing_points_in_ego, sorted_filepaths):
        pc_name = pc_path.stem + "_missing.npy"
        np.save(out_dir / pc_name, missing_points)


def main(config: ExtractMissingPointsConfig) -> None:
    extract_missing_points(
        output_root=config.output_root,
        data_root=config.data_dir,
        split=config.split,
    )


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            ExtractMissingPointsConfig,
            description=convert_markup_to_ansi(__doc__),  # type: ignore
        )
    )


if __name__ == "__main__":
    entrypoint()
