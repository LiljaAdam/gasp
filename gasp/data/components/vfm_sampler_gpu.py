import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

import torch
from torch import Tensor

import gasp.config.base_config as cfg
from gasp.common.types import PointCloudRange
from gasp.data.components.pose_utils import inverse as inverse3x4, to4x4
from gasp.model.ops.functions.interpolate import interpolate_trajectories

MAX_CAMERA_LIDAR_TIMEDIFF = 0.05  # seconds


def _transform(points: Tensor, poses: Tensor) -> Tensor:
    return (poses[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + poses[..., :3, 3]


@dataclass
class VFMFeaturesProjectionGPUSamplerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: VFMFeaturesProjectionGPUSampler)

    use_cached_features: bool = False
    """Whether to generate the features on the fly or not."""

    feature_extractor_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    """The model to use for feature extraction."""

    num_query_points: int = 100_000
    """Number of positive samples to generate."""

    buffer_size: float = 0.1
    """Buffer size behind the positive samples."""

    depth_scaling = 1.2
    """How far away (factor) a point can be to still be considered ok in background removal."""

    rough_camera_fov_deg: float = -1.0  # 130.0
    """Rough camera field of view in degrees. Disable with value less than 0."""

    feature_reduction_root: Path = Path("/outputs/feature_reductions")
    """Path to cached feature reductions."""

    @property
    def model_name(self) -> str:
        return self.feature_extractor_model.split("/")[-1].replace("-", "_")

    @property
    def reduction_info_path(self) -> Path:
        return self.feature_reduction_root / f"{self.model_name}_features" / "reduction_info.pt"


class VFMFeaturesProjectionGPUSampler(torch.nn.Module):
    def __init__(
        self,
        config: VFMFeaturesProjectionGPUSamplerConfig,
        point_cloud_range: PointCloudRange,
        feature_dim: int = -1,
        feature_layers: Tuple[int, ...] = (12,),
    ):
        super().__init__()
        self.config = config
        self.point_cloud_range = point_cloud_range
        self.feature_dim = feature_dim
        self.feature_layers = feature_layers
        self.num_layers = len(feature_layers)

        if feature_dim > 0:
            self.predicted_feature_dim = feature_dim * len(feature_layers)
        else:
            self.predicted_feature_dim = 768 * len(feature_layers)  # TODO: fix this

        self.rough_camera_fov = config.rough_camera_fov_deg * torch.pi / 180.0

        for i in range(3):
            try:
                self._init_feature_extractor()
                break
            except Exception as e:
                print(f"[{i} / 3] Error initializing feature extractor: {e}")
                time.sleep(1)
                if i == 2:
                    raise e

        if feature_layers[0] > 0:
            self.reduction_info = torch.load(self.config.reduction_info_path, weights_only=False)
            means = torch.stack(
                [self._fetch_reduction_info(layer_idx, 768, "mean") for layer_idx in self.feature_layers], dim=0
            )
            stds = torch.stack(
                [self._fetch_reduction_info(layer_idx, 768, "std") for layer_idx in self.feature_layers], dim=0
            )
            compression = torch.stack(
                [
                    self._fetch_reduction_info(layer_idx, self.feature_dim, "pca_compression_matrix")
                    for layer_idx in self.feature_layers
                ],
                dim=0,
            )
            self.register_buffer("reduction_means", means)
            self.register_buffer("reduction_stds", stds)
            self.register_buffer("reduction_compressions", compression)

    def _init_feature_extractor(self):
        if self.config.use_cached_features:
            self.feature_extractor = None
            return

        import torch.distributed as dist
        from transformers import AutoImageProcessor, AutoModel

        model_name = self.config.feature_extractor_model

        # Handle distributed training safely
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if is_dist:
            if rank == 0:
                # Rank 0 downloads and caches the model
                self.processor = AutoImageProcessor.from_pretrained(model_name, do_resize=False, do_rescale=True)
                self.model = AutoModel.from_pretrained(model_name, device_map=None).eval()
            # Synchronize — all other ranks wait until cache is ready
            dist.barrier()
            if rank != 0:
                # Other ranks load only from local cache (offline)
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, do_resize=False, do_rescale=True, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(model_name, device_map=None, local_files_only=True).eval()
        else:
            # Single-process (non-DDP) case
            self.processor = AutoImageProcessor.from_pretrained(model_name, do_resize=False, do_rescale=True)
            self.model = AutoModel.from_pretrained(model_name, device_map=None).eval()

        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Extra barrier to ensure all ranks have fully loaded before proceeding
        if is_dist:
            dist.barrier()

    def _move_feature_extractor_to_device(self, device: torch.device):
        if self.feature_extractor is not None and self.feature_extractor.model.device != device:
            self.feature_extractor.model.to(device)

    def _fetch_reduction_info(self, layer_idx: int, dim: int, info_type: str) -> Tensor:
        return torch.from_numpy(self.reduction_info[f"layer_{layer_idx}"][f"dim_{dim}"][info_type])

    def _remove_invalid_points(self, points: Tensor, vfm_times: Tensor, augmentation_angle: Tensor) -> Tensor:
        sanitized = []
        for pc, vfm_t, aug_ang in zip(points, vfm_times, augmentation_angle):
            if self.rough_camera_fov > 0:
                # remove points that can not end up inside the camera
                angles = torch.atan2(pc[..., 0], pc[..., 1])

                rough_camera_fov = self.rough_camera_fov + aug_ang * torch.pi / 180.0
                angles_in_fov_lower = angles >= -rough_camera_fov / 2
                angles_in_fov_upper = angles <= +rough_camera_fov / 2
                in_fov = torch.logical_and(angles_in_fov_lower, angles_in_fov_upper)
                pc = pc[in_fov]
            # remove points that have time too far away from any camera frame
            t = pc[..., -1]  # N
            time_diffs = t[:, None] - vfm_t[None, :]  # N, M
            in_time = (torch.abs(time_diffs) <= MAX_CAMERA_LIDAR_TIMEDIFF).any(dim=-1)
            pc = pc[in_time]

            sanitized.append(pc)

        return sanitized

    def _pad(self, points: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # pad the points to the same length
        max_length = max([p.shape[0] for p in points])
        out = torch.zeros(len(points), max_length, 4, dtype=points[0].dtype, device=points[0].device)
        valid = torch.zeros(len(points), max_length, dtype=torch.bool, device=points[0].device)
        for i, p in enumerate(points):
            out[i, : p.shape[0]] = p
            valid[i, : p.shape[0]] = True
        return out, valid

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def _get_features(self, vfm_images: Tensor) -> Tensor:
        b, m, c, h, w = vfm_images.shape
        h_feat, w_feat = h // 16, w // 16
        n = h_feat * w_feat

        # Prepare the images for the model
        vfm_images = vfm_images.reshape(b * m, c, h, w)

        inputs = self.processor(vfm_images, return_tensors="pt")

        # Get the features
        if self.feature_layers[0] > 0:
            outputs = self.model(**inputs, output_hidden_states=True)
            features = [outputs.hidden_states[layer_idx][:, 5:, :] for layer_idx in self.feature_layers]
            features = torch.stack(features, dim=0)

            means = self.reduction_means.unsqueeze(1).unsqueeze(1).repeat(1, b * m, n, 1)
            stds = self.reduction_stds.unsqueeze(1).unsqueeze(1).repeat(1, b * m, n, 1)

            features = (features - means) / (stds + 1e-8)

            features = features.view(self.num_layers, b * m * n, 768) @ self.reduction_compressions

        else:
            outputs = self.model(**inputs, output_hidden_states=False)
            features = outputs.last_hidden_state[:, 5:, :]
            features = features.view(b, m, h_feat, w_feat, 768)
            features = features[..., : self.feature_dim].contiguous()

        features = features.view(self.num_layers, b, m, h_feat, w_feat, self.feature_dim)

        features = features.permute(1, 2, 3, 4, 5, 0)

        features = features.reshape(b, m, h_feat, w_feat, self.predicted_feature_dim)

        return features

    @torch.no_grad()
    def sample(
        self,
        vfm_features: Tensor,  # B, M, H, W, C
        vfm_images: Tensor,  # B, M, H, W, C
        vfm_timestamps: Tensor,  # B, M
        ego_poses: Tensor,  # float64
        ego_pose_times: Tensor,  # float32, relative to current time
        lidar_extrinsics: Tensor,  # B, 4, 4 (float64)
        camera_extrinsics_inverse: Tensor,  # B, 4, 4 (float64)
        camera_intrinsics: Tensor,  # B, 3, 3 (float64)
        camera_distortion_params: Tensor,  # B, 8 (float64)
        positive_points: Tensor,  # B, M, 4
        augmentation_angle: Tensor,  #  B, M
        get_rgb: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.config.use_cached_features:
            assert vfm_features is not None, "VFM features are required when using cached features"
            assert not get_rgb, "RGB features are not supported when using cached features"
        else:
            assert vfm_images is not None, "VFM images should be provided when not using cached features"
            if get_rgb:
                vfm_features = vfm_images.permute(0, 1, 3, 4, 2)
            else:
                vfm_features = self._get_features(vfm_images)

        bs, device = len(vfm_features), vfm_features.device
        # subsample the points to the ones that could be in any camera
        points = self._remove_invalid_points(positive_points, vfm_timestamps, augmentation_angle)  # List[N, 4]
        points, valid = self._pad(points)  # B, N, 4

        # find what camera time the points are closest to
        time_diffs = torch.abs(points[..., -1][:, :, None] - vfm_timestamps[:, None, :])  # B, N, M

        # find the camera that is closest in time
        _, closest_camera = time_diffs.min(dim=-1)  # B, N ranging [0, M-1]

        # lets project the points to the camera
        query_times = torch.cat([points[..., -1], vfm_timestamps], dim=-1)  # B, N+M
        query_poses = interpolate_trajectories(ego_poses, ego_pose_times, query_times)  # B, N+M, 3, 4
        point_poses, vfm_poses = torch.split(query_poses, [points.shape[1], vfm_features.shape[1]], dim=1)
        # compensate the points to the time of the camera
        vfm_poses_inv = inverse3x4(vfm_poses)
        target_poses_inv = vfm_poses_inv[torch.arange(bs, device=device), closest_camera]  # B, N, 3, 4
        # compensate the points to the closest camera and that time
        # note that we keep the points when transformed to the current frame as we will use them later
        t = to4x4(point_poses) @ lidar_extrinsics.unsqueeze(1)
        points_in_current = _transform(points[..., :3], t.to(points.dtype))  # B, N, 3
        origins_in_current = t[:, :, :3, -1]  # B, N, 3
        # do the remaining transformation to the camera frame. QUESTION: is this ok?
        t = camera_extrinsics_inverse.unsqueeze(1) @ to4x4(target_poses_inv)
        points[..., :3] = _transform(points_in_current, t.to(points.dtype))  # B, N, 4 in camera frame

        # project the points to the camera
        xy_array = batched_project_kannala(points[..., :3], camera_intrinsics, camera_distortion_params)
        xyd_array = torch.cat([xy_array, points[..., 2:3]], dim=-1)
        # filter out the points that are outside the image
        vfm_size_w = vfm_features.shape[-2]
        vfm_size_h = vfm_features.shape[-3]
        inside_vfm_mask = torch.logical_and(
            torch.logical_and(xyd_array[..., 0] >= 0, xyd_array[..., 0] < vfm_size_w),
            torch.logical_and(xyd_array[..., 1] >= 0, xyd_array[..., 1] < vfm_size_h),
        )
        valid = torch.logical_and(valid, inside_vfm_mask)  # B, N
        dummy_point = torch.tensor([0, 0, 1e6], dtype=xyd_array.dtype, device=points.device)
        # We set the points that are outside the image to a dummy point to not interfere with the filtering
        xyd_array[~valid] = dummy_point
        foreground_masks = batched_depth_filtering(xyd_array, (vfm_size_w, vfm_size_h), self.config.depth_scaling)
        valid = torch.logical_and(valid, foreground_masks)  # B, N
        num_valid = valid.sum(dim=-1)  # B

        invalid_batch_mask = num_valid == 0
        # TODO: Change this to handle the case where one batch is ok and others are not ok
        # for no we just fail across all batches if one is invalid which is not very nice
        # but it is a start. Especially since we are using bs=1 for now.
        if invalid_batch_mask.any():
            if get_rgb:
                out_dim = 3
            else:
                out_dim = self.predicted_feature_dim
            return (
                torch.zeros((bs, self.config.num_query_points, 4), dtype=points.dtype, device=device),
                torch.zeros((bs, self.config.num_query_points, out_dim), device=device),
                torch.zeros((bs, self.config.num_query_points, 3), device=device),
                torch.zeros((bs, self.config.num_query_points), dtype=torch.int32, device=device),
                torch.zeros((bs,), dtype=torch.int32, device=device),  # if batch is valid
                vfm_features if not self.config.use_cached_features else None,  # B, M, H, W, C
            )

        # This nasty ass for loop is hard to get rid off. I dont have enough coffe to do it. If anyone has
        # the time, energy and grit, feel free to batchify this one. Note that since we are using bs=1 for
        # now it is not super prio.
        indices = []
        for i in range(bs):
            rand_ind = torch.randint(
                0, int(num_valid[i]), (self.config.num_query_points,), dtype=torch.int32, device=device
            )
            valid_ = valid[i].nonzero().squeeze(-1)
            indices.append(valid_[rand_ind])

        indices = torch.stack(indices).to(torch.int32)  # B, num_sampled_points

        # take out the valid point and other stuff we neewd
        batch_indices = torch.arange(bs, device=device)
        xy_array = xyd_array[batch_indices, indices, :2].to(
            torch.int32
        )  # to int for indexing | B, num_sampled_points, 2
        points_in_current = points_in_current[batch_indices, indices]  # B, num_sampled_points, 3
        origins_in_current = origins_in_current[batch_indices, indices]  # B, num_sampled_points, 3
        directions = points_in_current - origins_in_current
        directions = directions / directions.norm(dim=-1, keepdim=True)
        closest_camera = closest_camera[batch_indices, indices]  # B, num_sampled_points
        vfm_query_times = points[batch_indices, indices, -1]  # B, num_sampled_points
        rand_dist = torch.rand_like(vfm_query_times) * self.config.buffer_size
        vfm_query_points = points_in_current + directions * rand_dist[:, :, None]
        # get the features at the points
        vfm_point_features = vfm_features[batch_indices, closest_camera, xy_array[..., 1], xy_array[..., 0]]
        # use only the number of features that we want to predict
        vfm_point_features = vfm_point_features[:, :, : self.predicted_feature_dim]
        vfm_queries = torch.cat([vfm_query_points, vfm_query_times.unsqueeze(-1)], dim=-1).to(torch.float32)

        return (
            vfm_queries.to(torch.float32),  # B, N, 4
            vfm_point_features,  # B, N, C
            xyd_array[batch_indices, indices, :3],  # B, N, 3 (float32)
            torch.ones_like(vfm_query_times, dtype=torch.int32),  # B, N
            torch.ones((bs,), dtype=torch.int32, device=device),  # B
            vfm_features if not self.config.use_cached_features else None,  # B, M, H, W, C
        )


def batched_project_kannala(
    points_3d: torch.Tensor,
    camera_matrix: torch.Tensor,
    distortion_coefs: torch.Tensor,
) -> torch.Tensor:
    """Project 3d points to 2d using Kannala-Brandt model with batch support.

    Args:
        points_3d: 3d points in shape (B, N, 3)
        camera_matrix: Camera matrix in shape (B, 3, 4)
        distortion_coefs: Distortion coefficients in shape (B, 4)
        device: Optional device to perform computations on.

    Returns:
        2d points in shape (B, N, 2)
    """

    device = points_3d.device

    # Batch size and number of points
    B, N, _ = points_3d.shape

    # Create homogeneous points
    homogenous_points_3d = torch.cat([points_3d, torch.ones((B, N, 1), device=device)], dim=-1)

    # Compute norm of x and y coordinates
    norm_data = homogenous_points_3d[..., :2].norm(dim=-1)

    # Compute radial distortion
    radial = torch.atan2(norm_data, homogenous_points_3d[..., 2])
    radial2 = radial**2
    radial4 = radial2**2
    radial6 = radial4 * radial2
    radial8 = radial4**2

    # Apply distortion (broadcasting distortion coefficients)
    distortion_angle = radial * (
        1
        + distortion_coefs[:, 0, None] * radial2
        + distortion_coefs[:, 1, None] * radial4
        + distortion_coefs[:, 2, None] * radial6
        + distortion_coefs[:, 3, None] * radial8
    )

    # Compute distorted coordinates
    u_dist = distortion_angle * homogenous_points_3d[..., 0] / norm_data
    v_dist = distortion_angle * homogenous_points_3d[..., 1] / norm_data

    # Apply camera matrix
    pos_u = camera_matrix[:, 0, 0, None] * u_dist + camera_matrix[:, 0, 2, None]
    pos_v = camera_matrix[:, 1, 1, None] * v_dist + camera_matrix[:, 1, 2, None]

    # Stack coordinates
    projected_points = torch.stack((pos_u, pos_v), dim=-1)

    # Remove batch dimension if input was 2D
    return projected_points


def batched_depth_filtering(xyds: Tensor, cropped_size: Tuple[int, int], depth_scaling: float) -> Tensor:
    bs = len(xyds)
    # Convert inputs to tensors
    u_int = torch.floor(xyds[..., 0]).to(torch.int64)
    v_int = torch.floor(xyds[..., 1]).to(torch.int64)
    depth = xyds[..., 2].to(torch.float32)

    # Create linear indices
    linear_idx = v_int * cropped_size[0] + u_int  # B, N
    batched_offsets = torch.arange(bs, device=linear_idx.device)[:, None] * cropped_size[1] * cropped_size[0]  # B, 1
    linear_idx += batched_offsets

    # Find minimum depths for each unique pixel across all batches
    min_depths = torch.zeros(bs * cropped_size[1] * cropped_size[0], dtype=torch.float32, device=xyds.device)

    # Scatter minimum depths
    min_depths.scatter_reduce_(0, linear_idx.reshape(-1), depth.reshape(-1), reduce="amin", include_self=False)

    # Create masks based on depth scaling
    foreground_masks = depth < min_depths[linear_idx.reshape(-1)].reshape(bs, -1) * depth_scaling

    return foreground_masks
