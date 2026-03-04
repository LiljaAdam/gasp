# interpolate.py
import torch  # note that torch have to be imported prior to the custom cuda  # noqa: I001
import fast_interp_cpu
import fast_interp_cuda
import numpy as np
from torch import Tensor


def interpolate_rotations(rot0: Tensor, rot1: Tensor, alpha: Tensor) -> Tensor:
    """
    Interpolate rotation matrices using quaternion SLERP with a custom CUDA kernel.

    Args:
        rot0 (torch.Tensor): Tensor of shape (..., N, 3, 3), initial rotation matrices.
        rot1 (torch.Tensor): Tensor of shape (..., N, 3, 3), final rotation matrices.
        alpha (torch.Tensor or float): Interpolation factors, scalar or tensor of shape (N).

    Returns:
        torch.Tensor: Interpolated rotation matrices of shape (..., 3, 3).
    """
    if not rot0.is_cuda or not rot1.is_cuda or not alpha.is_cuda:
        raise ValueError("All input tensors must be CUDA tensors.")
    if rot1.shape != rot0.shape:
        raise ValueError("rot0 and rot1 must have the same shape.")
    if alpha.shape != rot0.shape[:-2] and alpha.numel() != 1:
        raise ValueError("alpha must be scalar or have shape matching rotations batch dimension.")

    rot_interp = torch.empty_like(rot0)
    fast_interp_cuda.interpolate_rotations(rot0.contiguous(), rot1.contiguous(), alpha.contiguous(), rot_interp)
    return rot_interp


def interpolate_poses(pose0: Tensor, pose1: Tensor, alpha: Tensor) -> Tensor:
    """
    Interpolate pose matrices (4x4 transformation matrices) using quaternion SLERP
    for rotation and linear interpolation for translation.

    Args:
        pose0 (torch.Tensor): Tensor of shape (..., N, 4, 4), initial pose matrices.
        pose1 (torch.Tensor): Tensor of shape (..., N, 4, 4), final pose matrices.
        alpha (torch.Tensor or float): Interpolation factors, scalar or tensor of shape (..., N).

    Returns:
        torch.Tensor: Interpolated pose matrices of shape (..., N, 4, 4).
    """
    if not pose0.is_cuda or not pose1.is_cuda or not alpha.is_cuda:
        raise ValueError("All input tensors must be CUDA tensors.")
    if pose1.shape != pose0.shape:
        raise ValueError("pose0 and pose1 must have the same shape.")
    if alpha.shape != pose0.shape[:-2]:
        raise ValueError("alpha must have shape matching poses batch dimension.")
    if pose0.shape[-2] == 3:
        pose0 = torch.cat([pose0, torch.zeros(*pose0.shape[:-2], 1, 4, device=pose0.device, dtype=pose0.dtype)], dim=-2)
        pose0[..., -1, -1] = 1.0
    if pose1.shape[-2] == 3:
        pose1 = torch.cat([pose1, torch.zeros(*pose1.shape[:-2], 1, 4, device=pose1.device, dtype=pose1.dtype)], dim=-2)
        pose1[..., -1, -1] = 1.0
    if pose0.shape[-2:] != (4, 4) or pose1.shape[-2:] != (4, 4):
        raise ValueError("pose0 and pose1 must be 4x4 matrices.")

    pose_interp = torch.empty_like(pose0)
    fast_interp_cuda.interpolate_poses(
        pose0.contiguous(), pose1.contiguous(), alpha.to(pose0.dtype).contiguous(), pose_interp
    )
    return pose_interp


def interpolate_trajectories(poses: Tensor, pose_times: Tensor, query_times: Tensor, clamp_frac: bool = True):
    """
    Interpolate trajectories (sequences of poses) with automatic CPU/CUDA dispatch.

    Args:
        poses (Tensor): Shape (..., N, 4, 4) pose matrices
        pose_times (Tensor): Shape (..., N) timestamps for poses
        query_times (Tensor): Shape (..., M) query timestamps
        clamp_frac (bool): Whether to clamp interpolation fraction to [0,1]

    Returns:
        Tensor: Shape (..., M, 4, 4) interpolated poses
    """
    if not torch.is_floating_point(poses):
        raise TypeError("poses must be a floating point tensor")
    if len({poses.device, pose_times.device, query_times.device}) > 1:
        raise ValueError("All input tensors must be on the same device")
    if pose_times.shape != poses.shape[:-2]:
        raise ValueError("pose_times must match poses batch and trajectory length dimensions")
    if query_times.shape[:-1] != poses.shape[:-3]:
        raise ValueError("query_times must match poses batch dimensions")
    if poses.shape[-2:] != (4, 4):
        raise ValueError("poses must be 4x4 matrices")

    poses, pose_times, query_times = poses.contiguous(), pose_times.contiguous(), query_times.contiguous()
    right_idx = torch.searchsorted(pose_times, query_times)
    pose_interp = torch.empty((*query_times.shape, 4, 4), device=poses.device, dtype=poses.dtype)
    pose_times, query_times = pose_times.to(poses), query_times.to(poses)

    if poses.is_cuda:
        fast_interp_cuda.interpolate_trajectories(poses, right_idx, pose_times, query_times, pose_interp, clamp_frac)
    else:
        fast_interp_cpu.interpolate_trajectories_cpu(poses, right_idx, pose_times, query_times, pose_interp, clamp_frac)

    return pose_interp


def interpolate_trajectories_np(
    poses: np.ndarray, pose_times: np.ndarray, query_times: np.ndarray, clamp_frac: bool = True
):
    """
    Interpolate trajectories (sequences of poses) using numpy.
    """
    return interpolate_trajectories(
        torch.from_numpy(poses), torch.from_numpy(pose_times), torch.from_numpy(query_times), clamp_frac
    ).numpy()
