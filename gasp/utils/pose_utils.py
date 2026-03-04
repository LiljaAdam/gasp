import math
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

_EPS = np.finfo(float).eps * 4.0


def transform_points(points, transform):
    """Transforms points by a transform."""
    points = points.clone()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    points[:, :3] = points[:, :3] @ rotations.swapaxes(-2, -1) + translations
    return points


def to4x4(pose: Tensor) -> Tensor:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def pose_inverse(pose: Tensor) -> Tensor:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def unit_vector(data: NDArray, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


# from https://github.com/naver/roma/blob/0da38a34f36c0c917c0a15ea36eee561c6096719/roma/mappings.py#L329
def rotmat_to_unitquat(R):
    """
    Converts rotation matrix to unit quaternion representation.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    batch_shape = R.shape[:-2]
    matrix = R.reshape((-1, 3, 3))
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return quat.reshape(batch_shape + (4,))


def unitquat_to_rotmat(quat):
    """
    Converts unit quaternion into rotation matrix representation.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = -x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = -x2 - y2 + z2 + w2
    return matrix


def quaternion_from_matrix(matrix: NDArray, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
        K = np.array(K)
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: NDArray, quat1: NDArray, fraction: float, spin: int = 0, shortestpath: bool = True
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


# adapted from https://github.com/naver/roma/blob/0da38a34f36c0c917c0a15ea36eee561c6096719/roma/utils.py#L333
def unitquat_slerp_fast(q0, q1, steps, shortest_arc=True):
    """
    Spherical linear interpolation between two unit quaternions.
    This function requires less computations than :func:`roma.utils.unitquat_slerp`,
    but is **unsuitable for extrapolation (i.e.** ``steps`` **must be within [0,1])**.

    Args:
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape A): interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 (A may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
    Returns:
        batch of interpolated quaternions (BxAx4 tensor).
    """
    batch_shape = q0.shape[:-1]
    q0 = q0.reshape((-1, 4))
    batch_shape1 = q1.shape[:-1]
    q1 = q1.reshape((-1, 4))

    assert batch_shape == batch_shape1
    assert batch_shape == steps.shape
    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(q0 * q1, dim=-1)
    if shortest_arc:
        # Flip some quaternions to perform shortest arc interpolation.
        q1 = q1.clone()
        q1[cos_omega < 0, :] = -q1[cos_omega < 0, :]
        cos_omega = torch.abs(cos_omega)
    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)
    nearby_quaternions_idx = nearby_quaternions.nonzero(as_tuple=True)[0]

    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - steps) * omega)
    beta = torch.sin(steps * omega)
    # Use linear interpolation for nearby quaternions
    alpha[nearby_quaternions_idx] = 1 - steps[nearby_quaternions_idx]
    beta[nearby_quaternions_idx] = steps[nearby_quaternions_idx]
    # Interpolation
    q = alpha.unsqueeze(-1) * q0 + beta.unsqueeze(-1) * q1
    # Normalization of the output
    q = q / torch.norm(q, dim=-1, keepdim=True)
    return q.reshape(batch_shape + (4,))


def quat_product(p, q):
    """
    Returns the product of two quaternions.

    Args:
        p, q (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L153

    vector = p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] + torch.cross(p[..., :3], q[..., :3], dim=-1)
    last = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], dim=-1)
    return torch.cat((vector, last[..., None]), dim=-1)


def unitquat_to_rotvec(quat, shortest_arc=True):
    """
    Converts unit quaternion into rotation vector representation.

    Based on the representation of a rotation of angle :math:`{\\theta}` and unit axis :math:`(x,y,z)`
    by the unit quaternions :math:`\pm [\sin({\\theta} / 2) (x i + y j + z k) + \cos({\\theta} / 2)]`.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
        shortest_arc (bool): if True, the function returns the smallest rotation vectors corresponding
            to the input 3D rotations, i.e. rotation vectors with a norm smaller than :math:`\pi`.
            If False, the function may return rotation vectors of norm larger than :math:`\pi`, depending on the sign of the input quaternions.
    Returns:
        batch of rotation vectors (...x3 tensor).
    Note:
        Behavior is undefined for inputs ``quat=torch.as_tensor([0.0, 0.0, 0.0, -1.0])`` and ``shortest_arc=False``,
        as any rotation vector of angle :math:`2 \pi` could be a valid representation in such case.
    """
    batch_shape = quat.shape[:-1]
    quat = quat.reshape((-1, 4))
    # We perform a copy to support auto-differentiation.
    quat = quat.clone()
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L1006-L1073
    if shortest_arc:
        # Enforce w > 0 to ensure 0 <= angle <= pi.
        # (Otherwise angle can be arbitrary within ]-2pi, 2pi]).
        quat[quat[:, 3] < 0] *= -1
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    angle = 2 * half_angle
    small_angle = torch.abs(angle) <= 1e-3
    large_angle = ~small_angle

    num_rotations = len(quat)
    scale = torch.empty(num_rotations, dtype=quat.dtype, device=quat.device)
    scale[small_angle] = 2 + angle[small_angle] ** 2 / 12 + 7 * angle[small_angle] ** 4 / 2880
    scale[large_angle] = angle[large_angle] / torch.sin(half_angle[large_angle])

    rotvec = scale[:, None] * quat[:, :3]
    return rotvec.reshape(batch_shape + (3,))


def rotvec_to_unitquat(rotvec):
    """
    Converts rotation vector into unit quaternion representation.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    batch_shape = rotvec.shape[:-1]
    rotvec = rotvec.reshape((-1, 3))
    num_rotations, D = rotvec.shape
    assert D == 3, "Input should be a Bx3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L621

    norms = torch.norm(rotvec, dim=-1)
    small_angle = norms <= 1e-3
    large_angle = ~small_angle

    scale = torch.empty((num_rotations,), device=rotvec.device, dtype=rotvec.dtype)
    scale[small_angle] = 0.5 - norms[small_angle] ** 2 / 48 + norms[small_angle] ** 4 / 3840
    scale[large_angle] = torch.sin(norms[large_angle] / 2) / norms[large_angle]

    quat = torch.empty((num_rotations, 4), device=rotvec.device, dtype=rotvec.dtype)
    quat[:, :3] = scale[:, None] * rotvec
    quat[:, 3] = torch.cos(norms / 2)
    return quat.reshape(batch_shape + (4,))


def unitquat_slerp(q0, q1, steps, shortest_arc=True):
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q0, q1 (Ax4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape A): interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (A may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.
    Returns:
        batch of interpolated quaternions (Ax4 tensor).
    Note:
        When considering quaternions as rotation representations,
        one should keep in mind that spherical interpolation is not necessarily performed along the shortest arc,
        depending on the sign of ``torch.sum(q0*q1,dim=-1)``.

        Behavior is undefined when using ``shortest_arc=False`` with antipodal quaternions.
    """
    # Relative rotation
    q0_conj = q0.clone()
    q0_conj[..., :3] *= -1
    rel_q = quat_product(q0_conj, q1)
    rel_rotvec = unitquat_to_rotvec(rel_q, shortest_arc=shortest_arc)
    # Relative rotations to apply
    rel_rotvecs = steps.reshape(steps.shape + (1,)) * rel_rotvec
    rots = rotvec_to_unitquat(rel_rotvecs.reshape(-1, 3)).reshape(*rel_rotvecs.shape[:-1], 4)
    interpolated_q = quat_product(q0, rots.to(q0))
    return interpolated_q


def quaternion_matrix(quaternion: NDArray) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def interpolate_trajectories(poses, pose_times, query_times, pose_valid_mask=None, clamp_frac=True):
    """Interpolate trajectory poses at query times using linear interpolation.

    Args:
        poses: A collection of poses to be interpolated N_times x N_trajectories.
        pose_times: The timestamps of the poses N_times.
        query_times: The timestamps to interpolate the poses at N_queries.
        pose_valid_mask: A mask indicating which poses are valid N_times x N_trajectories.

    Returns:
        The interpolated poses at the query times (M x 3 x 4)
        The indices of the queries used for interpolation (M).
        The indices of the tractories used for interpolation (M).
    """
    assert len(poses.shape) == 4, "Poses must be of shape [num_poses, num_actors, 3, 4]"
    # bugs are crawling like crazy if we only have one query time, fix with maybe squeeze
    qt = query_times if query_times.shape[-1] == 1 else query_times.squeeze()
    right_idx = torch.searchsorted(pose_times, qt)
    right_idx = right_idx.clamp(min=1, max=len(pose_times) - 1)
    left_idx = right_idx - 1

    # Compute the fraction between the left (previous) and right (after) timestamps
    right_time = pose_times[right_idx]
    left_time = pose_times[left_idx]
    time_diff = right_time - left_time + 1e-6
    fraction = (qt - left_time) / time_diff  # 0 = all left, 1 = all right
    if clamp_frac:
        fraction = fraction.clamp(0.0, 1.0)  # clamp to handle out of bounds

    if pose_valid_mask is None:
        pose_valid_mask = torch.ones_like(poses[..., 0, 0], dtype=torch.bool)
    trajs_to_sample = pose_valid_mask[left_idx] | pose_valid_mask[right_idx]  # [num_queries, n_trajs]
    query_idxs, object_idxs = torch.where(trajs_to_sample)  # 2 x [num_queries*n_valid_trajs]

    quat = rotmat_to_unitquat(poses[..., :3, :3])
    quat_left = quat[left_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    quat_right = quat[right_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    interp_fn = unitquat_slerp if (fraction < 0).any() or (fraction > 1).any() else unitquat_slerp_fast
    interp_quat = interp_fn(quat_left, quat_right, fraction[query_idxs])
    interp_rot = unitquat_to_rotmat(interp_quat)

    pos_left = poses[left_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    pos_right = poses[right_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    interp_pos = pos_left + (pos_right - pos_left) * fraction[query_idxs].unsqueeze(-1)

    interpolated_poses = torch.cat([interp_rot, interp_pos.unsqueeze(-1)], dim=-1)
    return interpolated_poses, query_idxs, object_idxs
