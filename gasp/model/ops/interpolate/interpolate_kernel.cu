#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quaternion_ops.cuh"

#define BLOCK_SIZE 256

template<typename scalar_t>
__global__ void interpolate_rotations_kernel(
    const scalar_t* __restrict__ rot0,
    const scalar_t* __restrict__ rot1,
    const scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ rot_interp,
    int N,
    bool alpha_is_scalar
) {
    // Each thread handles one rotation
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t a = alpha_is_scalar ? alpha[0] : alpha[idx];

    // Load rot0 and rot1 with stride 9 (3x3 matrices)
    const scalar_t *R0 = rot0 + idx * 9;
    const scalar_t *R1 = rot1 + idx * 9;

    // Convert rot0 and rot1 to quaternions
    scalar_t q0[4];
    scalar_t q1[4];
    mat3x3_to_quat<scalar_t>(R0, q0);
    mat3x3_to_quat<scalar_t>(R1, q1);

    // Perform SLERP interpolation
    scalar_t q_interp[4];
    quat_slerp<scalar_t>(q0, q1, a, q_interp);

    // Convert interpolated quaternion back to rotation matrix
    scalar_t *R_out = rot_interp + idx * 9;
    quat_to_mat3x3<scalar_t>(q_interp, R_out);
}

template<typename scalar_t>
__device__ void interpolate_pose_pair(
    const scalar_t* P0,  // left pose
    const scalar_t* P1,  // right pose
    scalar_t alpha,      // interpolation factor
    scalar_t* P_out      // output pose
) {
    // Extract the 3x3 rotation matrices (top-left block)
    scalar_t R0[9], R1[9];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R0[i*3 + j] = P0[i*4 + j];
            R1[i*3 + j] = P1[i*4 + j];
        }
    }

    // Interpolate rotation using quaternion slerp
    scalar_t q0[4], q1[4], q_interp[4], R_out[9];
    mat3x3_to_quat<scalar_t>(R0, q0);
    mat3x3_to_quat<scalar_t>(R1, q1);
    quat_slerp<scalar_t>(q0, q1, alpha, q_interp);
    quat_to_mat3x3<scalar_t>(q_interp, R_out);

    // Write rotation back to pose matrix with correct stride
    P_out[0] = R_out[0];  P_out[1] = R_out[1];  P_out[2] = R_out[2];
    P_out[4] = R_out[3];  P_out[5] = R_out[4];  P_out[6] = R_out[5];
    P_out[8] = R_out[6];  P_out[9] = R_out[7];  P_out[10] = R_out[8];

    // Linear interpolation for translation
    P_out[3] = (1.0 - alpha) * P0[3] + alpha * P1[3];
    P_out[7] = (1.0 - alpha) * P0[7] + alpha * P1[7];
    P_out[11] = (1.0 - alpha) * P0[11] + alpha * P1[11];

    // Set bottom row to [0, 0, 0, 1]
    P_out[12] = 0.0;
    P_out[13] = 0.0;
    P_out[14] = 0.0;
    P_out[15] = 1.0;
}

template<typename scalar_t>
__global__ void interpolate_poses_kernel(
    const scalar_t* __restrict__ pose0,
    const scalar_t* __restrict__ pose1,
    const scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ pose_interp,
    int N,
    bool alpha_is_scalar
) {
    // Each thread handles one pose
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t a = alpha_is_scalar ? alpha[0] : alpha[idx];

    // Load poses with stride 16 (4x4 matrices)
    const scalar_t *P0 = pose0 + idx * 16;
    const scalar_t *P1 = pose1 + idx * 16;
    scalar_t *P_out = pose_interp + idx * 16;

    interpolate_pose_pair(P0, P1, a, P_out);
}

template<typename scalar_t>
__global__ void interpolate_trajectories_kernel(
    const scalar_t* __restrict__ poses,      // [..., N, 4, 4]
    const int64_t* __restrict__ right_idx, // [..., M]
    const scalar_t* __restrict__ pose_times, // [..., N]
    const scalar_t* __restrict__ query_times,// [..., M]
    scalar_t* __restrict__ pose_interp,      // [..., M, 4, 4]
    int batch_size,                        // Product of batch dimensions
    int N,                                 // Number of poses per trajectory
    int M,                                 // Number of query times
    bool clamp_frac                        // Whether to clamp the interpolation fraction to [0, 1]
) {
    // Each thread handles one query time
    int batch_idx = blockIdx.x;
    int query_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || query_idx >= M) return;

    // Get base pointers for this batch
    const scalar_t* batch_poses = poses + batch_idx * N * 16;  // 4x4 matrix
    const scalar_t* batch_times = pose_times + batch_idx * N;
    const scalar_t* batch_queries = query_times + batch_idx * M;
    const int64_t* batch_right_idx = right_idx + batch_idx * M;
    scalar_t* batch_out = pose_interp + batch_idx * M * 16;

    // Get indices for this query
    int right = batch_right_idx[query_idx];
    right = min(max(right, 1), N - 1);  // Clamp to valid range
    int left = right - 1;

    // Compute interpolation fraction
    scalar_t query_time = batch_queries[query_idx];
    scalar_t left_time = batch_times[left];
    scalar_t right_time = batch_times[right];
    scalar_t alpha = (query_time - left_time) / (right_time - left_time + 1e-6);
    if (clamp_frac) alpha = min(max(alpha, 0.0), 1.0);

    // Get pose pointers
    const scalar_t* pose_left = batch_poses + left * 16;
    const scalar_t* pose_right = batch_poses + right * 16;
    scalar_t* pose_out = batch_out + query_idx * 16;

    interpolate_pose_pair(
        pose_left,
        pose_right,
        alpha,
        pose_out
    );
}

template<typename scalar_t>
void interpolate_rotations_launcher_impl(
    const at::Tensor& rot0,
    const at::Tensor& rot1,
    const at::Tensor& alpha,
    at::Tensor& rot_interp
) {
    const int N = rot0.size(0);  // Number of rotations to interpolate
    bool alpha_is_scalar = (alpha.numel() == 1);
    const int threads = BLOCK_SIZE;
    const int blocks = (N + threads - 1) / threads;

    interpolate_rotations_kernel<scalar_t><<<blocks, threads>>>(
        rot0.data_ptr<scalar_t>(),
        rot1.data_ptr<scalar_t>(),
        alpha.data_ptr<scalar_t>(),
        rot_interp.data_ptr<scalar_t>(),
        N,
        alpha_is_scalar
    );
}

void interpolate_rotations_launcher(
    const at::Tensor& rot0,
    const at::Tensor& rot1,
    const at::Tensor& alpha,
    at::Tensor& rot_interp
) {
    AT_DISPATCH_FLOATING_TYPES(rot0.scalar_type(), "interpolate_rotations_cuda", ([&] {
        interpolate_rotations_launcher_impl<scalar_t>(rot0, rot1, alpha, rot_interp);
    }));
}

template<typename scalar_t>
void interpolate_poses_launcher_impl(
    const at::Tensor& pose0,
    const at::Tensor& pose1,
    const at::Tensor& alpha,
    at::Tensor& pose_interp
) {
    int N = pose0.numel() / 16;
    bool alpha_is_scalar = (alpha.numel() == 1);

    const int threads = BLOCK_SIZE;
    const int blocks = (N + threads - 1) / threads;

    interpolate_poses_kernel<scalar_t><<<blocks, threads>>>(
        pose0.data_ptr<scalar_t>(),
        pose1.data_ptr<scalar_t>(),
        alpha.data_ptr<scalar_t>(),
        pose_interp.data_ptr<scalar_t>(),
        N,
        alpha_is_scalar
    );
}

void interpolate_poses_launcher(
    const at::Tensor& pose0,
    const at::Tensor& pose1,
    const at::Tensor& alpha,
    at::Tensor& pose_interp
) {
    AT_DISPATCH_FLOATING_TYPES(pose0.scalar_type(), "interpolate_poses_cuda", ([&] {
        interpolate_poses_launcher_impl<scalar_t>(pose0, pose1, alpha, pose_interp);
    }));
}

template<typename scalar_t>
void interpolate_trajectories_launcher_impl(
    const at::Tensor& poses,
    const at::Tensor& right_idx,
    const at::Tensor& pose_times,
    const at::Tensor& query_times,
    at::Tensor& pose_interp,
    const bool clamp_frac
) {
    // Calculate dimensions
    auto batch_dims = poses.sizes().slice(0, poses.dim() - 3);
    int batch_size = 1;
    for (int64_t dim : batch_dims) {
        batch_size *= dim;
    }
    int N = poses.size(-3);
    int M = query_times.size(-1);

    const int threads = 256;
    const int blocks_y = (M + threads - 1) / threads;
    dim3 blocks(batch_size, blocks_y);

    interpolate_trajectories_kernel<scalar_t><<<blocks, threads>>>(
        poses.data_ptr<scalar_t>(),
        right_idx.data_ptr<int64_t>(),
        pose_times.data_ptr<scalar_t>(),
        query_times.data_ptr<scalar_t>(),
        pose_interp.data_ptr<scalar_t>(),
        batch_size,
        N,
        M,
        clamp_frac
    );
}

void interpolate_trajectories_launcher(
    const at::Tensor& poses,
    const at::Tensor& right_idx,
    const at::Tensor& pose_times,
    const at::Tensor& query_times,
    at::Tensor& pose_interp,
    const bool clamp_frac
) {
    AT_DISPATCH_FLOATING_TYPES(poses.scalar_type(), "interpolate_trajectories_cuda", ([&] {
        interpolate_trajectories_launcher_impl<scalar_t>(
            poses, right_idx, pose_times, query_times, pose_interp, clamp_frac
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("interpolate_rotations", &interpolate_rotations_launcher, "Interpolate Rotations Kernel using Quaternion SLERP");
    m.def("interpolate_poses", &interpolate_poses_launcher, "Interpolate Poses Kernel");
    m.def("interpolate_trajectories", &interpolate_trajectories_launcher, "Interpolate Trajectories Kernel",
        py::arg("poses"), py::arg("right_idx"), py::arg("pose_times"),
        py::arg("query_times"), py::arg("pose_interp"), py::arg("clamp_frac"));
}