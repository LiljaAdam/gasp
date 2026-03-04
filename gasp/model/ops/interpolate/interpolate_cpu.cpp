#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define EPSILON 1e-6

template<typename scalar_t>
inline void mat3x3_to_quat(const scalar_t* R, scalar_t* q) {
    scalar_t tr = R[0] + R[4] + R[8];

    if (tr > 0) {
        scalar_t S = sqrt(tr + 1.0) * 2;
        q[0] = 0.25 * S;
        q[1] = (R[7] - R[5]) / S;
        q[2] = (R[2] - R[6]) / S;
        q[3] = (R[3] - R[1]) / S;
    } else if ((R[0] > R[4]) && (R[0] > R[8])) {
        scalar_t S = sqrt(1.0 + R[0] - R[4] - R[8]) * 2;
        q[0] = (R[7] - R[5]) / S;
        q[1] = 0.25 * S;
        q[2] = (R[1] + R[3]) / S;
        q[3] = (R[2] + R[6]) / S;
    } else if (R[4] > R[8]) {
        scalar_t S = sqrt(1.0 + R[4] - R[0] - R[8]) * 2;
        q[0] = (R[2] - R[6]) / S;
        q[1] = (R[1] + R[3]) / S;
        q[2] = 0.25 * S;
        q[3] = (R[5] + R[7]) / S;
    } else {
        scalar_t S = sqrt(1.0 + R[8] - R[0] - R[4]) * 2;
        q[0] = (R[3] - R[1]) / S;
        q[1] = (R[2] + R[6]) / S;
        q[2] = (R[5] + R[7]) / S;
        q[3] = 0.25 * S;
    }
}

template<typename scalar_t>
inline void quat_to_mat3x3(const scalar_t* q, scalar_t* R) {
    scalar_t x = q[1], y = q[2], z = q[3], w = q[0];
    scalar_t x2 = x*x, y2 = y*y, z2 = z*z, w2 = w*w;

    R[0] = w2 + x2 - y2 - z2;
    R[1] = 2*(x*y - w*z);
    R[2] = 2*(x*z + w*y);

    R[3] = 2*(x*y + w*z);
    R[4] = w2 - x2 + y2 - z2;
    R[5] = 2*(y*z - w*x);

    R[6] = 2*(x*z - w*y);
    R[7] = 2*(y*z + w*x);
    R[8] = w2 - x2 - y2 + z2;
}

template<typename scalar_t>
inline void quat_slerp(const scalar_t* q0, const scalar_t* q1, scalar_t t, scalar_t* result) {
    scalar_t dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];

    scalar_t q1_adj[4] = {q1[0], q1[1], q1[2], q1[3]};
    if (dot < 0) {
        dot = -dot;
        for (int i = 0; i < 4; i++) q1_adj[i] = -q1_adj[i];
    }

    if (dot > 0.9995) {
        for (int i = 0; i < 4; i++) {
            result[i] = q0[i] + t*(q1_adj[i] - q0[i]);
        }
        scalar_t norm = 1.0/sqrt(result[0]*result[0] + result[1]*result[1] +
                             result[2]*result[2] + result[3]*result[3]);
        for (int i = 0; i < 4; i++) result[i] *= norm;
        return;
    }

    scalar_t theta_0 = acos(dot);
    scalar_t theta = theta_0 * t;
    scalar_t sin_theta = sin(theta);
    scalar_t sin_theta_0 = sin(theta_0);

    scalar_t s0 = cos(theta) - dot * sin_theta / sin_theta_0;
    scalar_t s1 = sin_theta / sin_theta_0;

    for (int i = 0; i < 4; i++) {
        result[i] = s0 * q0[i] + s1 * q1_adj[i];
    }
}

template<typename scalar_t>
void interpolate_single_pose_cpu(
    const scalar_t* P0,
    const scalar_t* P1,
    scalar_t alpha,
    scalar_t* P_out
) {
    // Extract rotations
    scalar_t R0[9], R1[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R0[i*3 + j] = P0[i*4 + j];
            R1[i*3 + j] = P1[i*4 + j];
        }
    }

    // Convert to quaternions and interpolate
    scalar_t q0[4], q1[4], q_interp[4], R_out[9];
    mat3x3_to_quat(R0, q0);
    mat3x3_to_quat(R1, q1);
    quat_slerp(q0, q1, alpha, q_interp);
    quat_to_mat3x3(q_interp, R_out);

    // Write rotation back
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P_out[i*4 + j] = R_out[i*3 + j];
        }
    }

    // Interpolate translation
    scalar_t beta = 1.0 - alpha;
    P_out[3] = beta * P0[3] + alpha * P1[3];
    P_out[7] = beta * P0[7] + alpha * P1[7];
    P_out[11] = beta * P0[11] + alpha * P1[11];

    // Set bottom row
    P_out[12] = 0.0;
    P_out[13] = 0.0;
    P_out[14] = 0.0;
    P_out[15] = 1.0;
}

template<typename scalar_t>
void interpolate_trajectories_cpu_impl(
    const scalar_t* poses,
    const int64_t* right_idx,
    const scalar_t* pose_times,
    const scalar_t* query_times,
    scalar_t* pose_interp,
    int batch_size,
    int N,
    int M,
    bool clamp_frac
) {
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int query_idx = 0; query_idx < M; query_idx++) {
            const scalar_t* batch_poses = poses + batch_idx * N * 16;
            const scalar_t* batch_times = pose_times + batch_idx * N;
            const scalar_t* batch_queries = query_times + batch_idx * M;
            const int64_t* batch_right_idx = right_idx + batch_idx * M;
            scalar_t* batch_out = pose_interp + batch_idx * M * 16;

            int right = std::min(std::max(batch_right_idx[query_idx], (int64_t)1), (int64_t)(N-1));
            int left = right - 1;

            scalar_t query_time = batch_queries[query_idx];
            scalar_t left_time = batch_times[left];
            scalar_t right_time = batch_times[right];
            scalar_t alpha = (query_time - left_time) / (right_time - left_time + EPSILON);
            if (clamp_frac) {
                alpha = std::min(std::max(alpha, scalar_t(0)), scalar_t(1));
            }

            const scalar_t* pose_left = batch_poses + left * 16;
            const scalar_t* pose_right = batch_poses + right * 16;
            scalar_t* pose_out = batch_out + query_idx * 16;

            interpolate_single_pose_cpu(pose_left, pose_right, alpha, pose_out);
        }
    }
}

void interpolate_trajectories_cpu(
    const torch::Tensor& poses,
    const torch::Tensor& right_idx,
    const torch::Tensor& pose_times,
    const torch::Tensor& query_times,
    torch::Tensor& pose_interp,
    bool clamp_frac
) {
    // Calculate dimensions
    auto batch_dims = poses.sizes().slice(0, poses.dim() - 3);
    int batch_size = 1;
    for (int64_t dim : batch_dims) {
        batch_size *= dim;
    }
    int N = poses.size(-3);  // Number of poses per trajectory
    int M = query_times.size(-1);  // Number of queries per trajectory

    AT_DISPATCH_FLOATING_TYPES(poses.scalar_type(), "interpolate_trajectories_cpu", ([&] {
        interpolate_trajectories_cpu_impl<scalar_t>(
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
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("interpolate_trajectories_cpu", &interpolate_trajectories_cpu,
          "Fast CPU implementation of trajectory interpolation",
            py::arg("poses"), py::arg("right_idx"), py::arg("pose_times"),
            py::arg("query_times"), py::arg("pose_interp"), py::arg("clamp_frac")
);
}
