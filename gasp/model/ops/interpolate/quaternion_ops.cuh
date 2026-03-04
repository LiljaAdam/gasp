# pragma once

#include <cuda.h>
#include <cuda_runtime.h>


#define DOT_THRESHOLD 0.999

template<typename scalar_t>
__device__ __forceinline__ void quat_normalize(scalar_t *q) {
    scalar_t norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    q[0] /= norm;
    q[1] /= norm;
    q[2] /= norm;
    q[3] /= norm;
}

template<typename scalar_t>
__device__ __forceinline__ void mat3x3_to_quat(const scalar_t *R, scalar_t *q) {
    // Convert a rotation matrix to a quaternion (w, x, y, z)
    scalar_t trace = R[0] + R[4] + R[8];
    if (trace > 0.0) {
        scalar_t s = sqrt(trace + 1.0) * 2.0;
        q[0] = 0.25 * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if ((R[0] > R[4]) && (R[0] > R[8])) {
        scalar_t s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        scalar_t s = sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        scalar_t s = sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25 * s;
    }
    quat_normalize(q);
}

template<typename scalar_t>
__device__ __forceinline__ void quat_slerp(const scalar_t *q0, const scalar_t *q1, scalar_t alpha, scalar_t *q_out) {
    // Perform quaternion SLERP interpolation
    scalar_t dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];

    scalar_t q1_copy[4];
    if (dot < 0.0) {
        dot = -dot;
        q1_copy[0] = -q1[0];
        q1_copy[1] = -q1[1];
        q1_copy[2] = -q1[2];
        q1_copy[3] = -q1[3];
    } else {
        q1_copy[0] = q1[0];
        q1_copy[1] = q1[1];
        q1_copy[2] = q1[2];
        q1_copy[3] = q1[3];
    }

    if (dot > DOT_THRESHOLD) {
        // Linear interpolation for very close quaternions
        q_out[0] = q0[0] + alpha * (q1_copy[0] - q0[0]);
        q_out[1] = q0[1] + alpha * (q1_copy[1] - q0[1]);
        q_out[2] = q0[2] + alpha * (q1_copy[2] - q0[2]);
        q_out[3] = q0[3] + alpha * (q1_copy[3] - q0[3]);
        quat_normalize(q_out);
        return;
    }

    scalar_t theta = acos(dot);
    scalar_t sin_theta = sin(theta);

    scalar_t s0 = sin((1.0 - alpha) * theta) / sin_theta;
    scalar_t s1 = sin(alpha * theta) / sin_theta;

    q_out[0] = s0 * q0[0] + s1 * q1_copy[0];
    q_out[1] = s0 * q0[1] + s1 * q1_copy[1];
    q_out[2] = s0 * q0[2] + s1 * q1_copy[2];
    q_out[3] = s0 * q0[3] + s1 * q1_copy[3];

    quat_normalize(q_out);
}

template<typename scalar_t>
__device__ __forceinline__ void quat_to_mat3x3(const scalar_t *q, scalar_t *R) {
    // Convert a quaternion to a rotation matrix
    scalar_t w = q[0];
    scalar_t x = q[1];
    scalar_t y = q[2];
    scalar_t z = q[3];

    scalar_t ww = w * w;
    scalar_t xx = x * x;
    scalar_t yy = y * y;
    scalar_t zz = z * z;

    scalar_t wx = w * x;
    scalar_t wy = w * y;
    scalar_t wz = w * z;

    scalar_t xy = x * y;
    scalar_t xz = x * z;
    scalar_t yz = y * z;

    R[0] = ww + xx - yy - zz;
    R[1] = 2.0 * (xy - wz);
    R[2] = 2.0 * (xz + wy);

    R[3] = 2.0 * (xy + wz);
    R[4] = ww - xx + yy - zz;
    R[5] = 2.0 * (yz - wx);

    R[6] = 2.0 * (xz - wy);
    R[7] = 2.0 * (yz + wx);
    R[8] = ww - xx - yy + zz;
}
