// Adapted from
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal
// Copyright © 2024 Apple Inc.

#include <ATen/native/mps/kernels/RMSNorm.h>
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Keep x * inv_mean * weight in the accumulate type and cast once, matching the
// CPU composite -- casting before the weight multiply loses precision at half
// precision (#147203).
template <typename T>
inline T rms_norm_apply(T x, opmath_t<T> inv, T w) {
  using op_T = opmath_t<T>;
  return static_cast<T>((static_cast<op_T>(x) * inv) * static_cast<op_T>(w));
}

template <typename T>
[[kernel]] void rms_single_row(
    constant T* x,
    constant T* w,
    device T* out,
    device float* rstd,
    constant float& eps,
    constant uint& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float local_sums[simdgroup_size];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  const float inv_mean = metal::precise::rsqrt(
      threadgroup_sum(local_sums, acc, lid, lsize) / axis_size + eps);
  if (lid == 0) {
    rstd[gid] = inv_mean;
  }

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      out[i] = rms_norm_apply(x[i], inv_mean, w[i]);
    }
  } else {
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = rms_norm_apply(x[i], inv_mean, w[i]);
      }
    }
  }
}

template <typename T>
[[kernel]] void rms_looped(
    constant T* x,
    constant T* w,
    device T* out,
    device float* rstd,
    constant float& eps,
    constant uint& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float local_sums[simdgroup_size];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  const float inv_mean = metal::precise::rsqrt(
      threadgroup_sum(local_sums, acc, lid, lsize) / axis_size + eps);
  if (lid == 0) {
    rstd[gid] = inv_mean;
  }

  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        out[r + i] = rms_norm_apply(x[r + i], inv_mean, w[i + r]);
      }
    } else {
#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = rms_norm_apply(x[r + i], inv_mean, w[i + r]);
        }
      }
    }
  }
}

// grad_x[i] = w[i] * grad_out[i] * rstd
//             - x[i] * rstd^3 * sum_j(grad_out[j] * w[j] * x[j]) / axis_size
// grad_w[i] = sum over rows of grad_out[i] * x[i] * rstd
//
// grad_w needs a reduction across rows, so each threadgroup emits the partial
// sum for the rows it owns and the host reduces the partials.
template <typename T>
[[kernel]] void rms_backward_single_row(
    constant T* x,
    constant T* w,
    constant T* dy,
    constant float* rstd,
    device T* dx,
    device float* dw_partial,
    constant uint& axis_size,
    constant uint& num_rows,
    constant uint& compute_dx,
    constant uint& compute_dw,
    uint gid [[threadgroup_position_in_grid]],
    uint num_row_blocks [[threadgroups_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float local_sums[simdgroup_size];

  const uint col = lid * N_READS;
  float dw_acc[N_READS] = {};

  for (uint row = gid; row < num_rows; row += num_row_blocks) {
    const size_t offset = row * size_t(axis_size) + col;
    const float r = rstd[row];

    float xi[N_READS];
    float gi[N_READS] = {};
    float acc = 0;
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      const bool inbounds = col + i < axis_size;
      const float dyi = inbounds ? static_cast<float>(dy[offset + i]) : 0.0f;
      xi[i] = inbounds ? static_cast<float>(x[offset + i]) : 0.0f;
      if (compute_dx) {
        gi[i] = dyi * (inbounds ? static_cast<float>(w[col + i]) : 0.0f);
        acc += gi[i] * xi[i];
      }
      if (compute_dw) {
        // Accumulated here rather than with dx below, which would need a second
        // load of dy: only dx depends on the threadgroup reduction.
        dw_acc[i] += dyi * xi[i] * r;
      }
    }
    // compute_dx/compute_dw are uniform, so these branches never diverge. Metal
    // function constants would specialise them away at pipeline-creation time
    // and free the registers the dead path holds, which is likely worth some
    // performance -- nothing under mps/kernels/ uses them yet.
    if (compute_dx) {
      const float scale =
          threadgroup_sum(local_sums, acc, lid, lsize) * r * r * r / axis_size;

#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        if (col + i < axis_size) {
          dx[offset + i] = static_cast<T>(gi[i] * r - xi[i] * scale);
        }
      }
      // threadgroup_sum barriers before returning its scratch, not after, so
      // the next row must not overwrite it until every thread has read it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (compute_dw) {
#pragma unroll
    for (uint i = 0; i < N_READS; i++) {
      if (col + i < axis_size) {
        dw_partial[gid * size_t(axis_size) + col + i] = dw_acc[i];
      }
    }
  }
}

// A thread cannot hold its grad_w partials in registers at this axis_size, so
// it accumulates them straight into dw_partial, which the host zero-fills.
// Within a threadgroup each column is owned by exactly one thread, so a block
// can still span several rows without racing on that accumulation.
template <typename T>
[[kernel]] void rms_backward_looped(
    constant T* x,
    constant T* w,
    constant T* dy,
    constant float* rstd,
    device T* dx,
    device float* dw_partial,
    constant uint& axis_size,
    constant uint& num_rows,
    constant uint& compute_dx,
    constant uint& compute_dw,
    uint gid [[threadgroup_position_in_grid]],
    uint num_row_blocks [[threadgroups_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float local_sums[simdgroup_size];

  const uint col = lid * N_READS;
  const size_t partial_offset = gid * size_t(axis_size);

  for (uint row = gid; row < num_rows; row += num_row_blocks) {
    const size_t row_offset = row * size_t(axis_size);
    const float r = rstd[row];

    float acc = 0;
    for (uint base = 0; compute_dx && base < axis_size;
         base += lsize * N_READS) {
      if (base + col + N_READS <= axis_size) {
#pragma unroll
        for (uint i = 0; i < N_READS; i++) {
          const uint c = base + col + i;
          acc += static_cast<float>(dy[row_offset + c]) *
              static_cast<float>(w[c]) * static_cast<float>(x[row_offset + c]);
        }
      } else {
#pragma unroll
        for (uint i = 0; i < N_READS; i++) {
          const uint c = base + col + i;
          if (c < axis_size) {
            acc += static_cast<float>(dy[row_offset + c]) *
                static_cast<float>(w[c]) *
                static_cast<float>(x[row_offset + c]);
          }
        }
      }
    }
    const float scale = compute_dx
        ? threadgroup_sum(local_sums, acc, lid, lsize) * r * r * r / axis_size
        : 0.0f;

    for (uint base = 0; base < axis_size; base += lsize * N_READS) {
#pragma unroll
      for (uint i = 0; i < N_READS; i++) {
        const uint c = base + col + i;
        if (c < axis_size) {
          const float xi = static_cast<float>(x[row_offset + c]);
          const float dyi = static_cast<float>(dy[row_offset + c]);
          if (compute_dx) {
            dx[row_offset + c] =
                static_cast<T>(dyi * static_cast<float>(w[c]) * r - xi * scale);
          }
          if (compute_dw) {
            dw_partial[partial_offset + c] += dyi * xi * r;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The host sums the per-block grad_w partials with this rather than at::sum,
// whose MPSGraph cache is keyed on tensor shape: n_row_blocks varies with the
// input, so at::sum would compile and retain a graph per distinct shape.
template <typename T>
[[kernel]] void rms_backward_reduce_partials(
    constant float* dw_partial,
    device T* dw,
    constant uint& axis_size,
    constant uint& num_blocks,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[REDUCE_SLICES][REDUCE_COLS];

  const uint col = gid.x;
  float acc = 0;
  if (col < axis_size) {
    for (uint b = lid.y; b < num_blocks; b += REDUCE_SLICES) {
      acc += dw_partial[b * size_t(axis_size) + col];
    }
  }
  scratch[lid.y][lid.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid.y == 0 && col < axis_size) {
    float total = 0;
#pragma unroll
    for (uint s = 0; s < REDUCE_SLICES; s++) {
      total += scratch[s][lid.x];
    }
    dw[col] = static_cast<T>(total);
  }
}

// clang-format off
#define instantiate_rms_single_row(itype)                     \
  template [[host_name("rms_norm_" #itype)]] [[kernel]] void  \
  rms_single_row<itype>(                                      \
      constant itype* x,                                      \
      constant itype* w,                                      \
      device itype* out,                                      \
      device float* rstd,                                     \
      constant float& eps,                                    \
      constant uint& axis_size,                               \
      uint gid [[threadgroup_position_in_grid]],              \
      uint lid [[thread_position_in_threadgroup]],            \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_rms_looped(itype)                               \
  template [[host_name("rms_norm_looped_" #itype)]] [[kernel]] void \
  rms_looped<itype>(                                                \
      constant itype* x,                                            \
      constant itype* w,                                            \
      device itype* out,                                            \
      device float* rstd,                                           \
      constant float& eps,                                          \
      constant uint& axis_size,                                     \
      uint gid [[threadgroup_position_in_grid]],                    \
      uint lid [[thread_position_in_threadgroup]],                  \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_rms_backward_single_row(itype)                     \
  template [[host_name("rms_norm_backward_" #itype)]] [[kernel]] void  \
  rms_backward_single_row<itype>(                                      \
      constant itype* x,                                               \
      constant itype* w,                                               \
      constant itype* dy,                                              \
      constant float* rstd,                                            \
      device itype* dx,                                                \
      device float* dw_partial,                                        \
      constant uint& axis_size,                                        \
      constant uint& num_rows,                                         \
      constant uint& compute_dx,                                       \
      constant uint& compute_dw,                                       \
      uint gid [[threadgroup_position_in_grid]],                       \
      uint num_row_blocks [[threadgroups_per_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],                     \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_rms_backward_looped(itype)                                \
  template [[host_name("rms_norm_backward_looped_" #itype)]] [[kernel]] void  \
  rms_backward_looped<itype>(                                                 \
      constant itype* x,                                                      \
      constant itype* w,                                                      \
      constant itype* dy,                                                     \
      constant float* rstd,                                                   \
      device itype* dx,                                                       \
      device float* dw_partial,                                               \
      constant uint& axis_size,                                               \
      constant uint& num_rows,                                                \
      constant uint& compute_dx,                                              \
      constant uint& compute_dw,                                              \
      uint gid [[threadgroup_position_in_grid]],                              \
      uint num_row_blocks [[threadgroups_per_grid]],                          \
      uint lid [[thread_position_in_threadgroup]],                            \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_rms_reduce_partials(itype)                                       \
  template [[host_name("rms_norm_reduce_partials_" #itype)]] [[kernel]] void          \
  rms_backward_reduce_partials<itype>(                                                \
      constant float* dw_partial,                                                     \
      device itype* dw,                                                               \
      constant uint& axis_size,                                                       \
      constant uint& num_blocks,                                                      \
      uint2 gid [[thread_position_in_grid]],                                          \
      uint2 lid [[thread_position_in_threadgroup]]);

#define instantiate_rms(itype)                 \
  instantiate_rms_single_row(itype)            \
  instantiate_rms_looped(itype)                \
  instantiate_rms_backward_single_row(itype)   \
  instantiate_rms_backward_looped(itype)      \
  instantiate_rms_reduce_partials(itype)

instantiate_rms(float)
instantiate_rms(half)
instantiate_rms(bfloat)
