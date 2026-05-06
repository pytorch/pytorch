#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

template <typename T>
kernel void layer_norm_single_row(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant uint& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;

  // each threadgroup handles one full "row" of length axis_size
  uint row_offset = tg_id * axis_size;
  device T* x = input + row_offset + tid * N_READS;
  device T* out = output + row_offset + tid * N_READS;

  // partial sums for calculating mean & variance
  float partial_sum = 0.0f;
  float partial_sum_sq = 0.0f;
  uint base_lane = tid * N_READS;
  if (base_lane + N_READS <= axis_size) {
    float4 v4 = float4(x[0], x[1], x[2], x[3]);
    partial_sum = v4.x + v4.y + v4.z + v4.w;
    partial_sum_sq = dot(v4, v4);
  } else {
    int remaining = axis_size - base_lane;
    if (remaining >= 3) {
      float3 v3 = float3(x[0], x[1], x[2]);
      partial_sum = v3.x + v3.y + v3.z;
      partial_sum_sq = dot(v3, v3);
    } else if (remaining >= 2) {
      float2 v2 = float2(x[0], x[1]);
      partial_sum = v2.x + v2.y;
      partial_sum_sq = dot(v2, v2);
    } else if (remaining >= 1) {
      float v = x[0];
      partial_sum = v;
      partial_sum_sq = fma(v, v, partial_sum_sq);
    }
  }

  // threadgroup-wide reduction
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float shared_sum_sq[simdgroup_size];
  float sum = c10::metal::threadgroup_sum(shared_sum, partial_sum, tid, tptg);
  float sum_sq = c10::metal::threadgroup_sum(shared_sum_sq, partial_sum_sq, tid, tptg);
  float mean = sum / float(axis_size);
  float var = sum_sq / float(axis_size) - mean * mean;
  var = var < 1e-6 ? 0.0f : var;
  float inv_std = metal::precise::rsqrt(var + epsilon);

  // normalize and optional scale & shift
  if (base_lane + N_READS <= axis_size) {
#pragma unroll
    for (int i = 0; i < N_READS; i++) {
      float v = float(x[i]);
      float norm = (v - mean) * inv_std;
      uint lane_idx = base_lane + i;
      if (use_weight)
        norm *= float(weight[lane_idx]);
      if (use_bias)
        norm += float(bias[lane_idx]);
      out[i] = static_cast<T>(norm);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N_READS; i++) {
      uint lane_idx = base_lane + i;
      if (lane_idx < axis_size) {
        float v = float(x[i]);
        float norm = (v - mean) * inv_std;
        if (use_weight)
          norm *= float(weight[lane_idx]);
        if (use_bias)
          norm += float(bias[lane_idx]);
        out[i] = static_cast<T>(norm);
      }
    }
  }

  if (tid == 0) {
    meanOut[tg_id] = static_cast<T>(mean);
    rstdTensor[tg_id] = static_cast<T>(inv_std);
  }
}

template <typename T>
kernel void layer_norm_looped(
    device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* meanOut [[buffer(2)]],
    device T* rstdTensor [[buffer(3)]],
    constant uint& axis_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant int& use_weight [[buffer(6)]],
    constant int& use_bias [[buffer(7)]],
    device T* weight [[buffer(8)]],
    device T* bias [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;

  uint row_offset = tg_id * axis_size;
  device T* x = input + row_offset;
  device T* out = output + row_offset;

  float partial_sum = 0.0f;
  float partial_sum_sq = 0.0f;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
      float4 xi4 = float4(x[base], x[base + 1], x[base + 2], x[base + 3]);
      partial_sum += xi4.x + xi4.y + xi4.z + xi4.w;
      partial_sum_sq += dot(xi4, xi4);
    } else {
      int remaining = axis_size - base;
      if (remaining >= 3) {
        float3 v3 = float3(x[base], x[base + 1], x[base + 2]);
        partial_sum += v3.x + v3.y + v3.z;
        partial_sum_sq += dot(v3, v3);
      } else if (remaining >= 2) {
        float2 v2 = float2(x[base], x[base + 1]);
        partial_sum += v2.x + v2.y;
        partial_sum_sq += dot(v2, v2);
      } else if (remaining >= 1) {
        float v = x[base];
        partial_sum += v;
        partial_sum_sq = fma(v, v, partial_sum_sq);
      }
    }
  }

  // threadgroup-wide reduction
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float shared_sum_sq[simdgroup_size];
  float sum = c10::metal::threadgroup_sum(shared_sum, partial_sum, tid, lsize);
  float sum_sq = c10::metal::threadgroup_sum(shared_sum_sq, partial_sum_sq, tid, lsize);
  float mean = sum / float(axis_size);
  float var = sum_sq / float(axis_size) - mean * mean;
  var = var < 1e-6 ? 0.0f : var;
  float inv_std = metal::precise::rsqrt(var + epsilon);

  // write back normalized + scale/shift if needed
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= axis_size) {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        float xi = float(x[base + i]);
        float norm = (xi - mean) * inv_std;
        if (use_weight)
          norm *= float(weight[base + i]);
        if (use_bias)
          norm += float(bias[base + i]);
        out[base + i] = T(norm);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_READS; i++) {
        if (base + i < axis_size) {
          float xi = float(x[base + i]);
          float norm = (xi - mean) * inv_std;
          if (use_weight)
            norm *= float(weight[base + i]);
          if (use_bias)
            norm += float(bias[base + i]);
          out[base + i] = T(norm);
        }
      }
    }
  }

  if (tid == 0) {
    meanOut[tg_id] = T(mean);
    rstdTensor[tg_id] = T(inv_std);
  }
}

#define instantiate_layer_norm_single_row(DTYPE)                          \
  template [[host_name("layer_norm_single_row_" #DTYPE)]] [[kernel]] void \
  layer_norm_single_row<DTYPE>(                                           \
      device DTYPE * input [[buffer(0)]],                                 \
      device DTYPE * output [[buffer(1)]],                                \
      device DTYPE * meanOut [[buffer(2)]],                               \
      device DTYPE * rstdTensor [[buffer(3)]],                            \
      constant uint & axis_size [[buffer(4)]],                            \
      constant float& epsilon [[buffer(5)]],                              \
      constant int& use_weight [[buffer(6)]],                             \
      constant int& use_bias [[buffer(7)]],                               \
      device DTYPE* weight [[buffer(8)]],                                 \
      device DTYPE* bias [[buffer(9)]],                                   \
      uint tg_id [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_position_in_threadgroup]],                        \
      uint tptg [[threads_per_threadgroup]]);

#define instantiate_layer_norm_looped(DTYPE)                          \
  template [[host_name("layer_norm_looped_" #DTYPE)]] [[kernel]] void \
  layer_norm_looped<DTYPE>(                                           \
      device DTYPE * input [[buffer(0)]],                             \
      device DTYPE * output [[buffer(1)]],                            \
      device DTYPE * meanOut [[buffer(2)]],                           \
      device DTYPE * rstdTensor [[buffer(3)]],                        \
      constant uint & axis_size [[buffer(4)]],                        \
      constant float& epsilon [[buffer(5)]],                          \
      constant int& use_weight [[buffer(6)]],                         \
      constant int& use_bias [[buffer(7)]],                           \
      device DTYPE* weight [[buffer(8)]],                             \
      device DTYPE* bias [[buffer(9)]],                               \
      uint tg_id [[threadgroup_position_in_grid]],                    \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_layer_norm(DTYPE) \
  instantiate_layer_norm_single_row(DTYPE) instantiate_layer_norm_looped(DTYPE)

instantiate_layer_norm(float);
instantiate_layer_norm(half);
instantiate_layer_norm(bfloat);
// =============================================================
//  Backward: grad_input (dx)
//  Template parameter HAS_WEIGHT eliminates the runtime branch
//  and the use_weight buffer binding.
//  Two passes:
//    1. reduce sb = sum_j(gamma_j*dy_j) and ss = sum_j(gamma_j*dy_j*xhat_j)
//    2. dx_i = (rstd/N) * (N*gamma_i*dy_i - sb - xhat_i*ss)
//  In the fast path (base + N_READS <= N) xi, dyi, wi are cached in
//  registers to avoid a second round-trip to global memory.
// =============================================================
template <typename T, bool HAS_WEIGHT>
kernel void layer_norm_bwd_dx_single_row(
    device const T* dy [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const T* mean [[buffer(2)]],
    device const T* rstd [[buffer(3)]],
    device const T* weight [[buffer(4)]],
    device T* dx [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;

  uint row_off = tg_id * N;
  float mu = float(mean[tg_id]);
  float rs = float(rstd[tg_id]);

  float sb = 0.0f, ss = 0.0f;
  uint base = tid * N_READS;
  float dyi_r[N_READS], xi_r[N_READS], wi_r[N_READS];

  if (base + N_READS <= N) {
    for (int i = 0; i < N_READS; i++) {
      uint idx = base + i;
      dyi_r[i] = float(dy[row_off + idx]);
      xi_r[i] = float(x[row_off + idx]);
      wi_r[i] = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
      float xh = (xi_r[i] - mu) * rs;
      float wdy = wi_r[i] * dyi_r[i];
      sb += wdy;
      ss += wdy * xh;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      uint idx = base + i;
      if (idx < N) {
        dyi_r[i] = float(dy[row_off + idx]);
        xi_r[i] = float(x[row_off + idx]);
        wi_r[i] = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
        float xh = (xi_r[i] - mu) * rs;
        float wdy = wi_r[i] * dyi_r[i];
        sb += wdy;
        ss += wdy * xh;
      } else {
        dyi_r[i] = xi_r[i] = wi_r[i] = 0.0f;
      }
    }
  }

  threadgroup float shared_sb[simdgroup_size];
  threadgroup float shared_ss[simdgroup_size];
  sb = c10::metal::threadgroup_sum(shared_sb, sb, tid, tptg);
  ss = c10::metal::threadgroup_sum(shared_ss, ss, tid, tptg);

  float c = rs / float(N);
  float fN = float(N);
  if (base + N_READS <= N) {
    for (int i = 0; i < N_READS; i++) {
      uint idx = base + i;
      float xh = (xi_r[i] - mu) * rs;
      dx[row_off + idx] = T(c * (fN * wi_r[i] * dyi_r[i] - sb - xh * ss));
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      uint idx = base + i;
      if (idx < N) {
        float xh = (xi_r[i] - mu) * rs;
        dx[row_off + idx] = T(c * (fN * wi_r[i] * dyi_r[i] - sb - xh * ss));
      }
    }
  }
}

template <typename T, bool HAS_WEIGHT>
kernel void layer_norm_bwd_dx_looped(
    device const T* dy [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const T* mean [[buffer(2)]],
    device const T* rstd [[buffer(3)]],
    device const T* weight [[buffer(4)]],
    device T* dx [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  constexpr int MAX_CACHED = 8;

  uint row_off = tg_id * N;
  float mu = float(mean[tg_id]);
  float rs = float(rstd[tg_id]);

  // Compute elements this thread will process
  uint elems_per_thread =
      ((N + lsize * N_READS - 1) / (lsize * N_READS)) * N_READS;
  bool use_cache = (elems_per_thread <= MAX_CACHED);

  float cached_wdy[MAX_CACHED];
  float cached_xhat[MAX_CACHED];

  float sb = 0.0f, ss = 0.0f;
  uint ci = 0;
  for (uint r = 0; r < N; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= N) {
      for (int i = 0; i < N_READS; i++) {
        uint idx = base + i;
        float dyi = float(dy[row_off + idx]);
        float xi = float(x[row_off + idx]);
        float wi = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
        float xh = (xi - mu) * rs;
        float wdy = wi * dyi;
        if (use_cache) {
          cached_wdy[ci] = wdy;
          cached_xhat[ci] = xh;
          ci++;
        }
        sb += wdy;
        ss += wdy * xh;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        uint idx = base + i;
        if (idx < N) {
          float dyi = float(dy[row_off + idx]);
          float xi = float(x[row_off + idx]);
          float wi = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
          float xh = (xi - mu) * rs;
          float wdy = wi * dyi;
          if (use_cache) {
            cached_wdy[ci] = wdy;
            cached_xhat[ci] = xh;
            ci++;
          }
          sb += wdy;
          ss += wdy * xh;
        }
      }
    }
  }
  uint total_cached = ci;

  threadgroup float shared_sb[simdgroup_size];
  threadgroup float shared_ss[simdgroup_size];
  sb = c10::metal::threadgroup_sum(shared_sb, sb, tid, lsize);
  ss = c10::metal::threadgroup_sum(shared_ss, ss, tid, lsize);

  // Pass 2: compute dx
  float c = rs / float(N);
  float fN = float(N);
  if (use_cache) {
    // Fast path: read from cached wdy/xhat (no global re-read)
    ci = 0;
    for (uint r = 0; r < N; r += lsize * N_READS) {
      uint base = r + tid * N_READS;
      if (base + N_READS <= N) {
        for (int i = 0; i < N_READS; i++) {
          dx[row_off + base + i] =
              T(c * (fN * cached_wdy[ci] - sb - cached_xhat[ci] * ss));
          ci++;
        }
      } else {
        for (int i = 0; i < N_READS; i++) {
          if (base + i < N) {
            dx[row_off + base + i] =
                T(c * (fN * cached_wdy[ci] - sb - cached_xhat[ci] * ss));
            ci++;
          }
        }
      }
    }
  } else {
    // Fallback for very large N: re-read from global memory
    for (uint r = 0; r < N; r += lsize * N_READS) {
      uint base = r + tid * N_READS;
      if (base + N_READS <= N) {
        for (int i = 0; i < N_READS; i++) {
          uint idx = base + i;
          float dyi = float(dy[row_off + idx]);
          float xi = float(x[row_off + idx]);
          float wi = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
          float xh = (xi - mu) * rs;
          dx[row_off + idx] = T(c * (fN * wi * dyi - sb - xh * ss));
        }
      } else {
        for (int i = 0; i < N_READS; i++) {
          uint idx = base + i;
          if (idx < N) {
            float dyi = float(dy[row_off + idx]);
            float xi = float(x[row_off + idx]);
            float wi = HAS_WEIGHT ? float(weight[idx]) : 1.0f;
            float xh = (xi - mu) * rs;
            dx[row_off + idx] = T(c * (fN * wi * dyi - sb - xh * ss));
          }
        }
      }
    }
  }
}



#define instantiate_layer_norm_bwd_dx_single_row(DTYPE, HW, SFXW) \
  template [[host_name("layer_norm_bwd_dx_single_row_" #DTYPE     \
                       "_" #SFXW)]] [[kernel]] void               \
  layer_norm_bwd_dx_single_row<DTYPE, HW>(                        \
      device const DTYPE* dy [[buffer(0)]],                       \
      device const DTYPE* x [[buffer(1)]],                        \
      device const DTYPE* mean [[buffer(2)]],                     \
      device const DTYPE* rstd [[buffer(3)]],                     \
      device const DTYPE* weight [[buffer(4)]],                   \
      device DTYPE* dx [[buffer(5)]],                             \
      constant uint& N [[buffer(6)]],                             \
      uint tg_id [[threadgroup_position_in_grid]],                \
      uint tid [[thread_position_in_threadgroup]],                \
      uint tptg [[threads_per_threadgroup]]);

#define instantiate_layer_norm_bwd_dx_looped(DTYPE, HW, SFXW) \
  template [[host_name("layer_norm_bwd_dx_looped_" #DTYPE     \
                       "_" #SFXW)]] [[kernel]] void           \
  layer_norm_bwd_dx_looped<DTYPE, HW>(                        \
      device const DTYPE* dy [[buffer(0)]],                   \
      device const DTYPE* x [[buffer(1)]],                    \
      device const DTYPE* mean [[buffer(2)]],                 \
      device const DTYPE* rstd [[buffer(3)]],                 \
      device const DTYPE* weight [[buffer(4)]],               \
      device DTYPE* dx [[buffer(5)]],                         \
      constant uint& N [[buffer(6)]],                         \
      uint tg_id [[threadgroup_position_in_grid]],            \
      uint tid [[thread_position_in_threadgroup]],            \
      uint lsize [[threads_per_threadgroup]]);

#define instantiate_layer_norm_bwd_dw_db(DTYPE)                          \
  template [[host_name("layer_norm_bwd_dw_db_" #DTYPE)]] [[kernel]] void \
  layer_norm_bwd_dw_db<DTYPE>(                                           \
      device const DTYPE* dy [[buffer(0)]],                              \
      device const DTYPE* x [[buffer(1)]],                               \
      device const DTYPE* mean [[buffer(2)]],                            \
      device const DTYPE* rstd [[buffer(3)]],                            \
      device DTYPE* dgamma [[buffer(4)]],                                \
      device DTYPE* dbeta [[buffer(5)]],                                 \
      constant uint& N [[buffer(6)]],                                    \
      constant uint& M [[buffer(7)]],                                    \
      uint tg_id [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint tptg [[threads_per_threadgroup]]);

#define instantiate_layer_norm_bwd_dw_db_coalesced(DTYPE)                          \
  template [[host_name("layer_norm_bwd_dw_db_coalesced_" #DTYPE)]] [[kernel]] void \
  layer_norm_bwd_dw_db_coalesced<DTYPE>(                                           \
      device const DTYPE* dy [[buffer(0)]],                              \
      device const DTYPE* x [[buffer(1)]],                               \
      device const DTYPE* mean [[buffer(2)]],                            \
      device const DTYPE* rstd [[buffer(3)]],                            \
      device DTYPE* dgamma [[buffer(4)]],                                \
      device DTYPE* dbeta [[buffer(5)]],                                 \
      constant uint& N [[buffer(6)]],                                    \
      constant uint& M [[buffer(7)]],                                    \
      uint tg_id [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                       \
      uint tptg [[threads_per_threadgroup]]);

#define instantiate_layer_norm_bwd(DTYPE)                              instantiate_layer_norm_bwd_dx_single_row(DTYPE, true, w)                 instantiate_layer_norm_bwd_dx_single_row(DTYPE, false, nw)               instantiate_layer_norm_bwd_dx_looped(DTYPE, true, w)                     instantiate_layer_norm_bwd_dx_looped(DTYPE, false, nw)


instantiate_layer_norm_bwd(float);
instantiate_layer_norm_bwd(half);
instantiate_layer_norm_bwd(bfloat);

// =============================================================
//  M=1 specialization: dgamma[j] = dy[j] * xhat[j], dbeta[j] = dy[j]
//  No reduction needed — one row, one element per column.
// =============================================================
template <typename T>
kernel void layer_norm_bwd_dw_db_m1(
    device const T* dy [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const T* mean [[buffer(2)]],
    device const T* rstd [[buffer(3)]],
    device T* dgamma [[buffer(4)]],
    device T* dbeta [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= N) return;
  float mu = float(mean[0]);
  float rs = float(rstd[0]);
  float dyi = float(dy[gid]);
  float xi = float(x[gid]);
  float xh = (xi - mu) * rs;
  dgamma[gid] = T(dyi * xh);
  dbeta[gid] = T(dyi);
}

#define instantiate_layer_norm_bwd_dw_db_m1(DTYPE)                                \
  template [[host_name("layer_norm_bwd_dw_db_m1_" #DTYPE)]] [[kernel]] void       \
  layer_norm_bwd_dw_db_m1<DTYPE>(                                                  \
      device const DTYPE* dy [[buffer(0)]],                                        \
      device const DTYPE* x [[buffer(1)]],                                         \
      device const DTYPE* mean [[buffer(2)]],                                      \
      device const DTYPE* rstd [[buffer(3)]],                                      \
      device DTYPE* dgamma [[buffer(4)]],                                          \
      device DTYPE* dbeta [[buffer(5)]],                                           \
      constant uint& N [[buffer(6)]],                                              \
      uint gid [[thread_position_in_grid]]);

instantiate_layer_norm_bwd_dw_db_m1(float);
instantiate_layer_norm_bwd_dw_db_m1(half);
instantiate_layer_norm_bwd_dw_db_m1(bfloat);

// =============================================================
//  Tiled dw_db: two-phase column reduction
//  Phase 1 (layer_norm_bwd_dw_db_tiled): each threadgroup handles
//    a TILE_M × TILE_N block. Threads cooperatively reduce TILE_M
//    rows for TILE_N columns, writing partial sums to a temp buffer
//    of shape [num_row_blocks, N].
//  Phase 2 (layer_norm_bwd_dw_db_reduce): each thread sums one
//    column across num_row_blocks partials.
// =============================================================

constant constexpr uint TILE_M = 64;
constant constexpr uint TILE_N = 32;

template <typename T>
kernel void layer_norm_bwd_dw_db_tiled(
    device const T* dy         [[buffer(0)]],
    device const T* x          [[buffer(1)]],
    device const T* mean       [[buffer(2)]],
    device const T* rstd       [[buffer(3)]],
    device float* partial_dgam [[buffer(4)]],
    device float* partial_dbet [[buffer(5)]],
    constant uint& N           [[buffer(6)]],
    constant uint& M           [[buffer(7)]],
    uint2 tg_id  [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint2 tg_dim [[threadgroups_per_grid]]) {
  // tg_id.x = row-block index, tg_id.y = col-block index
  uint row_base = tg_id.x * TILE_M;
  uint col_base = tg_id.y * TILE_N;
  uint num_row_blocks = tg_dim.x;

  // Thread layout: TILE_N threads (one per column in the tile)
  uint col_local = tid;  // 0..TILE_N-1
  uint col = col_base + col_local;

  if (col >= N) return;

  float dgam = 0.0f;
  float dbet = 0.0f;

  // Each thread iterates over TILE_M rows (or fewer at the boundary)
  uint row_end = min(row_base + TILE_M, M);
  for (uint m = row_base; m < row_end; m++) {
    float mu = float(mean[m]);
    float rs = float(rstd[m]);
    float dyi = float(dy[m * N + col]);
    float xi  = float(x[m * N + col]);
    float xh  = (xi - mu) * rs;
    dgam += dyi * xh;
    dbet += dyi;
  }

  // Write partial to temp buffer: [row_block, col]
  uint partial_idx = tg_id.x * N + col;
  partial_dgam[partial_idx] = dgam;
  partial_dbet[partial_idx] = dbet;
}

template <typename T>
kernel void layer_norm_bwd_dw_db_reduce(
    device const float* partial_dgam [[buffer(0)]],
    device const float* partial_dbet [[buffer(1)]],
    device T* dgamma               [[buffer(2)]],
    device T* dbeta                [[buffer(3)]],
    constant uint& N               [[buffer(4)]],
    constant uint& num_row_blocks  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= N) return;
  float dgam = 0.0f;
  float dbet = 0.0f;
  for (uint rb = 0; rb < num_row_blocks; rb++) {
    dgam += partial_dgam[rb * N + gid];
    dbet += partial_dbet[rb * N + gid];
  }
  dgamma[gid] = T(dgam);
  dbeta[gid] = T(dbet);
}

#define instantiate_layer_norm_bwd_dw_db_tiled(DTYPE)                              \
  template [[host_name("layer_norm_bwd_dw_db_tiled_" #DTYPE)]] [[kernel]] void     \
  layer_norm_bwd_dw_db_tiled<DTYPE>(                                               \
      device const DTYPE* dy         [[buffer(0)]],                                \
      device const DTYPE* x          [[buffer(1)]],                                \
      device const DTYPE* mean       [[buffer(2)]],                                \
      device const DTYPE* rstd       [[buffer(3)]],                                \
      device float* partial_dgam     [[buffer(4)]],                                \
      device float* partial_dbet     [[buffer(5)]],                                \
      constant uint& N               [[buffer(6)]],                                \
      constant uint& M               [[buffer(7)]],                                \
      uint2 tg_id  [[threadgroup_position_in_grid]],                               \
      uint  tid    [[thread_index_in_threadgroup]],                                \
      uint2 tg_dim [[threadgroups_per_grid]]);

#define instantiate_layer_norm_bwd_dw_db_reduce(DTYPE)                             \
  template [[host_name("layer_norm_bwd_dw_db_reduce_" #DTYPE)]] [[kernel]] void    \
  layer_norm_bwd_dw_db_reduce<DTYPE>(                                              \
      device const float* partial_dgam [[buffer(0)]],                              \
      device const float* partial_dbet [[buffer(1)]],                              \
      device DTYPE* dgamma             [[buffer(2)]],                              \
      device DTYPE* dbeta              [[buffer(3)]],                              \
      constant uint& N                 [[buffer(4)]],                              \
      constant uint& num_row_blocks    [[buffer(5)]],                              \
      uint gid [[thread_position_in_grid]]);

instantiate_layer_norm_bwd_dw_db_tiled(float);
instantiate_layer_norm_bwd_dw_db_tiled(half);
instantiate_layer_norm_bwd_dw_db_tiled(bfloat);
instantiate_layer_norm_bwd_dw_db_reduce(float);
instantiate_layer_norm_bwd_dw_db_reduce(half);
instantiate_layer_norm_bwd_dw_db_reduce(bfloat);
