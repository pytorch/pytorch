#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

#include <ATen/native/mps/kernels/NormBackward.h>

template <typename T>
struct WeightNormOp {
  using opmath = opmath_t<T>;
  using extra_t = T;
  using acc_t = opmath;
  opmath g, norm;
  device T* grad_g;
  opmath grad(opmath dy, uint) const {
    return dy;
  }
  opmath dir(opmath x) const {
    return x;
  }
  METAL_FUNC NormBackwardCoeffs<T> coeffs(
      thread const NormBackwardRow<T>& row) const {
    const opmath rnorm = 1 / norm;
    const opmath dot =
        threadgroup_sum(row.tg_b, row.sum_prod, row.tid, row.lsize);
    if (row.tid == 0) {
      grad_g[row.index] = static_cast<T>(dot * rnorm);
    }
    return {g * rnorm, 0, dot * rnorm * rnorm};
  }
  static opmath column_term(
      opmath dy,
      opmath x,
      device const T*,
      device const T*,
      uint) {
    return dy * x;
  }
};

kernel void weight_norm_backward_cols_sum(
    device float* part [[buffer(0)]],
    constant uint2& splits_cols [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
  float acc = 0;
  for (uint p = 0; p < splits_cols.x; ++p) {
    acc += part[p * splits_cols.y + col];
  }
  part[col] = acc;
}

template <typename T, typename Idx>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void weight_norm_backward_cols(
    device const T* grad_w [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device const T* g [[buffer(2)]],
    device const T* norms [[buffer(3)]],
    device T* grad_v [[buffer(4)]],
    device T* grad_g [[buffer(5)]],
    device const opmath_t<T>* dots [[buffer(6)]],
    constant uint2& rows_cols [[buffer(7)]],
    constant uint& rows_per_tile [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  using opmath = opmath_t<T>;
  const uint col = gid.x, cols = rows_cols.y;
  const uint first = tg.y * rows_per_tile + lid.y;
  const uint last =
      uint(min(ulong(rows_cols.x), ulong(tg.y + 1) * rows_per_tile));
  const opmath rnorm = 1 / opmath(norms[col]);
  const opmath scale = opmath(g[col]) * rnorm;
  const opmath slope = dots[col] * rnorm * rnorm;
  if (tg.y == 0 && lid.y == 0) {
    grad_g[col] = static_cast<T>(dots[col] * rnorm);
  }
  for (uint row = first; row < last; row += lsize.y) {
    const Idx idx = col + Idx(row) * cols;
    grad_v[idx] =
        static_cast<T>(scale * (opmath(grad_w[idx]) - opmath(v[idx]) * slope));
  }
}

#define REGISTER_WEIGHT_NORM_BACKWARD(T)                                    \
  template [[host_name("weight_norm_backward_single_row_" #T)]] [[kernel]]  \
  decltype(norm_backward_single_row<T, WeightNormOp<T>>)                    \
      norm_backward_single_row<T, WeightNormOp<T>>;                         \
  template [[host_name("weight_norm_backward_looped_" #T)]] [[kernel]]      \
  decltype(norm_backward_looped<T, WeightNormOp<T>>)                        \
      norm_backward_looped<T, WeightNormOp<T>>;                             \
  template [[host_name("weight_norm_backward_cols_dot_" #T)]] [[kernel]]    \
  decltype(norm_backward_column_partials<T, WeightNormOp<T>>)               \
      norm_backward_column_partials<T, WeightNormOp<T>>;                    \
  template [[host_name("weight_norm_backward_cols_" #T "_u32")]] [[kernel]] \
  decltype(weight_norm_backward_cols<T, uint>)                              \
      weight_norm_backward_cols<T, uint>;                                   \
  template [[host_name("weight_norm_backward_cols_" #T "_u64")]] [[kernel]] \
  decltype(weight_norm_backward_cols<T, ulong>)                             \
      weight_norm_backward_cols<T, ulong>

REGISTER_WEIGHT_NORM_BACKWARD(float);
REGISTER_WEIGHT_NORM_BACKWARD(half);
REGISTER_WEIGHT_NORM_BACKWARD(bfloat);
