#include <ATen/native/mps/kernels/Distance.h>
#include <metal_stdlib>
using namespace metal;

inline uint row_start(uint i, uint n) {
  return n * i - i * (i + 1) / 2;
}

inline uint2 pair_from_condensed_index(uint k, uint n) {
  float n2 = static_cast<float>(n) - 0.5f;
  uint i = static_cast<uint>(
      n2 - sqrt(n2 * n2 - 2.0f * static_cast<float>(k) - 1.0f));
  if (i >= n) {
    i = n - 1;
  }

  while (i > 0 && k < row_start(i, n)) {
    i--;
  }
  while ((i + 1) < n && k >= row_start(i + 1, n)) {
    i++;
  }

  uint j = k - row_start(i, n) + i + 1;
  return uint2(i, j);
}

inline float signf(float v) {
  return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
}

inline float forward_distance_update(
    float agg,
    float diff_abs,
    float p,
    PdistMode mode) {
  if (mode == PdistMode::MODE_ZERO) {
    return agg + min(ceil(diff_abs), 1.0f);
  }
  if (mode == PdistMode::MODE_ONE) {
    return agg + diff_abs;
  }
  if (mode == PdistMode::MODE_TWO) {
    return agg + diff_abs * diff_abs;
  }
  if (mode == PdistMode::MODE_INF) {
    return max(agg, diff_abs);
  }
  return agg + pow(diff_abs, p);
}

inline float forward_distance_finish(float agg, float p, PdistMode mode) {
  if (mode == PdistMode::MODE_TWO) {
    return sqrt(agg);
  }
  if (mode == PdistMode::MODE_GENERAL) {
    return pow(agg, 1.0f / p);
  }
  return agg;
}

inline float backward_value(
    float diff,
    float grad,
    float dist,
    float p,
    PdistMode mode) {
  if (dist == 0.0f) {
    return 0.0f;
  }

  float diff_abs = fabs(diff);
  float diff_sign = signf(diff);

  if (mode == PdistMode::MODE_ONE) {
    return grad * diff_sign;
  }
  if (mode == PdistMode::MODE_LT_TWO) {
    if (diff == 0.0f && p < 1.0f) {
      return 0.0f;
    }
    return diff_sign * pow(diff_abs, p - 1.0f) * grad / pow(dist, p - 1.0f);
  }
  if (mode == PdistMode::MODE_TWO) {
    return grad * diff / dist;
  }
  if (mode == PdistMode::MODE_INF) {
    return grad * diff_sign * (1.0f - min(1.0f, ceil(fabs(diff_abs - dist))));
  }
  return diff * pow(diff_abs, p - 2.0f) * grad / pow(dist, p - 1.0f);
}

template <typename T>
kernel void pdist_forward_kernel(
    device T* result [[buffer(0)]],
    constant T* self [[buffer(1)]],
    constant PdistForwardParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint n = static_cast<uint>(params.n);
  const uint m = static_cast<uint>(params.m);
  const float p = params.p;
  const PdistMode mode_enum = static_cast<PdistMode>(params.mode);
  uint2 pair = pair_from_condensed_index(gid, n);
  uint i = pair.x;
  uint j = pair.y;

  float agg = 0.0f;
  for (uint x = 0; x < m; ++x) {
    float a = static_cast<float>(self[i * m + x]);
    float b = static_cast<float>(self[j * m + x]);
    agg = forward_distance_update(agg, fabs(a - b), p, mode_enum);
  }

  result[gid] = static_cast<T>(forward_distance_finish(agg, p, mode_enum));
}

template <typename T>
kernel void pdist_backward_kernel(
    device T* buffer [[buffer(0)]],
    constant T* grad [[buffer(1)]],
    constant T* self [[buffer(2)]],
    constant T* dist [[buffer(3)]],
    constant PdistBackwardParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint grad_stride = static_cast<uint>(params.grad_stride);
  const uint n = static_cast<uint>(params.n);
  const uint m = static_cast<uint>(params.m);
  const float p = params.p;
  const PdistMode mode_enum = static_cast<PdistMode>(params.mode);
  uint k = gid / m;
  uint x = gid % m;

  uint2 pair = pair_from_condensed_index(k, n);
  uint i = pair.x;
  uint j = pair.y;
  uint ib = j - i - 1;
  uint jb = static_cast<uint>(n - 2) - i;

  float grad_k = static_cast<float>(grad[k * grad_stride]);
  float dist_k = static_cast<float>(dist[k]);
  float a = static_cast<float>(self[i * m + x]);
  float b = static_cast<float>(self[j * m + x]);

  float res = backward_value(a - b, grad_k, dist_k, p, mode_enum);

  uint lhs = ((ib * n + i) * m) + x;
  uint rhs = ((jb * n + j) * m) + x;
  buffer[lhs] = static_cast<T>(res);
  buffer[rhs] = static_cast<T>(-res);
}

#define REGISTER_PDIST_FORWARD_OP(DTYPE)                      \
  template [[host_name("pdist_forward_" #DTYPE)]] kernel void \
  pdist_forward_kernel<DTYPE>(                                \
      device DTYPE * result [[buffer(0)]],                    \
      constant DTYPE * self [[buffer(1)]],                    \
      constant PdistForwardParams & params [[buffer(2)]],     \
      uint gid [[thread_position_in_grid]]);

#define REGISTER_PDIST_BACKWARD_OP(DTYPE)                      \
  template [[host_name("pdist_backward_" #DTYPE)]] kernel void \
  pdist_backward_kernel<DTYPE>(                                \
      device DTYPE * buffer [[buffer(0)]],                     \
      constant DTYPE * grad [[buffer(1)]],                     \
      constant DTYPE * self [[buffer(2)]],                     \
      constant DTYPE * dist [[buffer(3)]],                     \
      constant PdistBackwardParams & params [[buffer(4)]],     \
      uint gid [[thread_position_in_grid]]);

REGISTER_PDIST_FORWARD_OP(float);
REGISTER_PDIST_FORWARD_OP(half);
REGISTER_PDIST_FORWARD_OP(bfloat);

REGISTER_PDIST_BACKWARD_OP(float);
REGISTER_PDIST_BACKWARD_OP(half);
REGISTER_PDIST_BACKWARD_OP(bfloat);
