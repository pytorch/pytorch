#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void ampNonFiniteCheckAndUnscale(
    device T* data [[buffer(0)]],
    device T* foundInf [[buffer(1)]],
    constant T& invScale [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= numel) {
    return;
  }
  T val = data[tid];

  if (!isfinite(val)) {
    device atomic_float* flag = (device atomic_float*)foundInf;
    atomic_store_explicit(flag, 1, memory_order_relaxed);
  }

  data[tid] = (invScale == 1.0f ? val : val * invScale);
}

template <typename T>
kernel void ampUpdateScale(
    device T* scale [[buffer(0)]],
    device int* growth_tracker [[buffer(1)]],
    device T* foundInf [[buffer(2)]],
    constant T& scaleGrowthFactor [[buffer(3)]],
    constant T& scaleBackoffFactor [[buffer(4)]],
    constant uint& growthInterval [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  int flag = (int)(foundInf[0]);

  if (flag != 0) {
    *scale = *scale * scaleBackoffFactor;
    *growth_tracker = 0;
  } else {
    int g = *growth_tracker;
    g += 1;
    if (uint(g) >= growthInterval) {
      *scale = *scale * scaleGrowthFactor;
      g = 0;
    }
    *growth_tracker = g;
  }
}

#define INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(DTYPE)                  \
  template [[host_name("ampNonFiniteCheckAndUnscale_" #DTYPE)]] kernel void \
  ampNonFiniteCheckAndUnscale<DTYPE>(                                       \
      device DTYPE * data [[buffer(0)]],                                    \
      device DTYPE * foundInf [[buffer(1)]],                                \
      constant DTYPE & invScale [[buffer(2)]],                              \
      constant uint & numel [[buffer(3)]],                                  \
      uint tid [[thread_position_in_grid]])

#define INSTANTIATE_AMP_UPDATE_SCALE(DTYPE)                    \
  template [[host_name("ampUpdateScale_" #DTYPE)]] kernel void \
  ampUpdateScale<DTYPE>(                                       \
      device DTYPE * scale [[buffer(0)]],                      \
      device int* growth_tracker [[buffer(1)]],                \
      device DTYPE* foundInf [[buffer(2)]],                    \
      constant DTYPE& scaleGrowthFactor [[buffer(3)]],         \
      constant DTYPE& scaleBackoffFactor [[buffer(4)]],        \
      constant uint& growthInterval [[buffer(5)]],             \
      uint tid [[thread_position_in_grid]])

INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(float);
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(bfloat);
#endif

INSTANTIATE_AMP_UPDATE_SCALE(float);
INSTANTIATE_AMP_UPDATE_SCALE(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_AMP_UPDATE_SCALE(bfloat);
#endif
