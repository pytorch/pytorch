#include <metal_stdlib>
using namespace metal;

constant constexpr unsigned kmaxThreadGroups = 32;
constant constexpr unsigned kmaxTensors = 32;
constant constexpr unsigned kChunkSize = 65536;

template <typename T>
struct AmpNonFiniteCheckAndUnscaleArgs {
  metal::array<device T*, kmaxTensors> data [[id(0)]];
};

struct MetadataArguments {
  ulong numels[kmaxTensors];
  ulong threadgroup_to_tensor[kmaxThreadGroups];
  ulong threadgroup_to_chunk[kmaxThreadGroups];
};

template <typename T>
kernel void ampNonFiniteCheckAndUnscale(
    constant AmpNonFiniteCheckAndUnscaleArgs<T>& pointerArgs [[buffer(0)]],
    constant MetadataArguments& metadata [[buffer(1)]],
    device float& foundInf [[buffer(2)]],
    constant T& invScale [[buffer(3)]],
    uint local_tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  uint threadGroupSize = tgSize;
  uint tensor_index = metadata.threadgroup_to_tensor[group_id];
  uint chunk = metadata.threadgroup_to_chunk[group_id];
  uint numel = metadata.numels[tensor_index];

  uint offset = chunk * kChunkSize;
  uint chunk_size =
      ((offset + kChunkSize) > numel) ? (numel - offset) : kChunkSize;

  device T* data = pointerArgs.data[tensor_index];

  for (uint i = local_tid; i < chunk_size; i += threadGroupSize) {
    uint index = offset + i;
    T val = data[index];
    if (!isfinite(val)) {
      foundInf = 1.0f;
    }
    data[index] = (invScale == static_cast<T>(1.0) ? val : val * invScale);
  }
}

template <typename T>
kernel void ampNonFiniteCheckAndUnscaleSingle(
    device T* data [[buffer(0)]],
    device float& foundInf [[buffer(1)]],
    constant T& invScale [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  T val = data[tid];
  if (!isfinite(val)) {
    foundInf = 1.0f;
  }
  data[tid] = (invScale == T(1.0) ? val : val * invScale);
}

template <typename T>
kernel void ampUpdateScale(
    device T& scale [[buffer(0)]],
    device int& growth_tracker [[buffer(1)]],
    device float& foundInf [[buffer(2)]],
    constant T& scaleGrowthFactor [[buffer(3)]],
    constant T& scaleBackoffFactor [[buffer(4)]],
    constant int& growthInterval [[buffer(5)]]) {
  if (foundInf != 0.0f) {
    scale *= scaleBackoffFactor;
    growth_tracker = 0;
  } else {
    int g = growth_tracker + 1;
    if (g >= growthInterval) {
      scale *= scaleGrowthFactor;
      g = 0;
    }
    growth_tracker = g;
  }
}

#define INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(DTYPE)                  \
  template [[host_name("ampNonFiniteCheckAndUnscale_" #DTYPE)]] kernel void \
  ampNonFiniteCheckAndUnscale<DTYPE>(                                       \
      constant AmpNonFiniteCheckAndUnscaleArgs<DTYPE> &                     \
          pointerArgs [[buffer(0)]],                                        \
      constant MetadataArguments & metadata [[buffer(1)]],                  \
      device float& foundInf [[buffer(2)]],                                 \
      constant DTYPE& invScale [[buffer(3)]],                               \
      uint local_tid [[thread_position_in_threadgroup]],                    \
      uint tgSize [[threads_per_threadgroup]],                              \
      uint group_id [[threadgroup_position_in_grid]])

#define INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE_SINGLE(DTYPE)            \
  template                                                                   \
      [[host_name("ampNonFiniteCheckAndUnscaleSingle_" #DTYPE)]] kernel void \
      ampNonFiniteCheckAndUnscaleSingle<DTYPE>(                              \
          device DTYPE * data [[buffer(0)]],                                 \
          device float& foundInf [[buffer(1)]],                              \
          constant DTYPE& invScale [[buffer(2)]],                            \
          uint tid [[thread_position_in_grid]])

#define INSTANTIATE_AMP_UPDATE_SCALE(DTYPE)                    \
  template [[host_name("ampUpdateScale_" #DTYPE)]] kernel void \
  ampUpdateScale<DTYPE>(                                       \
      device DTYPE & scale [[buffer(0)]],                      \
      device int& growth_tracker [[buffer(1)]],                \
      device float& foundInf [[buffer(2)]],                    \
      constant DTYPE& scaleGrowthFactor [[buffer(3)]],         \
      constant DTYPE& scaleBackoffFactor [[buffer(4)]],        \
      constant int& growthInterval [[buffer(5)]])

INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(float);
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(half);
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE(bfloat);

INSTANTIATE_AMP_UPDATE_SCALE(float);
INSTANTIATE_AMP_UPDATE_SCALE(half);
INSTANTIATE_AMP_UPDATE_SCALE(bfloat);

INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE_SINGLE(float);
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE_SINGLE(half);
INSTANTIATE_AMP_NONFINITE_CHECK_AND_UNSCALE_SINGLE(bfloat);
