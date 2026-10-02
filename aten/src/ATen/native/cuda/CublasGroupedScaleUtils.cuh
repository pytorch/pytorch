#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

namespace at::native {

enum class CublasGroupedScaleLayout : uint8_t {
  Scalar,
  PerBatchScalar,
  Vec16UE4M3,
};

C10_HOST_DEVICE inline int64_t cublas_grouped_ceil_div(
    int64_t value,
    int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

C10_HOST_DEVICE inline int64_t cublas_grouped_round_up(
    int64_t value,
    int64_t multiple) {
  return cublas_grouped_ceil_div(value, multiple) * multiple;
}

C10_HOST_DEVICE inline bool cublas_grouped_scale_uses_pointer_array(
    CublasGroupedScaleLayout layout) {
  return layout != CublasGroupedScaleLayout::Scalar;
}

C10_HOST_DEVICE inline bool cublas_grouped_scale_is_blockwise(
    CublasGroupedScaleLayout layout) {
  return layout != CublasGroupedScaleLayout::Scalar &&
      layout != CublasGroupedScaleLayout::PerBatchScalar;
}

C10_HOST_DEVICE inline int64_t cublas_grouped_scale_size_bytes(
    CublasGroupedScaleLayout layout,
    int64_t inner,
    int64_t outer) {
  switch (layout) {
    case CublasGroupedScaleLayout::Scalar:
      return 0;
    case CublasGroupedScaleLayout::PerBatchScalar:
      return sizeof(float);
    case CublasGroupedScaleLayout::Vec16UE4M3:
      return cublas_grouped_round_up(outer, 128) *
          cublas_grouped_round_up(cublas_grouped_ceil_div(inner, 16), 4);
  }
  return 0;
}

} // namespace at::native
