#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

namespace at::native {

enum class CublasGroupedScaleLayout : uint8_t {
  Scalar,
  PerBatchScalar,
};

C10_HOST_DEVICE inline bool cublas_grouped_scale_uses_pointer_array(
    CublasGroupedScaleLayout layout) {
  return layout != CublasGroupedScaleLayout::Scalar;
}


} // namespace at::native
