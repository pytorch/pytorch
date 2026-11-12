#pragma once

#include <ATen/cuda/CUDABlas.h>

#include <c10/macros/Macros.h>

#include <cstdint>

namespace at::native {

using at::cuda::blas::CublasGroupedScaleLayout;

C10_HOST_DEVICE inline bool cublas_grouped_scale_uses_pointer_array(
    CublasGroupedScaleLayout layout) {
  return layout != CublasGroupedScaleLayout::Scalar;
}

} // namespace at::native
